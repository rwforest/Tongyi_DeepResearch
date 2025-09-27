import json
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union
import requests
from qwen_agent.tools.base import BaseTool, register_tool
from prompt import EXTRACTOR_PROMPT
from openai import OpenAI
import random
from urllib.parse import urlparse, unquote
import time
from transformers import AutoTokenizer
import tiktoken
import mlflow
from mlflow.entities import SpanType

# Import safe MLflow configuration
try:
    from mlflow_config import safe_trace_decorator
    print("📊 MLFLOW: Using safe MLflow tracing for visit")
except ImportError:
    def safe_trace_decorator(name, span_type=None):
        """Fallback decorator"""
        def decorator(func):
            try:
                return mlflow.trace(name=name, span_type=span_type)(func)
            except:
                return func
        return decorator

def get_visit_config():
    """Get visit tool configuration at runtime to ensure .env is loaded"""
    return {
        'timeout': int(os.getenv("VISIT_SERVER_TIMEOUT", 200)),
        'max_length': int(os.getenv("WEBCONTENT_MAXLENGTH", 150000)),
        'jina_keys': os.getenv("JINA_API_KEYS", ""),
        'api_key': os.environ.get("API_KEY"),
        'api_base': os.environ.get("API_BASE"),
        'model_name': os.environ.get("SUMMARY_MODEL_NAME", "")
    }


@staticmethod
def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

OSS_JSON_FORMAT = """# Response Formats
## visit_content
{"properties":{"rational":{"type":"string","description":"Locate the **specific sections/data** directly related to the user's goal within the webpage content"},"evidence":{"type":"string","description":"Identify and extract the **most relevant information** from the content, never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.","summary":{"type":"string","description":"Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal."}}}}"""


@register_tool('visit', allow_overwrite=True)
class Visit(BaseTool):
    # The `description` tells the agent the functionality of this tool.
    name = 'visit'
    description = 'Visit webpage(s) and return the summary of the content.'
    # The `parameters` tell the agent what input parameters the tool has.
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {
                    "type": "string"
                    },
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
        },
        "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s)."
        }
        },
        "required": ["url", "goal"]
    }
    # The `call` method is the main function of the tool.
    @safe_trace_decorator(name="visit_webpage", span_type=SpanType.TOOL)
    def call(self, params: Union[str, dict], predict_function=None, **kwargs) -> str:
        start_time = time.time()
        print(f"🌐 VISIT_CALL: Visit.call(predict_function={'SET' if predict_function else 'MISSING'})")

        # Set MLflow span inputs
        visit_span = mlflow.get_current_active_span()

        try:
            url = params["url"]
            goal = params["goal"]
        except:
            elapsed = time.time() - start_time
            error_msg = "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"
            print(f"🌐 VISIT_ERROR: {error_msg} (⏱️ {elapsed:.2f}s)")

            # Set MLflow span outputs for parameter error
            if visit_span:
                visit_span.set_outputs({
                    "error": error_msg,
                    "success": False,
                    "execution_time": elapsed
                })

            return error_msg

        # Set MLflow span inputs after successful parameter extraction
        if visit_span:
            visit_span.set_inputs({
                "url": url if isinstance(url, str) else url,
                "goal": goal,
                "predict_function_available": predict_function is not None
            })
            visit_span.set_attributes({
                "tool_type": "visit",
                "url_type": "single" if isinstance(url, str) else "multiple"
            })

        # Create log folder if it doesn't exist
        log_folder = "log"
        os.makedirs(log_folder, exist_ok=True)

        if isinstance(url, str):
            print(f"🌐 VISIT_MODE: single_url='{url}'")
            response = self.readpage_jina(url, goal, predict_function)
            elapsed = time.time() - start_time
            print(f"🌐 VISIT_RESULT: single_response_length={len(response)} (⏱️ {elapsed:.2f}s)")
        else:
            print(f"🌐 VISIT_MODE: multiple_urls={len(url)}")
            response = []
            assert isinstance(url, List)
            batch_start_time = time.time()
            for i, u in enumerate(url):
                url_start_time = time.time()
                if time.time() - batch_start_time > 900:
                    print(f"🌐 VISIT_TIMEOUT: batch processing exceeded 900s, aborting remaining URLs")
                    cur_response = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    cur_response += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                    cur_response += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                else:
                    try:
                        print(f"🌐 VISIT_BATCH: processing={i+1}/{len(url)} url='{u}'")
                        cur_response = self.readpage_jina(u, goal, predict_function)
                        url_elapsed = time.time() - url_start_time
                        print(f"🌐 VISIT_BATCH: result={i+1} length={len(cur_response)} (⏱️ {url_elapsed:.2f}s)")
                    except Exception as e:
                        url_elapsed = time.time() - url_start_time
                        cur_response = f"Error fetching {u}: {str(e)}"
                        print(f"🌐 VISIT_BATCH: error={i+1} {type(e).__name__}:{e} (⏱️ {url_elapsed:.2f}s)")
                response.append(cur_response)
            response = "\n=======\n".join(response)
            elapsed = time.time() - start_time
            print(f"🌐 VISIT_RESULT: combined_response_length={len(response)} (⏱️ {elapsed:.2f}s)")

        elapsed = time.time() - start_time
        print(f"🌐 VISIT_RETURN: final_response_length={len(response)} (⏱️ {elapsed:.2f}s)")

        # Set MLflow span outputs for successful completion
        if visit_span:
            visit_span.set_outputs({
                "response": response.strip(),
                "response_length": len(response),
                "success": True,
                "execution_time": elapsed,
                "urls_processed": len(url) if isinstance(url, list) else 1
            })

        return response.strip()
        
    def call_server(self, msgs, max_retries=2, predict_function=None):
        start_time = time.time()
        print(f"🌐 VISIT_API: call_server(predict_function={'SET' if predict_function else 'MISSING'}, msgs_count={len(msgs)})")

        # Use predict function if provided, otherwise fall back to OpenAI API
        if predict_function:
            result = self.call_server_with_predict(msgs, max_retries, predict_function)
        else:
            result = self.call_server_with_openai(msgs, max_retries)

        elapsed = time.time() - start_time
        print(f"🌐 VISIT_API: call_server completed (⏱️ {elapsed:.2f}s)")
        return result

    def call_server_with_predict(self, msgs, max_retries, predict_function):
        """Use predict function for content summarization"""
        start_time = time.time()
        print(f"🌐 VISIT_PREDICT: Using predict function for summarization")

        # Convert messages to prompt format for predict function
        prompt_parts = []
        for msg in msgs:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        # Add final assistant prompt
        prompt_parts.append("Assistant:")
        full_prompt = "\n\n".join(prompt_parts)

        for attempt in range(max_retries):
            try:
                attempt_start = time.time()
                print(f"🌐 VISIT_PREDICT: Attempt {attempt + 1}/{max_retries}")

                # Format input for predict function
                model_input = [{
                    "prompt": full_prompt,
                    "max_length": 2048,
                    "temperature": 0.7,
                    "presence_penalty": 1.1
                }]

                # Call predict function
                content = predict_function(model_input)
                attempt_elapsed = time.time() - attempt_start
                print(f"🌐 VISIT_PREDICT: Response length={len(content)} chars (⏱️ {attempt_elapsed:.2f}s)")

                if content and content.strip():
                    # Try to extract JSON from response
                    try:
                        json.loads(content.strip())
                        elapsed = time.time() - start_time
                        print(f"🌐 VISIT_PREDICT: Success on attempt {attempt + 1} (⏱️ {elapsed:.2f}s)")
                        return content.strip()
                    except:
                        # Extract JSON from string if it's embedded
                        left = content.find('{')
                        right = content.rfind('}')
                        if left != -1 and right != -1 and left <= right:
                            json_content = content[left:right+1]
                            try:
                                json.loads(json_content)  # Validate JSON
                                elapsed = time.time() - start_time
                                print(f"🌐 VISIT_PREDICT: JSON extracted on attempt {attempt + 1} (⏱️ {elapsed:.2f}s)")
                                return json_content
                            except:
                                pass

                        # If no valid JSON found, wrap the content
                        elapsed = time.time() - start_time
                        print(f"🌐 VISIT_PREDICT: Content wrapped on attempt {attempt + 1} (⏱️ {elapsed:.2f}s)")
                        return json.dumps({
                            "rational": "Content extracted from webpage",
                            "evidence": content.strip()[:2000],  # Limit length
                            "summary": "Webpage content processed but not in expected JSON format"
                        })
                else:
                    print(f"🌐 VISIT_PREDICT: Empty response on attempt {attempt + 1} (⏱️ {attempt_elapsed:.2f}s)")

            except Exception as e:
                attempt_elapsed = time.time() - attempt_start
                print(f"🌐 VISIT_PREDICT: Error attempt {attempt + 1}: {type(e).__name__}:{e} (⏱️ {attempt_elapsed:.2f}s)")
                if attempt == max_retries - 1:
                    elapsed = time.time() - start_time
                    print(f"🌐 VISIT_PREDICT: Final failure after {max_retries} attempts (⏱️ {elapsed:.2f}s)")
                    return json.dumps({
                        "rational": f"Predict function failed after {max_retries} attempts",
                        "evidence": f"Error: {str(e)}",
                        "summary": "Unable to process webpage content due to predict function error"
                    })

        elapsed = time.time() - start_time
        print(f"🌐 VISIT_PREDICT: No content returned after {max_retries} attempts (⏱️ {elapsed:.2f}s)")
        return ""

    def call_server_with_openai(self, msgs, max_retries):
        """Fallback to OpenAI API if predict function not available"""
        config = get_visit_config()
        api_key = config['api_key']
        url_llm = config['api_base']
        model_name = config['model_name']

        print(f"🌐 VISIT_OPENAI: call_server(api_key={'SET' if api_key else 'MISSING'}, base={'SET' if url_llm else 'MISSING'}, model='{model_name}')")

        if not api_key or not url_llm:
            return json.dumps({
                "rational": "Missing API configuration",
                "evidence": "API_KEY, API_BASE, or SUMMARY_MODEL_NAME not configured in environment",
                "summary": "Cannot process webpage content without proper API configuration"
            })

        client = OpenAI(
            api_key=api_key,
            base_url=url_llm,
        )
        for attempt in range(max_retries):
            try:
                chat_response = client.chat.completions.create(
                    model=model_name,
                    messages=msgs,
                    temperature=0.7
                )
                content = chat_response.choices[0].message.content
                if content:
                    try:
                        json.loads(content)
                    except:
                        # extract json from string
                        left = content.find('{')
                        right = content.rfind('}')
                        if left != -1 and right != -1 and left <= right:
                            content = content[left:right+1]
                    return content
            except Exception as e:
                print(f"🌐 VISIT_OPENAI: Error attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return ""
                continue


    def jina_readpage(self, url: str) -> str:
        """
        Read webpage content using Jina service.
        
        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            
        Returns:
            str: The webpage content or error message
        """
        max_retries = 3
        timeout = 50
        
        for attempt in range(max_retries):
            headers = {
                "Authorization": f"Bearer {get_visit_config()['jina_keys']}",
            }
            try:
                response = requests.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                    timeout=timeout
                )
                if response.status_code == 200:
                    webpage_content = response.text
                    return webpage_content
                else:
                    print(response.text)
                    raise ValueError("jina readpage error")
            except Exception as e:
                time.sleep(0.5)
                if attempt == max_retries - 1:
                    return "[visit] Failed to read page."
                
        return "[visit] Failed to read page."

    def html_readpage_jina(self, url: str) -> str:
        max_attempts = 8
        for attempt in range(max_attempts):
            content = self.jina_readpage(url)
            service = "jina"     
            print(service)
            if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
                return content
        return "[visit] Failed to read page."

    def readpage_jina(self, url: str, goal: str, predict_function=None) -> str:
        """
        Attempt to read webpage content by alternating between jina and aidata services.

        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            predict_function: Optional predict function to use instead of OpenAI API

        Returns:
            str: The webpage content or error message
        """

        # Create a lambda that passes the predict function to call_server
        if predict_function:
            summary_page_func = lambda msgs, max_retries=2: self.call_server(msgs, max_retries, predict_function)
            print(f"🌐 VISIT_JINA: Using predict function for summarization")
        else:
            summary_page_func = self.call_server
            print(f"🌐 VISIT_JINA: Using OpenAI API for summarization")

        max_retries = int(os.getenv('VISIT_SERVER_MAX_RETRIES', 1))

        content = self.html_readpage_jina(url)

        if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
            content = truncate_to_tokens(content, max_tokens=95000)
            messages = [{"role":"user","content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
            parse_retry_times = 0
            raw = summary_page_func(messages, max_retries=max_retries)
            summary_retries = 3
            while len(raw) < 10 and summary_retries >= 0:
                truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
                status_msg = (
                    f"[visit] Summary url[{url}] " 
                    f"attempt {3 - summary_retries + 1}/3, "
                    f"content length: {len(content)}, "
                    f"truncating to {truncate_length} chars"
                ) if summary_retries > 0 else (
                    f"[visit] Summary url[{url}] failed after 3 attempts, "
                    f"final truncation to 25000 chars"
                )
                print(status_msg)
                content = content[:truncate_length]
                extraction_prompt = EXTRACTOR_PROMPT.format(
                    webpage_content=content,
                    goal=goal
                )
                messages = [{"role": "user", "content": extraction_prompt}]
                raw = summary_page_func(messages, max_retries=max_retries)
                summary_retries -= 1

            parse_retry_times = 2
            if isinstance(raw, str):
                raw = raw.replace("```json", "").replace("```", "").strip()
            while parse_retry_times < 3:
                try:
                    raw = json.loads(raw)
                    break
                except:
                    raw = summary_page_func(messages, max_retries=max_retries)
                    parse_retry_times += 1
            
            if parse_retry_times >= 3:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            else:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
                useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"

            if len(useful_information) < 10 and summary_retries < 0:
                print("[visit] Could not generate valid summary after maximum retries")
                useful_information = "[visit] Failed to read page"
            
            return useful_information

        # If no valid content was obtained after all retries
        else:
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
            useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
            useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            return useful_information

    