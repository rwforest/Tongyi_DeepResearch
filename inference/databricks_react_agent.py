"""
Modified React Agent for Databricks
Replaces OpenAI SDK calls with predict() function
Based on inference/react_agent.py
"""

import json
import json5
import os
import time
import random
import datetime
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union, Callable
import mlflow
from mlflow.entities import SpanType

def load_environment_variables():
    """Load environment variables from .env file if available"""
    try:
        from dotenv import load_dotenv
        load_dotenv('.env')
        print("🔧 ENV_LOADED: Successfully loaded .env file")

        # Log loaded API keys status
        api_keys = {
            'SERPER_KEY_ID': os.environ.get('SERPER_KEY_ID'),
            'PERPLEXITY_API_KEY': os.environ.get('PERPLEXITY_API_KEY'),
            'JINA_API_KEYS': os.environ.get('JINA_API_KEYS'),
            'API_KEY': os.environ.get('API_KEY'),
            'SANDBOX_FUSION_ENDPOINT': os.environ.get('SANDBOX_FUSION_ENDPOINT')
        }

        for key, value in api_keys.items():
            status = 'SET' if value else 'MISSING'
            print(f"🔧 ENV_KEY: {key}={status}")

    except ImportError:
        print("🔧 ENV_WARNING: python-dotenv not available, using system environment")

# Load environment variables early
load_environment_variables()

from qwen_agent.llm.schema import Message
from qwen_agent.utils.utils import build_text_completion_prompt
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from qwen_agent.utils.utils import format_as_text_message, merge_generate_cfgs

# Import tools (now in same directory)
from tool_file import *
from tool_scholar import *
from tool_python import *
from tool_search import *
from tool_perplexity import *
from tool_visit import *
from prompt import *

OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'

MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 100))

TOOL_CLASS = [
    FileParser(),
    Scholar(),
    Visit(),
    Search(),
    PerplexitySearch(),
    PythonInterpreter(),
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")


class DatabricksMultiTurnReactAgent(FnCallAgent):
    """
    Modified MultiTurnReactAgent that uses predict() function instead of OpenAI SDK
    """

    def __init__(self,
                 predict_function: Callable[[str], str],
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 **kwargs):
        """
        Initialize with predict function instead of LLM config

        Args:
            predict_function: Your predict() function that takes a string and returns response
            function_list: List of available functions/tools
            llm: LLM configuration (for compatibility, but we'll use predict_function)
        """
        self.predict_function = predict_function
        self.llm_generate_cfg = llm.get("generate_cfg", {}) if llm else {}
        self.llm_local_path = llm.get("model", "databricks-model") if llm else "databricks-model"
        self.model = self.llm_local_path

        print(f"✅ Initialized DatabricksMultiTurnReactAgent with predict() function")
        print(f"   Available tools: {list(TOOL_MAP.keys())}")

    def sanity_check_output(self, content):
        return "<think>" in content and "</think>" in content

    @mlflow.trace(name="predict_function_call", span_type=SpanType.LLM)
    def call_predict(self, msgs, max_tries=3):
        """
        Use predict() function instead of OpenAI API

        Args:
            msgs: List of message dictionaries
            max_tries: Maximum retry attempts

        Returns:
            Generated response string
        """
        start_time = time.time()
        print(f"🤖 PREDICT_CALL: Starting predict function with {len(msgs)} messages")

        # Set MLflow span inputs
        span = mlflow.get_current_active_span()
        if span:
            span.set_inputs({
                "messages": msgs,
                "max_tries": max_tries,
                "message_count": len(msgs)
            })
            span.set_attributes({
                "model": "tongyi-deepresearch",
                "temperature": float(os.environ.get('TEMPERATURE', 0.85)),
                "presence_penalty": float(os.environ.get('PRESENCE_PENALTY', 1.1))
            })
        # Convert messages to prompt format for predict function
        prompt_parts = []

        for msg in msgs:
            # Handle both dict and string formats
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            elif isinstance(msg, str):
                # If it's a string, treat it as user message
                role = "user"
                content = msg
            else:
                print(f"WARNING: Unexpected message type: {type(msg)}")
                continue

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        # Add final assistant prompt
        prompt_parts.append("Assistant:")
        full_prompt = "\n\n".join(prompt_parts)

        # Retry logic
        for attempt in range(max_tries):
            try:
                attempt_start = time.time()
                print(f"--- Calling predict() function, attempt {attempt + 1}/{max_tries} ---")

                # Format input for your predict function (expects list of dicts)
                model_input = [{
                    "prompt": full_prompt,
                    "max_length": 2048,
                    "temperature": float(os.environ.get('TEMPERATURE', 0.85)),
                    "presence_penalty": float(os.environ.get('PRESENCE_PENALTY', 1.1))
                }]

                # Call your predict function with the expected format
                response = self.predict_function(model_input)
                attempt_elapsed = time.time() - attempt_start

                if response and response.strip():
                    elapsed = time.time() - start_time
                    print(f"🤖 PREDICT_SUCCESS: attempt={attempt + 1} response_length={len(response)} (⏱️ {attempt_elapsed:.2f}s total: {elapsed:.2f}s)")

                    # Set MLflow span outputs
                    if span:
                        span.set_outputs({
                            "response": response.strip(),
                            "response_length": len(response),
                            "attempts_used": attempt + 1,
                            "total_time": elapsed
                        })

                    return response.strip()
                else:
                    print(f"🤖 PREDICT_EMPTY: attempt={attempt + 1} received empty response (⏱️ {attempt_elapsed:.2f}s)")

            except Exception as e:
                attempt_elapsed = time.time() - attempt_start
                print(f"🤖 PREDICT_ERROR: attempt={attempt + 1} {type(e).__name__}:{e} (⏱️ {attempt_elapsed:.2f}s)")
                print(f"ERROR DETAILS: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()

            if attempt < max_tries - 1:
                sleep_time = 1 * (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)

        elapsed = time.time() - start_time
        print(f"🤖 PREDICT_FINAL_FAIL: Failed after {max_tries} attempts (⏱️ {elapsed:.2f}s)")

        # Set MLflow span outputs for failure case
        if span:
            span.set_outputs({
                "error": f"Failed after {max_tries} attempts",
                "attempts_used": max_tries,
                "total_time": elapsed
            })

        raise Exception(f"Predict function failed after {max_tries} attempts")

    @mlflow.trace(name="react_agent_session", span_type=SpanType.AGENT)
    def _run(self, data, model=None, planning_port=None):
        """
        Main run method - modified to use predict() function instead of server calls

        Args:
            data: Task data containing question and answer
            model: Model name (for compatibility)
            planning_port: Port (not used, for compatibility)

        Returns:
            Result dictionary with prediction and metadata
        """
        start_time = time.time()
        question = data['item']['question']
        answer = data['item']['answer']

        print(f"🔬 Processing question: {question[:100]}...")

        # Set MLflow span inputs for session
        session_span = mlflow.get_current_active_span()
        if session_span:
            session_span.set_inputs({
                "question": question,
                "reference_answer": answer,
                "max_llm_calls": MAX_LLM_CALL_PER_RUN
            })
            session_span.set_attributes({
                "agent_type": "react",
                "model": "tongyi-deepresearch"
            })

        self.user_prompt = question
        system_prompt = SYSTEM_PROMPT
        cur_date = today_date()
        system_prompt = system_prompt + str(cur_date)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round_count = 0

        while num_llm_calls_available > 0:
            # Check time limit (150 minutes)
            if time.time() - start_time > 150 * 60:
                prediction = 'No answer found after 2h30mins'
                termination = 'No answer found after 2h30mins'
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                return result

            round_count += 1
            num_llm_calls_available -= 1

            # Call predict function instead of server
            content = self.call_predict(messages)

            print(f'Round {round_count}: {content}')

            # Remove tool_response if present in content
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]

            messages.append({"role": "assistant", "content": content.strip()})

            # Check for tool calls
            if '<tool_call>' in content and '</tool_call>' in content:
                print(f"🔧 TOOL CALL TRACE: ========== TOOL CALL DETECTED ==========")
                print(f"🔧 TOOL CALL TRACE: Round {round_count}, LLM calls remaining: {num_llm_calls_available}")
                print(f"🔧 TOOL CALL TRACE: Full content length: {len(content)} chars")
                print(f"🔧 TOOL CALL TRACE: Content preview: {content[:200]}...")

                # Extract tool call with detailed tracing
                try:
                    tool_call_start = content.find('<tool_call>')
                    tool_call_end = content.find('</tool_call>')
                    print(f"🔧 TOOL CALL TRACE: Start index: {tool_call_start}, End index: {tool_call_end}")

                    if tool_call_start == -1 or tool_call_end == -1:
                        raise ValueError("Tool call tags not properly formed")

                    tool_call = content[tool_call_start + len('<tool_call>'):tool_call_end]
                    print(f"🔧 TOOL CALL TRACE: Extracted tool call: '{tool_call}'")
                    print(f"🔧 TOOL CALL TRACE: Tool call length: {len(tool_call)} chars")
                    print(f"🔧 TOOL CALL TRACE: Tool call type: {type(tool_call)}")

                    # Check for whitespace issues
                    stripped_tool_call = tool_call.strip()
                    if len(stripped_tool_call) != len(tool_call):
                        print(f"🔧 TOOL CALL TRACE: Whitespace detected - original: {len(tool_call)}, stripped: {len(stripped_tool_call)}")
                        tool_call = stripped_tool_call

                except Exception as extraction_e:
                    print(f"🚨 TOOL CALL EXTRACTION ERROR: {type(extraction_e).__name__}: {extraction_e}")
                    import traceback
                    traceback.print_exc()
                    result = f'Error: Failed to extract tool call: {str(extraction_e)}'
                    tool_response = f"<tool_response>\n{result}\n</tool_response>"
                    messages.append({"role": "user", "content": tool_response})
                    continue

                try:
                    # Handle Python code specially
                    if "python" in tool_call.lower():
                        print(f"🔧 TOOL CALL TRACE: ===== PYTHON CODE EXECUTION =====")
                        print(f"🔧 TOOL CALL TRACE: Searching for <code> tags in tool call...")

                        try:
                            # More robust code extraction
                            if '<code>' in content and '</code>' in content:
                                code_start = content.find('<code>')
                                code_end = content.find('</code>')
                                code_raw = content[code_start + len('<code>'):code_end].strip()
                                print(f"🔧 TOOL CALL TRACE: Extracted Python code ({len(code_raw)} chars):")
                                print(f"🔧 TOOL CALL TRACE: Code preview: {code_raw[:200]}...")

                                print(f"🔧 TOOL CALL TRACE: Calling PythonInterpreter tool...")
                                result = TOOL_MAP['PythonInterpreter'].call(code_raw)
                                print(f"🔧 TOOL CALL TRACE: Python execution result type: {type(result)}")
                                print(f"🔧 TOOL CALL TRACE: Python result length: {len(str(result))} chars")
                            else:
                                raise ValueError("No <code> tags found for Python execution")

                        except Exception as python_e:
                            print(f"🚨 PYTHON EXECUTION ERROR: {type(python_e).__name__}: {python_e}")
                            import traceback
                            traceback.print_exc()
                            result = f"[Python Interpreter Error]: {str(python_e)}"

                    else:
                        print(f"🔧 TOOL CALL TRACE: ===== JSON TOOL CALL PARSING =====")
                        print(f"🔧 TOOL CALL TRACE: Attempting to parse as JSON...")
                        print(f"🔧 TOOL CALL TRACE: Raw tool call for JSON parsing: '{tool_call}'")

                        # Check if it looks like valid JSON
                        if not tool_call.startswith('{') or not tool_call.endswith('}'):
                            print(f"🚨 JSON FORMAT WARNING: Tool call doesn't start/end with braces")
                            print(f"🚨 First char: '{tool_call[0] if tool_call else 'EMPTY'}', Last char: '{tool_call[-1] if tool_call else 'EMPTY'}'")

                        # Parse JSON tool call
                        try:
                            tool_call_parsed = json5.loads(tool_call)
                            print(f"🔧 TOOL CALL TRACE: ✅ JSON parsing successful")
                            print(f"🔧 TOOL CALL TRACE: Parsed JSON type: {type(tool_call_parsed)}")
                            print(f"🔧 TOOL CALL TRACE: Parsed JSON: {tool_call_parsed}")
                        except json5.JSONError as json_e:
                            print(f"🚨 JSON PARSING ERROR: {type(json_e).__name__}: {json_e}")
                            print(f"🚨 Problematic JSON: '{tool_call}'")
                            raise json_e

                        # Extract tool name and arguments
                        tool_name = tool_call_parsed.get('name', '')
                        tool_args = tool_call_parsed.get('arguments', {})
                        print(f"🔧 TOOL CALL TRACE: Extracted tool_name: '{tool_name}' (type: {type(tool_name)})")
                        print(f"🔧 TOOL CALL TRACE: Extracted tool_args: {tool_args} (type: {type(tool_args)})")

                        # Validate tool name
                        if not tool_name:
                            raise ValueError("Tool name is empty or missing")
                        if tool_name not in TOOL_MAP:
                            print(f"🚨 UNKNOWN TOOL WARNING: '{tool_name}' not in {list(TOOL_MAP.keys())}")

                        print(f"🔧 TOOL CALL TRACE: Calling custom_call_tool('{tool_name}', {tool_args})...")
                        result = self.custom_call_tool(tool_name, tool_args)
                        print(f"🔧 TOOL CALL TRACE: ✅ Tool execution completed")
                        print(f"🔧 TOOL CALL TRACE: Result type: {type(result)}")
                        print(f"🔧 TOOL CALL TRACE: Result length: {len(str(result))} chars")

                except Exception as e:
                    print(f"🚨 TOOL CALL EXCEPTION: {type(e).__name__}: {str(e)}")
                    print(f"🚨 Exception during tool call processing in round {round_count}")
                    print(f"🚨 Current tool call: '{tool_call}' (length: {len(tool_call)})")
                    print(f"🚨 Current content preview: {content[:300]}...")
                    import traceback
                    print("🚨 FULL TRACEBACK:")
                    traceback.print_exc()
                    result = f'Error: Tool call processing failed: {type(e).__name__}: {str(e)}'

                # Add tool response
                tool_response = f"<tool_response>\n{result}\n</tool_response>"
                messages.append({"role": "user", "content": tool_response})

                print(f"Tool executed: {result[:200]}..." if len(str(result)) > 200 else f"Tool result: {result}")

            # Check for final answer
            if '<answer>' in content and '</answer>' in content:
                termination = 'answer'
                break

            # Check if we're out of calls
            if num_llm_calls_available <= 0 and '<answer>' not in content:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            # Check token count (simplified)
            max_tokens = 108 * 1024
            token_count = self.count_tokens(messages)
            print(f"round: {round_count}, token count: {token_count}")

            if token_count > max_tokens:
                print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")

                messages[-1]['content'] = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:<think>your final thinking</think>\n<answer>your answer</answer>"

                content = self.call_predict(messages)
                messages.append({"role": "assistant", "content": content.strip()})

                if '<answer>' in content and '</answer>' in content:
                    prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                    termination = 'generate an answer as token limit reached'
                else:
                    prediction = messages[-1]['content']
                    termination = 'format error: generate an answer as token limit reached'

                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                return result

        # Extract final prediction
        if '<answer>' in messages[-1]['content']:
            prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'

        session_elapsed = time.time() - start_time
        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination
        }

        # Set MLflow span outputs for session completion
        if session_span:
            session_span.set_outputs({
                "prediction": prediction,
                "termination": termination,
                "rounds_completed": round_count,
                "llm_calls_used": MAX_LLM_CALL_PER_RUN - num_llm_calls_available,
                "session_time": session_elapsed,
                "message_count": len(messages)
            })

        print(f"🚀 SESSION_COMPLETE: rounds={round_count} prediction_length={len(prediction)} (⏱️ {session_elapsed:.2f}s)")
        return result

    @mlflow.trace(name="tool_execution", span_type=SpanType.TOOL)
    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        """Execute a tool with given arguments"""
        tool_start_time = time.time()
        print(f"🛠️ TRACE: custom_call_tool called")
        print(f"🛠️ TRACE: tool_name: '{tool_name}'")
        print(f"🛠️ TRACE: tool_args: {tool_args}")
        print(f"🛠️ TRACE: tool_args type: {type(tool_args)}")
        print(f"🛠️ TRACE: Available tools: {list(TOOL_MAP.keys())}")

        # Set MLflow span inputs for tool execution
        tool_span = mlflow.get_current_active_span()
        if tool_span:
            tool_span.set_inputs({
                "tool_name": tool_name,
                "tool_args": tool_args
            })
            tool_span.set_attributes({
                "tool_type": tool_name,
                "available_tools": list(TOOL_MAP.keys())
            })

        if tool_name in TOOL_MAP:
            try:
                print(f"🛠️ TRACE: Calling {tool_name} tool...")
                print(f"🛠️ TRACE: Tool object: {TOOL_MAP[tool_name]}")

                # For Visit tool, pass the predict function
                if tool_name == 'visit':
                    result = TOOL_MAP[tool_name].call(tool_args, predict_function=self.predict_function)
                    print(f"🛠️ TRACE: Visit tool called with predict function")
                else:
                    result = TOOL_MAP[tool_name].call(tool_args)

                tool_elapsed = time.time() - tool_start_time
                print(f"🛠️ TRACE: Tool {tool_name} completed (⏱️ {tool_elapsed:.2f}s)")
                print(f"🛠️ TRACE: Result type: {type(result)}")
                print(f"🛠️ TRACE: Result length: {len(str(result))} characters")
                print(f"🛠️ TRACE: Result preview: {str(result)[:200]}...")

                # Set MLflow span outputs for successful tool execution
                if tool_span:
                    tool_span.set_outputs({
                        "result": str(result),
                        "result_length": len(str(result)),
                        "execution_time": tool_elapsed,
                        "success": True
                    })

                return result
            except Exception as e:
                tool_elapsed = time.time() - tool_start_time
                error_msg = f"Tool {tool_name} execution failed: {str(e)}"
                print(f"🛠️ TRACE: Tool execution failed: {error_msg} (⏱️ {tool_elapsed:.2f}s)")
                print(f"🛠️ TRACE: Exception type: {type(e).__name__}")
                import traceback
                print(f"🛠️ TRACE: Full traceback:")
                traceback.print_exc()

                # Set MLflow span outputs for failed tool execution
                if tool_span:
                    tool_span.set_outputs({
                        "error": error_msg,
                        "exception_type": type(e).__name__,
                        "execution_time": tool_elapsed,
                        "success": False
                    })

                return error_msg
        else:
            tool_elapsed = time.time() - tool_start_time
            error_msg = f"Tool {tool_name} not found in available tools: {list(TOOL_MAP.keys())}"
            print(f"🛠️ TRACE: {error_msg} (⏱️ {tool_elapsed:.2f}s)")

            # Set MLflow span outputs for tool not found
            if tool_span:
                tool_span.set_outputs({
                    "error": error_msg,
                    "execution_time": tool_elapsed,
                    "success": False
                })

            return error_msg

    def count_tokens(self, messages):
        """
        Simplified token counting - just count characters and divide by 4
        For production, you might want to use a proper tokenizer
        """
        total_chars = 0
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get('content', '')
            elif isinstance(msg, str):
                content = msg
            else:
                content = str(msg)
            total_chars += len(str(content))
        return total_chars // 4  # Rough approximation