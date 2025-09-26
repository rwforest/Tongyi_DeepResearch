import json
import http.client
import os
import time
from typing import List, Union, Optional

from qwen_agent.tools.base import BaseTool, register_tool

def get_perplexity_key():
    """Get PERPLEXITY_API_KEY at runtime to ensure .env is loaded"""
    return os.environ.get('PERPLEXITY_API_KEY')

print(f"🧠 IMPORT: tool_perplexity.py key={'SET' if get_perplexity_key() else 'MISSING'}")


@register_tool("perplexity_search", allow_overwrite=True)
class PerplexitySearch(BaseTool):
    name = "perplexity_search"
    description = "Performs AI-powered web searches using Perplexity's Search API: supply an array 'query'; the tool retrieves comprehensive answers with citations for each query."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. Each query will get a comprehensive AI-generated answer with web citations."
            },
            "model": {
                "type": "string",
                "description": "Model to use for search. Options: 'sonar-pro', 'sonar-reasoning', or 'sonar-pro'",
                "default": "sonar-pro"
            }
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def perplexity_search_with_api(self, query: str, model: str = "sonar-pro"):
        """Execute search using Perplexity API"""
        PERPLEXITY_API_KEY = get_perplexity_key()
        print(f"🧠 API_CALL: perplexity_search(query='{query}', model='{model}', key={'SET' if PERPLEXITY_API_KEY else 'MISSING'})")

        # Check if API key is available
        if not PERPLEXITY_API_KEY:
            error_msg = "[Perplexity Error] PERPLEXITY_API_KEY environment variable not set. Please configure your Perplexity API key."
            print(f"🧠 API_ERROR: {error_msg}")
            return error_msg

        conn = http.client.HTTPSConnection("api.perplexity.ai", timeout=60)

        # Format request according to Perplexity API docs
        payload = json.dumps({
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ],
            "return_citations": True,
            "return_images": False,
            "return_related_questions": True,
            "search_domain_filter": [],
            "search_recency_filter": "month",  # month, week, day
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 2048
        })

        headers = {
            'Authorization': f'Bearer {PERPLEXITY_API_KEY}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        print(f"🧠 API_REQUEST: POST /chat/completions payload_size={len(payload)} model={model}")

        for attempt in range(3):
            try:
                conn.request("POST", "/chat/completions", payload, headers)
                res = conn.getresponse()
                print(f"🧠 API_RESPONSE: attempt={attempt+1} status={res.status}")

                if res.status == 200:
                    break
                elif res.status == 401:
                    error_msg = "Perplexity API authentication failed - check your API key"
                    print(f"🧠 API_AUTH_ERROR: {error_msg}")
                    return f"[Perplexity Error] {error_msg}"
                elif res.status == 429:
                    if attempt < 2:
                        wait_time = 2 ** attempt
                        print(f"🧠 API_RATELIMIT: attempt={attempt+1} waiting={wait_time}s")
                        time.sleep(wait_time)
                        continue
                    else:
                        error_msg = "Perplexity API rate limit exceeded"
                        print(f"🧠 API_RATELIMIT_FINAL: {error_msg}")
                        return f"[Perplexity Error] {error_msg}"
                else:
                    error_data = res.read()
                    error_msg = f"Perplexity API error: HTTP {res.status}"
                    print(f"🧠 API_HTTP_ERROR: status={res.status} response={error_data[:200]}")
                    return f"[Perplexity Error] {error_msg}"

            except Exception as e:
                print(f"🧠 API_RETRY: attempt={attempt+1}/3 failed={type(e).__name__}:{str(e)}")
                if attempt == 2:
                    final_error = f"Perplexity search failed after 3 attempts: {str(e)}"
                    print(f"🧠 API_TIMEOUT: {final_error}")
                    return f"[Perplexity Error] {final_error}"
                time.sleep(2 * (attempt + 1))
                conn = http.client.HTTPSConnection("api.perplexity.ai", timeout=60)
                continue

        # Parse response
        try:
            data = res.read()
            response = json.loads(data.decode("utf-8"))
            print(f"🧠 API_DATA: response_size={len(data)} response_keys={list(response.keys())}")

            if "choices" not in response:
                error_msg = f"Unexpected Perplexity API response format: {list(response.keys())}"
                print(f"🧠 API_FORMAT_ERROR: {error_msg}")
                return f"[Perplexity Error] {error_msg}"

            choice = response["choices"][0]
            message = choice["message"]
            content = message.get("content", "")

            # Extract citations if available
            citations = []
            if "citations" in response:
                citations = response["citations"]
            elif "citations" in choice:
                citations = choice["citations"]

            # Extract related questions if available
            related_questions = []
            if "related_questions" in response:
                related_questions = response["related_questions"]

            # Format the response
            formatted_response = self.format_perplexity_response(query, content, citations, related_questions)

            print(f"🧠 API_SUCCESS: query='{query}' content_length={len(content)} citations={len(citations)} related={len(related_questions)}")
            return formatted_response

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse Perplexity API response: {str(e)}"
            print(f"🧠 API_JSON_ERROR: {error_msg}")
            return f"[Perplexity Error] {error_msg}"
        except Exception as e:
            error_msg = f"Error processing Perplexity response: {str(e)}"
            print(f"🧠 API_PARSE_ERROR: {error_msg}")
            return f"[Perplexity Error] {error_msg}"

    def format_perplexity_response(self, query: str, content: str, citations: List, related_questions: List) -> str:
        """Format the Perplexity response into a readable format"""

        # Start with the main response
        formatted = f"# Perplexity Search Results for: '{query}'\n\n"
        formatted += f"## Answer\n{content}\n\n"

        # Add citations if available
        if citations:
            formatted += "## Sources\n"
            for i, citation in enumerate(citations, 1):
                if isinstance(citation, dict):
                    title = citation.get("title", "Unknown Title")
                    url = citation.get("url", "")
                    snippet = citation.get("text", "")
                    formatted += f"{i}. [{title}]({url})\n"
                    if snippet:
                        formatted += f"   {snippet[:200]}...\n"
                elif isinstance(citation, str):
                    formatted += f"{i}. {citation}\n"
                formatted += "\n"

        # Add related questions if available
        if related_questions:
            formatted += "## Related Questions\n"
            for i, question in enumerate(related_questions, 1):
                if isinstance(question, dict):
                    q_text = question.get("question", question.get("text", str(question)))
                else:
                    q_text = str(question)
                formatted += f"{i}. {q_text}\n"

        return formatted

    def search_with_perplexity(self, query: str, model: str = "sonar-pro"):
        """Wrapper for search functionality"""
        return self.perplexity_search_with_api(query, model)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        print(f"🧠 TOOL_CALL: PerplexitySearch.call(params_type={type(params)}, params={params})")

        try:
            query = params["query"]
            model = params.get("model", "sonar-pro")
            print(f"🧠 TOOL_EXTRACT: query={query} model={model} query_type={type(query)}")
        except Exception as e:
            error_msg = "[Perplexity] Invalid request format: Input must be a JSON object containing 'query' field"
            print(f"🧠 TOOL_ERROR: extract_failed={type(e).__name__}:{e} returning='{error_msg}'")
            return error_msg

        if isinstance(query, str):
            print(f"🧠 TOOL_MODE: single_query='{query}' model={model}")
            response = self.search_with_perplexity(query, model)
            print(f"🧠 TOOL_RESULT: single_response_length={len(response)}")
        else:
            print(f"🧠 TOOL_MODE: multiple_queries={len(query)} model={model}")
            assert isinstance(query, List)
            responses = []
            for i, q in enumerate(query):
                print(f"🧠 TOOL_BATCH: processing={i+1}/{len(query)} query='{q}'")
                response_single = self.search_with_perplexity(q, model)
                print(f"🧠 TOOL_BATCH: result={i+1} length={len(response_single)}")
                responses.append(response_single)

            response = "\n" + "="*80 + "\n".join(responses)
            print(f"🧠 TOOL_RESULT: combined_response_length={len(response)}")

        print(f"🧠 TOOL_RETURN: final_response_length={len(response)}")
        return response