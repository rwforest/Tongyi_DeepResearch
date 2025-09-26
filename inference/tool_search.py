import json
import http.client
import os
import time
from typing import List, Union, Optional

from qwen_agent.tools.base import BaseTool, register_tool


def get_serper_key():
    """Get SERPER_KEY at runtime to ensure .env is loaded"""
    return os.environ.get('SERPER_KEY_ID')

print(f"🔍 IMPORT: tool_search.py key={'SET' if get_serper_key() else 'MISSING'}")


@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
    def google_search_with_serp(self, query: str):
        SERPER_KEY = get_serper_key()
        print(f"🔍 API_CALL: google_search_with_serp(query='{query}', key={'SET' if SERPER_KEY else 'MISSING'})")

        # Check if API key is available
        if not SERPER_KEY:
            error_msg = f"[Search Error] SERPER_KEY_ID environment variable not set. Please configure your Serper API key."
            print(f"🔍 API_ERROR: {error_msg}")
            return error_msg

        conn = http.client.HTTPSConnection("google.serper.dev", timeout=30)
        payload = json.dumps({"q": query})
        headers = {'X-API-KEY': SERPER_KEY, 'Content-Type': 'application/json'}
        print(f"🔍 API_REQUEST: POST /search payload_size={len(payload)} timeout=30s")
        
        
        for i in range(5):
            try:
                conn.request("POST", "/search", payload, headers)
                res = conn.getresponse()
                print(f"🔍 API_RESPONSE: attempt={i+1} status={res.status}")
                break
            except Exception as e:
                print(f"🔍 API_RETRY: attempt={i+1}/5 failed={type(e).__name__}:{str(e)}")
                if i == 4:
                    final_error = f"Google search Timeout after 5 attempts, return None, Please try again later."
                    print(f"🔍 API_TIMEOUT: {final_error}")
                    return final_error
                time.sleep(2 * (i + 1))
                conn = http.client.HTTPSConnection("google.serper.dev", timeout=30)
                continue

        data = res.read()
        results = json.loads(data.decode("utf-8"))
        print(f"🔍 API_DATA: response_size={len(data)} results_keys={list(results.keys())}")

        try:
            if "organic" not in results:
                print(f"🔍 API_NODATA: No organic results for '{query}'")
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = list()
            for idx, page in enumerate(results["organic"], 1):
                date_published = "\nDate published: " + page["date"] if "date" in page else ""
                source = "\nSource: " + page["source"] if "source" in page else ""
                snippet = "\n" + page["snippet"] if "snippet" in page else ""
                redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                web_snippets.append(redacted_version)

            content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
            print(f"🔍 API_SUCCESS: query='{query}' results={len(web_snippets)} content_size={len(content)}")
            return content
        except Exception as e:
            error_msg = f"No results found for '{query}'. Try with a more general query."
            print(f"🔍 API_EXCEPTION: {type(e).__name__}:{str(e)} returning='{error_msg}'")
            return error_msg


    
    def search_with_serp(self, query: str):
        result = self.google_search_with_serp(query)
        return result

    def call(self, params: Union[str, dict], **kwargs) -> str:
        print(f"🔍 TOOL_CALL: Search.call(params_type={type(params)}, params={params})")

        try:
            query = params["query"]
            print(f"🔍 TOOL_EXTRACT: query={query} query_type={type(query)}")
        except Exception as e:
            error_msg = "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
            print(f"🔍 TOOL_ERROR: extract_failed={type(e).__name__}:{e} returning='{error_msg}'")
            return error_msg

        if isinstance(query, str):
            print(f"🔍 TOOL_MODE: single_query='{query}'")
            response = self.search_with_serp(query)
            print(f"🔍 TOOL_RESULT: single_response_length={len(response)}")
        else:
            print(f"🔍 TOOL_MODE: multiple_queries={len(query)}")
            assert isinstance(query, List)
            responses = []
            for i, q in enumerate(query):
                print(f"🔍 TOOL_BATCH: processing={i+1}/{len(query)} query='{q}'")
                response_single = self.search_with_serp(q)
                print(f"🔍 TOOL_BATCH: result={i+1} length={len(response_single)}")
                responses.append(response_single)

            response = "\n=======\n".join(responses)
            print(f"🔍 TOOL_RESULT: combined_response_length={len(response)}")

        print(f"🔍 TOOL_RETURN: final_response_length={len(response)}")
        return response

