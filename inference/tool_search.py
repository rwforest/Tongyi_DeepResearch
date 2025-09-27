import json
import http.client
import os
import time
from typing import List, Union, Optional

from qwen_agent.tools.base import BaseTool, register_tool


def get_serper_key():
    """Get SERPER_KEY at runtime to ensure .env is loaded"""
    return os.environ.get('SERPER_KEY_ID')

def get_search_config():
    """Get search configuration at runtime"""
    return {
        'use_perplexity': os.environ.get('USE_PERPLEXITY_SEARCH', 'false').lower() == 'true',
        'perplexity_key': os.environ.get('PERPLEXITY_API_KEY'),
        'perplexity_max_results': int(os.environ.get('PERPLEXITY_MAX_RESULTS', '10')),
        'serper_key': get_serper_key()
    }

search_config = get_search_config()
print(f"🔍 IMPORT: tool_search.py serper={'SET' if search_config['serper_key'] else 'MISSING'} perplexity={'SET' if search_config['perplexity_key'] else 'MISSING'} use_perplexity={search_config['use_perplexity']}")


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


    
    def perplexity_search_with_api(self, query: str, max_results: int = 10):
        """Execute search using Perplexity Search API (new dedicated search endpoint)"""
        config = get_search_config()
        PERPLEXITY_API_KEY = config['perplexity_key']

        print(f"🧠 SEARCH_PERPLEXITY: perplexity_search(query='{query}', max_results={max_results}, key={'SET' if PERPLEXITY_API_KEY else 'MISSING'})")

        # Check if API key is available
        if not PERPLEXITY_API_KEY:
            error_msg = "[Search Error] PERPLEXITY_API_KEY environment variable not set. Please configure your Perplexity API key."
            print(f"🧠 SEARCH_ERROR: {error_msg}")
            return error_msg

        conn = http.client.HTTPSConnection("api.perplexity.ai", timeout=60)

        # Format request according to new Perplexity Search API
        payload = json.dumps({
            "query": query,
            "max_results": min(max_results, 20),  # API limit is 20
            "max_tokens_per_page": 1024,
            "country": "US"  # Optional country code
        })

        headers = {
            'Authorization': f'Bearer {PERPLEXITY_API_KEY}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        print(f"🧠 SEARCH_REQUEST: POST /search payload_size={len(payload)} max_results={max_results}")

        for attempt in range(3):
            try:
                conn.request("POST", "/search", payload, headers)
                res = conn.getresponse()
                print(f"🧠 SEARCH_RESPONSE: attempt={attempt+1} status={res.status}")

                if res.status == 200:
                    break
                elif res.status == 401:
                    error_msg = "Perplexity API authentication failed - check your API key"
                    print(f"🧠 SEARCH_AUTH_ERROR: {error_msg}")
                    return f"[Search Error] {error_msg}"
                elif res.status == 429:
                    if attempt < 2:
                        wait_time = 2 ** attempt
                        print(f"🧠 SEARCH_RATELIMIT: attempt={attempt+1} waiting={wait_time}s")
                        time.sleep(wait_time)
                        continue
                    else:
                        error_msg = "Perplexity API rate limit exceeded"
                        print(f"🧠 SEARCH_RATELIMIT_FINAL: {error_msg}")
                        return f"[Search Error] {error_msg}"
                else:
                    error_data = res.read()
                    error_msg = f"Perplexity API error: HTTP {res.status}"
                    print(f"🧠 SEARCH_HTTP_ERROR: status={res.status} response={error_data[:200]}")
                    return f"[Search Error] {error_msg}"

            except Exception as e:
                print(f"🧠 SEARCH_RETRY: attempt={attempt+1}/3 failed={type(e).__name__}:{str(e)}")
                if attempt == 2:
                    final_error = f"Perplexity search failed after 3 attempts: {str(e)}"
                    print(f"🧠 SEARCH_TIMEOUT: {final_error}")
                    return f"[Search Error] {final_error}"
                time.sleep(2 * (attempt + 1))
                conn = http.client.HTTPSConnection("api.perplexity.ai", timeout=60)
                continue

        # Parse response from new Search API
        try:
            data = res.read()
            response = json.loads(data.decode("utf-8"))
            print(f"🧠 SEARCH_DATA: response_size={len(data)} response_keys={list(response.keys())}")

            if "results" not in response:
                error_msg = f"Unexpected Perplexity Search API response format: {list(response.keys())}"
                print(f"🧠 SEARCH_FORMAT_ERROR: {error_msg}")
                return f"[Search Error] {error_msg}"

            results = response["results"]

            # Format the response similar to Serper format for consistency
            web_snippets = []
            for idx, result in enumerate(results, 1):
                title = result.get("title", "Unknown Title")
                url = result.get("url", "")
                snippet = result.get("snippet", "")
                date = result.get("date", "")

                date_published = f"\nDate published: {date}" if date else ""
                result_text = f"{idx}. [{title}]({url}){date_published}\n{snippet}"
                web_snippets.append(result_text)

            content = f"A Perplexity search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)

            print(f"🧠 SEARCH_SUCCESS: query='{query}' results={len(results)} content_size={len(content)}")
            return content

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse Perplexity Search API response: {str(e)}"
            print(f"🧠 SEARCH_JSON_ERROR: {error_msg}")
            return f"[Search Error] {error_msg}"
        except Exception as e:
            error_msg = f"Error processing Perplexity search response: {str(e)}"
            print(f"🧠 SEARCH_PARSE_ERROR: {error_msg}")
            return f"[Search Error] {error_msg}"

    def search_with_serp(self, query: str):
        result = self.google_search_with_serp(query)
        return result

    def search_with_backend(self, query: str):
        """Search using the configured backend (Serper or Perplexity)"""
        config = get_search_config()

        if config['use_perplexity']:
            print(f"🔍 SEARCH_BACKEND: Using Perplexity Search API for query '{query}'")
            return self.perplexity_search_with_api(query, config['perplexity_max_results'])
        else:
            print(f"🔍 SEARCH_BACKEND: Using Serper for query '{query}'")
            return self.search_with_serp(query)

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
            response = self.search_with_backend(query)
            print(f"🔍 TOOL_RESULT: single_response_length={len(response)}")
        else:
            print(f"🔍 TOOL_MODE: multiple_queries={len(query)}")
            assert isinstance(query, List)
            responses = []
            for i, q in enumerate(query):
                print(f"🔍 TOOL_BATCH: processing={i+1}/{len(query)} query='{q}'")
                response_single = self.search_with_backend(q)
                print(f"🔍 TOOL_BATCH: result={i+1} length={len(response_single)}")
                responses.append(response_single)

            response = "\n=======\n".join(responses)
            print(f"🔍 TOOL_RESULT: combined_response_length={len(response)}")

        print(f"🔍 TOOL_RETURN: final_response_length={len(response)}")
        return response

