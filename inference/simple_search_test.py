#!/usr/bin/env python3
"""
Simple standalone search test to identify the exact issue
"""

import json
import http.client
import os
import time

def load_env_manually():
    """Manually load .env file if it exists"""
    env_file = '.env'
    if os.path.exists(env_file):
        print(f"✅ Found .env file")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
                    print(f"   Loaded: {key.strip()}")
    else:
        print(f"❌ .env file not found at {os.path.abspath(env_file)}")

def test_basic_search():
    """Test basic search functionality like your working version"""
    print("🔍 BASIC SEARCH TEST")
    print("=" * 50)

    # Load environment
    load_env_manually()

    # Check API key
    serper_key = os.environ.get('SERPER_KEY_ID')
    print(f"🔑 SERPER_KEY_ID: {serper_key[:10] + '...' if serper_key else 'NOT_FOUND'}")

    if not serper_key:
        print("❌ No API key found")
        return False

    query = "artificial intelligence"
    print(f"🔍 Testing search for: '{query}'")

    try:
        # Use your exact working version
        conn = http.client.HTTPSConnection("google.serper.dev", timeout=30)

        payload = json.dumps({
            "q": query
        })

        headers = {
            'X-API-KEY': serper_key,
            'Content-Type': 'application/json'
        }

        print("📡 Making request...")
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()

        print(f"📊 Response status: {res.status}")

        if res.status == 200:
            data = res.read()
            results = json.loads(data.decode("utf-8"))

            if "organic" in results:
                web_snippets = []
                for idx, page in enumerate(results["organic"][:3], 1):  # Just first 3
                    title = page.get('title', 'No title')
                    link = page.get('link', 'No link')
                    snippet = page.get('snippet', 'No snippet')

                    result_text = f"{idx}. [{title}]({link})\n{snippet}"
                    web_snippets.append(result_text)

                content = f"Search for '{query}' found {len(results['organic'])} results:\n\n" + "\n\n".join(web_snippets)
                print(f"✅ Search successful! {len(content)} characters")
                print(f"📄 First result: {results['organic'][0].get('title', 'No title')}")
                return True
            else:
                print(f"❌ No organic results. Keys: {list(results.keys())}")
                return False

        elif res.status == 401:
            print("❌ Authentication failed - check API key")
            return False
        elif res.status == 429:
            print("❌ Rate limit exceeded")
            return False
        else:
            print(f"❌ HTTP error: {res.status}")
            error_data = res.read()
            print(f"   Error response: {error_data}")
            return False

    except Exception as e:
        print(f"❌ Request failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_retries():
    """Test with retry logic like in the search tool"""
    print("\n🔄 RETRY SEARCH TEST")
    print("=" * 50)

    serper_key = os.environ.get('SERPER_KEY_ID')
    if not serper_key:
        print("❌ No API key")
        return False

    query = "test retry search"

    for attempt in range(3):
        try:
            print(f"🔄 Attempt {attempt + 1}/3")

            conn = http.client.HTTPSConnection("google.serper.dev", timeout=30)

            payload = json.dumps({"q": query})
            headers = {
                'X-API-KEY': serper_key,
                'Content-Type': 'application/json'
            }

            conn.request("POST", "/search", payload, headers)
            res = conn.getresponse()

            print(f"   Status: {res.status}")

            if res.status == 200:
                data = res.read()
                results = json.loads(data.decode("utf-8"))

                if "organic" in results:
                    print(f"✅ Retry test successful on attempt {attempt + 1}")
                    return True
                else:
                    print(f"   No organic results on attempt {attempt + 1}")
            else:
                print(f"   HTTP error {res.status} on attempt {attempt + 1}")

        except Exception as e:
            print(f"   Exception on attempt {attempt + 1}: {type(e).__name__}: {e}")

        if attempt < 2:
            sleep_time = 2 * (attempt + 1)
            print(f"   Sleeping {sleep_time}s before retry...")
            time.sleep(sleep_time)

    print("❌ All retry attempts failed")
    return False

def test_environment_variables():
    """Test environment variable handling"""
    print("\n🌍 ENVIRONMENT VARIABLE TEST")
    print("=" * 50)

    # Test direct access
    direct_serper = os.environ.get('SERPER_KEY_ID')
    print(f"Direct access: {direct_serper[:10] + '...' if direct_serper else 'None'}")

    # Test setting manually
    test_key = "test_value_12345"
    os.environ['TEST_KEY'] = test_key
    retrieved = os.environ.get('TEST_KEY')
    print(f"Manual set/get: {'✅ Works' if retrieved == test_key else '❌ Failed'}")

    # Check current working directory
    print(f"Current dir: {os.getcwd()}")

    # List .env files
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.env'):
                print(f"Found env file: {os.path.join(root, file)}")

if __name__ == "__main__":
    print("🧪 Simple Search Test Suite")
    print("=" * 60)

    # Test environment first
    test_environment_variables()

    # Test basic search
    basic_success = test_basic_search()

    # Test retry mechanism
    retry_success = test_with_retries()

    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print(f"Basic search: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"Retry search: {'✅ PASS' if retry_success else '❌ FAIL'}")

    if not basic_success:
        print("\n💡 Troubleshooting:")
        print("1. Check if SERPER_KEY_ID is set correctly")
        print("2. Verify network connectivity")
        print("3. Check API key validity at https://serper.dev/")