#!/usr/bin/env python3
"""
Test function for the Search tool
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

def test_search():
    """Test the Search tool functionality"""
    print("🔍 Testing Search Tool")
    print("=" * 50)

    # Check environment variable
    serper_key = os.environ.get('SERPER_KEY_ID')
    if serper_key:
        print(f"✅ SERPER_KEY_ID found: {serper_key[:10]}...")
    else:
        print("⚠️ SERPER_KEY_ID not found in environment")
        print("   Set it in your .env file or environment")

    # Test importing the tool
    try:
        from tool_search import Search
        print("✅ Search tool imported successfully")

        # Initialize the search tool
        search_tool = Search()
        print("✅ Search tool initialized")

        # Test search parameters
        test_queries = [
            "artificial intelligence",
            "machine learning transformers"
        ]

        test_params = {"query": test_queries}

        print(f"\n🔍 Testing search with queries: {test_queries}")
        print("-" * 40)

        # Call the search tool
        result = search_tool.call(test_params)

        if "[Search Error]" in result:
            print("❌ Search failed:")
            print(f"   {result}")

            if "SERPER_KEY_ID" in result:
                print("\n💡 To fix this:")
                print("   1. Go to https://serper.dev/")
                print("   2. Sign up and get your API key")
                print("   3. Add to inference/.env:")
                print("      SERPER_KEY_ID=your_api_key_here")
        else:
            print("✅ Search successful!")
            print(f"📄 Result length: {len(result)} characters")
            print("\n📝 First 200 characters of results:")
            print("-" * 40)
            print(result[:200] + "..." if len(result) > 200 else result)

    except ImportError as e:
        print(f"❌ Failed to import Search tool: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_api_key_only():
    """Test just the API key configuration"""
    print("🔑 Testing API Key Configuration")
    print("=" * 40)

    # Load environment
    load_dotenv('.env')

    serper_key = os.environ.get('SERPER_KEY_ID')

    if serper_key:
        print("✅ SERPER_KEY_ID is set")
        print(f"   Key preview: {serper_key[:8]}{'*' * 8}")

        # Test if it's not the default placeholder
        if serper_key.startswith('your_'):
            print("⚠️ API key appears to be a placeholder")
            print("   Replace with your actual Serper API key")
        else:
            print("✅ API key looks valid")

    else:
        print("❌ SERPER_KEY_ID not found")
        print("\n💡 Setup instructions:")
        print("   1. Copy inference/.env.example to inference/.env")
        print("   2. Get API key from https://serper.dev/")
        print("   3. Replace 'your_serper_key' with actual key")


def test_connection():
    """Test connection to Serper API using your working version"""
    print("🌐 Testing Connection to Serper API")
    print("=" * 40)

    import http.client
    import json

    serper_key = os.environ.get('SERPER_KEY_ID')

    if not serper_key or serper_key.startswith('your_'):
        print("❌ No valid API key found")
        return

    try:
        # Use your exact working version
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({
            "q": "test query"  # Simplified like your working version
        })
        headers = {
            'X-API-KEY': serper_key,
            'Content-Type': 'application/json'
        }

        print("🔄 Sending test request...")
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()

        print(f"📡 Response status: {res.status}")

        if res.status == 200:
            print("✅ Connection successful!")
            data = res.read()
            result = json.loads(data.decode("utf-8"))
            if "organic" in result:
                print(f"📄 Found {len(result['organic'])} search results")
                # Show first result title as example
                if result['organic']:
                    first_title = result['organic'][0].get('title', 'No title')
                    print(f"📝 First result: {first_title}")
            else:
                print("⚠️ Unexpected response format")
                print(f"📄 Response keys: {list(result.keys())}")
        elif res.status == 401:
            print("❌ Authentication failed - check your API key")
        elif res.status == 429:
            print("⚠️ Rate limit exceeded - try again later")
        else:
            print(f"❌ HTTP error: {res.status}")

    except Exception as e:
        print(f"❌ Connection failed: {e}")


if __name__ == "__main__":
    print("🧪 Search Tool Test Suite")
    print("=" * 60)

    # Run all tests
    test_api_key_only()
    print("\n")
    test_connection()
    print("\n")
    test_search()

    print("\n" + "=" * 60)
    print("✅ Test suite completed!")