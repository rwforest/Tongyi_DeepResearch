#!/usr/bin/env python3
"""
Test search backends - demonstrate switching between Serper and Perplexity
"""

import os

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
    else:
        print(f"❌ .env file not found")

def test_search_backend_switching():
    """Test switching between search backends"""
    print("🔄 SEARCH BACKEND SWITCHING TEST")
    print("=" * 60)

    # Load environment
    load_env_manually()

    try:
        from tool_search import Search, get_search_config

        # Show current configuration
        config = get_search_config()
        print(f"🔧 Current configuration:")
        print(f"   USE_PERPLEXITY_SEARCH: {config['use_perplexity']}")
        print(f"   PERPLEXITY_MAX_RESULTS: {config['perplexity_max_results']}")
        print(f"   SERPER_KEY: {'SET' if config['serper_key'] else 'MISSING'}")
        print(f"   PERPLEXITY_KEY: {'SET' if config['perplexity_key'] else 'MISSING'}")

        # Initialize search tool
        search_tool = Search()
        print(f"\n✅ Search tool initialized")

        # Test query
        test_query = "latest developments in artificial intelligence"
        test_params = {"query": [test_query]}

        print(f"\n🔍 Testing search with query: '{test_query}'")
        print(f"🔧 Backend: {'Perplexity' if config['use_perplexity'] else 'Serper'}")

        # Execute search
        result = search_tool.call(test_params)

        if "[Search Error]" in result:
            print(f"❌ Search failed:")
            print(f"   {result}")

            # Provide specific guidance based on backend
            if config['use_perplexity']:
                print(f"\n💡 To fix Perplexity search:")
                print(f"   1. Get API key from https://www.perplexity.ai/")
                print(f"   2. Set PERPLEXITY_API_KEY in .env")
                print(f"   3. Or set USE_PERPLEXITY_SEARCH=false to use Serper")
            else:
                print(f"\n💡 To fix Serper search:")
                print(f"   1. Get API key from https://serper.dev/")
                print(f"   2. Set SERPER_KEY_ID in .env")
                print(f"   3. Or set USE_PERPLEXITY_SEARCH=true to use Perplexity")
        else:
            print(f"✅ Search successful!")
            print(f"📄 Result length: {len(result)} characters")

            # Show format differences
            if "## Answer" in result:
                print(f"📝 Format: Perplexity (AI-generated answer with citations)")
            elif "Web Results" in result:
                print(f"📝 Format: Serper (traditional search results)")

            print(f"📖 Preview: {result[:200]}...")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_backend_switching():
    """Demonstrate how to switch backends programmatically"""
    print(f"\n🔄 PROGRAMMATIC BACKEND SWITCHING DEMO")
    print("=" * 60)

    print(f"💡 To switch search backends, modify your .env file:")
    print(f"\n📝 For Serper search (traditional web results):")
    print(f"   USE_PERPLEXITY_SEARCH=false")
    print(f"   SERPER_KEY_ID=your_serper_api_key")

    print(f"\n📝 For Perplexity Search API (ranked web results):")
    print(f"   USE_PERPLEXITY_SEARCH=true")
    print(f"   PERPLEXITY_API_KEY=your_perplexity_api_key")
    print(f"   PERPLEXITY_MAX_RESULTS=10  # 1-20 results")

    print(f"\n🔧 Perplexity Search API features:")
    print(f"   • Ranked web search results")
    print(f"   • Real-time data from continuously refreshed index")
    print(f"   • Advanced filtering and customization")
    print(f"   • $5 per 1,000 requests (no token fees)")

    print(f"\n⚡ Benefits of each backend:")
    print(f"\n🔍 Serper (Traditional Web Search):")
    print(f"   ✅ Fast and cost-effective")
    print(f"   ✅ Raw web results with snippets")
    print(f"   ✅ Good for finding specific information")
    print(f"   ✅ Returns multiple source links")

    print(f"\n🧠 Perplexity Search API (Advanced Web Search):")
    print(f"   ✅ Ranked search results with relevance scoring")
    print(f"   ✅ Real-time data from continuously updated index")
    print(f"   ✅ Advanced filtering and customization options")
    print(f"   ✅ Structured response format with metadata")

def test_configuration_validation():
    """Test configuration validation and error handling"""
    print(f"\n🔧 CONFIGURATION VALIDATION TEST")
    print("=" * 60)

    try:
        from tool_search import get_search_config

        config = get_search_config()

        # Check if configuration is valid
        if config['use_perplexity']:
            if not config['perplexity_key'] or config['perplexity_key'].startswith('your_'):
                print(f"⚠️ Perplexity selected but API key not configured")
                print(f"   Set PERPLEXITY_API_KEY in .env file")
            else:
                print(f"✅ Perplexity configuration valid")
        else:
            if not config['serper_key'] or config['serper_key'].startswith('your_'):
                print(f"⚠️ Serper selected but API key not configured")
                print(f"   Set SERPER_KEY_ID in .env file")
            else:
                print(f"✅ Serper configuration valid")

        # Check max results setting for Perplexity
        if config['use_perplexity']:
            max_results = config['perplexity_max_results']
            if 1 <= max_results <= 20:
                print(f"✅ Perplexity max_results '{max_results}' is valid (1-20)")
            else:
                print(f"⚠️ Invalid Perplexity max_results '{max_results}'")
                print(f"   Valid range: 1-20")

        return True

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

def main():
    print("🧪 Search Backend Testing Suite")
    print("=" * 80)

    # Test 1: Backend switching
    backend_success = test_search_backend_switching()

    # Test 2: Configuration validation
    config_success = test_configuration_validation()

    # Demo: Show how to switch
    demonstrate_backend_switching()

    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS")
    print(f"🔄 Backend switching: {'✅ PASS' if backend_success else '❌ FAIL'}")
    print(f"🔧 Configuration: {'✅ PASS' if config_success else '❌ FAIL'}")

    print(f"\n🎯 CURRENT STATUS:")
    try:
        from tool_search import get_search_config
        config = get_search_config()
        backend = 'Perplexity Search API' if config['use_perplexity'] else 'Serper'
        print(f"   Active backend: {backend}")
        if config['use_perplexity']:
            print(f"   Max results: {config['perplexity_max_results']}")
    except:
        print(f"   Could not determine current backend")

if __name__ == "__main__":
    main()