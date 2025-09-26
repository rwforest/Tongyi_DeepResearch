#!/usr/bin/env python3
"""
Debug script to compare independent vs agent-based tool calling
"""

import os
import sys

# Try to load environment variables if dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv('.env')
    print("✅ Loaded environment from .env")
except ImportError:
    print("⚠️ python-dotenv not available, using system environment")

def test_independent_search():
    """Test search tool independently"""
    print("🔍 INDEPENDENT SEARCH TEST")
    print("=" * 50)

    try:
        from tool_search import Search
        search_tool = Search()

        test_params = {"query": ["artificial intelligence"]}
        print(f"Calling search tool with params: {test_params}")

        result = search_tool.call(test_params)

        if "[Search Error]" in result:
            print(f"❌ Independent search failed: {result}")
            return False
        else:
            print(f"✅ Independent search successful: {len(result)} chars")
            return True

    except Exception as e:
        print(f"❌ Independent search exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_search():
    """Test search tool through agent"""
    print("\n🤖 AGENT-BASED SEARCH TEST")
    print("=" * 50)

    try:
        from databricks_react_agent import DatabricksMultiTurnReactAgent

        # Simple mock predict function for testing
        def mock_predict_function(model_input):
            """Mock predict function that requests search"""
            return '''<think>
I need to search for information about artificial intelligence.
</think>

I'll search for information about this topic.

<tool_call>
{"name": "search", "arguments": {"query": ["artificial intelligence"]}}
</tool_call>'''

        # Initialize agent
        llm_cfg = {
            'model': 'test-model',
            'generate_cfg': {}
        }

        agent = DatabricksMultiTurnReactAgent(
            predict_function=mock_predict_function,
            llm=llm_cfg
        )

        # Test direct tool calling through agent
        print("Testing agent.custom_call_tool() directly...")

        tool_result = agent.custom_call_tool("search", {"query": ["artificial intelligence"]})

        if "[Search Error]" in str(tool_result):
            print(f"❌ Agent search failed: {tool_result}")
            return False
        else:
            print(f"✅ Agent search successful: {len(str(tool_result))} chars")
            return True

    except Exception as e:
        print(f"❌ Agent search exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_comparison():
    """Compare environment between contexts"""
    print("\n🌍 ENVIRONMENT COMPARISON")
    print("=" * 50)

    # Check key environment variables
    env_vars = ['SERPER_KEY_ID', 'TEMPERATURE', 'PRESENCE_PENALTY']

    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}: {value[:10]}..." if len(value) > 10 else f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")

    # Check working directory
    print(f"📁 Working directory: {os.getcwd()}")

    # Check if .env file exists
    env_file = '.env'
    if os.path.exists(env_file):
        print(f"✅ .env file exists")
        with open(env_file, 'r') as f:
            lines = f.readlines()
            print(f"📄 .env has {len(lines)} lines")
    else:
        print(f"❌ .env file not found")

def test_argument_formats():
    """Test different argument formats"""
    print("\n📝 ARGUMENT FORMAT TEST")
    print("=" * 50)

    try:
        from tool_search import Search
        search_tool = Search()

        # Test different formats
        test_cases = [
            {"query": ["single query in array"]},
            {"query": "single string query"},
        ]

        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest case {i}: {test_case}")
            try:
                result = search_tool.call(test_case)
                if "[Search Error]" in result:
                    print(f"❌ Failed: {result}")
                else:
                    print(f"✅ Success: {len(result)} chars")
            except Exception as e:
                print(f"❌ Exception: {e}")

    except Exception as e:
        print(f"❌ Setup failed: {e}")

if __name__ == "__main__":
    print("🐛 Tool Calling Debug Suite")
    print("=" * 60)

    # Run environment check first
    test_environment_comparison()

    # Test argument formats
    test_argument_formats()

    # Test independent search
    independent_success = test_independent_search()

    # Test agent-based search
    agent_success = test_agent_search()

    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print(f"Independent search: {'✅ PASS' if independent_success else '❌ FAIL'}")
    print(f"Agent search: {'✅ PASS' if agent_success else '❌ FAIL'}")

    if independent_success and not agent_success:
        print("\n🔍 ANALYSIS: Tool works independently but fails in agent context")
        print("Possible causes:")
        print("1. Environment variable differences")
        print("2. Argument format differences")
        print("3. Threading or context issues")
        print("4. Import path differences")
    elif not independent_success:
        print("\n🔍 ANALYSIS: Tool has fundamental issues")
        print("Check API key and network connectivity")
    else:
        print("\n🔍 ANALYSIS: Both tests passed - issue may be elsewhere")