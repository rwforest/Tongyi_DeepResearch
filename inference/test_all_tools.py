#!/usr/bin/env python3
"""
Test all tools to verify API keys are properly loaded
"""

import os
import sys

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
        print(f"❌ .env file not found at {os.path.abspath(env_file)}")

def test_environment_setup():
    """Test environment variable loading"""
    print("🌍 ENVIRONMENT SETUP TEST")
    print("=" * 50)

    # Load environment manually first
    load_env_manually()

    # Test all required API keys
    api_keys = {
        'SERPER_KEY_ID': 'Search & Scholar tools',
        'PERPLEXITY_API_KEY': 'Perplexity search tool',
        'JINA_API_KEYS': 'Visit tool (web page reading)',
        'API_KEY': 'Visit tool (content summarization)',
        'API_BASE': 'Visit tool (content summarization)',
        'SUMMARY_MODEL_NAME': 'Visit tool (content summarization)',
        'SANDBOX_FUSION_ENDPOINT': 'Python interpreter tool',
        'DASHSCOPE_API_KEY': 'File parsing tool',
        'DASHSCOPE_API_BASE': 'File parsing tool'
    }

    missing_keys = []
    placeholder_keys = []

    for key, description in api_keys.items():
        value = os.environ.get(key)
        if not value:
            print(f"❌ {key}: Missing - {description}")
            missing_keys.append(key)
        elif value.startswith('your_'):
            print(f"⚠️ {key}: Placeholder - {description}")
            placeholder_keys.append(key)
        else:
            print(f"✅ {key}: Set - {description}")

    print(f"\n📊 Summary:")
    print(f"   ✅ Configured: {len(api_keys) - len(missing_keys) - len(placeholder_keys)}")
    print(f"   ⚠️ Placeholder: {len(placeholder_keys)}")
    print(f"   ❌ Missing: {len(missing_keys)}")

    return missing_keys, placeholder_keys

def test_tool_imports():
    """Test importing all tools"""
    print("\n🔧 TOOL IMPORT TEST")
    print("=" * 50)

    tools_to_test = [
        ('tool_search', 'Search'),
        ('tool_perplexity', 'PerplexitySearch'),
        ('tool_scholar', 'Scholar'),
        ('tool_visit', 'Visit'),
        ('tool_python', 'PythonInterpreter'),
        ('tool_file', 'FileParser')
    ]

    imported_tools = []
    failed_imports = []

    for module_name, class_name in tools_to_test:
        try:
            module = __import__(module_name)
            tool_class = getattr(module, class_name)
            tool_instance = tool_class()
            print(f"✅ {module_name}.{class_name}: Import successful")
            imported_tools.append((module_name, class_name, tool_instance))
        except Exception as e:
            print(f"❌ {module_name}.{class_name}: Import failed - {e}")
            failed_imports.append((module_name, class_name, str(e)))

    return imported_tools, failed_imports

def test_agent_initialization():
    """Test agent initialization with tools"""
    print("\n🤖 AGENT INITIALIZATION TEST")
    print("=" * 50)

    try:
        from databricks_react_agent import DatabricksMultiTurnReactAgent

        # Simple mock predict function
        def mock_predict_function(model_input):
            return "Mock response for testing"

        llm_cfg = {
            'model': 'test-model',
            'generate_cfg': {}
        }

        agent = DatabricksMultiTurnReactAgent(
            predict_function=mock_predict_function,
            llm=llm_cfg
        )

        print(f"✅ Agent initialized successfully")
        print(f"📋 Available tools: {list(agent.custom_call_tool.__globals__['TOOL_MAP'].keys())}")
        return True

    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tool_api_key_access():
    """Test that tools can access API keys at runtime"""
    print("\n🔑 TOOL API KEY ACCESS TEST")
    print("=" * 50)

    # Test each tool's API key access
    try:
        # Test Search tool
        from tool_search import get_serper_key
        serper_key = get_serper_key()
        print(f"✅ Search tool SERPER_KEY: {'SET' if serper_key else 'MISSING'}")

        # Test Perplexity tool
        from tool_perplexity import get_perplexity_key
        perplexity_key = get_perplexity_key()
        print(f"✅ Perplexity tool API_KEY: {'SET' if perplexity_key else 'MISSING'}")

        # Test Scholar tool
        from tool_scholar import get_scholar_serper_key
        scholar_key = get_scholar_serper_key()
        print(f"✅ Scholar tool SERPER_KEY: {'SET' if scholar_key else 'MISSING'}")

        # Test Visit tool
        from tool_visit import get_visit_config
        visit_config = get_visit_config()
        print(f"✅ Visit tool JINA_KEYS: {'SET' if visit_config['jina_keys'] else 'MISSING'}")
        print(f"✅ Visit tool API_KEY: {'SET' if visit_config['api_key'] else 'MISSING'}")

        # Test Python tool
        from tool_python import get_sandbox_endpoints
        endpoints = get_sandbox_endpoints()
        print(f"✅ Python tool SANDBOX_ENDPOINTS: {'SET' if endpoints else 'MISSING'}")

        return True

    except Exception as e:
        print(f"❌ Tool API key access test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 Comprehensive Tool Test Suite")
    print("=" * 80)

    # Test 1: Environment setup
    missing_keys, placeholder_keys = test_environment_setup()

    # Test 2: Tool imports
    imported_tools, failed_imports = test_tool_imports()

    # Test 3: Agent initialization
    agent_success = test_agent_initialization()

    # Test 4: Tool API key access
    api_key_access_success = test_tool_api_key_access()

    # Summary
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 80)

    print(f"🌍 Environment: {len(missing_keys) + len(placeholder_keys)} keys need attention")
    print(f"🔧 Tool imports: {len(imported_tools)} successful, {len(failed_imports)} failed")
    print(f"🤖 Agent init: {'✅ SUCCESS' if agent_success else '❌ FAILED'}")
    print(f"🔑 API key access: {'✅ SUCCESS' if api_key_access_success else '❌ FAILED'}")

    if missing_keys or placeholder_keys:
        print(f"\n💡 NEXT STEPS:")
        if placeholder_keys:
            print(f"   1. Replace placeholder values in .env:")
            for key in placeholder_keys:
                print(f"      - {key}")
        if missing_keys:
            print(f"   2. Add missing keys to .env:")
            for key in missing_keys:
                print(f"      - {key}")

    if failed_imports:
        print(f"\n🔧 IMPORT ISSUES:")
        for module_name, class_name, error in failed_imports:
            print(f"   - {module_name}.{class_name}: {error}")

    overall_success = (
        len(failed_imports) == 0 and
        agent_success and
        api_key_access_success
    )

    print(f"\n🎯 OVERALL STATUS: {'✅ READY' if overall_success else '⚠️ NEEDS SETUP'}")

if __name__ == "__main__":
    main()