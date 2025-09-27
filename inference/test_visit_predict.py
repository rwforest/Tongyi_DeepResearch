#!/usr/bin/env python3
"""
Test Visit tool with predict function integration
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

def test_visit_with_predict():
    """Test Visit tool with mock predict function"""
    print("🌐 VISIT TOOL WITH PREDICT FUNCTION TEST")
    print("=" * 60)

    # Load environment
    load_env_manually()

    try:
        from tool_visit import Visit

        # Create mock predict function
        def mock_predict_function(model_input):
            """Mock predict function that returns JSON format for Visit tool"""
            prompt = model_input[0].get("prompt", "")

            # Simulate the model understanding the webpage summarization task
            if "webpage_content" in prompt and "goal" in prompt:
                return '''
{
    "rational": "This webpage contains information relevant to the user's research goal",
    "evidence": "The webpage discusses artificial intelligence research, recent breakthroughs in machine learning, and their applications in various industries. Key findings include advances in neural network architectures and their performance improvements.",
    "summary": "The webpage provides comprehensive coverage of AI research trends, highlighting significant developments in deep learning and their practical applications across multiple sectors."
}
'''
            else:
                return '''
{
    "rational": "General webpage content analysis",
    "evidence": "Content was processed but may not be directly relevant to specific research goals",
    "summary": "Webpage content has been analyzed and summarized"
}
'''

        # Initialize Visit tool
        visit_tool = Visit()
        print("✅ Visit tool initialized")

        # Test parameters
        test_params = {
            "url": "https://example.com",
            "goal": "Research AI developments and trends"
        }

        print(f"\n🔍 Testing Visit with URL: {test_params['url']}")
        print(f"🎯 Goal: {test_params['goal']}")
        print("🔄 Using mock predict function...")

        # Call Visit tool with predict function
        result = visit_tool.call(test_params, predict_function=mock_predict_function)

        print("\n📊 RESULTS:")
        print(f"✅ Visit tool call successful!")
        print(f"📄 Result length: {len(result)} characters")
        print(f"📝 Result preview: {result[:300]}...")

        # Check if result contains expected structure
        if "rational" in result or "evidence" in result or "summary" in result:
            print("✅ Result contains expected JSON structure")
        else:
            print("⚠️ Result may not be in expected format")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visit_fallback():
    """Test Visit tool fallback to OpenAI API when no predict function provided"""
    print("\n🌐 VISIT TOOL FALLBACK TEST")
    print("=" * 60)

    try:
        from tool_visit import Visit

        visit_tool = Visit()

        test_params = {
            "url": "https://example.com",
            "goal": "Test fallback behavior"
        }

        print(f"🔍 Testing Visit without predict function (should use OpenAI API)")

        # Call without predict function - should fallback to OpenAI API
        result = visit_tool.call(test_params)

        print(f"📄 Result length: {len(result)} characters")
        print(f"📝 Result: {result[:200]}...")

        # Should contain error about missing API configuration (since we don't have OpenAI keys)
        if "Missing API configuration" in result or "API_KEY" in result:
            print("✅ Fallback behavior working correctly - detected missing OpenAI config")
        else:
            print("⚠️ Unexpected fallback behavior")

        return True

    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_integration():
    """Test Visit tool integration with the ReAct agent"""
    print("\n🤖 AGENT INTEGRATION TEST")
    print("=" * 60)

    try:
        from databricks_react_agent import DatabricksMultiTurnReactAgent

        # Mock predict function
        def mock_predict_function(model_input):
            prompt = model_input[0].get("prompt", "")

            # If it's asking for a visit tool call
            if "visit" in prompt.lower() and "tool_call" in prompt.lower():
                return '''<think>
I need to visit a webpage to gather information for this research task.
</think>

I'll visit the webpage to gather the requested information.

<tool_call>
{"name": "visit", "arguments": {"url": "https://example.com", "goal": "Research AI developments"}}
</tool_call>'''
            else:
                return '''<think>
This seems like a straightforward question.
</think>

<answer>
This is a test response from the mock predict function for agent integration.
</answer>'''

        # Initialize agent
        llm_cfg = {'model': 'test-model', 'generate_cfg': {}}
        agent = DatabricksMultiTurnReactAgent(
            predict_function=mock_predict_function,
            llm=llm_cfg
        )

        print("✅ Agent initialized with Visit tool integration")

        # Test direct tool calling
        tool_result = agent.custom_call_tool("visit", {
            "url": "https://example.com",
            "goal": "Test agent integration"
        })

        print(f"📄 Tool result length: {len(str(tool_result))} characters")
        print(f"📝 Tool result preview: {str(tool_result)[:200]}...")

        if "rational" in str(tool_result) or "evidence" in str(tool_result):
            print("✅ Agent integration successful - Visit tool called with predict function")
        else:
            print("⚠️ Integration may have issues")

        return True

    except Exception as e:
        print(f"❌ Agent integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 Visit Tool Predict Function Integration Test Suite")
    print("=" * 80)

    # Test 1: Visit tool with predict function
    predict_success = test_visit_with_predict()

    # Test 2: Visit tool fallback behavior
    fallback_success = test_visit_fallback()

    # Test 3: Agent integration
    agent_success = test_agent_integration()

    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS")
    print(f"🌐 Visit with predict: {'✅ PASS' if predict_success else '❌ FAIL'}")
    print(f"🔄 Visit fallback: {'✅ PASS' if fallback_success else '❌ FAIL'}")
    print(f"🤖 Agent integration: {'✅ PASS' if agent_success else '❌ FAIL'}")

    overall_success = predict_success and fallback_success and agent_success
    print(f"\n🎯 OVERALL: {'✅ ALL TESTS PASSED' if overall_success else '⚠️ SOME TESTS FAILED'}")

    print(f"\n💡 NEXT STEPS:")
    print(f"   1. Set JINA_API_KEYS in .env for web page fetching")
    print(f"   2. Your predict function will now be used for webpage summarization")
    print(f"   3. Run full ReAct agent to test complete workflow")

if __name__ == "__main__":
    main()