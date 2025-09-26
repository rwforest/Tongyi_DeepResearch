#!/usr/bin/env python3
"""
Test ReAct agent tool calling functionality
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

def test_tool_calling():
    """Test tool calling within the ReAct agent"""
    print("🤖 Testing ReAct Agent Tool Calling")
    print("=" * 60)

    # Test 1: Import and initialize agent
    try:
        from databricks_react_agent import DatabricksMultiTurnReactAgent
        print("✅ Successfully imported DatabricksMultiTurnReactAgent")

        # Simple mock predict function for testing
        def mock_predict_function(model_input):
            """Mock predict function that simulates tool calling"""
            print(f"🔮 Mock predict called with input type: {type(model_input)}")

            item = model_input[0]
            prompt = item.get("prompt", "")

            # Check if this is asking for a search
            if "search" in prompt.lower() or "find" in prompt.lower():
                # Simulate the model deciding to use search tool
                return '''<think>
I need to search for information about this topic. Let me use the search tool.
</think>

I'll search for information about this topic.

<tool_call>
{"name": "search", "arguments": {"query": ["artificial intelligence"]}}
</tool_call>'''
            else:
                # Simulate regular response
                return '''<think>
This seems like a straightforward question I can answer directly.
</think>

<answer>
This is a test response from the mock predict function.
</answer>'''

        # Initialize agent with mock function
        llm_cfg = {
            'model': 'test-model',
            'generate_cfg': {
                'max_input_tokens': 320000,
                'max_retries': 10,
                'temperature': 0.85,
                'top_p': 0.95,
                'presence_penalty': 1.1
            },
            'model_type': 'databricks'
        }

        agent = DatabricksMultiTurnReactAgent(
            predict_function=mock_predict_function,
            llm=llm_cfg,
            function_list=["search", "visit", "google_scholar", "PythonInterpreter"]
        )
        print("✅ Successfully initialized DatabricksMultiTurnReactAgent")

    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 2: Test individual tool calling
    print("\n" + "=" * 60)
    print("🔧 Testing Individual Tool Calling")

    try:
        from tool_search import Search
        search_tool = Search()

        # Test direct tool call
        test_params = {"query": ["test search"]}
        result = search_tool.call(test_params)

        if "[Search Error]" in result:
            print("❌ Direct tool call failed:")
            print(f"   {result}")
        else:
            print("✅ Direct tool call successful!")
            print(f"📄 Result length: {len(result)} characters")

    except Exception as e:
        print(f"❌ Direct tool call failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Test agent's tool calling mechanism
    print("\n" + "=" * 60)
    print("🔗 Testing Agent Tool Calling Mechanism")

    try:
        # Test the agent's custom_call_tool method directly
        tool_result = agent.custom_call_tool("search", {"query": ["test query"]})
        print("✅ Agent tool calling mechanism works!")
        print(f"📄 Tool result type: {type(tool_result)}")
        print(f"📄 Tool result length: {len(str(tool_result))} characters")

        if "[Search Error]" in str(tool_result):
            print("⚠️ Tool returned error (likely API key issue):")
            print(f"   {tool_result}")
        else:
            print("✅ Tool executed successfully through agent")

    except Exception as e:
        print(f"❌ Agent tool calling failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: Test full ReAct workflow (simplified)
    print("\n" + "=" * 60)
    print("🔄 Testing Full ReAct Workflow")

    try:
        # Create test data for agent
        test_data = {
            'item': {
                'question': 'Search for information about artificial intelligence',
                'answer': 'Test reference answer'
            }
        }

        print("🚀 Running simplified ReAct workflow...")
        print("   Question: Search for information about artificial intelligence")

        # This should trigger the search tool
        result = agent._run(test_data, model="test-model")

        print("✅ ReAct workflow completed!")
        print(f"📄 Result keys: {list(result.keys())}")
        print(f"📄 Prediction: {result.get('prediction', 'No prediction')[:200]}...")
        print(f"📄 Termination: {result.get('termination', 'Unknown')}")

    except Exception as e:
        print(f"❌ Full ReAct workflow failed: {e}")
        import traceback
        traceback.print_exc()


def test_tool_json_parsing():
    """Test JSON parsing for tool calls"""
    print("\n" + "=" * 60)
    print("🔍 Testing Tool Call JSON Parsing")

    import json5

    # Test various tool call formats
    test_cases = [
        '{"name": "search", "arguments": {"query": ["test"]}}',
        '{"name": "search", "arguments": {"query": "single query"}}',
        '{"name": "PythonInterpreter", "arguments": {}}',
    ]

    for i, test_case in enumerate(test_cases, 1):
        try:
            parsed = json5.loads(test_case)
            print(f"✅ Test case {i}: JSON parsed successfully")
            print(f"   Tool: {parsed.get('name', 'unknown')}")
            print(f"   Args: {parsed.get('arguments', {})}")
        except Exception as e:
            print(f"❌ Test case {i}: JSON parsing failed: {e}")


if __name__ == "__main__":
    print("🧪 ReAct Tool Calling Test Suite")
    print("=" * 80)

    # Run tests
    test_tool_calling()
    test_tool_json_parsing()

    print("\n" + "=" * 80)
    print("✅ ReAct tool calling test suite completed!")
    print("\n💡 If you see errors:")
    print("   1. Check that SERPER_KEY_ID is set in .env")
    print("   2. Ensure all imports work correctly")
    print("   3. Verify your predict function format")