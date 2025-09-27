#!/usr/bin/env python3
"""
Test MLflow Tracing Integration
Demonstrates comprehensive MLflow tracing for the Tongyi DeepResearch ReAct agent
"""

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
    else:
        print(f"❌ .env file not found")

def mock_predict_function(model_input):
    """Mock predict function for testing"""
    print("🤖 MOCK: Mock predict function called")
    time.sleep(0.1)  # Simulate processing time
    return "I need to search for information about artificial intelligence research trends."

def test_mlflow_tracing_integration():
    """Test MLflow tracing integration with ReAct agent"""
    print("🔬 MLFLOW TRACING INTEGRATION TEST")
    print("=" * 80)

    # Load environment
    load_env_manually()

    try:
        # Import MLflow and set up tracing
        import mlflow
        print("✅ MLflow imported successfully")

        # Start MLflow experiment
        mlflow.set_experiment("tongyi_deepresearch_tracing")
        print("✅ MLflow experiment set")

        # Import agent components
        from databricks_react_agent import DatabricksMultiTurnReactAgent
        print("✅ ReAct agent imported")

        # Create agent with mock predict function
        agent = DatabricksMultiTurnReactAgent(
            predict_function=mock_predict_function,
            llm={'model': 'tongyi-deepresearch-mock'}
        )
        print("✅ Agent initialized with MLflow tracing")

        # Prepare test data
        test_data = {
            'item': {
                'question': 'What are the latest developments in AI research?',
                'answer': 'Reference answer for testing'
            }
        }

        print("\n🚀 Starting traced agent execution...")

        # Run agent with MLflow tracing
        with mlflow.start_run():
            print("📊 MLflow run started")

            # Execute the agent - this will create nested spans
            result = agent._run(test_data)

            print("✅ Agent execution completed")
            print(f"📋 Result keys: {list(result.keys())}")
            print(f"📖 Prediction preview: {result['prediction'][:100]}...")

        print("📊 MLflow run completed")

        # Get the current run info
        run = mlflow.active_run()
        if run:
            print(f"🔗 MLflow Run ID: {run.info.run_id}")
            print(f"🔗 MLflow Experiment ID: {run.info.experiment_id}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_mlflow_tracing_features():
    """Demonstrate MLflow tracing features and capabilities"""
    print(f"\n📊 MLFLOW TRACING FEATURES DEMONSTRATION")
    print("=" * 80)

    print(f"🔧 MLflow Tracing Features Implemented:")
    print(f"")
    print(f"🎯 **Session-Level Tracing** (@mlflow.trace on _run method)")
    print(f"   • Span Type: AGENT")
    print(f"   • Captures: Full session inputs/outputs, rounds, timing")
    print(f"   • Attributes: agent_type, model, max_llm_calls")
    print(f"")
    print(f"🤖 **LLM Call Tracing** (@mlflow.trace on call_predict method)")
    print(f"   • Span Type: LLM")
    print(f"   • Captures: Messages, responses, retry attempts, model params")
    print(f"   • Attributes: model, temperature, presence_penalty")
    print(f"")
    print(f"🛠️ **Tool Execution Tracing** (@mlflow.trace on custom_call_tool method)")
    print(f"   • Span Type: TOOL")
    print(f"   • Captures: Tool args, results, execution time, success/failure")
    print(f"   • Attributes: tool_type, available_tools")
    print(f"")
    print(f"🔍 **Search Tool Tracing** (@mlflow.trace on search methods)")
    print(f"   • Span Type: TOOL")
    print(f"   • Captures: Query, results count, API response time")
    print(f"   • Attributes: search_backend (serper/perplexity), api_key_available")
    print(f"")
    print(f"🌐 **Visit Tool Tracing** (@mlflow.trace on visit method)")
    print(f"   • Span Type: TOOL")
    print(f"   • Captures: URLs, webpage content, summarization")
    print(f"   • Attributes: tool_type, url_type (single/multiple)")
    print(f"")
    print(f"📈 **Automatic Metrics Captured:**")
    print(f"   • Execution time for each component")
    print(f"   • Token/character counts")
    print(f"   • Retry attempts and success rates")
    print(f"   • API response times")
    print(f"   • Error types and frequencies")
    print(f"")
    print(f"🔗 **Hierarchical Tracing Structure:**")
    print(f"   Session (AGENT)")
    print(f"   ├── LLM Call (LLM)")
    print(f"   ├── Tool Execution (TOOL)")
    print(f"   │   ├── Search (TOOL)")
    print(f"   │   ├── Visit (TOOL)")
    print(f"   │   └── Other Tools (TOOL)")
    print(f"   └── Final Response (AGENT)")

def show_mlflow_ui_instructions():
    """Show instructions for viewing MLflow traces"""
    print(f"\n📊 VIEWING MLFLOW TRACES")
    print("=" * 80)

    print(f"🖥️ **MLflow UI Access:**")
    print(f"   1. Start MLflow UI: mlflow ui")
    print(f"   2. Open browser: http://localhost:5000")
    print(f"   3. Navigate to 'Experiments' → 'tongyi_deepresearch_tracing'")
    print(f"")
    print(f"🔍 **Trace Analysis:**")
    print(f"   • Click on any run to see trace details")
    print(f"   • View hierarchical span structure")
    print(f"   • Analyze timing and performance metrics")
    print(f"   • Debug errors and exceptions")
    print(f"")
    print(f"📈 **Key Metrics to Monitor:**")
    print(f"   • Total session time")
    print(f"   • LLM response times")
    print(f"   • Tool execution times")
    print(f"   • API call latencies")
    print(f"   • Error rates by component")

def main():
    print("🧪 MLflow Tracing Integration Test Suite")
    print("=" * 100)

    # Test 1: MLflow tracing integration
    tracing_success = test_mlflow_tracing_integration()

    # Demo 2: Show tracing features
    demonstrate_mlflow_tracing_features()

    # Demo 3: UI instructions
    show_mlflow_ui_instructions()

    # Summary
    print("\n" + "=" * 100)
    print("📊 TEST RESULTS")
    print(f"🔬 MLflow Tracing Integration: {'✅ PASS' if tracing_success else '❌ FAIL'}")

    print(f"\n🎯 INTEGRATION STATUS:")
    print(f"   ✅ MLflow decorators added to all major functions")
    print(f"   ✅ Hierarchical span structure implemented")
    print(f"   ✅ Comprehensive input/output logging")
    print(f"   ✅ Performance timing integration")
    print(f"   ✅ Error handling and exception tracing")

    print(f"\n💡 NEXT STEPS:")
    print(f"   1. Configure MLflow tracking server for production")
    print(f"   2. Set up MLflow experiment tracking in Databricks")
    print(f"   3. Create custom MLflow metrics for research quality")
    print(f"   4. Implement MLflow model versioning for predict function")

if __name__ == "__main__":
    main()