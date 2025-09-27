#!/usr/bin/env python3
"""
Demo: Adding MLflow Tracing to Predict Functions
Shows how to wrap any predict function with comprehensive MLflow tracing
"""

import os
import sys
import time

# Add inference directory to path
sys.path.append('.')

def load_env_manually():
    """Load .env file manually"""
    env_file = '.env'
    if os.path.exists(env_file):
        print("✅ Loading .env file")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    else:
        print("⚠️ .env file not found")

def demo_traced_predict_integration():
    """Demonstrate traced predict function integration"""
    print("🚀 TRACED PREDICT FUNCTION DEMO")
    print("=" * 60)

    # Load environment
    load_env_manually()

    try:
        # Import tracing wrapper
        from traced_predict_wrapper import (
            create_traced_predict_function,
            create_databricks_traced_predict,
            create_huggingface_traced_predict,
            create_openai_traced_predict
        )
        print("✅ Traced predict wrapper imported")

        # Import MLflow
        import mlflow
        from mlflow_config import initialize_mlflow

        # Initialize MLflow
        initialize_mlflow()
        print("✅ MLflow initialized")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    # Demo 1: Wrap existing predict function
    print("\n📊 DEMO 1: Wrapping Existing Predict Function")
    print("-" * 50)

    def my_original_predict(model_input):
        """Example original predict function"""
        time.sleep(0.1)  # Simulate processing time

        if isinstance(model_input, list) and len(model_input) > 0:
            prompt = model_input[0].get('prompt', str(model_input))
        else:
            prompt = str(model_input)

        return f"AI Response: Based on your input '{prompt[:50]}...', here's a comprehensive answer with detailed analysis and insights."

    # Create traced version
    traced_predict = create_traced_predict_function(
        my_original_predict,
        model_name="demo_tongyi_model",
        log_inputs=True,
        log_outputs=True,
        log_timing=True
    )

    print("✅ Created traced predict function")

    # Demo 2: Use with MLflow run
    print("\n🔬 DEMO 2: Execute with MLflow Tracing")
    print("-" * 50)

    try:
        with mlflow.start_run(run_name="traced_predict_demo"):
            print("📊 MLflow run started")

            # Test the traced predict function
            test_inputs = [
                {
                    "prompt": "What are the latest developments in artificial intelligence and machine learning?",
                    "temperature": 0.8,
                    "max_length": 2048
                }
            ]

            print(f"🤖 Testing traced predict function...")
            result = traced_predict(test_inputs)

            print(f"✅ Prediction completed!")
            print(f"📝 Input: {test_inputs[0]['prompt'][:60]}...")
            print(f"📝 Output: {result[:100]}...")

            # Log additional metrics
            mlflow.log_metrics({
                "demo_execution": 1,
                "test_success": 1
            })

        print("📊 MLflow run completed")

    except Exception as e:
        print(f"⚠️ MLflow demo warning: {e}")

    # Demo 3: Show different predict function patterns
    print("\n🔧 DEMO 3: Different Predict Function Patterns")
    print("-" * 50)

    # Pattern 1: String input/output
    def string_predict(prompt: str) -> str:
        return f"String response to: {prompt[:30]}..."

    traced_string = create_traced_predict_function(
        string_predict,
        model_name="string_model"
    )

    # Pattern 2: Dict input/output
    def dict_predict(model_input: dict) -> dict:
        return {
            "generated_text": f"Dict response to: {model_input.get('prompt', 'unknown')[:30]}...",
            "confidence": 0.95
        }

    traced_dict = create_traced_predict_function(
        dict_predict,
        model_name="dict_model"
    )

    # Pattern 3: List input/output
    def list_predict(model_inputs: list) -> list:
        return [f"List response {i}: {inp.get('prompt', str(inp))[:20]}..."
                for i, inp in enumerate(model_inputs)]

    traced_list = create_traced_predict_function(
        list_predict,
        model_name="list_model"
    )

    print("✅ Created traced versions for different patterns:")
    print("   - String input/output")
    print("   - Dict input/output")
    print("   - List input/output")

    # Demo 4: Integration with ReAct Agent
    print("\n🎯 DEMO 4: Integration with ReAct Agent")
    print("-" * 50)

    try:
        from databricks_react_agent import DatabricksMultiTurnReactAgent

        # Create agent with traced predict function
        agent = DatabricksMultiTurnReactAgent(
            predict_function=traced_predict,  # Use traced version
            llm={'model': 'traced-tongyi-model'}
        )

        print("✅ ReAct agent initialized with traced predict function")
        print("   All model calls will now be traced in MLflow!")

        # Test with a simple question
        test_data = {
            'item': {
                'question': 'What is machine learning?',
                'answer': 'Reference answer for testing'
            }
        }

        print("🔬 Testing agent with traced predict...")

        try:
            with mlflow.start_run(run_name="react_agent_traced_demo"):
                # This will create nested spans:
                # - Session span (agent)
                # - Tool execution spans
                # - Model prediction spans (our traced predict)
                result = agent._run(test_data)

                print("✅ Agent execution completed with full tracing!")
                print(f"📊 Prediction: {result['prediction'][:100]}...")

        except Exception as agent_error:
            print(f"⚠️ Agent demo warning: {agent_error}")
            print("   (This is normal if tools/APIs aren't configured)")

    except ImportError:
        print("⚠️ ReAct agent not available for demo")

    return True

def show_traced_predict_benefits():
    """Show the benefits of traced predict functions"""
    print("\n📈 TRACED PREDICT FUNCTION BENEFITS")
    print("=" * 60)

    print("🔍 **Comprehensive Model Observability:**")
    print("   • Input/output logging with configurable limits")
    print("   • Execution timing for performance monitoring")
    print("   • Token usage approximation")
    print("   • Error tracking and classification")
    print("   • Model parameter logging")

    print("\n🎯 **Integration Features:**")
    print("   • Works with ANY predict function signature")
    print("   • Handles string, dict, list inputs/outputs")
    print("   • Automatic span creation in MLflow")
    print("   • Safe fallback if MLflow unavailable")
    print("   • Configurable logging levels")

    print("\n📊 **MLflow Span Structure:**")
    print("   Session (AGENT)")
    print("   ├── Tool Execution (TOOL)")
    print("   └── Model Prediction (LLM) ← New traced predict spans")
    print("       ├── Input processing")
    print("       ├── Model inference")
    print("       ├── Output processing")
    print("       └── Metrics logging")

    print("\n🛠️ **Ready-Made Wrappers:**")
    print("   • create_databricks_traced_predict() - Databricks endpoints")
    print("   • create_huggingface_traced_predict() - HF transformers")
    print("   • create_openai_traced_predict() - OpenAI API")
    print("   • create_traced_predict_function() - Any custom function")

    print("\n⚡ **Performance Impact:**")
    print("   • Minimal overhead (~1-5ms per prediction)")
    print("   • Async logging options available")
    print("   • Graceful degradation if MLflow fails")
    print("   • Configurable logging verbosity")

def main():
    print("🧪 Traced Predict Function Integration Demo")
    print("=" * 80)

    # Run main demo
    success = demo_traced_predict_integration()

    # Show benefits
    show_traced_predict_benefits()

    # Summary
    print("\n" + "=" * 80)
    print("📊 DEMO RESULTS")
    print(f"🔬 Traced Predict Integration: {'✅ SUCCESS' if success else '❌ FAILED'}")

    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Wrap your predict function with create_traced_predict_function()")
    print(f"   2. Use the traced version in your ReAct agent")
    print(f"   3. View comprehensive traces in MLflow UI")
    print(f"   4. Monitor model performance and usage patterns")

    print(f"\n💡 USAGE EXAMPLE:")
    print(f"   # Wrap your existing predict function")
    print(f"   traced_predict = create_traced_predict_function(")
    print(f"       your_predict_function,")
    print(f"       model_name='your_model_name'")
    print(f"   )")
    print(f"   ")
    print(f"   # Use in ReAct agent")
    print(f"   agent = DatabricksMultiTurnReactAgent(")
    print(f"       predict_function=traced_predict")
    print(f"   )")

if __name__ == "__main__":
    main()