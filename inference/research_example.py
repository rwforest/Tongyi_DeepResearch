#!/usr/bin/env python3
"""
Example of using the research system with automatic result saving
"""

from databricks_react_inference import DatabricksReactInference

def example_research_with_saving():
    """Example of research that automatically saves results and properly manages MLflow runs"""

    # You need to provide your predict function
    def your_predict_function(model_input):
        # Replace with your actual predict function
        # This is just a placeholder
        return "This is a placeholder response"

    # Initialize the inference system
    inference_system = DatabricksReactInference(predict_function=your_predict_function)

    # Research a question - MLflow run will start and end automatically!
    question = "What are the latest developments in transformer architectures?"

    print(f"🔬 Researching: {question}")
    print("📊 MLflow run will start automatically...")

    result = inference_system.research_question(question)

    # MLflow run automatically ended ✅
    # Results automatically saved to files ✅
    # Results logged to MLflow experiment ✅

    print(f"\n✅ Research completed!")
    print(f"📊 MLflow run automatically ended")
    print(f"📝 Answer: {result['prediction'][:200]}...")
    print(f"🔚 Status: {result['termination']}")

    return result

def example_manual_run_control():
    """Example of manual MLflow run control"""

    def your_predict_function(model_input):
        return "This is a placeholder response"

    inference_system = DatabricksReactInference(predict_function=your_predict_function)

    # Option 1: Let the system manage runs automatically
    print("🔧 Option 1: Automatic run management")
    result1 = inference_system.research_question("What is quantum computing?")
    print("✅ Run ended automatically\n")

    # Option 2: Manual run control with custom name
    print("🔧 Option 2: Manual run control")
    result2 = inference_system.research_question_with_run_control(
        "What is machine learning?",
        run_name="my_custom_research_run"
    )
    print("✅ Run ended with context manager\n")

    # Option 3: Disable auto-end for custom control
    print("🔧 Option 3: Disable auto-end")
    import mlflow
    with mlflow.start_run(run_name="fully_manual_run"):
        result3 = inference_system.research_question(
            "What is artificial intelligence?",
            auto_end_run=False  # Don't auto-end, we'll manage it
        )
        # Run continues...
        print("📊 Run still active, doing more work...")
        # Run ends when context exits
    print("✅ Run ended when context exited")

    return result1, result2, result3

if __name__ == "__main__":
    example_research_with_saving()