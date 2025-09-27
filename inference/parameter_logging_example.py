#!/usr/bin/env python3
"""
MLflow Parameter Logging Example
Shows how every function call logs its parameters automatically
"""

import mlflow
from traced_predict_wrapper import create_traced_predict_function
from mlflow_config import safe_trace_decorator, ensure_mlflow_initialized, set_span_attributes_safely
from mlflow.entities import SpanType

def demo_predict_function(model_input, temperature=0.7, max_tokens=1000, **kwargs):
    """
    Demo predict function - ALL parameters will be logged to MLflow
    """
    # Simulate model prediction
    if isinstance(model_input, list):
        prompt = str(model_input[0]) if model_input else "empty"
    else:
        prompt = str(model_input)

    return f"Response to '{prompt[:50]}...' (temp={temperature}, max_tokens={max_tokens})"

@safe_trace_decorator(name="complex_research_function", span_type=SpanType.AGENT, log_params=True)
def research_with_parameters(
    query: str,
    search_depth: int = 3,
    include_sources: bool = True,
    filters: dict = None,
    model_config: dict = None
):
    """
    Example research function - ALL parameters automatically logged
    """
    print(f"🔬 Research: {query}")
    print(f"   Depth: {search_depth}")
    print(f"   Include sources: {include_sources}")
    print(f"   Filters: {filters}")
    print(f"   Model config: {model_config}")

    return {
        'query': query,
        'results': f"Found {search_depth * 5} results",
        'sources_included': include_sources
    }

def demonstrate_parameter_logging():
    """
    Comprehensive demonstration of parameter logging capabilities
    """
    print("🔧 PARAMETER LOGGING DEMONSTRATION")
    print("=" * 60)

    # Initialize MLflow
    ensure_mlflow_initialized()

    # Create traced predict function
    traced_predict = create_traced_predict_function(
        demo_predict_function,
        model_name="demo_model",
        log_parameters=True  # Enable parameter logging
    )

    # Start MLflow run
    with mlflow.start_run(run_name="parameter_logging_demo"):
        print("📊 MLflow run started - all parameters will be logged!")

        # Example 1: Simple parameters
        print("\n1️⃣ Testing simple parameters...")
        result1 = traced_predict(
            "What is machine learning?",
            temperature=0.8,
            max_tokens=1500
        )
        print(f"   Result: {result1}")

        # Example 2: Complex nested parameters
        print("\n2️⃣ Testing complex nested parameters...")
        complex_input = {
            'prompt': 'Explain deep learning',
            'context': ['neural networks', 'backpropagation', 'gradients'],
            'settings': {
                'model': 'transformer',
                'layers': 24,
                'attention_heads': 16
            }
        }

        result2 = traced_predict(
            [complex_input],
            temperature=0.9,
            max_tokens=2000,
            stream=False,
            return_metadata=True
        )
        print(f"   Result: {result2}")

        # Example 3: Function with complex parameters
        print("\n3️⃣ Testing function with complex parameters...")
        research_result = research_with_parameters(
            query="Latest developments in AI",
            search_depth=5,
            include_sources=True,
            filters={
                'date_range': '2024-01-01 to 2024-12-31',
                'domains': ['arxiv.org', 'scholar.google.com'],
                'languages': ['en', 'zh']
            },
            model_config={
                'name': 'tongyi-deepresearch',
                'version': '2.0',
                'capabilities': ['search', 'summarize', 'analyze'],
                'limits': {
                    'max_tokens': 4000,
                    'timeout': 30
                }
            }
        )
        print(f"   Research result: {research_result}")

        print("\n✅ Parameter logging demonstration complete!")
        print("📊 Check your MLflow experiment to see all logged parameters:")
        print("   • Function arguments (query, temperature, max_tokens, etc.)")
        print("   • Complex data structures (nested dicts, lists)")
        print("   • Data types and sizes")
        print("   • Privacy-safe truncation for large parameters")
        print("   • Automatic serialization of complex objects")

def show_logged_parameters():
    """
    Show what parameters get logged for different data types
    """
    print("\n📋 PARAMETER LOGGING REFERENCE")
    print("=" * 60)

    examples = {
        "String": "Simple string parameter",
        "Integer": 42,
        "Float": 3.14159,
        "Boolean": True,
        "List": ['item1', 'item2', 'item3'],
        "Dict": {'key1': 'value1', 'key2': 42, 'nested': {'deep': 'value'}},
        "Large String": "x" * 1000,  # Will be truncated
        "Complex Object": {'model': 'gpt-4', 'config': {'temp': 0.7, 'tokens': 1000}}
    }

    print("Parameter types and how they're logged:")
    for name, value in examples.items():
        print(f"\n🔧 {name}:")
        print(f"   Value: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
        print(f"   Logged as:")

        if isinstance(value, str):
            if len(value) > 500:
                print(f"     • {name}: {value[:500]}... (truncated)")
                print(f"     • {name}_length: {len(value)}")
            else:
                print(f"     • {name}: {value}")
        elif isinstance(value, (int, float, bool)):
            print(f"     • {name}: {value}")
        elif isinstance(value, list):
            print(f"     • {name}_type: list")
            print(f"     • {name}_length: {len(value)}")
            print(f"     • {name}_first_type: {type(value[0]).__name__}")
        elif isinstance(value, dict):
            print(f"     • {name}_type: dict")
            print(f"     • {name}_size: {len(value)}")
            print(f"     • {name}_keys: {list(value.keys())[:5]}")

if __name__ == "__main__":
    # Run the demonstration
    demonstrate_parameter_logging()
    show_logged_parameters()

    print("\n" + "=" * 80)
    print("🎯 SUMMARY: Comprehensive Parameter Logging")
    print("✅ ALL function parameters are automatically captured and logged")
    print("✅ Complex data types are intelligently serialized")
    print("✅ Privacy controls prevent logging of sensitive large data")
    print("✅ Easy to integrate with existing functions using decorators")
    print("✅ Works seamlessly with Databricks Unity Catalog experiments")
    print("=" * 80)