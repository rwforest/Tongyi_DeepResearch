#!/usr/bin/env python3
"""
Traced Predict Function Wrapper
Adds comprehensive MLflow tracing to any predict function
"""

import time
import functools
import inspect
import json
from typing import Callable, Any, Dict, List, Union
import mlflow
from mlflow.entities import SpanType

# Import safe MLflow configuration
try:
    from mlflow_config import (
        safe_trace_decorator, set_span_attributes_safely, set_span_inputs_safely, set_span_outputs_safely,
        ensure_mlflow_initialized, get_global_mlflow_config
    )
    print("📊 MLFLOW: Using robust MLflow configuration for predict tracing")
    MLFLOW_AVAILABLE = True
except ImportError:
    print("📊 MLFLOW: Using basic MLflow configuration for predict tracing")
    MLFLOW_AVAILABLE = False

    def safe_trace_decorator(name, span_type=None):
        """Fallback decorator"""
        def decorator(func):
            try:
                return mlflow.trace(name=name, span_type=span_type)(func)
            except:
                return func
        return decorator

    def set_span_attributes_safely(attributes):
        try:
            span = mlflow.get_current_active_span()
            if span:
                span.set_attributes(attributes)
        except:
            pass

    def set_span_inputs_safely(inputs):
        try:
            span = mlflow.get_current_active_span()
            if span:
                span.set_inputs(inputs)
        except:
            pass

    def set_span_outputs_safely(outputs):
        try:
            span = mlflow.get_current_active_span()
            if span:
                span.set_outputs(outputs)
        except:
            pass

    def ensure_mlflow_initialized():
        """Fallback function"""
        return True

def capture_function_parameters(func: Callable, args: tuple, kwargs: dict) -> Dict[str, Any]:
    """
    Capture all parameters passed to a function including defaults

    Args:
        func: The function being called
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Dictionary of all parameter names and values
    """
    try:
        # Get function signature
        sig = inspect.signature(func)

        # Bind arguments to parameters
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Convert to serializable format
        params = {}
        for param_name, param_value in bound.arguments.items():
            try:
                # Handle different parameter types
                if isinstance(param_value, (str, int, float, bool, type(None))):
                    params[param_name] = param_value
                elif isinstance(param_value, (list, tuple)):
                    # Log list/tuple length and first few items
                    params[f"{param_name}_type"] = type(param_value).__name__
                    params[f"{param_name}_length"] = len(param_value)
                    if len(param_value) > 0:
                        params[f"{param_name}_first_item_type"] = type(param_value[0]).__name__
                        if len(str(param_value[0])) < 200:
                            params[f"{param_name}_first_item"] = str(param_value[0])[:200]
                elif isinstance(param_value, dict):
                    # Log dict keys and size
                    params[f"{param_name}_type"] = "dict"
                    params[f"{param_name}_keys"] = list(param_value.keys())[:10]  # First 10 keys
                    params[f"{param_name}_size"] = len(param_value)
                elif hasattr(param_value, '__dict__'):
                    # Log object type and attributes
                    params[f"{param_name}_type"] = type(param_value).__name__
                    params[f"{param_name}_class"] = param_value.__class__.__name__
                else:
                    # Fallback to string representation
                    params[f"{param_name}_type"] = type(param_value).__name__
                    param_str = str(param_value)
                    if len(param_str) < 200:
                        params[param_name] = param_str
                    else:
                        params[f"{param_name}_preview"] = param_str[:200] + "..."

            except Exception as e:
                # If parameter can't be serialized, log its type
                params[f"{param_name}_type"] = type(param_value).__name__
                params[f"{param_name}_serialization_error"] = str(e)[:100]

        return params

    except Exception as e:
        return {"parameter_capture_error": str(e)}

def safe_serialize_for_mlflow(obj: Any, max_length: int = 500) -> Any:
    """
    Safely serialize objects for MLflow logging with size limits

    Args:
        obj: Object to serialize
        max_length: Maximum string length for serialization

    Returns:
        MLflow-compatible serialized object
    """
    try:
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            if isinstance(obj, str) and len(obj) > max_length:
                return obj[:max_length] + f"... (truncated from {len(obj)} chars)"
            return obj
        elif isinstance(obj, (list, tuple)):
            return {
                "type": type(obj).__name__,
                "length": len(obj),
                "preview": [safe_serialize_for_mlflow(item, max_length//2) for item in obj[:3]]
            }
        elif isinstance(obj, dict):
            serialized = {}
            for k, v in list(obj.items())[:10]:  # First 10 items
                key = str(k)[:50]  # Limit key length
                serialized[key] = safe_serialize_for_mlflow(v, max_length//2)
            if len(obj) > 10:
                serialized["_truncated"] = f"... and {len(obj) - 10} more items"
            return serialized
        else:
            return {
                "type": type(obj).__name__,
                "string_repr": str(obj)[:max_length]
            }
    except Exception as e:
        return {"serialization_error": str(e)[:100]}

def create_traced_predict_function(
    predict_function: Callable,
    model_name: str = "unknown",
    log_inputs: bool = True,
    log_outputs: bool = True,
    log_timing: bool = True,
    log_parameters: bool = True,
    max_input_log_length: int = 500,
    max_output_log_length: int = 500
) -> Callable:
    """
    Create a traced version of any predict function

    Args:
        predict_function: Original predict function to wrap
        model_name: Name of the model for tracing
        log_inputs: Whether to log input prompts
        log_outputs: Whether to log output responses
        log_timing: Whether to log timing metrics
        log_parameters: Whether to log all function parameters
        max_input_log_length: Maximum characters to log from input
        max_output_log_length: Maximum characters to log from output

    Returns:
        Wrapped predict function with MLflow tracing
    """

    @safe_trace_decorator(name="model_prediction", span_type=SpanType.LLM)
    @functools.wraps(predict_function)
    def traced_predict(*args, **kwargs) -> Any:
        """
        Traced version of predict function with comprehensive logging
        """
        # Ensure MLflow is initialized with global config
        ensure_mlflow_initialized()

        start_time = time.time()

        # Get current MLflow span
        span = mlflow.get_current_active_span()

        # Capture all function parameters using MLflow span methods
        if log_parameters and span:
            try:
                function_params = capture_function_parameters(predict_function, args, kwargs)
                # Use span.set_attributes() for function parameters
                span.set_attributes({
                    f"predict_param_{k}": v for k, v in function_params.items()
                })
                print(f"🤖 PREDICT_TRACE: Set {len(function_params)} parameters as span attributes for {model_name}")
            except Exception as e:
                print(f"🤖 PREDICT_TRACE: Parameter capture warning: {e}")

        # Get model_input (first argument for backward compatibility)
        model_input = args[0] if args else kwargs.get('model_input', {})

        # Process different input formats
        if isinstance(model_input, list) and len(model_input) > 0:
            # Handle list input (common format)
            first_item = model_input[0]
            if isinstance(first_item, dict):
                prompt = first_item.get('prompt', str(first_item))
                model_params = {
                    'temperature': first_item.get('temperature', 0.85),
                    'max_length': first_item.get('max_length', 2048),
                    'presence_penalty': first_item.get('presence_penalty', 1.1)
                }
            else:
                prompt = str(first_item)
                model_params = {}
        elif isinstance(model_input, dict):
            # Handle dict input
            prompt = model_input.get('prompt', str(model_input))
            model_params = {k: v for k, v in model_input.items() if k != 'prompt'}
        elif isinstance(model_input, str):
            # Handle string input
            prompt = model_input
            model_params = {}
        else:
            # Handle other types
            prompt = str(model_input)
            model_params = {}

        # Set MLflow span inputs
        if span:
            span_inputs = {
                "input_type": type(model_input).__name__,
                "input_length": len(str(model_input)),
                "prompt_length": len(prompt)
            }

            if log_inputs:
                span_inputs["prompt_preview"] = prompt[:max_input_log_length]
                if len(prompt) > max_input_log_length:
                    span_inputs["prompt_truncated"] = True

            span.set_inputs(span_inputs)

            # Set model attributes
            span.set_attributes({
                "model_name": model_name,
                "prediction_function": predict_function.__name__,
                **model_params
            })

        # Set additional span attributes for model parameters
        if span:
            span.set_attributes({
                "model_name": model_name,
                "input_length": len(prompt),
                **{f"model_param_{k}": v for k, v in model_params.items()}
            })

        # Execute the original predict function
        try:
            print(f"🤖 PREDICT_TRACE: Calling {model_name} with input length {len(prompt)}")

            response = predict_function(*args, **kwargs)

            execution_time = time.time() - start_time

            # Process response
            if isinstance(response, list) and len(response) > 0:
                response_text = str(response[0])
            else:
                response_text = str(response)

            print(f"🤖 PREDICT_TRACE: {model_name} responded with {len(response_text)} chars (⏱️ {execution_time:.2f}s)")

            # Set MLflow span outputs
            if span:
                span_outputs = {
                    "response_length": len(response_text),
                    "execution_time": execution_time,
                    "success": True
                }

                if log_outputs:
                    span_outputs["response_preview"] = response_text[:max_output_log_length]
                    if len(response_text) > max_output_log_length:
                        span_outputs["response_truncated"] = True

                span.set_outputs(span_outputs)

            # Set metrics as span attributes
            if span and log_timing:
                span.set_attributes({
                    "prediction_time": execution_time,
                    "input_tokens_approx": len(prompt) // 4,  # Rough approximation
                    "output_tokens_approx": len(response_text) // 4,
                    "tokens_per_second": (len(response_text) // 4) / execution_time if execution_time > 0 else 0
                })

            return response

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Predict function failed: {str(e)}"

            print(f"🤖 PREDICT_TRACE: {model_name} failed after {execution_time:.2f}s: {error_msg}")

            # Set MLflow span outputs for error
            if span:
                span.set_outputs({
                    "error": error_msg,
                    "exception_type": type(e).__name__,
                    "execution_time": execution_time,
                    "success": False
                })

            # Set error attributes on span
            if span:
                span.set_attributes({
                    "prediction_time": execution_time,
                    "prediction_errors": 1,
                    "error_occurred": True
                })

            # Re-raise the exception
            raise e

    return traced_predict

def create_databricks_traced_predict(
    endpoint_url: str,
    token: str,
    model_name: str = "databricks_model",
    **trace_kwargs
) -> Callable:
    """
    Create traced predict function for Databricks model serving endpoint

    Args:
        endpoint_url: Databricks model serving endpoint URL
        token: Databricks access token
        model_name: Model name for tracing
        **trace_kwargs: Additional tracing configuration

    Returns:
        Traced predict function for Databricks endpoint
    """
    import requests

    def databricks_predict(model_input: Any) -> str:
        """Databricks model serving prediction"""
        if isinstance(model_input, list) and len(model_input) > 0:
            first_item = model_input[0]
            if isinstance(first_item, dict):
                prompt = first_item.get('prompt', str(first_item))
                temperature = first_item.get('temperature', 0.85)
                max_tokens = first_item.get('max_length', 2048)
            else:
                prompt = str(first_item)
                temperature = 0.85
                max_tokens = 2048
        else:
            prompt = str(model_input)
            temperature = 0.85
            max_tokens = 2048

        headers = {"Authorization": f"Bearer {token}"}
        data = {
            "inputs": [prompt],
            "params": {
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        }

        response = requests.post(endpoint_url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["predictions"][0]

    return create_traced_predict_function(
        databricks_predict,
        model_name=model_name,
        **trace_kwargs
    )

def create_huggingface_traced_predict(
    model_name: str,
    tokenizer=None,
    model=None,
    device: str = "auto",
    **trace_kwargs
) -> Callable:
    """
    Create traced predict function for HuggingFace transformers

    Args:
        model_name: HuggingFace model name
        tokenizer: Pre-loaded tokenizer (optional)
        model: Pre-loaded model (optional)
        device: Device to use for inference
        **trace_kwargs: Additional tracing configuration

    Returns:
        Traced predict function for HuggingFace model
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    def huggingface_predict(model_input: Any) -> str:
        """HuggingFace model prediction"""
        if isinstance(model_input, list) and len(model_input) > 0:
            first_item = model_input[0]
            if isinstance(first_item, dict):
                prompt = first_item.get('prompt', str(first_item))
                temperature = first_item.get('temperature', 0.85)
                max_length = first_item.get('max_length', 2048)
            else:
                prompt = str(first_item)
                temperature = 0.85
                max_length = 2048
        else:
            prompt = str(model_input)
            temperature = 0.85
            max_length = 2048

        inputs = tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from the output
        response = generated_text[len(prompt):].strip()
        return response

    return create_traced_predict_function(
        huggingface_predict,
        model_name=model_name,
        **trace_kwargs
    )

def create_openai_traced_predict(
    api_key: str,
    model_name: str = "gpt-3.5-turbo",
    base_url: str = None,
    **trace_kwargs
) -> Callable:
    """
    Create traced predict function for OpenAI API

    Args:
        api_key: OpenAI API key
        model_name: OpenAI model name
        base_url: Custom base URL (optional)
        **trace_kwargs: Additional tracing configuration

    Returns:
        Traced predict function for OpenAI API
    """
    import openai

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def openai_predict(model_input: Any) -> str:
        """OpenAI API prediction"""
        if isinstance(model_input, list) and len(model_input) > 0:
            first_item = model_input[0]
            if isinstance(first_item, dict):
                prompt = first_item.get('prompt', str(first_item))
                temperature = first_item.get('temperature', 0.85)
                max_tokens = first_item.get('max_length', 2048)
            else:
                prompt = str(first_item)
                temperature = 0.85
                max_tokens = 2048
        else:
            prompt = str(model_input)
            temperature = 0.85
            max_tokens = 2048

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    return create_traced_predict_function(
        openai_predict,
        model_name=model_name,
        **trace_kwargs
    )

# Example usage functions
def example_traced_predict_usage():
    """Example of how to use traced predict functions"""

    print("📊 TRACED PREDICT FUNCTION EXAMPLES")
    print("=" * 50)

    # Example 1: Wrap existing predict function
    def my_original_predict(model_input):
        # Your existing predict function
        return f"Response to: {model_input}"

    traced_predict = create_traced_predict_function(
        my_original_predict,
        model_name="my_custom_model",
        log_inputs=True,
        log_outputs=True
    )

    print("✅ Example 1: Wrapped existing function")

    # Example 2: Databricks endpoint
    # traced_databricks = create_databricks_traced_predict(
    #     endpoint_url="https://your-workspace.databricks.com/serving-endpoints/your-model/invocations",
    #     token="your-token",
    #     model_name="my_databricks_model"
    # )
    print("✅ Example 2: Databricks endpoint wrapper available")

    # Example 3: HuggingFace model
    # traced_hf = create_huggingface_traced_predict(
    #     model_name="microsoft/DialoGPT-medium",
    #     model_name="dialogpt"
    # )
    print("✅ Example 3: HuggingFace wrapper available")

    # Example 4: OpenAI API
    # traced_openai = create_openai_traced_predict(
    #     api_key="your-api-key",
    #     model_name="gpt-3.5-turbo"
    # )
    print("✅ Example 4: OpenAI wrapper available")

if __name__ == "__main__":
    example_traced_predict_usage()