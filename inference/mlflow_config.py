#!/usr/bin/env python3
"""
MLflow Configuration for Tongyi DeepResearch Agent
Handles experiment setup, timeout configuration, and tracing optimization
"""

import os
import time
import inspect
import functools
from typing import Callable, Any, Dict
import mlflow
from mlflow.entities import SpanType

def configure_mlflow_for_databricks():
    """Configure MLflow specifically for Databricks environment"""
    try:
        # Set MLflow tracking URI for Databricks
        databricks_host = os.environ.get('DATABRICKS_HOST')
        databricks_token = os.environ.get('DATABRICKS_TOKEN')

        if databricks_host and databricks_token:
            mlflow.set_tracking_uri("databricks")
            print("📊 MLFLOW: Configured for Databricks environment")
        else:
            print("📊 MLFLOW: Using local MLflow tracking")

    except Exception as e:
        print(f"📊 MLFLOW: Warning - Databricks configuration failed: {e}")

def setup_mlflow_experiment(experiment_name="tongyi_deepresearch_agent",
                           catalog=None, schema=None, volume=None,
                           artifact_location=None, tracking_uri=None,
                           registry_uri=None):
    """Setup MLflow experiment with proper configuration including Databricks Unity Catalog

    Args:
        experiment_name: Name of the experiment
        catalog: Databricks catalog name
        schema: Databricks schema name
        volume: Databricks volume name
        artifact_location: Custom artifact location (overrides volume settings)
        tracking_uri: MLflow tracking URI (defaults to "databricks" if on Databricks)
        registry_uri: MLflow registry URI (defaults to "databricks-uc" if on Databricks)
    """
    try:
        # Configure timeout settings
        os.environ.setdefault('MLFLOW_TRACKING_TIMEOUT', '30')
        os.environ.setdefault('MLFLOW_HTTP_REQUEST_TIMEOUT', '30')

        # Set tracking and registry URIs
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            print(f"📊 MLFLOW: Tracking URI set to '{tracking_uri}'")
        elif os.environ.get('DATABRICKS_HOST'):
            mlflow.set_tracking_uri("databricks")
            print("📊 MLFLOW: Tracking URI set to 'databricks'")

        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
            print(f"📊 MLFLOW: Registry URI set to '{registry_uri}'")
        elif os.environ.get('DATABRICKS_HOST'):
            mlflow.set_registry_uri("databricks-uc")
            print("📊 MLFLOW: Registry URI set to 'databricks-uc'")

        # Configure artifact location for Databricks volumes
        if not artifact_location and catalog and schema and volume:
            artifact_location = f"dbfs:/Volumes/{catalog}/{schema}/{volume}"
            print(f"📊 MLFLOW: Artifact location set to '{artifact_location}'")

        # Create experiment if it doesn't exist
        try:
            existing_experiment = mlflow.get_experiment_by_name(experiment_name)
            if existing_experiment is None:
                if artifact_location:
                    experiment_id = mlflow.create_experiment(
                        name=experiment_name,
                        artifact_location=artifact_location
                    )
                    print(f"📊 MLFLOW: Created experiment '{experiment_name}' with artifact location")
                else:
                    experiment_id = mlflow.create_experiment(name=experiment_name)
                    print(f"📊 MLFLOW: Created experiment '{experiment_name}'")
            else:
                print(f"📊 MLFLOW: Using existing experiment '{experiment_name}'")

            # Set the experiment
            mlflow.set_experiment(experiment_name)
            print(f"📊 MLFLOW: Active experiment set to '{experiment_name}'")

        except Exception as exp_error:
            print(f"📊 MLFLOW: Using default experiment due to: {exp_error}")

        # Disable autologging to prevent conflicts
        mlflow.autolog(disable=True)
        print("📊 MLFLOW: Autologging disabled for manual tracing")

        return True

    except Exception as e:
        print(f"📊 MLFLOW: Configuration warning: {e}")
        return False

def create_tracing_context(session_name="react_session"):
    """Create MLflow tracing context with error handling"""
    try:
        # Start a new run if not already active
        if not mlflow.active_run():
            run = mlflow.start_run(run_name=f"{session_name}_{int(time.time())}")
            print(f"📊 MLFLOW: Started run {run.info.run_id}")
            return run
        else:
            print("📊 MLFLOW: Using existing active run")
            return mlflow.active_run()

    except Exception as e:
        print(f"📊 MLFLOW: Warning - could not create tracing context: {e}")
        return None

def safe_trace_decorator(name, span_type=SpanType.UNKNOWN, log_params=True):
    """Decorator that safely adds MLflow tracing with fallback and parameter logging"""
    def decorator(func):
        try:
            # Create traced version with parameter logging
            traced_func = mlflow.trace(name=name, span_type=span_type)(func)

            if log_params:
                @functools.wraps(traced_func)
                def wrapper_with_params(*args, **kwargs):
                    try:
                        # Get the current span and set parameters as attributes
                        span = mlflow.get_current_active_span()
                        if span:
                            # Capture function parameters
                            params = capture_function_parameters(func, args, kwargs)
                            # Set as span attributes with function name prefix
                            span.set_attributes({f"{func.__name__}_{k}": v for k, v in params.items()})

                            # Also set function inputs for better tracing
                            span.set_inputs({
                                "function_name": func.__name__,
                                "args_count": len(args),
                                "kwargs_keys": list(kwargs.keys())
                            })

                            print(f"📊 MLFLOW: Set {len(params)} parameters as span attributes for {func.__name__}")
                    except Exception as e:
                        print(f"📊 MLFLOW: Parameter logging warning for {func.__name__}: {e}")

                    return traced_func(*args, **kwargs)
                return wrapper_with_params
            else:
                return traced_func

        except Exception as e:
            print(f"📊 MLFLOW: Warning - tracing disabled for {name}: {e}")
            # Return original function if tracing fails
            return func
    return decorator

def capture_function_parameters(func: Callable, args: tuple, kwargs: dict) -> Dict[str, Any]:
    """
    Capture all parameters passed to a function including defaults

    Args:
        func: The function being called
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Dictionary of all parameter names and values (MLflow-safe)
    """
    try:
        # Get function signature
        sig = inspect.signature(func)

        # Bind arguments to parameters
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Convert to MLflow-safe format
        params = {}
        for param_name, param_value in bound.arguments.items():
            try:
                # Handle different parameter types for MLflow compatibility
                if param_value is None:
                    params[param_name] = "None"
                elif isinstance(param_value, (str, int, float, bool)):
                    if isinstance(param_value, str) and len(param_value) > 500:
                        params[param_name] = param_value[:500] + "... (truncated)"
                        params[f"{param_name}_length"] = len(param_value)
                    else:
                        params[param_name] = param_value
                elif isinstance(param_value, (list, tuple)):
                    params[f"{param_name}_type"] = type(param_value).__name__
                    params[f"{param_name}_length"] = len(param_value)
                    if len(param_value) > 0:
                        params[f"{param_name}_first_type"] = type(param_value[0]).__name__
                elif isinstance(param_value, dict):
                    params[f"{param_name}_type"] = "dict"
                    params[f"{param_name}_size"] = len(param_value)
                    params[f"{param_name}_keys"] = str(list(param_value.keys())[:5])  # First 5 keys
                elif hasattr(param_value, '__dict__'):
                    params[f"{param_name}_type"] = type(param_value).__name__
                    params[f"{param_name}_class"] = param_value.__class__.__name__
                else:
                    params[f"{param_name}_type"] = type(param_value).__name__
                    param_str = str(param_value)
                    if len(param_str) < 200:
                        params[param_name] = param_str
                    else:
                        params[f"{param_name}_str_preview"] = param_str[:200]

            except Exception as e:
                params[f"{param_name}_error"] = f"Serialization failed: {str(e)[:50]}"

        return params

    except Exception as e:
        return {"parameter_capture_error": str(e)[:100]}

def log_metrics_safely(metrics_dict):
    """Safely log metrics to MLflow with error handling (deprecated - use span.set_attributes())"""
    try:
        if mlflow.active_run():
            mlflow.log_metrics(metrics_dict)
            print(f"📊 MLFLOW: Logged metrics: {list(metrics_dict.keys())}")
    except Exception as e:
        print(f"📊 MLFLOW: Warning - could not log metrics: {e}")

def log_parameters_safely(params_dict):
    """Safely log parameters to MLflow with error handling (deprecated - use span.set_attributes())"""
    try:
        if mlflow.active_run():
            mlflow.log_params(params_dict)
            print(f"📊 MLFLOW: Logged parameters: {list(params_dict.keys())}")
    except Exception as e:
        print(f"📊 MLFLOW: Warning - could not log parameters: {e}")

def set_span_attributes_safely(attributes_dict):
    """Safely set attributes on current MLflow span"""
    try:
        span = mlflow.get_current_active_span()
        if span:
            span.set_attributes(attributes_dict)
            print(f"📊 MLFLOW: Set span attributes: {list(attributes_dict.keys())}")
    except Exception as e:
        print(f"📊 MLFLOW: Warning - could not set span attributes: {e}")

def set_span_inputs_safely(inputs_dict):
    """Safely set inputs on current MLflow span"""
    try:
        span = mlflow.get_current_active_span()
        if span:
            span.set_inputs(inputs_dict)
            print(f"📊 MLFLOW: Set span inputs: {list(inputs_dict.keys())}")
    except Exception as e:
        print(f"📊 MLFLOW: Warning - could not set span inputs: {e}")

def set_span_outputs_safely(outputs):
    """Safely set outputs on current MLflow span"""
    try:
        span = mlflow.get_current_active_span()
        if span:
            span.set_outputs(outputs)
            print(f"📊 MLFLOW: Set span outputs")
    except Exception as e:
        print(f"📊 MLFLOW: Warning - could not set span outputs: {e}")

def end_run_safely():
    """Safely end MLflow run"""
    try:
        if mlflow.active_run():
            mlflow.end_run()
            print("📊 MLFLOW: Run ended successfully")
    except Exception as e:
        print(f"📊 MLFLOW: Warning - could not end run: {e}")

# Global MLflow configuration
_GLOBAL_MLFLOW_CONFIG = {
    'experiment_name': 'tongyi_deepresearch_agent',
    'tracking_timeout': 30,
    'http_timeout': 30,
    'disable_autolog': True,
    'safe_mode': True,  # Enable safe mode for production
    'tracking_uri': None,
    'registry_uri': None,
    'catalog': None,
    'schema': None,
    'volume': None,
    'artifact_location': None,
    'initialized': False
}

def set_global_mlflow_config(experiment_name=None, catalog=None, schema=None,
                           volume=None, artifact_location=None, tracking_uri=None,
                           registry_uri=None, **kwargs):
    """Set global MLflow configuration that will be used by all modules

    Args:
        experiment_name: MLflow experiment name
        catalog: Databricks catalog name
        schema: Databricks schema name
        volume: Databricks volume name
        artifact_location: Custom artifact location
        tracking_uri: MLflow tracking URI
        registry_uri: MLflow registry URI
        **kwargs: Additional configuration options
    """
    global _GLOBAL_MLFLOW_CONFIG

    if experiment_name:
        _GLOBAL_MLFLOW_CONFIG['experiment_name'] = experiment_name
    if catalog:
        _GLOBAL_MLFLOW_CONFIG['catalog'] = catalog
    if schema:
        _GLOBAL_MLFLOW_CONFIG['schema'] = schema
    if volume:
        _GLOBAL_MLFLOW_CONFIG['volume'] = volume
    if artifact_location:
        _GLOBAL_MLFLOW_CONFIG['artifact_location'] = artifact_location
    if tracking_uri:
        _GLOBAL_MLFLOW_CONFIG['tracking_uri'] = tracking_uri
    if registry_uri:
        _GLOBAL_MLFLOW_CONFIG['registry_uri'] = registry_uri

    # Update additional config
    _GLOBAL_MLFLOW_CONFIG.update(kwargs)

    print(f"📊 MLFLOW: Global configuration updated")
    print(f"   Experiment: {_GLOBAL_MLFLOW_CONFIG['experiment_name']}")
    if _GLOBAL_MLFLOW_CONFIG['catalog']:
        print(f"   Catalog: {_GLOBAL_MLFLOW_CONFIG['catalog']}")
        print(f"   Schema: {_GLOBAL_MLFLOW_CONFIG['schema']}")
        print(f"   Volume: {_GLOBAL_MLFLOW_CONFIG['volume']}")

def get_global_mlflow_config():
    """Get the current global MLflow configuration"""
    return _GLOBAL_MLFLOW_CONFIG.copy()

def ensure_mlflow_initialized():
    """Ensure MLflow is initialized with global configuration"""
    global _GLOBAL_MLFLOW_CONFIG

    if not _GLOBAL_MLFLOW_CONFIG['initialized']:
        config = _GLOBAL_MLFLOW_CONFIG
        success = setup_mlflow_experiment(
            experiment_name=config['experiment_name'],
            catalog=config['catalog'],
            schema=config['schema'],
            volume=config['volume'],
            artifact_location=config['artifact_location'],
            tracking_uri=config['tracking_uri'],
            registry_uri=config['registry_uri']
        )
        _GLOBAL_MLFLOW_CONFIG['initialized'] = success
        return success

    return True

# Configuration for different environments (backward compatibility)
MLFLOW_CONFIG = {
    'experiment_name': 'tongyi_deepresearch_agent',
    'tracking_timeout': 30,
    'http_timeout': 30,
    'disable_autolog': True,
    'safe_mode': True  # Enable safe mode for production
}

def initialize_mlflow(config=None):
    """Initialize MLflow with given configuration"""
    if config is None:
        config = MLFLOW_CONFIG

    print("📊 MLFLOW: Initializing MLflow configuration...")

    # Setup experiment
    success = setup_mlflow_experiment(config['experiment_name'])

    if success:
        print("📊 MLFLOW: ✅ MLflow configured successfully")
    else:
        print("📊 MLFLOW: ⚠️ MLflow configured with warnings")

    return success

# Auto-initialize when module is imported
if __name__ != "__main__":
    initialize_mlflow()