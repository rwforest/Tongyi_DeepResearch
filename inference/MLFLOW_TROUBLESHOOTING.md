# MLflow Tracing Troubleshooting Guide

## Common MLflow Issues and Solutions

### 1. **Default Experiment Warning**
```
WARNING mlflow.tracing.processor.mlflow: Creating a trace within the default experiment with id '0'
```

**Solution:**
```python
# Set a specific experiment before tracing
mlflow.set_experiment("your_experiment_name")

# Or use the robust configuration
from mlflow_config import initialize_mlflow
initialize_mlflow()
```

### 2. **Timeout Issues**
```
WARNING mlflow.tracking.default_experiment.registry: Encountered unexpected error while getting experiment_id: Timed out after 0:08:20
```

**Solutions:**
```python
# Set timeout environment variables
os.environ['MLFLOW_TRACKING_TIMEOUT'] = '30'
os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = '30'

# Or use robust configuration that handles timeouts
from mlflow_config import initialize_mlflow
initialize_mlflow()
```

### 3. **Databricks Configuration**
```python
# For Databricks environments
os.environ['DATABRICKS_HOST'] = 'https://your-workspace.cloud.databricks.com'
os.environ['DATABRICKS_TOKEN'] = 'your_token'
mlflow.set_tracking_uri("databricks")
```

### 4. **Safe Tracing Implementation**

Use the safe decorator pattern:
```python
from mlflow_config import safe_trace_decorator

@safe_trace_decorator(name="function_name", span_type=SpanType.LLM)
def your_function():
    # Function implementation
    pass
```

This ensures:
- Graceful fallback if MLflow fails
- No interruption to core functionality
- Proper error logging

## Configuration Best Practices

### Environment Setup
```bash
# .env file
MLFLOW_TRACKING_TIMEOUT=30
MLFLOW_HTTP_REQUEST_TIMEOUT=30
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your_databricks_token
```

### Production Configuration
```python
# Initialize MLflow with production settings
from mlflow_config import MLFLOW_CONFIG

config = {
    'experiment_name': 'production_tongyi_deepresearch',
    'tracking_timeout': 60,
    'http_timeout': 60,
    'safe_mode': True
}

initialize_mlflow(config)
```

## Debugging MLflow Issues

### 1. **Check MLflow Status**
```python
try:
    import mlflow
    print(f"MLflow version: {mlflow.__version__}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Active run: {mlflow.active_run()}")
except Exception as e:
    print(f"MLflow issue: {e}")
```

### 2. **Verify Experiment Setup**
```python
try:
    experiment = mlflow.get_experiment_by_name("your_experiment")
    if experiment:
        print(f"Experiment ID: {experiment.experiment_id}")
    else:
        print("Experiment not found")
except Exception as e:
    print(f"Experiment error: {e}")
```

### 3. **Test Tracing**
```python
# Simple tracing test
try:
    with mlflow.start_run():
        with mlflow.start_span(name="test_span") as span:
            span.set_inputs({"test": "input"})
            span.set_outputs({"test": "output"})
        print("✅ Basic tracing works")
except Exception as e:
    print(f"❌ Tracing failed: {e}")
```

## Performance Optimization

### 1. **Disable Autologging**
```python
# Disable conflicting autologging
mlflow.autolog(disable=True)
```

### 2. **Batch Logging**
```python
# Log metrics in batches
metrics = {
    'execution_time': 1.23,
    'token_count': 150,
    'success_rate': 0.95
}
mlflow.log_metrics(metrics)
```

### 3. **Async Logging** (For high-frequency calls)
```python
# Use background logging for high-frequency operations
import threading

def log_async(metrics):
    thread = threading.Thread(
        target=lambda: mlflow.log_metrics(metrics)
    )
    thread.start()
```

## Error Recovery Patterns

### 1. **Graceful Degradation**
```python
def safe_mlflow_operation(operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except Exception as e:
        print(f"MLflow warning: {e}")
        return None  # Continue without MLflow
```

### 2. **Retry Logic**
```python
def retry_mlflow_operation(operation, max_retries=3):
    for attempt in range(max_retries):
        try:
            return operation()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"MLflow failed after {max_retries} attempts: {e}")
                return None
            time.sleep(2 ** attempt)  # Exponential backoff
```

## Testing MLflow Integration

### 1. **Unit Tests**
```python
def test_mlflow_tracing():
    """Test MLflow tracing without external dependencies"""
    with mlflow.start_run():
        # Test basic tracing functionality
        assert mlflow.active_run() is not None
```

### 2. **Integration Tests**
```python
def test_agent_with_mlflow():
    """Test agent execution with MLflow tracing"""
    # Run the test_mlflow_tracing.py script
    result = subprocess.run(['python', 'test_mlflow_tracing.py'])
    assert result.returncode == 0
```

## Monitoring and Observability

### 1. **MLflow UI Access**
```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# For Databricks
# Use the Databricks MLflow UI directly
```

### 2. **Metrics to Monitor**
- Trace completion rate
- Average execution time per component
- Error frequency by component
- API response times

### 3. **Alerting**
```python
# Set up alerts for critical failures
def check_mlflow_health():
    try:
        mlflow.search_experiments()
        return True
    except:
        # Send alert
        return False
```

## Common Integration Patterns

### 1. **Context Managers**
```python
class MLflowTraceContext:
    def __enter__(self):
        self.run = mlflow.start_run()
        return self.run

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run()
```

### 2. **Decorator Pattern**
```python
def traced_function(span_name=None, span_type=SpanType.UNKNOWN):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with mlflow.start_span(name=span_name or func.__name__) as span:
                span.set_inputs({"args": args, "kwargs": kwargs})
                result = func(*args, **kwargs)
                span.set_outputs({"result": str(result)})
                return result
        return wrapper
    return decorator
```

## Troubleshooting Checklist

- [ ] MLflow package installed and importable
- [ ] Experiment configured (not using default)
- [ ] Timeout settings configured
- [ ] Network connectivity to tracking server
- [ ] Proper credentials for Databricks (if used)
- [ ] Safe decorators implemented
- [ ] Error handling in place
- [ ] Fallback mechanisms working
- [ ] UI accessible and showing traces
- [ ] Performance impact acceptable