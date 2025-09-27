# Tongyi Deep Research - Databricks Integration

This directory contains the Databricks-compatible version of the Tongyi Deep Research agent, modified to use your custom predict() function instead of vLLM servers.

## 🚀 **Quick Start**

### 1. **Environment Setup**
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your actual API keys
nano .env
```

### 2. **Essential API Keys**
Get these API keys for basic functionality:
- **SERPER_KEY_ID**: Web search (get from [serper.dev](https://serper.dev/))
- **PERPLEXITY_API_KEY**: AI-powered search (get from [perplexity.ai](https://www.perplexity.ai/))

### 3. **Test Setup**
```bash
# Test all tools and API keys
python3 test_all_tools.py

# Test search functionality specifically
python3 simple_search_test.py

# Test Visit tool with predict function
python3 test_visit_predict.py
```

## 🏗️ **Architecture**

### **Main Components**

| File | Purpose | Key Features |
|------|---------|--------------|
| `databricks_react_agent.py` | **Main ReAct Agent** | Uses your predict() function, comprehensive tracing |
| `databricks_multi_react.py` | **Batch Processor** | Processes multiple research questions |
| `databricks_react_inference.py` | **High-level Interface** | Easy-to-use research methods |

### **Research Tools**

| Tool | File | Purpose | API Required |
|------|------|---------|--------------|
| **Search** | `tool_search.py` | **Configurable**: Serper OR Perplexity | SERPER_KEY_ID or PERPLEXITY_API_KEY |
| **Perplexity** | `tool_perplexity.py` | Dedicated AI search with citations | PERPLEXITY_API_KEY |
| **Scholar** | `tool_scholar.py` | Academic paper search | SERPER_KEY_ID |
| **Visit** | `tool_visit.py` | Webpage analysis | JINA_API_KEYS + Your predict() |
| **Python** | `tool_python.py` | Code execution | SANDBOX_FUSION_ENDPOINT |
| **Files** | `tool_file.py` | Document parsing | DASHSCOPE_API_KEY |

## 🔧 **Configuration**

### **Environment Variables (.env)**

#### **Required for Basic Functionality**
```bash
# Search APIs (choose one or both)
SERPER_KEY_ID=your_serper_api_key        # For traditional web search
PERPLEXITY_API_KEY=your_perplexity_api_key  # For AI-powered search

# Search Backend Configuration
USE_PERPLEXITY_SEARCH=false      # Set to 'true' to use Perplexity Search API instead of Serper
PERPLEXITY_MAX_RESULTS=10        # Number of results from Perplexity Search (1-20)

# Model Configuration
TEMPERATURE=0.85
PRESENCE_PENALTY=1.1
```

#### **Optional but Recommended**
```bash
# Web Page Reading
JINA_API_KEYS=your_jina_api_key

# Code Execution
SANDBOX_FUSION_ENDPOINT=your_sandbox_endpoints

# File Parsing
DASHSCOPE_API_KEY=your_dashscope_key
DASHSCOPE_API_BASE=your_dashscope_base
```

#### **Legacy OpenAI Fallback (for Visit tool)**
```bash
# Only needed if you want OpenAI fallback instead of predict()
API_KEY=your_openai_api_key
API_BASE=your_openai_base_url
SUMMARY_MODEL_NAME=your_model_name
```

## 🎯 **Usage**

### **Basic Research Query**
```python
from databricks_react_agent import DatabricksMultiTurnReactAgent

# Your predict function
def my_predict_function(model_input):
    # Your Databricks model inference logic
    return model_response

# Initialize agent
agent = DatabricksMultiTurnReactAgent(
    predict_function=my_predict_function,
    llm={'model': 'your-model-name'}
)

# Run research
data = {
    'item': {
        'question': 'What are the latest developments in AI agents?',
        'answer': 'Reference answer if available'
    }
}

result = agent._run(data)
print(result['prediction'])
```

### **High-Level Interface**
```python
from databricks_react_inference import research_question

# Simple research query
answer = research_question(
    "What are the recent breakthroughs in quantum computing?",
    predict_function=my_predict_function
)
```

### **Batch Processing**
```python
from databricks_multi_react import run_batch_inference

questions = [
    "Latest AI research trends",
    "Quantum computing applications",
    "Climate change solutions"
]

results = run_batch_inference(
    questions=questions,
    predict_function=my_predict_function
)
```

## 🔍 **Key Features**

### **🔄 Configurable Search Backends**
- **Flexible Search**: Switch between Serper (traditional) and Perplexity (AI-powered)
- **Runtime Configuration**: Change backends via environment variable
- **Automatic Switching**: Set once, works throughout the entire workflow

#### **Search Backend Comparison**
| Feature | Serper | Perplexity Search API |
|---------|--------|----------------------|
| **Speed** | ⚡ Fast | ⚡ Fast |
| **Cost** | 💰 Low | 💰💰 $5/1k requests |
| **Format** | 📄 Raw web results | 📊 Ranked search results |
| **Data** | 🔄 Standard indexing | 🚀 Real-time continuous index |
| **Best For** | General web search | Advanced research with ranking |

### **🧠 Predict Function Integration**
- **Main Agent**: Uses your predict() function for reasoning
- **Visit Tool**: Uses your predict() function for webpage summarization
- **Unified Model**: Same Tongyi DeepResearch model throughout workflow
- **MLflow Tracing**: Optional comprehensive observability for predict functions

### **🛠️ Tool Calling**
- **Dynamic API Keys**: Tools load environment variables at runtime
- **Comprehensive Tracing**: Single-line logging for debugging
- **Error Handling**: Graceful fallbacks and detailed error messages
- **MLflow Integration**: Full observability with GenAI tracing decorators

### **📊 Research Capabilities**
- **Web Search**: Real-time search via Serper API
- **AI Search**: Perplexity API with citations
- **Academic Search**: Google Scholar integration
- **Web Analysis**: Automatic webpage reading and summarization
- **Code Execution**: Python code running in sandboxed environment
- **Document Processing**: PDF, Word, and other file formats

## 🧪 **Testing**

### **Comprehensive Test Suite**
```bash
# Test all components
python3 test_all_tools.py
```
**Output**: Shows status of all API keys, tool imports, and integrations

### **Individual Tests**
```bash
# Test search functionality
python3 simple_search_test.py

# Test search backend switching
python3 test_search_backends.py

# Test ReAct agent tool calling
python3 test_react_tools.py

# Test Visit tool predict integration
python3 test_visit_predict.py
```

### **Debug Tool Calling**
```bash
# Compare independent vs agent-based tool calling
python3 debug_tool_calling.py
```

## 🔧 **Troubleshooting**

### **Common Issues**

#### **"No module named 'qwen_agent'"**
```bash
pip install qwen-agent json5 requests python-dotenv
```

#### **"SERPER_KEY_ID environment variable not set"**
1. Get API key from [serper.dev](https://serper.dev/)
2. Add to `.env`: `SERPER_KEY_ID=your_actual_key`

#### **"Tool call failed"**
- Check API keys are set correctly
- Run `test_all_tools.py` to diagnose issues
- Check tracing output for specific error details

#### **Search timeouts**
- Verify internet connectivity
- Check if API quotas are exceeded
- Try different search queries

### **Debugging with Tracing**

The system includes comprehensive tracing:
- `🔧 ENV_KEY:` - Environment variable status
- `🔍 API_CALL:` - Search tool operations
- `🧠 API_CALL:` - Perplexity tool operations
- `🌐 VISIT_PREDICT:` - Visit tool using predict function
- `🛠️ TRACE:` - Agent tool calling
- `🔧 TOOL CALL TRACE:` - Detailed tool call parsing
- `⏱️ TIMING:` - Execution time insights for all functions
- `📊 MLFLOW:` - Comprehensive observability with GenAI tracing

## 📊 **MLflow Observability**

### **GenAI Tracing Integration**
The system uses MLflow's GenAI tracing decorators for comprehensive observability:

```python
@mlflow.trace(name="react_agent_session", span_type=SpanType.AGENT)
def _run(self, data, model=None, planning_port=None):
    # Session-level tracing with nested spans
```

### **Hierarchical Span Structure**
```
Session (AGENT)
├── LLM Call (LLM) - predict() function execution
├── Tool Execution (TOOL)
│   ├── Search (TOOL) - Serper/Perplexity API calls
│   ├── Visit (TOOL) - Webpage analysis
│   └── Other Tools (TOOL)
└── Final Response (AGENT)
```

### **Automatic Metrics Captured**
- **Performance**: Execution time for each component
- **Usage**: Token/character counts, API call frequencies
- **Reliability**: Retry attempts, success rates, error types
- **Quality**: Response lengths, tool success rates

### **Testing MLflow Integration**
```bash
# Test MLflow tracing integration
python3 test_mlflow_tracing.py

# View traces in MLflow UI
mlflow ui
# Open http://localhost:5000
```

### **MLflow Troubleshooting**
Common issues and solutions:

#### **Default Experiment Warning**
```python
# Use specific experiment instead of default
mlflow.set_experiment("tongyi_deepresearch_agent")

# Or use robust configuration
from mlflow_config import initialize_mlflow
initialize_mlflow()
```

#### **Timeout Issues**
```bash
# Set timeout environment variables in .env
MLFLOW_TRACKING_TIMEOUT=30
MLFLOW_HTTP_REQUEST_TIMEOUT=30
```

#### **Safe Tracing**
All tracing uses safe decorators that gracefully degrade if MLflow fails:
```python
@safe_trace_decorator(name="function_name", span_type=SpanType.LLM)
def your_function():
    # Core functionality continues even if tracing fails
```

For detailed troubleshooting, see `MLFLOW_TROUBLESHOOTING.md`

### **Traced Predict Functions**
Add comprehensive observability to any predict function:

```python
from traced_predict_wrapper import create_traced_predict_function

# Wrap any predict function with MLflow tracing
traced_predict = create_traced_predict_function(
    your_predict_function,
    model_name="your-model",
    log_inputs=True,
    log_outputs=True
)

# Use in ReAct agent for complete observability
agent = DatabricksMultiTurnReactAgent(
    predict_function=traced_predict
)
```

**Benefits:**
- Input/output logging with privacy controls
- Execution timing and performance metrics
- Token usage approximation
- Error tracking and classification
- Works with any predict function signature

**Pre-built wrappers available for:**
- Databricks model serving endpoints
- HuggingFace transformers
- OpenAI API
- Any custom inference function

## 📁 **File Organization**

```
inference/
├── README.md                          # This file
├── .env.example                       # Environment template
├── .env                               # Your API keys (never commit!)
│
├── databricks_react_agent.py          # Main ReAct agent
├── databricks_multi_react.py          # Batch processing
├── databricks_react_inference.py      # High-level interface
│
├── tool_search.py                     # Web search tool
├── tool_perplexity.py                 # Perplexity AI search
├── tool_scholar.py                    # Academic search
├── tool_visit.py                      # Webpage analysis
├── tool_python.py                     # Code execution
├── tool_file.py                       # Document parsing
│
├── test_all_tools.py                  # Comprehensive tests
├── simple_search_test.py              # Search API tests
├── test_react_tools.py                # Agent integration tests
├── test_visit_predict.py              # Visit tool tests
├── test_mlflow_tracing.py             # MLflow tracing tests
├── traced_predict_demo.py             # Traced predict function demo
├── debug_tool_calling.py              # Debug utilities
│
├── requirements.txt                   # Full dependencies
├── requirements-minimal.txt           # Minimal dependencies
│
├── mlflow_config.py                   # Robust MLflow configuration
├── traced_predict_wrapper.py          # MLflow predict function wrappers
├── MLFLOW_TROUBLESHOOTING.md          # MLflow troubleshooting guide
│
└── SETUP_GUIDE.md                     # Detailed setup instructions
```

## 🎯 **Next Steps**

1. **Set up API keys** in `.env` file
2. **Run tests** to verify configuration
3. **Try basic research query** with your predict function
4. **Add MLflow tracing** to your predict function for observability
5. **Explore advanced features** like batch processing
6. **Check out the notebook** `databricks_react_inference.ipynb`

## 📚 **Additional Resources**

- **Paper**: [ReSum: Unlocking Long-Horizon Search Intelligence](https://huggingface.co/papers/2509.13313)
- **Model**: [Tongyi-DeepResearch-30B-A3B](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B)
- **Security**: See `../SECURITY.md` for API key protection guidelines

## 🤝 **Support**

- **Configuration Issues**: Run `test_all_tools.py` for diagnosis
- **API Problems**: Check individual tool tests
- **Integration Questions**: See tracing output for debugging

---

**⚠️ Security Note**: Never commit your `.env` file! It contains sensitive API keys that should remain private.