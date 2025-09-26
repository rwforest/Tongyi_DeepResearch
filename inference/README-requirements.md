# Inference Requirements Guide

This folder contains the Databricks ReAct Inference System with two requirements files:

## Requirements Files

### 1. `requirements-minimal.txt` (Recommended)
Install only the core dependencies:
```bash
pip install -r requirements-minimal.txt
```

**Includes:**
- Core system: `python-dotenv`, `requests`, `tqdm`, `json5`
- Qwen Agent framework: `qwen-agent`
- Data processing: `numpy`, `pandas`

**Your predict function:** Add based on your setup:
- HuggingFace: `pip install transformers torch`
- Databricks SDK: `pip install databricks-sdk`
- MLflow: `pip install mlflow`
- OpenAI API: `pip install openai`

### 2. `requirements.txt` (Full installation)
Install all dependencies for complete functionality:
```bash
pip install -r requirements.txt
```

**Includes everything above plus:**
- Python code execution: `sandbox-fusion`
- Document processing: `pdfplumber`, `python-pptx`, `openpyxl`
- Image processing: `pillow`, `opencv-python-headless`
- Advanced file parsing: `dashscope` (IDP/Alibaba Cloud removed)
- Tokenization: `tiktoken`, `transformers`

## Tool Dependencies

| Tool | Required Packages | Description |
|------|------------------|-------------|
| **Web Search** | None (uses `requests`) | Search via Serper API |
| **Google Scholar** | None (uses `requests`) | Academic paper search |
| **Visit (Webpage)** | `tiktoken` (optional) | Webpage content extraction |
| **Python Interpreter** | `sandbox-fusion` | Safe code execution |
| **File Parser** | `pdfplumber`, `pillow`, `python-pptx`, `openpyxl` | Document processing |
| **Video Analysis** | `pillow`, `opencv-python-headless` | Video/image analysis |
| **Dashscope Tools** | `dashscope` | Advanced file parsing (IDP removed) |

## Installation Strategies

### Strategy 1: Minimal Setup (Recommended)
```bash
# Install core system
pip install -r requirements-minimal.txt

# Add your predict function dependencies
pip install transformers torch  # For HuggingFace
# OR
pip install databricks-sdk      # For Databricks
# OR
pip install mlflow              # For MLflow

# Add tools as needed
pip install sandbox-fusion      # For Python interpreter
pip install pdfplumber pillow   # For document processing
```

### Strategy 2: Full Setup
```bash
# Install everything
pip install -r requirements.txt

# Add your predict function
pip install transformers torch  # Or your choice
```

### Strategy 3: Docker/Conda Environment
```dockerfile
# Dockerfile example
FROM python:3.9
COPY requirements-minimal.txt .
RUN pip install -r requirements-minimal.txt
RUN pip install transformers torch  # Your choice
```

## API Keys Required (Optional)

Set these in `.env` file for full functionality:

```env
# Web search (recommended)
SERPER_KEY_ID=your_serper_key

# Webpage reading (optional)
JINA_API_KEYS=your_jina_keys

# Advanced file parsing (optional)
DASHSCOPE_API_KEY=your_dashscope_key
```

## Testing Installation

```python
# Test core system
from databricks_react_agent import DatabricksMultiTurnReactAgent
from databricks_multi_react import run_databricks_multi_react
print("✅ Core system working!")

# Test your predict function
def my_predict_function(prompt: str) -> str:
    return "Test response"

agent = DatabricksMultiTurnReactAgent(
    predict_function=my_predict_function,
    llm={'model': 'test'},
    function_list=["search"]
)
print("✅ Agent initialized successfully!")
```

## Troubleshooting

**Import Error: `qwen-agent`**
```bash
pip install qwen-agent==0.0.26
```

**Import Error: `sandbox-fusion`**
```bash
pip install sandbox-fusion==0.3.7
```

**Tools not working:**
- Check API keys in `.env` file
- Install tool-specific packages from table above
- Tools fall back gracefully when dependencies missing

## Next Steps

1. Choose your installation strategy
2. Install requirements
3. Set up your predict function
4. Configure API keys (optional)
5. Run the notebook or Python scripts!