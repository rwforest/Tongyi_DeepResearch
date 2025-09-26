# Tongyi Deep Research + Databricks Setup Guide

## 🔧 Environment Setup

### 1. API Key Configuration

You need a Serper API key for web search functionality:

1. Go to [https://serper.dev/](https://serper.dev/)
2. Sign up for a free account
3. Get your API key from the dashboard
4. Edit `inference/.env` and replace `your_serper_key` with your actual key:

```bash
# Edit this line in inference/.env:
SERPER_KEY_ID=your_actual_serper_api_key_here
```

### 2. Test Search Functionality

Run the standalone search test:

```bash
cd inference
python3 simple_search_test.py
```

Expected output with valid API key:
```
✅ Search successful! [number] characters
📄 First result: [actual search result title]
```

### 3. Test Tool Calling Integration

Once search works independently, test the full ReAct agent:

```bash
cd inference
python3 test_react_tools.py
```

## 🐛 Debugging Search Issues

Based on our analysis, here's the debugging flow:

### Issue: Search tool works independently but fails in agent context

**Root Cause Analysis:**
- ✅ HTTP connection to Serper API works
- ✅ Environment variables load correctly
- ✅ JSON parsing works properly
- ✅ Retry mechanism functions
- ❌ **Primary issue**: Missing/invalid API key

### Current Status
1. **Environment Loading**: ✅ Working - `.env` file loads properly
2. **HTTP Connectivity**: ✅ Working - reaches Serper API
3. **API Authentication**: ❌ **Need valid API key**
4. **Tool Integration**: 🔄 **Ready to test once API key is set**

### Error Patterns Explained

| Error Message | Cause | Solution |
|---------------|--------|----------|
| `No module named 'qwen_agent'` | Missing dependencies | Install: `pip install qwen-agent` |
| `No module named 'json5'` | Missing dependencies | Install: `pip install json5` |
| `SERPER_KEY_ID environment variable not set` | No .env file | Copy `.env.example` to `.env` |
| `{"message":"Unauthorized.","statusCode":403}` | Invalid API key | Get real key from serper.dev |
| `Search attempt failed: 'str' object has no attribute 'get'` | Message format issue | Fixed in latest code |
| `expected string or bytes-like object, got 'NoneType'` | Environment variable not loaded in tool context | Fixed by proper .env setup |

## 🎯 Next Steps

1. **Set valid SERPER_KEY_ID** in `inference/.env`
2. **Install missing dependencies** if needed:
   ```bash
   pip install qwen-agent json5 python-dotenv
   ```
3. **Run tests** to verify everything works:
   ```bash
   python3 simple_search_test.py    # Test search API
   python3 test_react_tools.py      # Test full integration
   ```

## 📁 Key Files

- `inference/.env` - Environment configuration
- `inference/tool_search.py` - Search tool with comprehensive tracing
- `inference/databricks_react_agent.py` - Main agent with predict() integration
- `inference/simple_search_test.py` - Standalone search testing
- `inference/test_react_tools.py` - Full integration testing

## 🔍 Architecture Summary

```
User's predict() function
         ↓
DatabricksMultiTurnReactAgent
         ↓
Tool calling (search, scholar, python, etc.)
         ↓
Search Tool → Serper API (requires valid key)
```

The integration is **ready** - the only missing piece is the Serper API key for web search functionality.