# MCP Local Agent Project

A collection of Python scripts demonstrating how to build AI agents using the Model Context Protocol (MCP) with both cloud (Claude) and local (Ollama) language models.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Files Structure](#files-structure)
- [Usage](#usage)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project demonstrates how to:
- Create an MCP server with custom tools
- Connect to the server using different clients (Claude API, Ollama, or direct calls)
- Build AI agents that can use tools to solve problems
- Run everything locally without cloud dependencies (using Ollama)

## ✨ Features

- **MCP Server** (`server.py`): Provides tools like calculator and time checker
- **Claude Client** (`client.py`): Uses Anthropic's Claude API (cloud-based)
- **Ollama Client** (`client_ollama.py`): Uses Ollama for fully local operation
- **Direct Tool Caller** (`direct_call.py`): Call MCP tools without any LLM

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Install Python Dependencies

```bash
pip install mcp anthropic ollama
```

### Step 2: Setup for Cloud (Optional - for Claude)

If you want to use the Claude client:

1. Get an API key from [Anthropic](https://console.anthropic.com/)
2. Edit `client.py` and replace `"your-api-key-here"` with your actual API key

Or set it as an environment variable:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Step 3: Setup for Local (For Ollama)

1. **Install Ollama:**
   - Visit [https://ollama.ai](https://ollama.ai)
   - Download and install for your operating system
   - Or on Linux: `curl -fsSL https://ollama.com/install.sh | sh`

2. **Pull a model:**
   ```bash
   ollama pull llama3.2
   ```
   
   Other recommended models:
   ```bash
   ollama pull llama3.1    # Larger, more capable
   ollama pull mistral     # Fast and efficient
   ollama pull phi3        # Lightweight
   ```

3. **Verify Ollama is running:**
   ```bash
   ollama list
   ```

## 📁 Files Structure

```
.
├── server.py              # MCP server with tools (calculate, get_time)
├── client.py              # Client using Claude API (cloud)
├── client_ollama.py       # Client using Ollama (fully local)
├── direct_call.py         # Direct tool caller (no LLM)
└── README.md             # This file
```

## 🚀 Usage

### Option 1: Using Claude (Cloud)

```bash
# With default query
python client.py

# With custom query
python client.py "What is 100 divided by 4? What time is it?"
```

### Option 2: Using Ollama (Fully Local)

```bash
# With default query
python client_ollama.py

# With custom query
python client_ollama.py "Calculate 50 * 2 and tell me the current time"

# General questions (no tools needed)
python client_ollama.py "What is the capital of France?"
```

### Option 3: Direct Tool Calls (No LLM)

```bash
# Run demo of all tools
python direct_call.py

# Call calculate tool
python direct_call.py calculate "25 * 4 + 10"

# Call get_time tool
python direct_call.py get_time
```

## 💡 Examples

### Example 1: Math Calculation

```bash
python client_ollama.py "What is 123 * 456?"
```

**Output:**
```
User: What is 123 * 456?
Using model: llama3.2

Calling tool: calculate
Arguments: {'expression': '123 * 456'}

Tool result: Result: 56088

Ollama: The result of 123 multiplied by 456 is 56,088.
```

### Example 2: Multiple Tools

```bash
python client_ollama.py "Calculate 100/5 and tell me what time it is"
```

**Output:**
```
User: Calculate 100/5 and tell me what time it is
Using model: llama3.2

Calling tool: calculate
Arguments: {'expression': '100/5'}

Tool result: Result: 20.0

Calling tool: get_time
Arguments: {}

Tool result: Current time: 2025-10-20 14:30:45

Ollama: 100 divided by 5 equals 20. The current time is 2:30:45 PM on October 20, 2025.
```

### Example 3: General Question (No Tools)

```bash
python client_ollama.py "Tell me a fun fact about Python programming"
```

**Output:**
```
User: Tell me a fun fact about Python programming
Using model: llama3.2

Ollama: Python was named after the British comedy group Monty Python, not the snake! Guido van Rossum, Python's creator, was a fan of the show and wanted a name that was short, unique, and slightly mysterious.
```

### Example 4: Direct Tool Call

```bash
python direct_call.py calculate "2**10"
```

**Output:**
```
Available tools on MCP server:
  - calculate: Perform basic arithmetic calculations
  - get_time: Get the current time

Calling tool: calculate
Arguments: {'expression': '2**10'}

Result:
  Result: 1024
```

## 🛠️ Troubleshooting

### Ollama Connection Error

**Error:** `Error connecting to Ollama: ...`

**Solutions:**
1. Check if Ollama is running: `ollama list`
2. Start Ollama service if needed
3. Make sure you have a model installed: `ollama pull llama3.2`
4. Update the ollama package: `pip install --upgrade ollama`

### MCP Import Error

**Error:** `cannot import Client from mcp.client`

**Solution:**
Make sure you have the latest MCP package:
```bash
pip install --upgrade mcp
```

### Model Not Found

**Error:** `Model 'llama3.2' not found`

**Solution:**
Pull the model first:
```bash
ollama pull llama3.2
```

Or change the model in `client_ollama.py` to one you have installed.

### Claude API Key Error

**Error:** `API key not found`

**Solution:**
1. Get an API key from [Anthropic Console](https://console.anthropic.com/)
2. Set it in `client.py` or as environment variable:
   ```bash
   export ANTHROPIC_API_KEY="your-key-here"
   ```

## 📝 Adding Custom Tools

To add your own tools to the MCP server, edit `server.py`:

```python
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        # ... existing tools ...
        types.Tool(
            name="your_tool_name",
            description="What your tool does",
            inputSchema={
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "Parameter description"
                    }
                },
                "required": ["param1"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None):
    # ... existing tool handlers ...
    
    elif name == "your_tool_name":
        # Your tool implementation
        result = do_something(arguments["param1"])
        return [types.TextContent(type="text", text=f"Result: {result}")]
```

## 🤝 Contributing

Feel free to extend this project by:
- Adding more tools to the MCP server
- Supporting additional LLM providers
- Improving error handling
- Adding more examples

## 📄 License

This project is open source and available for educational purposes.

## 🔗 Resources

- [MCP Documentation](https://docs.claude.com/en/docs/mcp)
- [Anthropic API Docs](https://docs.anthropic.com/)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [Model Context Protocol Specification](https://spec.modelcontextprotocol.io/)