### Prerequisites

- Python 3.8 or higher

### Installation
#### pip
pip install mcp anthropic ollama flask

#### local model:
1. **Install Ollama:**
   - on Linux: `curl -fsSL https://ollama.com/install.sh | sh`

2. **Pull a model:**
```bash
   ollama pull llama3.2
   ```
```bash
   ollama pull mistral
   ```


### running
run the server
```bash
python mcp_server.py
```

run the clienty with a query, for example:
```bash

python client_ollama.py "What is 25 * 4 + 10? Also, what time is it?"
```
```bash

python client_ollama.py "What is the capital of France?"
```
```bash

python client_ollama.py "Combien font 100 diviser par 5"
```
