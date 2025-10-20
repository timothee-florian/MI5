install:
pip install mcp anthropic ollama

local model:
ollama pull llama3.2
ollama pull mistral


run:
run the server
pyhton ollama pull llama3.2

run the clienty with a query, for example:

python client_ollama.py "What is 25 * 4 + 10? Also, what time is it?"

python client_ollama.py "What is the capital of France?"

python client_ollama.py "Combien font 100 diviser par 5"
