import chromadb
from chromadb.utils import embedding_functions
import ollama

# Initialize ChromaDB client
client = chromadb.Client()

# Set up Ollama embedding function
# Make sure Ollama is running locally with: ollama serve
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="nomic-embed-text" #"llama3"  
)

# Create or get a collection
collection = client.get_or_create_collection(
    name="my_collection",
    embedding_function=ollama_ef
)

# Add documents to the collection
documents = [
    "ChromaDB is a vector database for AI applications",
    "Ollama allows you to run LLMs locally",
    "Vector databases store embeddings for semantic search",
    "Python is a popular programming language for AI"
]

collection.add(
    documents=documents,
    ids=[f"id{i}" for i in range(len(documents))],
    metadatas=[{"source": f"doc{i}"} for i in range(len(documents))]
)

# Query the collection
query_text = "What is a vector database?"
results = collection.query(
    query_texts=[query_text],
    n_results=2
)

print(f"Query: {query_text}\n")
print("Results:")
for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
    print(f"{i+1}. {doc}")
    print(f"   Distance: {distance}\n")

# Optional: Use Ollama for LLM generation after retrieval
import requests

def query_ollama(prompt, model="llama3"):
    response = ollama.generate(model=model, prompt=prompt)
    return response['response']


# RAG example: Retrieve context and generate response
context = "\n".join(results['documents'][0])
prompt = f"Based on this context:\n{context}\n\nAnswer: {query_text}"
print(f"Prompt: {prompt}")
answer = query_ollama(prompt)
print(f"LLM Response:\n{answer}")