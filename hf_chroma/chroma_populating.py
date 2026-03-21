import chromadb
from chromadb.utils import embedding_functions


from chromadb import Documents, EmbeddingFunction, Embeddings
from transformers import AutoModel, AutoTokenizer
import torch
import chromadb
from download import local_dir
from constants import embeded_model, main_dir, CHROMA_PATH, COLLECTION_NAME
import os
COLLECTION_NAME = "my_collection"

class QwenEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        self.model.eval()
    
    def __call__(self, input: Documents) -> Embeddings:
        inputs = self.tokenizer(
            input,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy().tolist()

# Use it

if __name__ == '__main__':
    qwen_ef = QwenEmbeddingFunction(os.path.join(main_dir, embeded_model))


    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    # ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    #     model_name= "CHROMA_MODEL
    # )
    # ollama_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    #     model_name= CHROMA_MODEL,
    #     api_key=None 
    # )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=qwen_ef
    )

    # documents = [
    #     "ChromaDB is a vector database for AI applications",
    #     "Ollama allows you to run LLMs locally",
    #     "Vector databases store embeddings for semantic search",
    #     "Python is a popular programming language for AI"
    # ]

# ChromaDB is a vector database for AI applications
# Ollama allows you to run LLMs locally
# Vector databases store embeddings for semantic search
# Python is a popular programming language for AI
# Napoleon was french
# I love anja

    documents = []
    while True:
        document = input('Give me an info?')
        if document =='':
            break
        documents += document.split('\n# ')
    colection_size =collection.count()
    if len(documents) > 0:
        collection.add(
            documents=documents,
            ids=[f"id{i}" for i in range(colection_size, len(documents)+colection_size)],
            metadatas=[{"source": f"doc{i}"} for i in range(len(documents))]
        )