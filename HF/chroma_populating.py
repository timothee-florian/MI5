
import chromadb
from chromadb.utils import embedding_functions
CHROMA_PATH = "/home/tim/Desktop/MI5/HF/chroma_db_HF"

from chromadb import Documents, EmbeddingFunction, Embeddings
from transformers import AutoModel, AutoTokenizer
import torch
import chromadb

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
qwen_ef = QwenEmbeddingFunction("./qwen_model")


if __name__ == '__main__':
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    # ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    #     model_name= "CHROMA_MODEL
    # )
    # ollama_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    #     model_name= CHROMA_MODEL,
    #     api_key=None 
    # )

    collection = chroma_client.get_or_create_collection(
        name="my_collection",
        embedding_function=qwen_ef
    )

    # documents = [
    #     "ChromaDB is a vector database for AI applications",
    #     "Ollama allows you to run LLMs locally",
    #     "Vector databases store embeddings for semantic search",
    #     "Python is a popular programming language for AI"
    # ]
    documents = []
    while True:
        document = input('Give me an info?')
        if document =='':
            break
        documents += [document]
    colection_size =collection.count()
    if len(documents) > 0:
        collection.add(
            documents=documents,
            ids=[f"id{i}" for i in range(colection_size, len(documents)+colection_size)],
            metadatas=[{"source": f"doc{i}"} for i in range(len(documents))]
        )
