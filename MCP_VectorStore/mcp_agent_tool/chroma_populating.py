#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 19:57:46 2025

@author: tim
"""

import chromadb
from chromadb.utils import embedding_functions
CHROMA_PATH = "/home/tim/Desktop/MI5/mcp_agent_tool/chroma_db"
CHROMA_MODEL = "nomic-embed-text" #"llama3"  
if __name__ == '__main__':
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        model_name= CHROMA_MODEL
    )

    collection = chroma_client.get_or_create_collection(
        name="my_collection",
        embedding_function=ollama_ef
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
