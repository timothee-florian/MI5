#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 19:17:10 2025

@author: tim
"""

import asyncio
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server
from langchain_core.tools import Tool as LangChainTool
from typing import Optional
import re
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions
from chroma_populating import CHROMA_PATH, qwen_ef
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# ollama_ef = embedding_functions.OllamaEmbeddingFunction(
#     model_name= CHROMA_MODEL
# )

collection = chroma_client.get_or_create_collection(
        name="my_collection",
        embedding_function=qwen_ef
    )

# Query the collection
# query_text = "What is a vector database?"

def retrieve_info(query: str) -> str:
    """Retrieve info in the vector database the most related to the query. 
    Use embedding to check similarity between the query and all document saved in the database."""
   

    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    print(f"Query: {query}\n")
    print("Results:")
    out = [f"out for {query}:\n"]
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
        print(f"{i+1}. {doc}")
        print(f"   Distance: {distance}\n")
        out += [f"{doc}: {distance}"]
    return "\n".join(out)#"\n".join(results['documents'][0])

# Define the agent tools
def calculator(query: str) -> str:
    """Perform basic math operations"""
    try:
        # Remove all letters, keep only numbers, operators, and basic math symbols
        cleaned = re.sub(r'[a-zA-Z]', '', query)
        result = eval(cleaned)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

def get_time(query: str) -> str:
    """Get current time"""
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


# LangChain tools
langchain_tools = [
    LangChainTool(
        name="Calculator",
        func=calculator,
        description="Useful for math calculations. Input should be a math expression."
    ),
    LangChainTool(
        name="GetTime",
        func=get_time,
        description="Returns current date and time. No input needed."
    ),
    LangChainTool(
        name="RetrieveInfo",
        func=retrieve_info,
        description="Returns information saved as documents into a Vector store."
        )
]

# Simple rule-based agent
class SimpleRuleBasedAgent:
    def __init__(self, tools):
        self.tools = {tool.name: tool for tool in tools}
    
    def run(self, query: str) -> str:
        query_lower = query.lower()
        print(f'query_lower: {query_lower}')
        if any(word in query_lower for word in ['calculate', 'math', '+', '-', '*', '/']):
            for tool_name, tool in self.tools.items():
                if tool_name == "Calculator" :
                    return tool.func(query)
        
        elif any(word in query_lower for word in ['time', 'date', 'clock']):
            for tool_name, tool in self.tools.items():
                if tool_name == "GetTime":
                    return tool.func(query)
                
        elif any(word in query_lower for word in ['tell', 'info']):
            for tool_name, tool in self.tools.items():
                if tool_name == "RetrieveInfo":
                    return tool.func(query)
        
        
        return "I don't understand that query. I can: calculate math, get time, or info retrieval."


agent = SimpleRuleBasedAgent(langchain_tools)

# Create MCP server
app = Server("rule-based-agent")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="run_agent",
            description="Run the rule-based agent with a query. Can calculate math or get time.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to process (e.g., 'calculate 5+3', 'what time is it', 'Get me information about Napoleon' (RAG))"
                    }
                },
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    if name == "run_agent":
        query = arguments.get("query", "")
        result = agent.run(query)
        return [TextContent(type="text", text=result)]
    
    raise ValueError(f"Unknown tool: {name}")

async def main():
    """Run the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())