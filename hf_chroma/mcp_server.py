import asyncio
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server
from langchain_core.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.agents import create_agent
from typing import Any, List, Optional
import re
from datetime import datetime
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
from chroma_populating import COLLECTION_NAME, QwenEmbeddingFunction
from download import model_name, local_dir
import os
from constants import embeded_model, main_dir, llm_model, CHROMA_PATH

# Global variables for Qwen LLM model
qwen_llm_model = None
qwen_llm_tokenizer = None

def initialize_qwen_llm(model_name="Qwen/Qwen2.5-7B-Instruct"):
    """Initialize the Qwen LLM model and tokenizer for chat/generation"""
    global qwen_llm_model, qwen_llm_tokenizer
    
    print(f"Loading Qwen LLM model: {model_name}")
    
    qwen_llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    qwen_llm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    
    if not torch.cuda.is_available():
        qwen_llm_model = qwen_llm_model.to('cpu')
    
    print("Qwen LLM model loaded successfully!")

# Initialize ChromaDB with custom Qwen embedding function
print(f"Connecting to ChromaDB at: {CHROMA_PATH}")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

qwen_ef = QwenEmbeddingFunction(os.path.join(main_dir, embeded_model))



collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=qwen_ef
)

print(f"ChromaDB collection loaded. Total documents: {collection.count()}")

# Define tools using @tool decorator for LangChain 1.1.2
@tool
def retrieve_info(query: str) -> str:
    """Retrieve information from the vector database that is most related to the query. 
    Use this tool when the user asks for information, facts, or knowledge about a topic.
    Input should be a clear query or question."""
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=3
        )

        print(f"\n[RETRIEVE_INFO] Query: {query}")
        print("Results:")
        
        if not results['documents'][0]:
            return f"No relevant information found for: {query}"
        
        out = [f"Information retrieved for '{query}':\n"]
        for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
            print(f"{i+1}. {doc} (Distance: {distance})")
            relevance = 1 - distance  # Convert distance to relevance score
            out.append(f"{i+1}. {doc} (Relevance: {relevance:.2f})")
        
        return "\n".join(out)
    except Exception as e:
        return f"Error retrieving information: {str(e)}"

@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations. 
    Use this tool when the user asks to calculate, compute, or solve math problems.
    Input should be a mathematical expression like '5+3', '10*25', or '(100-50)/2'."""
    
    try:
        # Remove letters but keep numbers, operators, and math symbols
        cleaned = re.sub(r'[a-zA-Z]', '', expression).strip()
        
        if not cleaned:
            return "Error: No valid mathematical expression found"
        
        # Safe evaluation
        result = eval(cleaned)
        print(f"[CALCULATOR] {cleaned} = {result}")
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_time(query: str = "") -> str:
    """Get the current date and time.
    Use this tool when the user asks about the current time, date, or day.
    No specific input needed."""
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[GET_TIME] {current_time}")
    return f"Current time: {current_time}"

# Qwen Chat Model for LangChain
class QwenChatModel(BaseChatModel):
    """Chat model wrapper for Qwen LLM"""
    
    max_tokens: int = 512
    temperature: float = 0.7
    
    @property
    def _llm_type(self) -> str:
        return "qwen-chat"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Generate response from Qwen model"""
        if qwen_llm_model is None or qwen_llm_tokenizer is None:
            raise RuntimeError("Qwen LLM model not initialized")
        
        # Convert LangChain messages to Qwen format
        qwen_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                qwen_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                qwen_messages.append({"role": "assistant", "content": msg.content})
            else:
                qwen_messages.append({"role": "user", "content": str(msg.content)})
        
        # Apply chat template
        text = qwen_llm_tokenizer.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = qwen_llm_tokenizer([text], return_tensors="pt").to(qwen_llm_model.device)
        
        with torch.no_grad():
            outputs = qwen_llm_model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                top_p=0.9,
                pad_token_id=qwen_llm_tokenizer.eos_token_id,
            )
        
        response = qwen_llm_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Handle stop sequences
        if stop:
            for stop_seq in stop:
                if stop_seq in response:
                    response = response[:response.index(stop_seq)]
        
        return response.strip()
    
    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any):
        """Streaming not implemented"""
        raise NotImplementedError("Streaming not supported")
    
    @property
    def _identifying_params(self) -> dict:
        return {"model_name": "qwen", "max_tokens": self.max_tokens, "temperature": self.temperature}

def generate_qwen_response(prompt: str, max_tokens: int = 512) -> str:
    """Generate a direct response from Qwen LLM model"""
    if qwen_llm_model is None or qwen_llm_tokenizer is None:
        raise RuntimeError("Qwen LLM model not initialized")
    
    messages = [{"role": "user", "content": prompt}]
    text = qwen_llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = qwen_llm_tokenizer([text], return_tensors="pt").to(qwen_llm_model.device)
    
    with torch.no_grad():
        outputs = qwen_llm_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=qwen_llm_tokenizer.eos_token_id
        )
    
    response = qwen_llm_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

# Agent function using LangChain 1.1.2
def run_langchain_agent(query: str, use_agent: bool = True) -> dict:
    """Run the LangChain agent with tools"""
    
    print(f"\n{'='*60}")
    print(f"AGENT PROCESSING: {query}")
    print(f"{'='*60}\n")
    
    if not use_agent:
        # Simple rule-based fallback
        return run_simple_agent(query)
    
    try:
        # Initialize Qwen chat model
        chat_model = QwenChatModel(max_tokens=512, temperature=0.3)
        
        # Define tools
        tools = [retrieve_info, calculator, get_time]
        
        # Create agent
        agent = create_agent(
            model=chat_model,
            tools=tools,
            system_prompt="""You are a helpful assistant with access to tools.

Available tools:
- retrieve_info: Search the knowledge base for information about topics
- calculator: Perform mathematical calculations
- get_time: Get the current date and time

Think step by step about which tool(s) to use to answer the user's question.
If the question requires multiple steps, use multiple tools in sequence.""",
            debug=True
        )
        
        # Invoke agent
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})
        
        # Extract response
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            output = last_message.content if hasattr(last_message, 'content') else str(last_message)
        else:
            output = str(result)
        
        return {
            "output": output,
            "success": True,
            "agent_type": "langchain"
        }
        
    except Exception as e:
        import traceback
        error_msg = f"LangChain agent error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        
        # Fallback to simple agent
        print("Falling back to simple rule-based agent...")
        return run_simple_agent(query)

def run_simple_agent(query: str) -> dict:
    """Simple rule-based agent as fallback"""
    
    print(f"\n{'='*60}")
    print(f"SIMPLE AGENT PROCESSING: {query}")
    print(f"{'='*60}\n")
    
    query_lower = query.lower()
    result = ""
    
    # Check for calculator
    if any(word in query_lower for word in ['calculate', 'math', 'compute', '+', '-', '*', '/', 'sum']):
        print("→ Using CALCULATOR tool")
        result = calculator.invoke(query)
    
    # Check for time
    elif any(word in query_lower for word in ['time', 'date', 'clock', 'today', 'now']):
        print("→ Using GET_TIME tool")
        result = get_time.invoke(query)
    
    # Check for information retrieval
    elif any(word in query_lower for word in ['tell', 'info', 'information', 'about', 'what', 'who', 'explain']):
        print("→ Using RETRIEVE_INFO tool")
        rag_result = retrieve_info.invoke(query)
        
        # Use Qwen to synthesize a natural response
        print("→ Using QWEN to generate natural response")
        synthesis_prompt = f"""Based on the following retrieved information, provide a clear and concise answer to the question: "{query}"

Retrieved Information:
{rag_result}

Please provide a natural, well-structured answer."""
        
        result = generate_qwen_response(synthesis_prompt, max_tokens=400)
    
    else:
        result = "I don't understand that query. I can: calculate math, get time, or info retrieval."
    
    print(f"{'='*60}\n")
    
    return {
        "output": result,
        "success": True,
        "agent_type": "simple"
    }

# Create MCP server
app = Server("rule-based-agent")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools"""
    return [
        Tool(
            name="run_agent",
            description="Run the agent with a query. The agent can retrieve information from the knowledge base, perform calculations, or get the current time. It will intelligently decide which tool(s) to use.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or task for the agent (e.g., 'Tell me about Napoleon', 'calculate 125 * 48', 'what time is it?')"
                    },
                    "use_langchain": {
                        "type": "boolean",
                        "description": "Use LangChain agent (true) or simple rule-based agent (false). Default: true",
                        "default": True
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="retrieve_info",
            description="Directly retrieve information from the vector database without agent reasoning",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search in the knowledge base"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="calculate",
            description="Directly perform a mathematical calculation without agent reasoning",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '25 * 4 + 10')"
                    }
                },
                "required": ["expression"]
            }
        ),
        Tool(
            name="get_time",
            description="Directly get the current date and time without agent reasoning",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle MCP tool calls"""
    
    try:
        if name == "run_agent":
            query = arguments.get("query", "")
            use_langchain = arguments.get("use_langchain", True)
            
            if not query:
                return [TextContent(type="text", text="Error: No query provided")]
            
            result = run_langchain_agent(query, use_agent=use_langchain)
            
            response = f"**Agent Response ({result['agent_type']}):**\n{result['output']}"
            return [TextContent(type="text", text=response)]
        
        elif name == "retrieve_info":
            query = arguments.get("query", "")
            if not query:
                return [TextContent(type="text", text="Error: No query provided")]
            
            result = retrieve_info.invoke(query)
            return [TextContent(type="text", text=result)]
        
        elif name == "calculate":
            expression = arguments.get("expression", "")
            if not expression:
                return [TextContent(type="text", text="Error: No expression provided")]
            
            result = calculator.invoke(expression)
            return [TextContent(type="text", text=result)]
        
        elif name == "get_time":
            result = get_time.invoke("")
            return [TextContent(type="text", text=result)]
        
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
            
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        return [TextContent(type="text", text=error_msg)]

async def main():
    """Run the MCP server"""
    # Initialize Qwen LLM model first
    initialize_qwen_llm(os.path.join(main_dir, llm_model)) # Change to your model path if needed
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())