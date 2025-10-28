# client_langgraph_hf.py
"""
MCP Client with LangGraph using HuggingFace Transformers (fully pip-installable)
Install dependencies: 
  pip install mcp langchain-huggingface langgraph langchain-core transformers torch
  
No external services needed - everything runs locally!
"""

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
import asyncio
import sys
import json
import re

# Global MCP session (will be initialized in async context)
mcp_session = None
mcp_tools_list = None

def create_langchain_tool_from_mcp(mcp_tool):
    """Create a LangChain tool from an MCP tool definition"""
    
    @tool
    async def mcp_tool_wrapper(**kwargs) -> str:
        """Dynamically created tool from MCP server"""
        try:
            result = await mcp_session.call_tool(mcp_tool.name, arguments=kwargs)
            return result.content[0].text
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Set the tool metadata
    mcp_tool_wrapper.name = mcp_tool.name
    mcp_tool_wrapper.description = mcp_tool.description
    mcp_tool_wrapper.args_schema = mcp_tool.inputSchema
    
    return mcp_tool_wrapper

def create_agent_graph(llm, langchain_tools: list = None):
    """Create a LangGraph agent with MCP tools"""
    
    if langchain_tools is None:
        langchain_tools = []
    
    # Build tool descriptions for system prompt
    tool_descriptions = "\n".join([
        f"- '{t.name}': {t.description}" for t in langchain_tools
    ])
    
    # Create a simple tool calling wrapper for the LLM
    async def call_model(state: MessagesState):
        messages = state["messages"]
        
        # Build a prompt with tool information
        tool_prompt = f"""You are a helpful assistant with access to the following tools:

{tool_descriptions}

IMPORTANT: Only use tools when necessary. For general knowledge questions, answer directly.

To use a tool, respond EXACTLY in this format:
TOOL_CALL: tool_name
ARGUMENTS: {{"arg1": "value1", "arg2": "value2"}}

If you don't need to use a tool, just answer the question directly.

Examples:
- Question: "What is the capital of France?" -> Answer: "The capital of France is Paris."
- Question: "Calculate 12345 * 67890" -> TOOL_CALL: calculate
ARGUMENTS: {{"expression": "12345 * 67890"}}
- Question: "What time is it?" -> TOOL_CALL: get_time
ARGUMENTS: {{}}

"""
        
        # Get last user message
        user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
            elif isinstance(msg, ToolMessage):
                # Already processed tools, now generate final answer
                tool_prompt = "Based on the tool results above, provide a final answer to the user's question.\n\n"
                break
        
        # Build full prompt
        full_prompt = tool_prompt
        
        # Add conversation history
        for msg in messages:
            if isinstance(msg, HumanMessage):
                full_prompt += f"\nUser: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                full_prompt += f"\nAssistant: {msg.content}\n"
            elif isinstance(msg, ToolMessage):
                full_prompt += f"\nTool Result ({msg.name}): {msg.content}\n"
        
        full_prompt += "\nAssistant: "
        
        # Get response from LLM
        response_text = await asyncio.to_thread(llm.invoke, full_prompt)
        
        # Parse for tool calls
        tool_calls = []
        if "TOOL_CALL:" in response_text:
            # Extract tool calls
            lines = response_text.split('\n')
            tool_name = None
            args_str = None
            
            for i, line in enumerate(lines):
                if line.startswith("TOOL_CALL:"):
                    tool_name = line.replace("TOOL_CALL:", "").strip()
                if line.startswith("ARGUMENTS:"):
                    args_str = line.replace("ARGUMENTS:", "").strip()
                    if tool_name and args_str:
                        try:
                            args = json.loads(args_str) if args_str != "{}" else {}
                            tool_calls.append({
                                "name": tool_name,
                                "args": args,
                                "id": f"call_{len(tool_calls)}"
                            })
                        except:
                            pass
                        tool_name = None
                        args_str = None
            
            # Create AI message with tool calls
            ai_msg = AIMessage(content=response_text, tool_calls=tool_calls)
        else:
            # Regular response without tool calls
            ai_msg = AIMessage(content=response_text)
        
        return {"messages": [ai_msg]}
    
    # Create the tool node that handles parallel tool execution
    async def tool_node_executor(state: MessagesState):
        """Execute tools and return results"""
        messages = state["messages"]
        last_message = messages[-1]
        
        tool_responses = []
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                # Find and execute the tool
                tool_to_execute = None
                for t in langchain_tools:
                    if t.name == tool_name:
                        tool_to_execute = t
                        break
                
                if tool_to_execute:
                    try:
                        result = await tool_to_execute.ainvoke(tool_args)
                        tool_responses.append(
                            ToolMessage(
                                content=str(result),
                                tool_call_id=tool_id,
                                name=tool_name
                            )
                        )
                    except Exception as e:
                        tool_responses.append(
                            ToolMessage(
                                content=f"Error: {str(e)}",
                                tool_call_id=tool_id,
                                name=tool_name
                            )
                        )
        
        return {"messages": tool_responses}
    
    # Define routing function
    def should_continue(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        
        # If there are tool calls, continue to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        # Otherwise, end
        return END
    
    # Build the graph
    workflow = StateGraph(MessagesState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node_executor)
    
    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    app = workflow.compile()
    
    return app

async def run_langgraph_agent(user_query: str, model_name: str = "google/flan-t5-small"):
    """Run a LangGraph agent using MCP tools with HuggingFace model"""
    
    global mcp_session, mcp_tools_list
    
    print(f"Loading model: {model_name}")
    print("(First run will download the model - this may take a while)\n")
    
    # Initialize HuggingFace pipeline
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Create pipeline
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.7,
            do_sample=True
        )
        
        # Create LangChain LLM
        llm = HuggingFacePipeline(pipeline=pipe)
        print("✓ Model loaded successfully\n")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nTrying a smaller model...")
        # Fallback to even smaller model
        pipe = pipeline("text2text-generation", model="google/flan-t5-small")
        llm = HuggingFacePipeline(pipeline=pipe)
    
    # Define server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],
        env=None
    )
    
    # Connect to MCP server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            mcp_session = session
            
            # Get available tools from MCP server
            mcp_tools_list = await session.list_tools()
            print(f"Connected to MCP server with {len(mcp_tools_list.tools)} tools available:")
            
            # Create LangChain tools from MCP tools
            langchain_tools = []
            for mcp_tool in mcp_tools_list.tools:
                print(f"  - {mcp_tool.name}: {mcp_tool.description}")
                lc_tool = create_langchain_tool_from_mcp(mcp_tool)
                langchain_tools.append(lc_tool)
            
            print()
            
            # Create the agent graph with dynamically loaded tools
            app = create_agent_graph(llm, langchain_tools)
            
            # Prepare input
            print(f"User: {user_query}\n")
            
            # Run the agent
            inputs = {"messages": [HumanMessage(content=user_query)]}
            
            # Stream the agent's execution
            print("Agent thinking...\n")
            
            async for event in app.astream(inputs, stream_mode="values"):
                messages = event["messages"]
                last_message = messages[-1]
                
                # Print tool calls
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    for tool_call in last_message.tool_calls:
                        print(f"🔧 Calling tool: {tool_call['name']}")
                        print(f"   Arguments: {tool_call['args']}\n")
                
                # Print tool results
                if isinstance(last_message, ToolMessage):
                    print(f"📊 Tool result: {last_message.content}\n")
                
                # Print final AI response
                if isinstance(last_message, AIMessage) and last_message.content:
                    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
                        # Clean up response
                        response = last_message.content
                        if "TOOL_CALL:" not in response:
                            print(f"🤖 Agent: {response}")

def main():
    """Main entry point"""
    
    print("=" * 60)
    print("MCP LangGraph Agent with HuggingFace Transformers")
    print("=" * 60)
    print()
    
    # Choose model - these are good lightweight options:
    # "google/flan-t5-small" - 80M params (fastest, less capable)
    # "google/flan-t5-base" - 250M params (balanced)
    # "google/flan-t5-large" - 780M params (better quality, slower)
    
    model = "google/flan-t5-base"
    
    # Default query if none provided
    default_query = "What is 25 * 4 + 10?"
    
    user_query = sys.argv[1] if len(sys.argv) > 1 else default_query
    
    # Run the agent
    asyncio.run(run_langgraph_agent(user_query, model))

if __name__ == "__main__":
    main()