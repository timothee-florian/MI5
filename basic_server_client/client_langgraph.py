# client_langgraph.py
"""
MCP Client with LangGraph Agent for fully local operation
Install dependencies: 
  pip install mcp langchain-ollama langgraph langchain-core
  
Make sure Ollama is installed and running:
  https://ollama.ai
  
Pull a model:
  ollama pull llama3.2
"""

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
import asyncio
import sys
import json

# Global MCP session (will be initialized in async context)
mcp_session = None

# Define LangChain tools that wrap MCP tools
@tool
async def calculate(expression: str) -> str:
    """Perform basic arithmetic calculations
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")
    """
    try:
        result = await mcp_session.call_tool("calculate", arguments={"expression": expression})
        return result.content[0].text
    except Exception as e:
        return f"Error: {str(e)}"

@tool
async def get_time() -> str:
    """Get the current time"""
    try:
        result = await mcp_session.call_tool("get_time", arguments={})
        return result.content[0].text
    except Exception as e:
        return f"Error: {str(e)}"

def create_agent_graph(model_name: str = "llama3.2"):
    """Create a LangGraph agent with MCP tools"""
    
    # Initialize the LLM
    llm = ChatOllama(model=model_name, temperature=0)
    
    # Bind tools to the LLM
    tools = [calculate, get_time]
    llm_with_tools = llm.bind_tools(tools)
    
    # Define the agent function
    def call_model(state: MessagesState):
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # Create the tool node
    tool_node = ToolNode(tools)
    
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
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    app = workflow.compile()
    
    return app

async def run_langgraph_agent(user_query: str, model: str = "llama3.2"):
    """Run a LangGraph agent using MCP tools"""
    
    global mcp_session
    
    # Define server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
        env=None
    )
    
    # Connect to MCP server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            mcp_session = session
            
            # Get available tools from MCP server
            tools_list = await session.list_tools()
            print(f"Connected to MCP server with {len(tools_list.tools)} tools available\n")
            print(f"Using model: {model}\n")
            
            # Create the agent graph
            app = create_agent_graph(model)
            
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
                        print(f"Calling tool: {tool_call['name']}")
                        print(f"Arguments: {tool_call['args']}\n")
                
                # Print tool results
                if isinstance(last_message, ToolMessage):
                    print(f"Tool result: {last_message.content}\n")
                
                # Print final AI response
                if isinstance(last_message, AIMessage) and last_message.content:
                    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
                        print(f"Agent: {last_message.content}")

def main():
    """Main entry point"""
    # Check Ollama availability

    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="llama3.2")
    
    # Choose model
    model = "llama3.2"  # Change to llama3.1, mistral, phi3, etc.
    
    # Default query if none provided
    default_query = "What is 25 * 4 + 10? Also, what time is it?"
    
    user_query = sys.argv[1] if len(sys.argv) > 1 else default_query
    
    # Run the agent
    asyncio.run(run_langgraph_agent(user_query, model))

if __name__ == "__main__":
    main()