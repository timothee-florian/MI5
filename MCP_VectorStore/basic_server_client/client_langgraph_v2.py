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
import asyncio
import sys
import json

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

def create_agent_graph(model_name: str = "llama3.2", langchain_tools: list = None):
    """Create a LangGraph agent with MCP tools"""
    
    if langchain_tools is None:
        langchain_tools = []
    
    # Initialize the LLM with better parameters for tool use
    llm = ChatOllama(
        model=model_name, 
        temperature=0,
        num_ctx=4096  # Larger context for better reasoning
    )
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(langchain_tools)
    
    # Build tool descriptions for system prompt
    tool_descriptions = "\n".join([
        f"- '{t.name}': {t.description}" for t in langchain_tools
    ])
    
    # Define the agent function with system message
    async def call_model(state: MessagesState):
        messages = state["messages"]
        
        # Add system message to guide tool usage if first call
        if len(messages) == 1:
            system_prompt = {
                "role": "system",
                "content": f"""You are a helpful assistant with access to tools. 

Available tools:
{tool_descriptions}

IMPORTANT RULES:
- Only use tools when they are NECESSARY to answer the question
- Do NOT use tools for general knowledge questions that you already know
- Use 'calculate' tool ONLY for mathematical computations that need evaluation
- Use 'get_time' tool ONLY when asked about current time/date
- For questions about facts, geography, history, or general knowledge, answer directly without tools

Examples:
- "What is the capital of France?" -> Answer directly: "Paris" (NO TOOLS)
- "What is 2+2?" -> Answer directly: "4" (NO TOOLS for simple math)
- "Calculate 12345 * 67890" -> Use calculate tool (complex calculation)
- "What time is it?" -> Use get_time tool (real-time data needed)"""
            }
            messages = [system_prompt] + messages
        
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}
    
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

async def run_langgraph_agent(user_query: str, model: str = "llama3.2"):
    """Run a LangGraph agent using MCP tools"""
    
    global mcp_session, mcp_tools_list
    
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
            mcp_tools_list = await session.list_tools()
            print(f"Connected to MCP server with {len(mcp_tools_list.tools)} tools available:")
            
            # Create LangChain tools from MCP tools
            langchain_tools = []
            for mcp_tool in mcp_tools_list.tools:
                print(f"  - {mcp_tool.name}: {mcp_tool.description}")
                lc_tool = create_langchain_tool_from_mcp(mcp_tool)
                langchain_tools.append(lc_tool)
            
            print(f"\nUsing model: {model}\n")
            
            # Create the agent graph with dynamically loaded tools
            app = create_agent_graph(model, langchain_tools)
            
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
                        print(f"🤖 Agent: {last_message.content}🤖 ")

def main():
    """Main entry point"""
    # Check Ollama availability
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="llama3.2")
        # Quick test
        llm.invoke("test")
        print("✓ Ollama is running\n")
    except Exception as e:
        print(f"✗ Error connecting to Ollama: {e}\n")
        print("Make sure Ollama is installed and running:")
        print("1. Install from https://ollama.ai")
        print("2. Pull a model: ollama pull llama3.2")
        print("3. Verify: ollama list\n")
        return
    
    # Choose model
    model = "llama3.2"  # Change to llama3.1, mistral, phi3, etc.
    
    # Default query if none provided
    default_query = "What is 25 * 4 + 10? Also, what time is it?"
    
    user_query = sys.argv[1] if len(sys.argv) > 1 else default_query
    
    # Run the agent
    asyncio.run(run_langgraph_agent(user_query, model))

if __name__ == "__main__":
    main()