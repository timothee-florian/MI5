# client_ollama.py
"""
MCP Client with Ollama for fully local agent
Install dependencies: 
  pip install mcp ollama
  
Make sure Ollama is installed and running:
  https://ollama.ai
  
Pull a model:
  ollama pull llama3.2
"""

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import ollama
import asyncio
import sys
import json

async def run_agent(user_query: str, model: str = "llama3.2"):
    """Run a fully local agent using MCP and Ollama"""
    
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
            
            # Get available tools from MCP server
            tools_list = await session.list_tools()
            print(f"Connected to MCP server with {len(tools_list.tools)} tools available\n")
            
            # Convert MCP tools to Ollama format
            ollama_tools = []
            for tool in tools_list.tools:
                ollama_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
            
            # Start conversation
            messages = [
                {"role": "user", "content": user_query}
            ]
            
            print(f"User: {user_query}\n")
            print(f"Using model: {model}\n")
            
            # Maximum iterations to prevent infinite loops
            max_iterations = 10
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                
                # Get response from Ollama
                response = ollama.chat(
                    model=model,
                    messages=messages,
                    tools=ollama_tools,
                )
                
                # Add assistant response to messages
                messages.append(response['message'])
                
                # Check if the model wants to use a tool
                if not response['message'].get('tool_calls'):
                    # No more tool calls, print final response
                    print(f"Claude (Ollama): {response['message']['content']}")
                    break
                
                # Process tool calls
                for tool_call in response['message']['tool_calls']:
                    tool_name = tool_call['function']['name']
                    tool_args = tool_call['function']['arguments']
                    
                    print(f"Calling tool: {tool_name}")
                    print(f"Arguments: {tool_args}\n")
                    
                    # Execute tool via MCP
                    try:
                        tool_result = await session.call_tool(
                            tool_name,
                            arguments=tool_args
                        )
                        
                        # Extract text from tool result
                        result_text = ""
                        for content in tool_result.content:
                            if hasattr(content, 'text'):
                                result_text += content.text
                        
                        print(f"Tool result: {result_text}\n")
                        
                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "content": result_text,
                        })
                        
                    except Exception as e:
                        error_msg = f"Error calling tool {tool_name}: {str(e)}"
                        print(error_msg)
                        messages.append({
                            "role": "tool",
                            "content": error_msg,
                        })
            
            if iteration >= max_iterations:
                print("Warning: Reached maximum iterations")

def main():

    default_query = "What is 25 * 4 + 10? Also, what time is it?"
    model = 'llama3.2'#'mistral'#
    # mistral doesn't work as well
    user_query = sys.argv[1] if len(sys.argv) > 1 else default_query
    
    asyncio.run(run_agent(user_query, model))

if __name__ == "__main__":
    main()