# direct_call.py
"""
Direct MCP Tool Caller - No LLM required
Just calls MCP server functions directly
Install dependencies: 
  pip install mcp
"""

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
import sys

async def call_mcp_tool(tool_name: str, arguments: dict = None):
    """Call an MCP tool directly without using an LLM"""
    
    # Define server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_rule_based_agent_without_llm.py"],
        env=None
    )
    
    # Connect to MCP server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            
            # List available tools
            tools_list = await session.list_tools()
            print(f"Available tools on MCP server:")
            for tool in tools_list.tools:
                print(f"  - {tool.name}: {tool.description}")
            print()
            
            # Call the specified tool
            print(f"Calling tool: {tool_name}")
            if arguments:
                print(f"Arguments: {arguments}")
            print()
            
            try:
                # Execute tool via MCP
                tool_result = await session.call_tool(
                    tool_name,
                    arguments=arguments or {}
                )
                
                # Extract and print result
                print("Result:")
                for content in tool_result.content:
                    if hasattr(content, 'text'):
                        print(f"  {content.text}")
                
            except Exception as e:
                print(f"Error calling tool: {e}")

async def demo_all_tools():
    """Demo calling all available tools"""
    
    # Define server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_rule_based_agent_without_llm_but_chroma.py"],#["mcp_rule_based_agent_without_llm.py"],
        env=None
    )
    
    # Connect to MCP server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            
            print("=== MCP Direct Tool Caller Demo ===\n")
            
            # List available tools
            tools_list = await session.list_tools()
            print(f"Found {len(tools_list.tools)} tools:\n")
            
            # Demo each tool
            # 1. Calculate tool
            print("1. Testing 'calculate' tool:")
            print("   Expression: 25 * 4 + 10")
            result = await session.call_tool(
                "run_agent",
                arguments={"query": "calculate 25 * 4 + 10"}
            )
            for content in result.content:
                if hasattr(content, 'text'):
                    print(f"   {content.text}")
            print()
            
            # 2. Get time tool
            print("2. Testing 'get_time' tool:")
            result = await session.call_tool(
                "run_agent",
                arguments={"query": 'you are a speaking clock'}
            )
            for content in result.content:
                if hasattr(content, 'text'):
                    print(f"   {content.text}")
            print()
            
            # 3. Another calculation
            print("3. Testing 'calculate' with division:")
            print("   Expression: 100 / 5")
            result = await session.call_tool(
                "run_agent",
                arguments={"query": "100 / 5"}
            )

            # 4. testing RAG
            print("4. testing the chroma RAG")
            query = "Tell me info about vector database."
            print(query)
            result = await session.call_tool(
                "run_agent",
                arguments={"query": query}
            )
            for content in result.content:
                if hasattr(content, 'text'):
                    print(f"   {content.text}")
            print()

def main():
    """Main entry point"""
    if len(sys.argv) == 1:
        # No arguments - run demo
        print("Running demo of all tools...\n")
        asyncio.run(demo_all_tools())
    elif sys.argv[1] == "calculate":
        # Call calculate tool
        if len(sys.argv) < 3:
            print("Usage: python direct_call.py calculate '<expression>'")
            print("Example: python direct_call.py calculate '10 + 5 * 2'")
            return
        expression = sys.argv[2]
        asyncio.run(call_mcp_tool("calculate", {"expression": expression}))
    elif sys.argv[1] == "get_time":
        # Call get_time tool
        asyncio.run(call_mcp_tool("get_time"))


if __name__ == "__main__":
    main()