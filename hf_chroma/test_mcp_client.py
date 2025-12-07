from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
import sys

async def call_mcp_tool(tool_name: str, arguments: dict = None):
    """Call an MCP tool directly"""
    
    # Define server parameters - UPDATE THIS PATH TO YOUR MCP SERVER FILE
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],  # Your MCP server file
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
                print("="*60)
                for content in tool_result.content:
                    if hasattr(content, 'text'):
                        print(content.text)
                print("="*60)
                
            except Exception as e:
                print(f"Error calling tool: {e}")
                import traceback
                traceback.print_exc()

async def demo_all_tools():
    """Demo calling all available tools"""
    
    # Define server parameters - UPDATE THIS PATH TO YOUR MCP SERVER FILE
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],  # Your MCP server file
        env=None
    )
    
    # Connect to MCP server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            print("Initializing MCP connection...")
            await session.initialize()
            
            print("\n" + "="*60)
            print("=== MCP AGENT TEST CLIENT ===")
            print("="*60 + "\n")
            
            # List available tools
            tools_list = await session.list_tools()
            print(f"Found {len(tools_list.tools)} tools available:\n")
            for i, tool in enumerate(tools_list.tools, 1):
                print(f"{i}. {tool.name}")
                print(f"   Description: {tool.description}\n")
            
            print("="*60)
            print("RUNNING TESTS")
            print("="*60 + "\n")
            
            # Test 1: Direct calculation
            print("TEST 1: Direct calculation tool")
            print("-" * 40)
            print("Expression: 25 * 4 + 10")
            result = await session.call_tool(
                "calculate",
                arguments={"expression": "25 * 4 + 10"}
            )
            for content in result.content:
                if hasattr(content, 'text'):
                    print(f"Result: {content.text}")
            print()
            
            # Test 2: Direct time query
            print("TEST 2: Direct time query")
            print("-" * 40)
            result = await session.call_tool(
                "get_time",
                arguments={}
            )
            for content in result.content:
                if hasattr(content, 'text'):
                    print(f"Result: {content.text}")
            print()
            
            # Test 3: Direct RAG retrieval
            print("TEST 3: Direct RAG retrieval")
            print("-" * 40)
            query = "vector database"
            print(f"Query: {query}")
            result = await session.call_tool(
                "retrieve_info",
                arguments={"query": query}
            )
            for content in result.content:
                if hasattr(content, 'text'):
                    print(f"Result: {content.text}")
            print()
            
            # Test 4: Agent with calculation (simple mode)
            print("TEST 4: Agent with calculation (simple rule-based)")
            print("-" * 40)
            query = "calculate 156 * 23"
            print(f"Query: {query}")
            result = await session.call_tool(
                "run_agent",
                arguments={"query": query, "use_langchain": False}
            )
            for content in result.content:
                if hasattr(content, 'text'):
                    print(f"Result: {content.text}")
            print()
            
            # Test 5: Agent with time query (simple mode)
            print("TEST 5: Agent with time query (simple rule-based)")
            print("-" * 40)
            query = "what time is it?"
            print(f"Query: {query}")
            result = await session.call_tool(
                "run_agent",
                arguments={"query": query, "use_langchain": False}
            )
            for content in result.content:
                if hasattr(content, 'text'):
                    print(f"Result: {content.text}")
            print()
            
            # Test 6: Agent with RAG query (simple mode)
            print("TEST 6: Agent with RAG query (simple rule-based)")
            print("-" * 40)
            query = "Tell me info about Napoleon"
            print(f"Query: {query}")
            result = await session.call_tool(
                "run_agent",
                arguments={"query": query, "use_langchain": False}
            )
            for content in result.content:
                if hasattr(content, 'text'):
                    print(f"Result: {content.text}")
            print()
            
            # Test 7: Agent with LangChain (complex query)
            print("TEST 7: Agent with LangChain (intelligent routing)")
            print("-" * 40)
            query = "Tell me about the French emperor"
            print(f"Query: {query}")
            result = await session.call_tool(
                "run_agent",
                arguments={"query": query, "use_langchain": True}
            )
            for content in result.content:
                if hasattr(content, 'text'):
                    print(f"Result: {content.text}")
            print()
            
            # # Test 8: Complex multi-step query
            # print("TEST 8: Complex multi-step query with LangChain")
            # print("-" * 40)
            # query = "Calculate 100 * 25 and tell me what time it is"
            # print(f"Query: {query}")
            # result = await session.call_tool(
            #     "run_agent",
            #     arguments={"query": query, "use_langchain": True}
            # )
            # for content in result.content:
            #     if hasattr(content, 'text'):
            #         print(f"Result: {content.text}")
            # print()
            
            print("="*60)
            print("ALL TESTS COMPLETED")
            print("="*60)

async def interactive_mode():
    """Interactive mode for testing queries"""
    
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print("\n" + "="*60)
            print("=== INTERACTIVE MCP AGENT TEST ===")
            print("="*60)
            print("\nAvailable commands:")
            print("  - Type your query to use the agent")
            print("  - 'langchain on/off' - toggle LangChain agent")
            print("  - 'list' - list available tools")
            print("  - 'quit' - exit")
            print()
            
            use_langchain = True
            
            while True:
                try:
                    query = input(f"\n[{'LangChain' if use_langchain else 'Simple'}] Query: ").strip()
                    
                    if not query:
                        continue
                    
                    if query.lower() == 'quit':
                        print("Goodbye!")
                        break
                    
                    if query.lower() == 'langchain on':
                        use_langchain = True
                        print("✓ LangChain agent enabled")
                        continue
                    
                    if query.lower() == 'langchain off':
                        use_langchain = False
                        print("✓ Simple rule-based agent enabled")
                        continue
                    
                    if query.lower() == 'list':
                        tools_list = await session.list_tools()
                        print("\nAvailable tools:")
                        for tool in tools_list.tools:
                            print(f"  - {tool.name}: {tool.description}")
                        continue
                    
                    # Call the agent
                    result = await session.call_tool(
                        "run_agent",
                        arguments={"query": query, "use_langchain": use_langchain}
                    )
                    
                    print("\nResult:")
                    print("-" * 60)
                    for content in result.content:
                        if hasattr(content, 'text'):
                            print(content.text)
                    print("-" * 60)
                    
                except KeyboardInterrupt:
                    print("\n\nGoodbye!")
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()

def main():
    """Main entry point"""
    if len(sys.argv) == 1:
        # No arguments - run demo
        print("Running comprehensive test suite...\n")
        asyncio.run(demo_all_tools())
    elif sys.argv[1] == "interactive":
        # Interactive mode
        asyncio.run(interactive_mode())
    elif sys.argv[1] == "calculate":
        # Direct calculation
        if len(sys.argv) < 3:
            print("Usage: python test_mcp_client.py calculate '<expression>'")
            print("Example: python test_mcp_client.py calculate '10 + 5 * 2'")
            return
        expression = sys.argv[2]
        asyncio.run(call_mcp_tool("calculate", {"expression": expression}))
    elif sys.argv[1] == "time":
        # Get time
        asyncio.run(call_mcp_tool("get_time"))
    elif sys.argv[1] == "retrieve":
        # Retrieve info
        if len(sys.argv) < 3:
            print("Usage: python test_mcp_client.py retrieve '<query>'")
            print("Example: python test_mcp_client.py retrieve 'Napoleon'")
            return
        query = sys.argv[2]
        asyncio.run(call_mcp_tool("retrieve_info", {"query": query}))
    elif sys.argv[1] == "agent":
        # Use agent
        if len(sys.argv) < 3:
            print("Usage: python test_mcp_client.py agent '<query>'")
            print("Example: python test_mcp_client.py agent 'Tell me about Napoleon'")
            return
        query = sys.argv[2]
        use_langchain = "--simple" not in sys.argv
        asyncio.run(call_mcp_tool("run_agent", {"query": query, "use_langchain": use_langchain}))
    else:
        print("Usage:")
        print("  python test_mcp_client.py                    # Run full test suite")
        print("  python test_mcp_client.py interactive        # Interactive mode")
        print("  python test_mcp_client.py calculate '<expr>' # Direct calculation")
        print("  python test_mcp_client.py time               # Get current time")
        print("  python test_mcp_client.py retrieve '<query>' # Retrieve info")
        print("  python test_mcp_client.py agent '<query>'    # Use agent (add --simple for rule-based)")

if __name__ == "__main__":
    main()