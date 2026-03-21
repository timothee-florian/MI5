# server.py
"""
MCP Server Implementation
Install dependencies: pip install mcp
"""

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
import asyncio
import requests

# Create an MCP server
server = Server("local-agent")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """Define available tools for the agent"""
    return [
        types.Tool(
            name="calculate",
            description="Perform basic arithmetic calculations",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        ),
        types.Tool(
            name="get_time",
            description="Get the current time",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
        ,
        types.Tool(
            name="get_lucky_number",
            description="Get my secret lucky number from the internet",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool calls"""
    if name == "calculate":
        try:
            result = eval(arguments["expression"])
            return [types.TextContent(type="text", text=f"Result: {result} :)")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "get_time":
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return [types.TextContent(type="text", text=f"Current time: {current_time}")]
    
    elif name == "get_lucky_number":
        import requests
        url = "http://127.0.0.1:5000/number"
        response = requests.get(url)
        response.raise_for_status() 
        data = response.json()
        return [types.TextContent(type="text", text=f"Lucky number: {data}")]

    raise ValueError(f"Unknown tool: {name}")

async def main():
    """Run the MCP server using stdio transport"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="local-agent",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())