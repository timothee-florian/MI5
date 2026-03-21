# server_rest.py
"""
MCP Server that fetches data from Flask REST API
Install dependencies: pip install mcp requests
"""

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
import asyncio
import requests
import json

# REST API base URL (make sure flask_server.py is running)
API_BASE_URL = "http://localhost:5000/api"

# Create an MCP server
server = Server("rest-api-agent")

def make_request(endpoint: str, params: dict = None) -> str:
    """Make HTTP request to Flask API"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to API server. Make sure flask_server.py is running on port 5000."
    except requests.exceptions.Timeout:
        return "Error: Request timed out"
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """Define available tools for the agent"""
    return [
        types.Tool(
            name="get_users",
            description="Get list of all users from the API",
            inputSchema={
                "type": "object",
                "properties": {},
            }
        ),
        types.Tool(
            name="get_user_by_id",
            description="Get a specific user by their ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "The ID of the user to retrieve"
                    }
                },
                "required": ["user_id"]
            }
        ),
        types.Tool(
            name="get_products",
            description="Get list of all products from the API",
            inputSchema={
                "type": "object",
                "properties": {},
            }
        ),
        types.Tool(
            name="get_product_by_id",
            description="Get a specific product by its ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "The ID of the product to retrieve"
                    }
                },
                "required": ["product_id"]
            }
        ),
        types.Tool(
            name="get_weather",
            description="Get current weather information for a city",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name (optional, random if not provided)"
                    }
                }
            }
        ),
        types.Tool(
            name="get_server_time",
            description="Get current server time and date",
            inputSchema={
                "type": "object",
                "properties": {},
            }
        ),
        types.Tool(
            name="get_random_number",
            description="Get a random number within a range",
            inputSchema={
                "type": "object",
                "properties": {
                    "min": {
                        "type": "integer",
                        "description": "Minimum value (default: 1)"
                    },
                    "max": {
                        "type": "integer",
                        "description": "Maximum value (default: 100)"
                    }
                }
            }
        ),
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool calls"""
    
    if arguments is None:
        arguments = {}
    
    if name == "get_users":
        result = await asyncio.to_thread(make_request, "/users")
        return [types.TextContent(type="text", text=result)]
    
    elif name == "get_user_by_id":
        user_id = arguments.get("user_id")
        result = await asyncio.to_thread(make_request, f"/users/{user_id}")
        return [types.TextContent(type="text", text=result)]
    
    elif name == "get_products":
        result = await asyncio.to_thread(make_request, "/products")
        return [types.TextContent(type="text", text=result)]
    
    elif name == "get_product_by_id":
        product_id = arguments.get("product_id")
        result = await asyncio.to_thread(make_request, f"/products/{product_id}")
        return [types.TextContent(type="text", text=result)]
    
    elif name == "get_weather":
        city = arguments.get("city")
        params = {"city": city} if city else {}
        result = await asyncio.to_thread(make_request, "/weather", params)
        return [types.TextContent(type="text", text=result)]
    
    elif name == "get_server_time":
        result = await asyncio.to_thread(make_request, "/time")
        return [types.TextContent(type="text", text=result)]
    
    elif name == "get_random_number":
        min_val = arguments.get("min", 1)
        max_val = arguments.get("max", 100)
        params = {"min": min_val, "max": max_val}
        result = await asyncio.to_thread(make_request, "/random", params)
        return [types.TextContent(type="text", text=result)]
    
    raise ValueError(f"Unknown tool: {name}")

async def main():
    """Run the MCP server using stdio transport"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="rest-api-agent",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())