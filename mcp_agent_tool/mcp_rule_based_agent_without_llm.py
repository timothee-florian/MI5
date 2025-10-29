import asyncio
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server
from langchain_core.tools import Tool as LangChainTool
from typing import Optional
import re
from datetime import datetime

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

def string_length(query: str) -> str:
    """Get length of a string"""
    return f"Length: {len(query)}"

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
        name="StringLength",
        func=string_length,
        description="Returns the length of a string. Input should be the string."
    )
]

# Simple rule-based agent
class SimpleRuleBasedAgent:
    def __init__(self, tools):
        self.tools = {tool.name: tool for tool in tools}
    
    def run(self, query: str) -> str:
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['calculate', 'math', '+', '-', '*', '/']):
            for tool_name, tool in self.tools.items():
                if 'calculator' in tool_name.lower():
                    return tool.func(query)
        
        elif any(word in query_lower for word in ['time', 'date', 'clock']):
            for tool_name, tool in self.tools.items():
                if 'time' in tool_name.lower():
                    return tool.func(query)
        
        
        return "I don't understand that query. I can: calculate math, get time, or count string length."

# Initialize the agent
agent = SimpleRuleBasedAgent(langchain_tools)

# Create MCP server
app = Server("rule-based-agent")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="run_agent",
            description="Run the rule-based agent with a query. Can calculate math, get time, or count string length.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to process (e.g., 'calculate 5+3', 'what time is it')"
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