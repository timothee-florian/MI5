from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# 1. Define tools
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 72°F."

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

# 2. Initialize the model
model = ChatAnthropic(model="claude-sonnet-4-20250514", api_key="YOUR_API_KEY")

# 3. Create the agent
agent = create_react_agent(
    model=model,
    tools=[get_weather, calculate],
    prompt="You are a helpful assistant."
)

# 4. Run the agent
response = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Paris?"}]
})

print(response["messages"][-1].content)