from dapr_agents import tool, Agent
from dotenv import load_dotenv

load_dotenv()
@tool
def weather() -> str:
    """Get current weather."""
    return "It's 72°F and sunny"

weather_agent = Agent(
    name="WeatherAgent",
    role="Weather Assistant",
    instructions=["Help users with weather information"],
    tools=[weather]
)

response = weather_agent.run("What's the weather?")
print(response)