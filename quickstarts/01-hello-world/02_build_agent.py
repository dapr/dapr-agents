from dapr_agents import tool, Agent
from dotenv import load_dotenv

load_dotenv()
@tool
def my_weather_func() -> str:
    """Get current weather."""
    return "It's 72°F and sunny"

weather_agent = Agent(
    name="WeatherAgent",
    role="Weather Assistant",
    instructions=["Help users with weather information"],
    tools=[my_weather_func]
)

response = weather_agent.run("What's the weather?")
print(response)