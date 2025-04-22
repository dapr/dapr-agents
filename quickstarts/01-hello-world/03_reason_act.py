import asyncio
from dapr_agents import tool, ReActAgent
from dotenv import load_dotenv

load_dotenv()

@tool
def search_weather(city: str) -> str:
    """Get weather information for a city."""
    weather_data = {"london": "rainy", "paris": "sunny"}
    return weather_data.get(city.lower(), "Unknown")

@tool
def get_activities(weather: str) -> str:
    """Get activity recommendations."""
    activities = {"rainy": "Visit museums", "sunny": "Go hiking"}
    return activities.get(weather.lower(), "Stay comfortable")

async def main():
    react_agent = ReActAgent(
        name="TravelAgent",
        role="Travel Assistant",
        instructions=["Check weather, then suggest activities"],
        tools=[search_weather, get_activities]
    )

    result = await react_agent.run("What should I do in London today?")
    if result:
        print("Result:", result)

if __name__ == "__main__":
    asyncio.run(main())
