import asyncio
from dapr_agents import tool, Agent
from dapr_agents import OpenAIChatClient
from dotenv import load_dotenv

load_dotenv()


@tool
def my_weather_func() -> str:
    """Get current weather."""
    return "It's 72°F and sunny"


async def main():
    weather_agent = Agent(
        name="WeatherAgent",
        role="Weather Assistant",
        instructions=["Help users with weather information"],
        tools=[my_weather_func],
        llm=OpenAIChatClient(model="gpt-3.5-turbo"),
    )
    try:
        response = await weather_agent.run("What's the weather?")
        print(response)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
