import asyncio
from dapr_agents import tool, DurableAgent, MemoryStore
from dapr_agents import OpenAIChatClient


@tool
def my_weather_func() -> str:
    """Get current weather."""
    return "It's 72°F and sunny"


async def main():
    weather_agent = DurableAgent(
        name="WeatherAgent",
        role="Weather Assistant",
        instructions=["Help users with weather information"],
        tools=[my_weather_func],
        message_bus_name="messagepubsub",
        memory_store=MemoryStore(
            name="statestore",
            # Optional
            local_directory="./local-state",
            session_id="session",
        ),
        llm=OpenAIChatClient(model="gpt-3.5-turbo"),
    )
    try:
        # Can override session_id per run
        response = await weather_agent.run(
            "What's the weather?",
            # session_id="new_session"  # Optional: override default session
        )
        print(response)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
