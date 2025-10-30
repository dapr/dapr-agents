from dapr_agents import DurableAgent, MemoryStore
from dotenv import load_dotenv
from weather_tools import tools
import asyncio
import logging


async def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    # 1️⃣ Instantiate your agent
    weather_agent = DurableAgent(
        role="Weather Assistant",
        name="Stevie",
        goal="Help humans get weather and location info using smart tools.",
        instructions=[
            "Respond clearly and helpfully to weather-related questions.",
            "Use tools when appropriate to fetch weather data.",
        ],
        message_bus_name="messagepubsub",
        memory_store=MemoryStore(
            name="statestore",
            # Optional
            local_directory="./local-state",
            session_id="session",
        ),
        tools=tools,
    )
    # 2️⃣ Start the agent service
    await weather_agent.run("What's the weather in Boston?")


if __name__ == "__main__":
    asyncio.run(main())
