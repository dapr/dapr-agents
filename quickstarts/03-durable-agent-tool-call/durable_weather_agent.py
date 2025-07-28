from dapr_agents import DurableAgent
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
        state_store_name="workflowstatestore",
        state_key="workflow_state",
        agents_registry_store_name="agentstatestore",
        agents_registry_key="agents_registry",
        tools=tools,
    )
    # 2️⃣ Start the agent service
    await weather_agent.run("What's the weather in Boston?")


if __name__ == "__main__":
    asyncio.run(main())
