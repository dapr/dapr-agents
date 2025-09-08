from dapr_agents import DurableAgent
from dapr_agents.llm.dapr import DaprChatClient
from dotenv import load_dotenv
from weather_tools import tools
import asyncio
import logging
import os


async def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    # Ensure default Dapr LLM component is set (e.g., "openai")
    os.environ.setdefault("DAPR_LLM_COMPONENT_DEFAULT", "openai")

    agent = DurableAgent(
        role="Weather Assistant",
        name="Stevie",
        goal="Help humans get weather and location info using smart tools.",
        instructions=[
            "Respond clearly and helpfully to weather-related questions.",
            "Use tools when appropriate to fetch weather data.",
        ],
        message_bus_name="pubsub",#"messagepubsub",
        state_store_name="statestore",#"workflowstatestore",
        state_key="workflow_state",
        agents_registry_store_name="statestore",#"workflowstatestore",
        agents_registry_key="agents_registry",
        tools=tools,
        llm=DaprChatClient(),
    )

    await agent.run("What's the weather in Boston?")


if __name__ == "__main__":
    asyncio.run(main())


