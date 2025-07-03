from dapr_agents import DurableAgent
from dotenv import load_dotenv
import asyncio
import logging


async def main():
    try:
        # Create the Weather Agent using those tools
        weather_agent = DurableAgent(
            role="Weather Assistant",
            name="Stevie",
            goal="Help humans get weather and location info using smart tools.",
            instructions=[
                "Respond clearly and helpfully to weather-related questions.",
                "Use tools when appropriate to fetch or simulate weather data.",
                "You may sometimes jump after answering the weather question.",
            ],
            message_bus_name="messagepubsub",
            state_store_name="workflowstatestore",
            state_key="workflow_state",
            agents_registry_store_name="agentstatestore",
            agents_registry_key="agents_registry",
        ).as_service(port=8001)

        # Start the FastAPI agent service
        await weather_agent.start()
    except Exception as e:
        print(f"Error starting service: {e}")


if __name__ == "__main__":
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())
