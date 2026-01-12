import asyncio
import logging

from agent_tools import tools
from dotenv import load_dotenv

from dapr_agents import DurableAgent
from dapr_agents.workflow.runners import AgentRunner


async def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    # Instantiate your agent
    weather_agent = DurableAgent(
        role="Weather Assistant",
        name="Stevie",
        goal="Help humans get weather and location info using smart tools.",
        instructions=[
            "Respond clearly and helpfully to weather-related questions.",
            "Use tools when appropriate to fetch weather data.",
        ],
        tools=tools,
    )

    # Create an AgentRunner to execute the workflow
    runner = AgentRunner()

    try:
        prompt = "What's the weather in Boston?"

        # Run the workflow and wait for completion
        result = await runner.run(
            weather_agent,
            payload={"task": prompt},
        )

        print(f"\n✅ Final Result:\n{result}\n", flush=True)

    except Exception as e:
        logger.error(f"Error running workflow: {e}", exc_info=True)
        raise
    finally:
        # Then shut down runner (unwire/close clients)
        runner.shutdown(weather_agent)

        exit(0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
