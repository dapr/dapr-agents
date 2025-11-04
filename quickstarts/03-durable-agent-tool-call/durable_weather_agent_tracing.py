import asyncio
import logging

from agent_tools import tools
from dotenv import load_dotenv

from dapr_agents import DurableAgent
from dapr_agents.workflow.runners import AgentRunner

logging.basicConfig(level=logging.INFO)

load_dotenv()


async def main():
    from phoenix.otel import register

    logger = logging.getLogger(__name__)

    # Register OpenTelemetry tracer provider
    tracer_provider = register(
        project_name="dapr-agentic-workflows",
        protocol="http/protobuf",
    )
    # Initialize Dapr Agents observability instrumentor
    from dapr_agents.observability import DaprAgentsInstrumentor

    instrumentor = DaprAgentsInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

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
    # Start the agent (registers workflows with the runtime)
    weather_agent.start()

    # Create an AgentRunner to execute the workflow
    runner = AgentRunner()

    try:
        prompt = "What's the weather in Boston?"

        # Run the workflow and wait for completion
        result = await runner.run(
            weather_agent,
            payload={"task": prompt},
        )

        print(f"\n🎯 Final result: {result}")
        print("📊 Check Phoenix UI at http://localhost:6006 for traces")

        return result

    except Exception as e:
        logger.error(f"Error running workflow: {e}", exc_info=True)
        raise
    finally:
        # Stop agent first (tears down durabletask runtime)
        weather_agent.stop()
        # Then shut down runner (unwire/close clients)
        runner.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
