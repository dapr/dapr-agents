import asyncio
import logging

from dotenv import load_dotenv
from weather_tools import tools

from dapr_agents import DurableAgent

logging.basicConfig(level=logging.INFO)

load_dotenv()


async def main():
    from phoenix.otel import register

    # Register OpenTelemetry tracer provider
    tracer_provider = register(
        project_name="dapr-agentic-workflows",
        protocol="http/protobuf",
    )
    # Initialize Dapr Agents observability instrumentor
    from dapr_agents.observability import DaprAgentsInstrumentor

    instrumentor = DaprAgentsInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider, skip_dep_check=True)

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
    result = await weather_agent.run("What's the weather in Boston?")

    print(f"\n🎯 Final result: {result}")
    print("📊 Check Phoenix UI at http://localhost:6006 for traces")

    return result


if __name__ == "__main__":
    asyncio.run(main())
