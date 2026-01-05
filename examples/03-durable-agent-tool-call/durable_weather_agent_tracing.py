import asyncio
import logging
import os
from agent_tools import tools
from dotenv import load_dotenv

from dapr_agents import DurableAgent
from dapr_agents.workflow.runners import AgentRunner
from dapr_agents.agents.configs import (
    AgentMemoryConfig,
    AgentRegistryConfig,
    AgentStateConfig,
)
from dapr_agents.llm.openai import OpenAIChatClient
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService
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

    registry = AgentRegistryConfig(
        store=StateStoreService(
            store_name=os.getenv("REGISTRY_STATE_STORE", "agentregistrystore")
        ),
        team_name="default",
    )
    legolas_state = AgentStateConfig(
        store=StateStoreService(store_name="workflowstatestore", key_prefix="stevie:")
    )
    legolas_memory = AgentMemoryConfig(
        store=ConversationDaprStateMemory(
            store_name="memorystore",
            session_id=f"default-session",
        )
    )
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
        state=legolas_state,
        registry=registry,
        memory=legolas_memory,

    )
    # Create an AgentRunner to execute the workflow
    runner = AgentRunner()

    try:
        prompt = (
            "What's the current weather in Boston, MA, then compute (14*7)+23, and finally "
            "search for the official tourism site for Boston?"
        )

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
        # Then shut down runner (unwire/close clients)
        runner.shutdown(weather_agent)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
