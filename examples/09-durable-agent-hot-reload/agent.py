import asyncio
import logging
import os
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.agents.configs import (
    AgentConfigurationConfig,
    AgentStateConfig,
    AgentPubSubConfig,
)
from dapr_agents.storage.daprstores.stateservice import StateStoreService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """
    This example demonstrates a Durable Agent that hot-reloads its configuration
    (role, goal, and instructions) from a Dapr Configuration Store.
    """

    # 1. Define the configuration subscription
    # We'll watch for keys related to our agent's persona
    config = AgentConfigurationConfig(
        store_name="configstore",
        keys=["agent_role", "agent_goal", "agent_instructions"],
    )

    # 2. Infrastructure Setup
    # Note: These component names should match your dapr run environment
    state_config = AgentStateConfig(
        store=StateStoreService(store_name="agent-workflow")
    )

    pubsub_config = AgentPubSubConfig(pubsub_name="agent-pubsub")

    # 3. Initialize the Agent
    # We start with some baseline defaults
    agent = DurableAgent(
        name="config-aware-agent",
        role="Base Assistant",
        goal="Wait for configuration updates",
        instructions=["Initial instruction"],
        configuration=config,
        state=state_config,
        pubsub=pubsub_config,
    )

    logger.info("=== Agent Initialized ===")
    logger.info(f"Role: {agent.profile.role}")
    logger.info(f"Goal: {agent.profile.goal}")

    # 4. Start the Agent Runtime
    # This will trigger the configuration subscription defined in AgentBase
    agent.start()

    logger.info("Agent started and subscribed to 'configstore'.")
    logger.info("To trigger a hot-reload, update the value in Redis:")
    logger.info('redis-cli SET agent_role "Expert Researcher"')

    try:
        # Keep the agent running to observe hot-reloads
        while True:
            await asyncio.sleep(10)
            logger.info(
                f"Current Persona: [{agent.profile.role}] - {agent.profile.goal}"
            )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
