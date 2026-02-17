import asyncio
import logging
import os
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.agents.configs import AgentConfigurationConfig, AgentStateConfig
from dapr_agents.storage.daprstores.stateservice import StateStoreService

logging.basicConfig(level=logging.INFO)


async def main():
    logger = logging.getLogger(__name__)
    # Configuration for hot-reloading
    # This assumes a Dapr configuration store named 'configstore' is configured
    config = AgentConfigurationConfig(
        store_name="configstore",
        keys=["agent_role", "agent_goal", "agent_instructions"],
    )

    # Durable state configuration
    state_config = AgentStateConfig(
        store=StateStoreService(store_name="agent-workflow")
    )

    # Initialize the agent with configuration subscription
    agent = DurableAgent(
        name="hot-reload-agent",
        role="Original Role",
        goal="Original Goal",
        instructions=["Original Instruction 1"],
        configuration=config,
        state=state_config,
    )

    logger.info(f"Agent initialized with role: {agent.profile.role}")

    # Start the agent (this starts the workflow runtime)
    agent.start()

    logger.info("Agent started. You can now update the configuration in Dapr.")
    logger.info("To hot-reload a single field:")
    logger.info('redis-cli SET agent_role "New Hot-Reloaded Role"')
    logger.info('redis-cli SET agent_goal "New Hot-Reloaded Goal"')
    logger.info(
        "redis-cli SET agent_instructions "
        '"[\\"New Hot-Reloaded Instruction 1\\", \\"New Hot-Reloaded Instruction 2\\"]"'
    )

    # When running under integration tests, exit early after confirming startup.
    if os.environ.get("INTEGRATION_TEST"):
        logger.info("INTEGRATION_TEST mode: exiting after startup confirmation.")
        agent.stop()
        return

    try:
        # Keep the process alive to receive updates
        while True:
            await asyncio.sleep(5)
            logger.info(f"Current role: {agent.profile.role}")
            logger.info(f"Current goal: {agent.profile.goal}")
            logger.info(f"Current instructions: {agent.profile.instructions}")
    except KeyboardInterrupt:
        pass
    finally:
        agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
