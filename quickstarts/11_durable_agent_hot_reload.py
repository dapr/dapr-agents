import asyncio
import logging
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.agents.configs import (
    ConfigKey,
    RuntimeSubscriptionConfig,
    AgentStateConfig,
)
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.runners import AgentRunner

logging.basicConfig(level=logging.INFO)


async def main():
    logger = logging.getLogger(__name__)
    # Configuration for hot-reloading
    # This assumes a Dapr configuration store named 'runtime-config' is configured.
    # See https://docs.dapr.io/reference/components-reference/supported-configuration-stores/
    # for supported backends.
    config = RuntimeSubscriptionConfig(
        store_name="runtime-config",
        keys=[ConfigKey.AGENT_ROLE, ConfigKey.AGENT_GOAL, ConfigKey.AGENT_INSTRUCTIONS],
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

    # Start the agent via the runner (internally calls agent.start(), which sets
    # up the configuration subscription and starts the workflow runtime).
    runner = AgentRunner()
    try:
        runner.subscribe(agent)
        logger.info("Agent started. You can now update the configuration in Dapr.")
        logger.info("To hot-reload a single field:")
        logger.info('redis-cli SET agent_role "New Hot-Reloaded Role"')
        logger.info('redis-cli SET agent_goal "New Hot-Reloaded Goal"')
        logger.info(
            "redis-cli SET agent_instructions "
            '"[\\"New Hot-Reloaded Instruction 1\\", \\"New Hot-Reloaded Instruction 2\\"]"'
        )

        # Keep the process alive to receive updates
        while True:
            await asyncio.sleep(5)
            logger.info(f"Current role: {agent.profile.role}")
            logger.info(f"Current goal: {agent.profile.goal}")
            logger.info(f"Current instructions: {agent.profile.instructions}")
    except KeyboardInterrupt:
        pass
    finally:
        runner.shutdown(agent)


if __name__ == "__main__":
    asyncio.run(main())
