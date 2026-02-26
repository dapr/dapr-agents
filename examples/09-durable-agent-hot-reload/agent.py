import asyncio
import logging
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.agents.configs import (
    ConfigKey,
    RuntimeSubscriptionConfig,
    AgentStateConfig,
    AgentPubSubConfig,
)
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow.runners import AgentRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def on_config_change(key: str, value):
    """Optional callback invoked after each successful config update."""
    logger.info(f"[callback] Configuration changed: {key} = {value}")


async def main():
    """
    This example demonstrates a Durable Agent that hot-reloads its configuration
    (role, goal, instructions, style_guidelines, max_iterations) from a Dapr
    Configuration Store.

    On startup the agent loads any pre-existing values from the store before
    subscribing to live changes. This works with any Dapr configuration backend
    (Redis, PostgreSQL, etc.).
    """

    # 1. Define the configuration subscription
    config = RuntimeSubscriptionConfig(
        store_name="runtime-config",
        keys=[
            ConfigKey.AGENT_ROLE,
            ConfigKey.AGENT_GOAL,
            ConfigKey.AGENT_INSTRUCTIONS,
            ConfigKey.STYLE_GUIDELINES,
            ConfigKey.MAX_ITERATIONS,
        ],
        on_config_change=on_config_change,
    )

    # 2. Infrastructure Setup
    state_config = AgentStateConfig(
        store=StateStoreService(store_name="workflowstatestore")
    )

    pubsub_config = AgentPubSubConfig(pubsub_name="agent-pubsub")

    # 3. Initialize the Agent
    agent = DurableAgent(
        name="config-aware-agent",
        role="Base Assistant",
        goal="Wait for configuration updates",
        instructions=["Initial instruction"],
        style_guidelines=["Be concise"],
        configuration=config,
        state=state_config,
        pubsub=pubsub_config,
    )

    logger.info("=== Agent Initialized ===")
    logger.info(f"Role: {agent.profile.role}")
    logger.info(f"Goal: {agent.profile.goal}")

    # 4. Start the agent via the runner. subscribe() calls agent.start()
    #    internally, which sets up the configuration subscription and starts
    #    the workflow runtime.
    runner = AgentRunner()
    try:
        runner.subscribe(agent)

        logger.info("Agent runtime started and subscribed to configuration store.")
        logger.info("To trigger a hot-reload, update the value in your config store:")
        logger.info('  Redis:      redis-cli SET agent_role "Expert Researcher"')
        logger.info(
            "  PostgreSQL: UPDATE configuration SET value='Expert Researcher', "
            "version=(version::int+1)::text WHERE key='agent_role';"
        )

        while True:
            await asyncio.sleep(10)
            logger.info(
                f"Current Persona: [{agent.profile.role}] - {agent.profile.goal} "
                f"(max_iterations={agent.execution.max_iterations})"
            )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        runner.shutdown(agent)


if __name__ == "__main__":
    asyncio.run(main())
