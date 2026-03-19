#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import asyncio
import logging
from dapr_agents import DurableAgent
from dapr_agents.agents.configs import (
    RuntimeConfigKey,
    RuntimeSubscriptionConfig,
    AgentStateConfig,
)
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents import AgentRunner

logging.basicConfig(level=logging.INFO)


async def main():
    logger = logging.getLogger(__name__)
    # Configuration for hot-reloading
    # This assumes a Dapr configuration store named 'runtime-config' is configured.
    # See https://docs.dapr.io/reference/components-reference/supported-configuration-stores/
    # for supported backends.
    config = RuntimeSubscriptionConfig(
        store_name="runtime-config",
        keys=[
            RuntimeConfigKey.AGENT_ROLE,
            RuntimeConfigKey.AGENT_GOAL,
            RuntimeConfigKey.AGENT_INSTRUCTIONS,
        ],
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
