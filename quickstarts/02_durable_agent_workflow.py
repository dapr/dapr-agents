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

from dapr_agents.llm import DaprChatClient

from dapr_agents import DurableAgent
from dapr_agents.agents.configs import (
    AgentMemoryConfig,
    AgentStateConfig,
)
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents import AgentRunner
from dapr_agents.workflow.utils.core import wait_for_shutdown
from function_tools import slow_weather_func


async def main() -> None:
    weather_agent = DurableAgent(
        name="WeatherAgent",
        role="Weather Assistant",
        instructions=["Help users with weather information"],
        tools=[slow_weather_func],
        # Configure this agent to use Dapr Conversation API.
        llm=DaprChatClient(component_name="llm-provider"),
        # Configure the agent to use Dapr State Store for conversation history.
        memory=AgentMemoryConfig(
            store=ConversationDaprStateMemory(
                store_name="agent-memory",
            )
        ),
        # This is where the execution state is stored.
        state=AgentStateConfig(
            store=StateStoreService(store_name="agent-workflow"),
        ),
        # No pub/sub config needed — this agent is triggered via the Dapr Workflow API
        # or by other Dapr workflows, not by pub/sub messages.
    )

    runner = AgentRunner()
    try:
        # workflow() starts the workflow runtime without wiring pub/sub or HTTP routes.
        # The agent is available to be triggered by external Dapr workflows or
        # direct Dapr Workflow API calls.
        runner.workflow(weather_agent)
        await wait_for_shutdown()
    finally:
        runner.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
