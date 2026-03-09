#!/usr/bin/env python3

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

from dotenv import load_dotenv

from dapr_agents import DurableAgent
from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentMemoryConfig,
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
)
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.tool.mcp import MCPClient
from dapr_agents.workflow.runners import AgentRunner


async def _load_mcp_tools() -> list:
    client = MCPClient()
    await client.connect_sse("local", url="http://localhost:8000/sse")
    return client.get_all_tools()


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    try:
        tools = asyncio.run(_load_mcp_tools())
    except Exception:
        logging.exception("Failed to load MCP tools via SSE")
        return

    # asyncio.run closes its loop; create a fresh default loop for sync workflow wiring.
    asyncio.set_event_loop(asyncio.new_event_loop())

    pubsub = AgentPubSubConfig(
        pubsub_name="messagepubsub",
        agent_topic="weather.requests",
        broadcast_topic="agents.broadcast",
    )
    state = AgentStateConfig(
        store=StateStoreService(store_name="agentstatestore"),
    )
    registry = AgentRegistryConfig(
        store=StateStoreService(store_name="agentregistrystore"),
        team_name="weather-team",
    )
    execution = AgentExecutionConfig(max_iterations=4)
    memory = AgentMemoryConfig(
        store=ConversationDaprStateMemory(
            store_name="conversationstore",
        )
    )

    agent = DurableAgent(
        name="Stevie",
        role="Weather Assistant",
        goal="Help humans get weather, travel, and location details using smart tools.",
        instructions=[
            "Answer clearly and helpfully.",
            "Call MCP tools when extra data improves accuracy.",
        ],
        tools=tools,
        pubsub=pubsub,
        registry=registry,
        execution=execution,
        memory=memory,
        state=state,
    )

    runner = AgentRunner()
    try:
        runner.serve(agent, port=8001)
    finally:
        runner.shutdown(agent)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
