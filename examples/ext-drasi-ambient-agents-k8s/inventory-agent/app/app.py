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

from __future__ import annotations

import asyncio
import logging
import os

from dapr.ext.workflow.aio import DaprMCPClient

from dapr_agents import AgentRunner

from dapr_agents.ext.drasi import enable_drasi, DrasiWorkflowTool

from agent import make_agent
from dapr_agents.llm.anthropic import client

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

AGENT_MCP_COMPONENT = os.getenv("AGENT_MCP_COMPONENT", "agent-mcp")
AGENT_PUBSUB_COMPONENT = os.getenv("AGENT_PUBSUB_COMPONENT", "agent-pubsub")
DRASI_TOPIC = "drasi-events"


async def _load_mcp_tools() -> list:
    client = DaprMCPClient()
    try:
        await client.connect(AGENT_MCP_COMPONENT)
        tool_defs = client.get_all_tools()
        logger.info(
            f"Loaded MCP tools from '{AGENT_MCP_COMPONENT}': "
            f"{[tool_def.name for tool_def in tool_defs]}"
        )
        # Convert to ``DrasiWorkflowTool`` instances
        return [DrasiWorkflowTool.from_mcp_tool_def(tool_def) for tool_def in tool_defs]
    except Exception as exc:
        raise RuntimeError(
            f"Could not load MCP tools from server '{AGENT_MCP_COMPONENT}'"
        ) from exc


def main() -> None:
    tools = asyncio.run(_load_mcp_tools())

    # Get a fresh event loop
    asyncio.set_event_loop(asyncio.new_event_loop())

    agent = make_agent(tools=tools)

    enable_drasi(
        agent,
        mcpserver=AGENT_MCP_COMPONENT,
        pubsub=AGENT_PUBSUB_COMPONENT,
        topic=DRASI_TOPIC,
    )

    runner = AgentRunner()
    try:
        runner.serve(agent)
    finally:
        runner.shutdown(agent)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
