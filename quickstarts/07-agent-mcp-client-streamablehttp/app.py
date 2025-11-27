#!/usr/bin/env python3
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
    try:
        await client.connect_streamable_http(
            server_name="local",
            url="http://localhost:8000/mcp/",
        )
        return client.get_all_tools()
    finally:
        try:
            await client.close()
        except RuntimeError as exc:
            if "Attempted to exit cancel scope" not in str(exc):
                raise


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    try:
        tools = asyncio.run(_load_mcp_tools())
    except Exception:
        logging.exception("Failed to load MCP tools via streamable HTTP")
        return

    # asyncio.run closes its loop; create a fresh default for sync wiring.
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
            session_id="weather-session",
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
