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

import dapr.ext.workflow as wf

import json

wfr = wf.WorkflowRuntime()

@wfr.workflow(name='list_tools')
def list_tools(ctx: wf.DaprWorkflowContext):
    tools = yield ctx.call_child_workflow(
        workflow='ListTools',
        app_id='mcp-proxy',
    )

    return tools.get("tools", [])


async def _load_mcp_tools() -> list:
    wfr.start()
    wf_client = wf.DaprWorkflowClient()

    instance_id = wf_client.schedule_new_workflow(workflow=list_tools)
    tools = wf_client.wait_for_workflow_completion(instance_id)
    wfr.shutdown()

    return json.loads(tools.serialized_output)


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    try:
        tools = asyncio.run(_load_mcp_tools())
    except Exception:
        logging.exception("Failed to load MCP tools via Workflows")
        return

    logging.info(f"Loaded {len(tools)} tools from MCP")

    asyncio.set_event_loop(asyncio.new_event_loop())

    agent = DurableAgent(
        name="josh",
        role="Dapr assistant",
        goal="Help humans get interact with Dapr.",
        instructions=[
            "Answer clearly and helpfully.",
            "Call MCP tools when extra data improves accuracy.",
        ],
        tools=tools,
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


    #pubsub = AgentPubSubConfig(
    #    pubsub_name="messagepubsub",
    #    agent_topic="weather.requests",
    #    broadcast_topic="agents.broadcast",
    #)
    #state = AgentStateConfig(
    #    store=StateStoreService(store_name="agentstatestore"),
    #)
    #registry = AgentRegistryConfig(
    #    store=StateStoreService(store_name="agentregistrystore"),
    #    team_name="weather-team",
    #)
    #execution = AgentExecutionConfig(max_iterations=4)
    #memory = AgentMemoryConfig(
    #    store=ConversationDaprStateMemory(
    #        store_name="conversationstore",
    #        session_id="weather-session",
    #    )
    #)
