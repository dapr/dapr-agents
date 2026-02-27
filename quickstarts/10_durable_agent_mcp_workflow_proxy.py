#!/usr/bin/env python3
"""DurableAgent + MCP tools over Dapr Workflows (multi-app).

This quickstart demonstrates:
  1) Listing MCP tools by calling the proxy app's `ListTools` workflow
  2) Exposing those tools to a DurableAgent
  3) Executing tool calls from inside the agent workflow using the same
     orchestrator workflow context (child workflow calls to the proxy).

Prereqs:
  - Run the mcp-wf-proxy Go app from the user's example with Dapr app-id `mcp-proxy`
  - Ensure the proxy registers workflows named `ListTools` and `CallTool`
"""

import asyncio
import logging

from dotenv import load_dotenv

from dapr_agents import DurableAgent
from dapr_agents.tool.mcp import WorkflowMCPClient
from dapr_agents.workflow.runners import AgentRunner
from dapr_agents.llm import DaprChatClient

async def load_tools() -> list:
    client = WorkflowMCPClient(
        proxy_app_id="mcp-proxy",
        server_name="dapr",
        list_tools_workflow="ListTools",
        call_tool_workflow="CallTool",
    )
    tools = await client.load_tools()
    return tools


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    tools = asyncio.run(load_tools())
    logging.info("Loaded %d MCP tools via workflow proxy", len(tools))
    if tools:
        logging.info("Tools: %s", ", ".join(t.name for t in tools))

    agent = DurableAgent(
        name="workflow-mcp-agent",
        role="Dapr assistant",
        instructions=[
            "Answer clearly and helpfully.",
            "Use tools when they help you get accurate information.",
        ],
        tools=tools,
        goal="Help users by calling MCP tools through Dapr workflows.",
        llm=DaprChatClient(component_name="openai"),
    )

    runner = AgentRunner()
    try:
        runner.serve(agent, port=8001)
    finally:
        runner.shutdown(agent)


if __name__ == "__main__":
    main()
