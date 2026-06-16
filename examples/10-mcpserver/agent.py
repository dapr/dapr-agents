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

import dapr.ext.workflow as wf

from dapr_agents import AgentRunner, DurableAgent

from middleware_workflows import (
    audit_log_workflow,
    audit_log_write,
    input_validation_check,
    input_validation_workflow,
    rate_limit_check,
    rate_limit_workflow,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcpserver-weather")

# MCPServer tools are auto-discovered from the sidecar metadata.
# To manually control which servers and tools are loaded:
#
#   from dapr.ext.workflow import DaprMCPClient
#   from dapr_agents.tool.mcp import mcp_tool_def_to_workflow_tool
#   client = DaprMCPClient(timeout_in_seconds=30)
#   client.connect("weather")
#   tools = [mcp_tool_def_to_workflow_tool(t) for t in client.get_all_tools()]
#   agent = DurableAgent(name="WeatherAgent", tools=tools, runtime=workflow_runtime, ...)


async def main() -> None:
    # The sidecar only accepts one gRPC worker stream, so middleware
    # workflows and the DurableAgent must share a WorkflowRuntime.
    workflow_runtime = wf.WorkflowRuntime()

    workflow_runtime.register_workflow(rate_limit_workflow)
    workflow_runtime.register_activity(rate_limit_check)
    workflow_runtime.register_workflow(input_validation_workflow)
    workflow_runtime.register_activity(input_validation_check)
    workflow_runtime.register_workflow(audit_log_workflow)
    workflow_runtime.register_activity(audit_log_write)

    logger.info("Middleware workflows registered on shared WorkflowRuntime.")

    # The `weather` MCPServer is auto-discovered from the sidecar
    # metadata API; no manual client wiring required.
    agent = DurableAgent(
        name="WeatherAgent",
        role="Weather assistant",
        goal="Answer weather questions using the Weather MCP server.",
        instructions=[
            "Use get_weather for current conditions.",
            "Use get_forecast for multi-day forecasts.",
        ],
        runtime=workflow_runtime,
    )

    try:
        async with AgentRunner() as runner:
            await runner.run(
                agent,
                payload={"task": "What's the weather like in Tokyo today?"},
                wait=True,
            )
    finally:
        workflow_runtime.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
