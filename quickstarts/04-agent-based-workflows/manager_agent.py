from __future__ import annotations

import logging

import dapr.ext.workflow as wf
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from dapr_agents import DurableAgent, tool
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.workflow.runners.agent import AgentRunner

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = DaprChatClient(component_name="openai")
runtime = wf.WorkflowRuntime()


class CallAgentArgs(BaseModel):
    message: str = Field(description="Message to send to the target agent.")


@tool(args_model=CallAgentArgs)
def call_extractor(message: str, _workflow_ctx: wf.DaprWorkflowContext) -> str:
    """Extract the destination city from a user request."""
    result = yield _workflow_ctx.call_child_workflow(
        workflow="agent_workflow",
        input={"task": message},
        app_id="extractor",
    )
    return result.get("content", "") if result else ""


@tool(args_model=CallAgentArgs)
def call_planner(message: str, _workflow_ctx: wf.DaprWorkflowContext) -> str:
    """Create a short trip outline based on a destination."""
    result = yield _workflow_ctx.call_child_workflow(
        workflow="agent_workflow",
        input={"task": message},
        app_id="planner",
    )
    return result.get("content", "") if result else ""


@tool(args_model=CallAgentArgs)
def call_expander(message: str, _workflow_ctx: wf.DaprWorkflowContext) -> str:
    """Expand a trip outline into a detailed itinerary."""
    result = yield _workflow_ctx.call_child_workflow(
        workflow="agent_workflow",
        input={"task": message},
        app_id="expander",
    )
    return result.get("content", "") if result else ""


def main() -> None:
    manager = DurableAgent(
        name="ManagerAgent",
        role="Trip manager",
        goal="Coordinate specialized agents to build a complete itinerary.",
        instructions=[
            "Use tools to call specialist agents to assist user",
            "Call the extractor get the destination from user message",
            "Call the planner to using the destination.",
            "Call the expander last using the outline.",
        ],
        tools=[call_extractor, call_planner, call_expander],
        llm=llm,
        runtime=runtime,
    )

    runner = AgentRunner()
    try:
        runner.serve(manager, port=8004)
    finally:
        runner.shutdown(manager)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutting down manager agent...")
