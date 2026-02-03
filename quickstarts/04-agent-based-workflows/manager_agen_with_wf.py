from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, Optional

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

_DEFAULT_TIMEOUT_SECONDS = 300


@runtime.workflow(name="manager_call_agent")
def manager_call_agent(ctx: wf.DaprWorkflowContext, payload: Dict[str, Any]) -> str:
    """Call a target agent workflow as a child workflow."""
    message = payload.get("message", "")
    app_id = payload.get("app_id", "")
    if not app_id:
        raise ValueError("Missing app_id for manager_call_agent.")
    result = yield ctx.call_child_workflow(
        workflow="agent_workflow",
        input={"task": message},
        app_id=app_id,
    )
    return result.get("content", "") if result else ""


class CallAgentArgs(BaseModel):
    message: str = Field(description="Message to send to the target agent.")


def _extract_content(payload: Optional[str]) -> str:
    if not payload:
        return ""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return str(payload)
    if isinstance(data, dict):
        return str(data.get("content", data))
    return str(data)


def _run_agent_call(app_id: str, message: str) -> str:
    client = wf.DaprWorkflowClient()
    try:
        instance_id = f"manager-{app_id}-{uuid.uuid4().hex}"
        client.schedule_new_workflow(
            workflow=manager_call_agent,
            input={"message": message, "app_id": app_id},
            instance_id=instance_id,
        )
        state = client.wait_for_workflow_completion(
            instance_id,
            timeout_in_seconds=_DEFAULT_TIMEOUT_SECONDS,
            fetch_payloads=True,
        )
    finally:
        client.close()

    if not state:
        raise RuntimeError(f"No workflow state returned for {instance_id}.")
    status = getattr(state.runtime_status, "name", str(state.runtime_status))
    if status != "COMPLETED":
        raise RuntimeError(f"Manager workflow failed with status {status}.")
    return _extract_content(getattr(state, "serialized_output", None))


@tool(args_model=CallAgentArgs)
def call_extractor(message: str) -> str:
    """Extract the destination city from a user request."""
    return _run_agent_call("extractor", message)


@tool(args_model=CallAgentArgs)
def call_planner(message: str) -> str:
    """Create a short trip outline based on a destination."""
    return _run_agent_call("planner", message)


@tool(args_model=CallAgentArgs)
def call_expander(message: str) -> str:
    """Expand a trip outline into a detailed itinerary."""
    return _run_agent_call("expander", message)


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
