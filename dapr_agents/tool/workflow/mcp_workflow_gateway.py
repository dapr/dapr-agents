import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from dapr_agents.tool.workflow.tool_context import WorkflowContextInjectedTool

logger = logging.getLogger(__name__)


def make_mcp_gateway_via_child_workflow_tool(
    *,
    target_app_id: str,
    gateway_workflow_name: str,
    name: str = "McpGatewayCall",
) -> WorkflowContextInjectedTool:
    """
    Create an AgentTool that calls an MCP "gateway workflow" hosted in another Dapr app,
    using the *current* workflow context (passed in by the agent) to schedule a
    multi-app child workflow.

    This uses Dapr's multi-application child workflow routing via `app_id`. :contentReference[oaicite:1]{index=1}

    The called workflow is expected to:
      - Receive input: {"tool": <str>, "arguments": <dict>}
      - Return: a JSON-serializable result (string/dict/etc.) representing the MCP call result.
    """

    class Args(BaseModel):
        tool: str = Field(..., description="Name of the MCP tool to call on the remote side.")
        arguments: Dict[str, Any] = Field(
            default_factory=dict,
            description="Arguments to pass to the remote MCP tool.",
        )
        instance_id: Optional[str] = Field(
            default=None,
            description=(
                "Optional child workflow instance id. Use when you need idempotency / "
                "dedupe semantics at the child-workflow level."
            ),
        )

    def _executor(ctx: Any, tool: str, arguments: Dict[str, Any], instance_id: Optional[str] = None) -> Any:
        """
        Schedule a child workflow on the target app id, passing the MCP tool call as input.

        NOTE: This is intentionally a *sync* function that returns a workflow Task.
        In a Dapr workflow orchestrator, you'd typically do:
            result = yield tool_obj(ctx=ctx, tool="X", arguments={...})
        """
        payload = {"tool": tool, "arguments": arguments}

        logger.debug(
            "Scheduling child workflow '%s' on app_id='%s' for MCP tool='%s' args=%s instance_id=%s",
            gateway_workflow_name,
            target_app_id,
            tool,
            arguments,
            instance_id,
        )

        # Dapr multi-app child workflow call: route execution to target_app_id. :contentReference[oaicite:2]{index=2}
        # The Python context API shown in docs:
        #   yield ctx.call_child_workflow(workflow='Workflow2', input='my-input', app_id='App2') :contentReference[oaicite:3]{index=3}
        if instance_id:
            return ctx.call_child_workflow(
                workflow=gateway_workflow_name,
                input=payload,
                instance_id=instance_id,
                app_id=target_app_id,
            )

        return ctx.call_child_workflow(
            workflow=gateway_workflow_name,
            input=payload,
            app_id=target_app_id,
        )

    return WorkflowContextInjectedTool(
        name=name,
        description=(
            f"Gateway tool that calls MCP via child workflow '{gateway_workflow_name}' "
            f"hosted by Dapr app '{target_app_id}'."
        ),
        func=_executor,
        args_model=Args,
    )
