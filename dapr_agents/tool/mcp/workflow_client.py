from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set

import dapr.ext.workflow as wf
from pydantic import BaseModel, Field

from mcp.types import Tool as MCPTool

from dapr_agents.tool import WorkflowContextInjectedTool
from dapr_agents.types import ToolError

logger = logging.getLogger(__name__)


class WorkflowMCPClient(BaseModel):
    """MCP tool discovery + execution via Dapr Workflows.

    This client is meant for the "mcp-wf-proxy" pattern where another Dapr app
    hosts workflows named `ListTools` and `CallTool` that proxy MCP operations.

    Key behavior:
        - Tool discovery uses a workflow call to the proxy app (`ListTools`).
        - Tool execution uses workflow context injection: each wrapped tool
          schedules a child workflow (`CallTool`) on the proxy app.

    The resulting objects returned by :meth:`load_tools` are
    :class:`~dapr_agents.tool.workflow_context.WorkflowContextInjectedTool`
    instances, which DurableAgent can run inside its orchestrator by passing
    the orchestrator `ctx`.
    """

    proxy_app_id: str = Field(..., description="Dapr app-id of the MCP workflow proxy")
    server_name: str = Field(
        default="mcp",
        description="Prefix namespace for tool names exposed to the LLM",
    )
    list_tools_workflow: str = Field(
        default="ListTools", description="Remote workflow name that lists tools"
    )
    call_tool_workflow: str = Field(
        default="CallTool", description="Remote workflow name that calls a tool"
    )
    allowed_tools: Optional[Set[str]] = Field(
        default=None,
        description="Optional set of MCP tool names to include (None means all)",
    )

    def _coerce_json(self, raw: Any) -> Any:
        """Coerce raw workflow output into a Python object."""
        if raw is None:
            return None
        if isinstance(raw, str):
            raw = raw.strip()
            if not raw:
                return None
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # Some workflow runtimes may wrap JSON in quotes or return non-JSON.
                return raw
        return raw

    def _parse_mcp_tool(self, tool_obj: Any) -> MCPTool:
        """Parse an MCP Tool from dict/obj."""
        if isinstance(tool_obj, MCPTool):
            return tool_obj
        if hasattr(MCPTool, "model_validate"):
            return MCPTool.model_validate(tool_obj)  # type: ignore[attr-defined]
        # Pydantic v1 fallback
        return MCPTool.parse_obj(tool_obj)  # type: ignore[attr-defined]

    def _wrap_tool(self, mcp_tool: MCPTool) -> WorkflowContextInjectedTool:
        """Wrap an MCP tool as a workflow-context injected tool."""
        public_name = f"{self.server_name}_{mcp_tool.name}"
        description = f"{mcp_tool.description or ''} (from MCP via workflows: {self.proxy_app_id})"

        # Build args model from JSON schema (same approach as MCPClient)
        args_model = None
        if getattr(mcp_tool, "inputSchema", None):
            try:
                from dapr_agents.tool.mcp.schema import create_pydantic_model_from_schema

                args_model = create_pydantic_model_from_schema(
                    mcp_tool.inputSchema, f"{public_name}Args"
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to create args schema for tool '%s': %s", public_name, e
                )

        remote_tool_name = mcp_tool.name

        # This executor is *orchestrator-only*: it must receive a workflow context.
        def executor(ctx: wf.DaprWorkflowContext, **kwargs: Any):
            # Match the Go proxy's expected mcp.CallToolParams shape
            payload = {"name": remote_tool_name, "arguments": kwargs}
            return ctx.call_child_workflow(
                workflow=self.call_tool_workflow,
                app_id=self.proxy_app_id,
                input=payload,
            )

        return WorkflowContextInjectedTool(
            name=public_name,
            description=description,
            func=executor,
            args_model=args_model,
        )

    def tools_from_list_tools_result(self, raw_result: Any) -> List[WorkflowContextInjectedTool]:
        """Convert a ListTools workflow result into DurableAgent-compatible tools.

        Args:
            raw_result: Either a dict matching `mcp.ListToolsResult` (from the
                workflow call), or a JSON string of that dict.

        Returns:
            List of WorkflowContextInjectedTool instances.
        """
        result_obj = self._coerce_json(raw_result)
        if result_obj is None:
            return []

        # The Go SDK returns mcp.ListToolsResult which serializes like:
        # {"tools": [...], "nextCursor": "..."}
        tools_list = None
        if isinstance(result_obj, dict):
            tools_list = result_obj.get("tools")
        elif isinstance(result_obj, list):
            # Some wrappers may return just the list
            tools_list = result_obj
        else:
            raise ToolError(
                f"Unexpected ListTools result type: {type(result_obj).__name__}"
            )

        if not tools_list:
            return []

        converted: List[WorkflowContextInjectedTool] = []
        for tool_item in tools_list:
            try:
                mcp_tool = self._parse_mcp_tool(tool_item)
                if self.allowed_tools and mcp_tool.name not in self.allowed_tools:
                    continue
                converted.append(self._wrap_tool(mcp_tool))
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to convert MCP tool from workflow: %s", e)
        return converted

    async def load_tools(self) -> List[WorkflowContextInjectedTool]:
        """Convenience method: list tools by running a local workflow that calls the proxy.

        This mirrors the pattern in the user's example (schedule a workflow,
        call remote child workflow, then parse the serialized output), but returns
        DurableAgent-compatible tools.
        """

        runtime = wf.WorkflowRuntime()

        @runtime.workflow(name=f"_wf_mcp_list_tools_{self.proxy_app_id}")
        def _wf_list_tools(ctx: wf.DaprWorkflowContext):
            # Call the remote proxy workflow directly
            result = yield ctx.call_child_workflow(
                workflow=self.list_tools_workflow,
                app_id=self.proxy_app_id,
                input={},
            )
            return result

        runtime.start()
        wf_client = wf.DaprWorkflowClient()
        try:
            instance_id = wf_client.schedule_new_workflow(workflow=_wf_list_tools)
            completion = wf_client.wait_for_workflow_completion(instance_id)
            raw = completion.serialized_output
            return self.tools_from_list_tools_result(raw)
        finally:
            runtime.shutdown()
