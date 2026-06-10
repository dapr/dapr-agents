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

"""Adapter from python-sdk :class:`MCPToolDef` to dapr-agents
:class:`WorkflowContextInjectedTool`.

The Dapr Python SDK's :class:`dapr.ext.workflow.DaprMCPClient` (and its async
counterpart in :mod:`dapr.ext.workflow.aio`) owns workflow scheduling and
tool-catalogue caching. This module only handles the dapr-agents-specific
adaptation: turning each :class:`MCPToolDef` into a
:class:`WorkflowContextInjectedTool` whose executor schedules the per-tool
``dapr.internal.mcp.<server>.CallTool.<tool>`` workflow as a child workflow.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel

from dapr.ext.workflow import MCPToolDef
from dapr.ext.workflow.mcp_schema import create_pydantic_model_from_schema
from dapr_agents.tool.workflow.tool_context import WorkflowContextInjectedTool

logger = logging.getLogger(__name__)


def _build_args_model(
    name: str, schema: Optional[Dict[str, Any]]
) -> Optional[Type[BaseModel]]:
    """Build a pydantic args model from an MCP input schema, if any."""
    if not schema:
        return None
    try:
        return create_pydantic_model_from_schema(schema, f"{name}Args")
    except Exception as exc:
        logger.warning(
            "Could not build args model for tool '%s': %s — "
            "tool will accept unvalidated **kwargs",
            name,
            exc,
        )
        return None


def mcp_tool_def_to_workflow_tool(
    tool_def: "MCPToolDef",
) -> WorkflowContextInjectedTool:
    """Convert an :class:`MCPToolDef` into a :class:`WorkflowContextInjectedTool`.

    The returned tool's executor schedules the per-tool CallTool workflow
    (``dapr.internal.mcp.<server>.CallTool.<tool>``) as a child workflow when
    invoked from within a parent orchestrator.

    Args:
        tool_def: A framework-agnostic MCP tool definition from the python-sdk.

    Returns:
        A :class:`WorkflowContextInjectedTool` ready for registration on an
        :class:`~dapr_agents.tool.executor.AgentToolExecutor`.
    """
    args_model = _build_args_model(tool_def.name, tool_def.input_schema)
    wf_name = tool_def.call_tool_workflow

    def _executor(
        ctx: Any,
        _source_agent: Optional[str] = None,
        _child_instance_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        payload = {"arguments": kwargs}
        if _child_instance_id:
            return ctx.call_child_workflow(
                workflow=wf_name,
                input=payload,
                instance_id=_child_instance_id,
            )
        return ctx.call_child_workflow(workflow=wf_name, input=payload)

    return WorkflowContextInjectedTool(
        name=tool_def.name,
        description=tool_def.description,
        func=_executor,
        args_model=args_model,
    )
