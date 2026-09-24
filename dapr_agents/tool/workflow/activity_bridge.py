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

"""
Call workflow-backed tools from inside a workflow activity.

``WorkflowContextInjectedTool`` instances (agents-as-tools and Dapr
``MCPServer`` tools) normally run in the workflow body, where they schedule
a child workflow through the workflow context. An ``AgentExecutorBase``
runs its whole tool loop inside the ``run_executor`` activity, which has no
workflow context. ``bridge_workflow_tools`` wraps such tools in plain
``AgentTool`` instances that schedule the same workflow through a
``DaprWorkflowClient`` and wait for its result.

Child instance ids are derived from the parent instance, the run round, the
tool name, its arguments and how often that exact call was made in the run.
A retried activity therefore re-attaches to the child workflow a previous
attempt started instead of starting a second one.

Trade-offs:

* The tool workflows are top-level instances, not child workflows:
  terminating or purging the parent does not cascade to them, and one keeps
  running after the caller gives up waiting and raises ``ToolError``.
* Because ids are deterministic, a tool workflow that ended ``FAILED`` or
  ``TERMINATED`` is re-attached on every retry of the same call, so that
  call keeps failing until the parent run moves on.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import inspect
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from dapr_agents.tool.base import AgentTool
from dapr_agents.tool.utils.serialization import serialize_tool_result
from dapr_agents.tool.workflow.agent_tool import AgentWorkflowTool
from dapr_agents.tool.workflow.ask_user_tool import ASK_USER_TOOL_NAME
from dapr_agents.tool.workflow.tool_context import WorkflowContextInjectedTool
from dapr_agents.types import ToolError

logger = logging.getLogger(__name__)

DEFAULT_CHILD_WORKFLOW_TIMEOUT_SECONDS = 600
_CHILD_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "dapr-agents/activity-tool")
_FAILED_STATUSES = frozenset({"FAILED", "TERMINATED"})


@dataclass(frozen=True)
class ChildWorkflowRequest:
    """The child workflow a workflow-backed tool asked to schedule."""

    workflow: str
    input: Any = None
    instance_id: Optional[str] = None
    app_id: Optional[str] = None


class _RequestRecorder:
    """Stands in for ``DaprWorkflowContext``: records the child workflow call."""

    is_replaying = False

    def call_child_workflow(
        self,
        workflow: Any,
        *,
        input: Any = None,
        instance_id: Optional[str] = None,
        app_id: Optional[str] = None,
        **_: Any,
    ) -> ChildWorkflowRequest:
        name = workflow if isinstance(workflow, str) else workflow.__name__
        return ChildWorkflowRequest(
            workflow=name, input=input, instance_id=instance_id, app_id=app_id
        )


@dataclass(frozen=True)
class ActivityToolContext:
    """
    Where bridged tools run.

    Attributes:
        client_factory: Returns the ``DaprWorkflowClient`` used to schedule
            and await child workflows.
        instance_id: The calling workflow instance (seeds child ids).
        round: The executor run round within that instance (seeds child ids).
        source_agent: Name of the calling agent, forwarded to child agents.
        stream_context: The caller's stream context, forwarded to child
            agents so they emit into the same session stream.
        timeout_seconds: How long to wait for a child workflow.
    """

    client_factory: Callable[[], Any]
    instance_id: str
    round: int = 0
    source_agent: Optional[str] = None
    stream_context: Optional[Dict[str, Any]] = None
    timeout_seconds: int = DEFAULT_CHILD_WORKFLOW_TIMEOUT_SECONDS


class _ChildIds:
    """Deterministic child instance ids for one activity attempt."""

    def __init__(self, context: ActivityToolContext) -> None:
        self._seed = f"{context.instance_id}:{context.round}"
        self._counts: Dict[str, int] = collections.Counter()

    def next(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        call_key = f"{tool_name}:{json.dumps(arguments, sort_keys=True, default=str)}"
        occurrence = self._counts[call_key]
        self._counts[call_key] = occurrence + 1
        return str(
            uuid.uuid5(_CHILD_ID_NAMESPACE, f"{self._seed}:{call_key}:{occurrence}")
        )


def _child_stream_context(context: ActivityToolContext) -> Optional[Dict[str, Any]]:
    parent = context.stream_context
    if not parent:
        return None
    return {
        **parent,
        "parent_agent": context.source_agent,
        "parent_instance_id": context.instance_id,
        "depth": int(parent.get("depth", 0)) + 1,
    }


def _run_child_workflow(
    client: Any, request: ChildWorkflowRequest, timeout_seconds: int
) -> Any:
    """Schedule ``request`` unless it already exists, then wait for its output."""
    instance_id = str(request.instance_id)
    if client.get_workflow_state(instance_id, fetch_payloads=False) is None:
        try:
            client.schedule_new_workflow(
                workflow=request.workflow, input=request.input, instance_id=instance_id
            )
        except Exception:
            # A concurrent attempt may have scheduled it first; only a
            # still-missing instance is a real failure.
            if client.get_workflow_state(instance_id, fetch_payloads=False) is None:
                raise
    try:
        state = client.wait_for_workflow_completion(
            instance_id, fetch_payloads=True, timeout_in_seconds=timeout_seconds
        )
    except TimeoutError as exc:
        raise ToolError(
            f"Child workflow {request.workflow} ({instance_id}) did not finish "
            f"within {timeout_seconds}s."
        ) from exc
    if state is None:
        raise ToolError(f"Child workflow {request.workflow} ({instance_id}) not found.")
    status = state.runtime_status.name
    if status in _FAILED_STATUSES:
        details = getattr(state, "failure_details", None)
        reason = getattr(details, "message", None) or status.lower()
        raise ToolError(f"Child workflow {request.workflow} failed: {reason}")
    output = state.serialized_output
    return json.loads(output) if output else None


def _accepted_kwargs(tool: AgentTool, hidden: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the hidden kwargs the tool's function can take.

    Tools built directly on ``WorkflowContextInjectedTool`` (for example
    ``make_mcp_gateway_via_child_workflow_tool``) may accept only ``ctx``.
    """
    try:
        params = inspect.signature(tool.func).parameters if tool.func else {}
    except (TypeError, ValueError):
        return hidden
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return hidden
    return {k: v for k, v in hidden.items() if k == "ctx" or k in params}


def _record_request(
    tool: WorkflowContextInjectedTool,
    context: ActivityToolContext,
    child_id: str,
    arguments: Dict[str, Any],
) -> ChildWorkflowRequest:
    """Run ``tool`` against a recorder and return the child workflow it asked for."""
    hidden: Dict[str, Any] = {
        "ctx": _RequestRecorder(),
        "_source_agent": context.source_agent,
        "_child_instance_id": child_id,
    }
    stream_context = _child_stream_context(context)
    if isinstance(tool, AgentWorkflowTool) and stream_context:
        hidden["_stream_context"] = stream_context
    request = tool.run(**_accepted_kwargs(tool, hidden), **arguments)
    if not isinstance(request, ChildWorkflowRequest):
        if inspect.isgenerator(request):
            request.close()
        raise ToolError(
            f"Tool '{tool.name}' cannot run inside an activity: it does not "
            "schedule a single child workflow."
        )
    if request.app_id:
        # DaprWorkflowClient schedules workflows on this app only, so a
        # cross-app child would start locally, where it is not registered.
        raise ToolError(
            f"Tool '{tool.name}' cannot run inside an activity: it targets "
            f"workflow '{request.workflow}' on app '{request.app_id}', and "
            "cross-app child workflows need the workflow body."
        )
    if request.instance_id is None:
        return dataclasses.replace(request, instance_id=child_id)
    return request


def _bridge(
    tool: WorkflowContextInjectedTool,
    context: ActivityToolContext,
    child_ids: _ChildIds,
) -> AgentTool:
    async def call(**arguments: Any) -> str:
        child_id = child_ids.next(tool.name, arguments)
        request = _record_request(tool, context, child_id, arguments)
        output = await asyncio.to_thread(
            _run_child_workflow,
            context.client_factory(),
            request,
            context.timeout_seconds,
        )
        return serialize_tool_result(output)

    return AgentTool(
        name=tool.name,
        description=tool.description,
        func=call,
        args_model=tool.args_model,
        source=tool.source,
    )


def _bridgeable(tool: AgentTool) -> bool:
    if tool.name == ASK_USER_TOOL_NAME:
        logger.debug(
            "Not exposing %s to the executor (needs the workflow body).", tool.name
        )
        return False
    if isinstance(tool, AgentWorkflowTool) and tool.target_app_id:
        # DaprWorkflowClient schedules workflows on this app only.
        logger.warning(
            "Not exposing agent tool %r to the executor: cross-app agents "
            "(target_app_id=%r) cannot be called from an activity.",
            tool.name,
            tool.target_app_id,
        )
        return False
    return True


def bridge_workflow_tools(
    tools: Iterable[AgentTool], context: ActivityToolContext
) -> List[AgentTool]:
    """
    Return ``tools`` with workflow-backed tools made callable from an activity.

    Plain ``AgentTool`` instances are returned as-is. Workflow-backed tools
    are wrapped so they schedule their child workflow through a workflow
    client. Tools that need the workflow body (``ask_user``) and cross-app
    agents are left out. Other tools that schedule their child workflow on
    another app (for example ``make_mcp_gateway_via_child_workflow_tool``)
    raise a ``ToolError`` when called, because the workflow client can only
    schedule on this app.

    Args:
        tools: The agent's tools.
        context: Where and for whom the bridged tools run.

    Returns:
        Tools that can be called from inside a workflow activity.
    """
    child_ids = _ChildIds(context)
    bridged: List[AgentTool] = []
    for tool in tools:
        if not _bridgeable(tool):
            continue
        if isinstance(tool, WorkflowContextInjectedTool):
            bridged.append(_bridge(tool, context, child_ids))
        else:
            bridged.append(tool)
    return bridged


__all__ = [
    "ActivityToolContext",
    "ChildWorkflowRequest",
    "DEFAULT_CHILD_WORKFLOW_TIMEOUT_SECONDS",
    "bridge_workflow_tools",
]
