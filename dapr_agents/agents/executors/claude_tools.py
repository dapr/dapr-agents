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
Bridges between dapr-agents tools/hooks and the Claude Agent SDK.

* ``build_tool_server`` exposes ``AgentTool`` instances to Claude as an
  in-process SDK MCP server.
* ``ToolGate`` turns dapr-agents ``before_tool_call`` hooks into a Claude
  ``PreToolUse`` hook, including the durable approval flow: a
  ``RequireApproval`` decision defers the call (ending the run) unless a
  decision for that exact ``tool_use_id`` was passed in for the resume.

Only import this module after ``claude_agent_sdk`` is known to be installed.
"""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server

from dapr_agents.agents.executors.event import ToolCallDecision
from dapr_agents.hooks import (
    BeforeToolHook,
    Deny,
    HookDecision,
    Mutate,
    Proceed,
    RequireApproval,
    Skip,
    ToolHookContext,
)
from dapr_agents.tool.base import AgentTool

logger = logging.getLogger(__name__)

_EMPTY_SCHEMA: Dict[str, Any] = {"type": "object", "properties": {}}


def _input_schema(agent_tool: AgentTool) -> Dict[str, Any]:
    if agent_tool.args_model is None:
        return dict(_EMPTY_SCHEMA)
    schema = agent_tool.args_model.model_json_schema()
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return schema


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)


def _to_sdk_tool(agent_tool: AgentTool) -> SdkMcpTool[Any]:
    async def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = await agent_tool.arun(**(args or {}))
        except Exception as exc:  # noqa: BLE001 - reported to the model
            logger.warning("Tool %s failed: %s", agent_tool.name, exc)
            return {"content": [{"type": "text", "text": str(exc)}], "is_error": True}
        return {"content": [{"type": "text", "text": _result_text(result)}]}

    return SdkMcpTool(
        name=agent_tool.name,
        description=agent_tool.description,
        input_schema=_input_schema(agent_tool),
        handler=handler,
    )


def build_tool_server(name: str, tools: Sequence[AgentTool]) -> Any:
    """Return an SDK MCP server config exposing ``tools`` under ``name``."""
    return create_sdk_mcp_server(
        name=name, version="1.0.0", tools=[_to_sdk_tool(t) for t in tools]
    )


@dataclass(frozen=True)
class PendingApproval:
    """Why a deferred call needs a decision (from ``RequireApproval``)."""

    timeout_seconds: Optional[int] = None
    instructions: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "instructions": self.instructions,
            "reason": self.reason,
        }


def _allow(updated_input: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "hookEventName": "PreToolUse",
        "permissionDecision": "allow",
    }
    if updated_input is not None:
        output["updatedInput"] = updated_input
    return {"hookSpecificOutput": output}


def _deny(reason: str) -> Dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _defer() -> Dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "defer",
        }
    }


class ToolGate:
    """
    Per-run ``PreToolUse`` callback applying dapr-agents tool hooks.

    One gate is created per ``run`` call; it records the approval details
    of any call it defers so the executor can report them in the ``paused``
    event. Decisions are matched strictly by ``tool_use_id``.

    dapr-agents tools that the hooks let through are allowed explicitly;
    other tools (built-in, external MCP) fall through to the CLI's normal
    permission rules (``allowed_tools``, ``permission_mode``).

    Args:
        hooks: ``before_tool_call`` hooks; the first non-``Proceed``
            decision wins, mirroring ``DurableAgent``.
        decisions: Decisions supplied to resume a paused run.
        tool_prefix: ``mcp__<server>__`` prefix of the dapr-agents tools,
            stripped so hooks see the original tool names.
    """

    def __init__(
        self,
        *,
        hooks: Sequence[BeforeToolHook],
        decisions: Mapping[str, ToolCallDecision],
        tool_prefix: str,
    ) -> None:
        self._hooks = tuple(hooks)
        self._decisions = dict(decisions)
        self._tool_prefix = tool_prefix
        self._deferred: Dict[str, PendingApproval] = {}

    @property
    def enabled(self) -> bool:
        """Whether the gate needs to be registered for this run."""
        return bool(self._hooks or self._decisions)

    def deferred_approval(self, tool_use_id: str) -> Optional[PendingApproval]:
        """Approval details recorded when ``tool_use_id`` was deferred."""
        return self._deferred.get(tool_use_id)

    def display_name(self, tool_name: str) -> Tuple[str, str]:
        """Return ``(name, source)`` for hooks and events."""
        if self._tool_prefix and tool_name.startswith(self._tool_prefix):
            return tool_name[len(self._tool_prefix) :], "local"
        if tool_name.startswith("mcp__"):
            return tool_name, "mcp"
        return tool_name, "claude"

    async def __call__(
        self, input_data: Mapping[str, Any], tool_use_id: Optional[str], _ctx: Any
    ) -> Dict[str, Any]:
        call_id = str(tool_use_id or input_data.get("tool_use_id") or "")
        decision = self._decisions.get(call_id) if call_id else None
        if decision is not None:
            if decision.approved:
                return _allow()
            return _deny(decision.reason or "Rejected by human approver")

        name, source = self.display_name(str(input_data.get("tool_name", "")))
        tool_input = input_data.get("tool_input")
        context = ToolHookContext(
            step_name=name,
            source=source,
            payload=dict(tool_input) if isinstance(tool_input, Mapping) else {},
            tool_call_id=call_id,
        )
        try:
            outcome = await self._evaluate(context)
        except Exception as exc:  # noqa: BLE001 - fail closed on hook errors
            logger.exception("before_tool_call hook failed for %s", name)
            return _deny(f"Tool call blocked: policy hook failed ({exc})")
        output = self._to_output(call_id, outcome)
        if not output and source == "local":
            # dapr-agents tools are not pre-allowed while the gate is active
            # (see ``ClaudeAgentExecutor``), so they are allowed here. If the
            # CLI ever ignores a ``defer`` it then falls back to its normal
            # permission check instead of running the call unapproved.
            return _allow()
        return output

    async def _evaluate(self, context: ToolHookContext) -> HookDecision:
        for hook in self._hooks:
            outcome = hook(context)
            if inspect.isawaitable(outcome):
                outcome = await outcome
            if outcome is not None and not isinstance(outcome, Proceed):
                return outcome
        return Proceed()

    def _to_output(self, call_id: str, decision: HookDecision) -> Dict[str, Any]:
        if isinstance(decision, RequireApproval):
            if not call_id:
                return _deny("Approval required but the call has no id")
            self._deferred[call_id] = PendingApproval(
                timeout_seconds=decision.timeout_seconds,
                instructions=decision.instructions,
                reason=decision.reason,
            )
            return _defer()
        if isinstance(decision, Deny):
            return _deny(decision.reason or "Blocked by policy")
        if isinstance(decision, Skip):
            return _deny(f"Skipped by policy: {_result_text(decision.result)}")
        if isinstance(decision, Mutate) and decision.payload is not None:
            return _allow(dict(decision.payload))
        return {}
