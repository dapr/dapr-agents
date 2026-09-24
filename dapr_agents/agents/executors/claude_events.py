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
Translate Claude Agent SDK messages into ``AgentEvent`` values.

Mapping (one ``ClaudeEventMapper`` per run):

* ``SystemMessage(init)`` -> ``session`` (earliest point the id is known).
* ``StreamEvent`` ``content_block_delta``/``text_delta`` -> ``text_delta``.
* ``AssistantMessage`` ``TextBlock`` -> ``message``; ``ToolUseBlock`` /
  ``ServerToolUseBlock`` -> ``tool_call``. The CLI emits one assistant
  message per content block, so text blocks arrive one at a time.
* ``UserMessage`` ``ToolResultBlock`` -> ``tool_result``.
* ``ResultMessage`` -> ``session`` checkpoint, then exactly one terminal
  event: ``complete``, ``paused`` (a deferred tool call) or ``error``.

Subagent traffic (``parent_tool_use_id`` set) is skipped. Terminal events
are returned separately so the executor can drain the SDK stream (the
session store receives a final append during shutdown) before yielding
them.

Only import this module after ``claude_agent_sdk`` is known to be installed.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    ServerToolResultBlock,
    ServerToolUseBlock,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from dapr_agents.agents.executors.event import (
    EVENT_COMPLETE,
    EVENT_ERROR,
    EVENT_MESSAGE,
    EVENT_PAUSED,
    EVENT_SESSION,
    EVENT_TEXT_DELTA,
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    METADATA_RETRYABLE,
    AgentEvent,
)

DisplayName = Callable[[str], Tuple[str, str]]
ApprovalLookup = Callable[[str], Optional[Dict[str, Any]]]

_ERROR_FIELDS = ("subtype", "terminal_reason", "stop_reason", "api_error_status")
# Failures that repeat on every retry of the same run.
_TERMINAL_SUBTYPES = frozenset(
    {"error_max_turns", "error_max_budget_usd", "error_max_structured_output_retries"}
)
_TERMINAL_REASONS = frozenset({"prompt_too_long"})
_TERMINAL_API_STATUSES = frozenset({400, 401, 403, 404, 413})


def is_retryable_result(message: ResultMessage) -> bool:
    """Whether a failed ``ResultMessage`` may succeed if the run is retried."""
    return not (
        message.subtype in _TERMINAL_SUBTYPES
        or message.terminal_reason in _TERMINAL_REASONS
        or message.api_error_status in _TERMINAL_API_STATUSES
    )


def tool_result_text(content: Any) -> str:
    """Flatten SDK tool-result content (str or MCP content list) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if len(texts) == len(content):
            return "\n".join(texts)
    return json.dumps(content, default=str)


class ClaudeEventMapper:
    """
    Stateful per-run translator from SDK messages to ``AgentEvent``.

    Args:
        session_id: The session id the run was started or resumed with.
        display_name: Maps a Claude tool name to ``(name, source)``.
        approval_for: Returns the approval details recorded for a deferred
            ``tool_use_id`` (from ``ToolGate``), if any.
        include_text_deltas: Emit ``text_delta`` events.
        prior_cost_usd: Session cost before this run, used to report the
            per-run cost delta (``total_cost_usd`` is session-cumulative).
            It is read from the session store transcript; without a store
            it is ``None`` and a resumed run's ``cost_usd`` is the session
            total.
    """

    def __init__(
        self,
        *,
        session_id: str,
        display_name: DisplayName,
        approval_for: ApprovalLookup,
        include_text_deltas: bool,
        prior_cost_usd: Optional[float] = None,
        tool_names: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.session_id = session_id
        self._display_name = display_name
        self._approval_for = approval_for
        self._include_text_deltas = include_text_deltas
        self._prior_cost = prior_cost_usd
        self._tool_names: Dict[str, str] = dict(tool_names or {})
        self._last_text: Optional[str] = None
        self.terminal: Optional[AgentEvent] = None

    # ------------------------------------------------------------------
    def map(self, message: Any) -> List[AgentEvent]:
        """Return the non-terminal events for one SDK message.

        A ``ResultMessage`` sets ``self.terminal`` instead of returning it.
        """
        if isinstance(message, StreamEvent):
            return self._stream_event(message)
        if isinstance(message, AssistantMessage):
            return self._assistant(message)
        if isinstance(message, UserMessage):
            return self._user(message)
        if isinstance(message, SystemMessage):
            return self._system(message)
        if isinstance(message, ResultMessage):
            return self._result(message)
        return []

    def _event(self, type_: Any, content: Any, **metadata: Any) -> AgentEvent:
        return AgentEvent(
            type=type_,
            content=content,
            session_id=self.session_id,
            metadata=metadata,
        )

    def _checkpoint(self, stage: str) -> AgentEvent:
        return self._event(
            EVENT_SESSION, {"session_id": self.session_id, "stage": stage}
        )

    def _observe_session(self, session_id: Optional[str]) -> None:
        if session_id:
            self.session_id = session_id

    # ------------------------------------------------------------------
    def _system(self, message: SystemMessage) -> List[AgentEvent]:
        if message.subtype != "init":
            return []
        self._observe_session(message.data.get("session_id"))
        return [self._checkpoint("init")]

    def _stream_event(self, message: StreamEvent) -> List[AgentEvent]:
        if not self._include_text_deltas or message.parent_tool_use_id:
            return []
        event = message.event or {}
        delta = event.get("delta") or {}
        if event.get("type") != "content_block_delta":
            return []
        if delta.get("type") != "text_delta" or not delta.get("text"):
            return []
        self._observe_session(message.session_id)
        return [self._event(EVENT_TEXT_DELTA, str(delta["text"]))]

    def _tool_call(self, block_id: str, name: str, arguments: Any) -> AgentEvent:
        display, source = self._display_name(name)
        self._tool_names[block_id] = display
        return self._event(
            EVENT_TOOL_CALL,
            {
                "id": block_id,
                "name": display,
                "arguments": dict(arguments or {}),
            },
            source=source,
        )

    def _assistant(self, message: AssistantMessage) -> List[AgentEvent]:
        if message.parent_tool_use_id:
            return []
        self._observe_session(message.session_id)
        events: List[AgentEvent] = []
        for block in message.content:
            if isinstance(block, TextBlock) and block.text:
                self._last_text = block.text
                events.append(
                    self._event(
                        EVENT_MESSAGE,
                        {"role": "assistant", "content": block.text},
                        message_id=message.message_id,
                        model=message.model,
                    )
                )
            elif isinstance(block, (ToolUseBlock, ServerToolUseBlock)):
                events.append(self._tool_call(block.id, block.name, block.input))
            elif isinstance(block, ServerToolResultBlock):
                events.append(self._tool_result(block.tool_use_id, block.content))
        return events

    def _tool_result(
        self, tool_use_id: str, content: Any, is_error: bool = False
    ) -> AgentEvent:
        return self._event(
            EVENT_TOOL_RESULT,
            {
                "tool_call_id": tool_use_id,
                "name": self._tool_names.get(tool_use_id, ""),
                "result": tool_result_text(content),
                "is_error": is_error,
            },
        )

    def _user(self, message: UserMessage) -> List[AgentEvent]:
        if message.parent_tool_use_id or isinstance(message.content, str):
            return []
        return [
            self._tool_result(block.tool_use_id, block.content, bool(block.is_error))
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]

    # ------------------------------------------------------------------
    def _usage_metadata(self, message: ResultMessage) -> Dict[str, Any]:
        total = message.total_cost_usd
        cost = None
        if total is not None:
            cost = max(total - (self._prior_cost or 0.0), 0.0)
        return {
            "usage": message.usage or {},
            "model_usage": message.model_usage or {},
            "cost_usd": cost,
            "session_total_cost_usd": total,
            "num_turns": message.num_turns,
            "duration_ms": message.duration_ms,
            "stop_reason": message.stop_reason,
            "terminal_reason": message.terminal_reason,
        }

    def _result(self, message: ResultMessage) -> List[AgentEvent]:
        self._observe_session(message.session_id)
        metadata = self._usage_metadata(message)
        deferred = message.deferred_tool_use
        if message.is_error:
            self.terminal = self._error_event(message, metadata)
        elif deferred is not None:
            display, source = self._display_name(deferred.name)
            content = {
                "tool_call_id": deferred.id,
                "name": display,
                "arguments": dict(deferred.input or {}),
                "approval": self._approval_for(deferred.id),
                "source": source,
            }
            self.terminal = self._event(EVENT_PAUSED, content, **metadata)
        else:
            text = message.result or self._last_text or ""
            self.terminal = self._event(
                EVENT_COMPLETE, {"role": "assistant", "content": text}, **metadata
            )
        return [self._checkpoint("result")]

    def _error_event(
        self, message: ResultMessage, metadata: Dict[str, Any]
    ) -> AgentEvent:
        details = {f: getattr(message, f, None) for f in _ERROR_FIELDS}
        errors = list(message.errors or [])
        reason = "; ".join(errors) or message.result or "Claude run failed"
        summary = ", ".join(f"{k}={v}" for k, v in details.items() if v is not None)
        return self._event(
            EVENT_ERROR,
            f"Claude run failed ({summary}): {reason}",
            errors=errors,
            **{**metadata, **details, METADATA_RETRYABLE: is_retryable_result(message)},
        )

    def error(self, text: str, **metadata: Any) -> AgentEvent:
        """Build an ``error`` event for failures outside a ``ResultMessage``."""
        return self._event(EVENT_ERROR, text, **metadata)
