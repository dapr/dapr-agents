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
Event types emitted by ``AgentExecutorBase`` implementations.

``AgentEvent`` is the wire format that executors yield as their async
stream. ``AgentEventType`` enumerates the legal discriminator values,
and the ``EVENT_*`` constants mirror them so callers can reference them
without magic strings (e.g. inside a ``match``/``case`` block).

Paused runs (tool-call approval)
--------------------------------
Executors that set ``AgentExecutorBase.supports_tool_approval`` may end a
run with a ``paused`` event instead of ``complete``/``error``. A paused run
is waiting for a decision on one deferred tool call; the event content is a
``dict`` with ``tool_call_id``, ``name``, ``arguments`` and an optional
``approval`` dict (``timeout_seconds``, ``instructions``, ``reason``) taken
from the ``RequireApproval`` hook decision that paused it.

The caller resumes the run by calling ``run`` again with the same
``session_id`` and the decision under ``context[CONTEXT_TOOL_DECISIONS]``,
a JSON-safe mapping ``{tool_call_id: ToolCallDecision.to_dict()}``. The
prompt is ignored while a paused call is being resumed. A resumed run may
pause again (for example when the model issues another gated call), so
callers must loop until they see ``complete`` or ``error`` and must only
apply a decision to the ``tool_call_id`` it was made for.

This is additive: executors that never pause keep the original
``complete``/``error`` terminal contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Optional

AgentEventType = Literal[
    "text_delta",
    "tool_call",
    "tool_result",
    "message",
    "session",
    "complete",
    "error",
    "paused",
]

# Typed event-name constants for use by executor consumers (e.g. the
# DurableAgent ``_consume_executor`` dispatch). Match the ``AgentEventType``
# literal so that ``match``/``case`` blocks can reference them without
# magic strings.
EVENT_TEXT_DELTA: AgentEventType = "text_delta"
EVENT_TOOL_CALL: AgentEventType = "tool_call"
EVENT_TOOL_RESULT: AgentEventType = "tool_result"
EVENT_MESSAGE: AgentEventType = "message"
EVENT_SESSION: AgentEventType = "session"
EVENT_COMPLETE: AgentEventType = "complete"
EVENT_ERROR: AgentEventType = "error"
EVENT_PAUSED: AgentEventType = "paused"

# Terminal event types: a run's stream ends with exactly one of these.
TERMINAL_EVENT_TYPES = frozenset({EVENT_COMPLETE, EVENT_ERROR, EVENT_PAUSED})

# ``context`` key under which callers pass tool-call decisions when resuming
# a paused run. The value maps ``tool_call_id`` to ``ToolCallDecision.to_dict()``.
CONTEXT_TOOL_DECISIONS = "tool_decisions"


@dataclass(frozen=True)
class AgentEvent:
    """
    A single event emitted by an ``AgentExecutorBase`` during a run.

    Attributes:
        type: Discriminator for the event. See ``AgentEventType``.
        content: Event payload. Shape is defined per ``type``:

            * ``text_delta`` — partial assistant text (``str``).
            * ``tool_call`` — ``dict`` with ``id``, ``name``, ``arguments``.
            * ``tool_result`` — ``dict`` with ``tool_call_id``, ``result``.
            * ``message`` — a fully-formed message ``dict`` matching
              ``dapr_agents.types.message.MessageContent``.
            * ``session`` — opaque checkpoint payload (provider-defined).
            * ``complete`` — the final assistant message ``dict``.
            * ``error`` — error message (``str``) or ``Exception``.
            * ``paused`` — ``dict`` with ``tool_call_id``, ``name``,
              ``arguments`` and optional ``approval`` describing the
              deferred tool call awaiting a decision (see module docs).
        session_id: Session identifier for multi-turn continuation.
        metadata: Free-form metadata (e.g. OpenTelemetry trace context).
    """

    type: AgentEventType
    content: Any
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallDecision:
    """
    A decision on a deferred tool call, passed back to resume a paused run.

    Attributes:
        tool_call_id: Identifier from the ``paused`` event's content.
        approved: ``True`` runs the tool; ``False`` rejects it and the model
            is told the call was blocked.
        reason: Optional explanation, shown to the model on rejection.
    """

    tool_call_id: str
    approved: bool
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dict (workflow activity inputs must be JSON)."""
        return {"approved": self.approved, "reason": self.reason}

    @classmethod
    def from_dict(
        cls, tool_call_id: str, data: Mapping[str, Any]
    ) -> "ToolCallDecision":
        """Build a decision from the dict produced by ``to_dict``."""
        reason = data.get("reason")
        return cls(
            tool_call_id=tool_call_id,
            approved=data.get("approved") is True,
            reason=str(reason) if reason is not None else None,
        )


def tool_decisions_from_context(
    context: Optional[Mapping[str, Any]],
) -> Dict[str, ToolCallDecision]:
    """
    Extract tool-call decisions from a ``run`` context.

    Args:
        context: The ``context`` passed to ``AgentExecutorBase.run``.

    Returns:
        Mapping of ``tool_call_id`` to ``ToolCallDecision``; empty when the
        context carries no (or malformed) decisions.
    """
    raw = (context or {}).get(CONTEXT_TOOL_DECISIONS)
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(tc_id): ToolCallDecision.from_dict(str(tc_id), data)
        for tc_id, data in raw.items()
        if isinstance(data, Mapping)
    }
