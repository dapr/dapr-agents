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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

AgentEventType = Literal[
    "text_delta",
    "tool_call",
    "tool_result",
    "message",
    "session",
    "complete",
    "error",
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
        session_id: Session identifier for multi-turn continuation.
        metadata: Free-form metadata (e.g. OpenTelemetry trace context).
    """

    type: AgentEventType
    content: Any
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
