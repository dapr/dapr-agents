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
Abstractions for stateful agent runtimes.

Where :class:`~dapr_agents.llm.chat.ChatClientBase` models the stateless
function ``messages -> completion``, :class:`AgentExecutorBase` models
runtimes that own the full agent loop — session state, tool dispatch,
multi-turn reasoning — and emit typed :class:`AgentEvent` values as an
async stream.

Typical integration targets include the Claude Agent SDK, LangGraph,
AutoGen, and the OpenAI Assistants API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Literal, Optional

AgentEventType = Literal[
    "text_delta",
    "tool_call",
    "tool_result",
    "message",
    "session",
    "complete",
    "error",
]


@dataclass(frozen=True)
class AgentEvent:
    """
    A single event emitted by an :class:`AgentExecutorBase` during a run.

    Attributes:
        type: Discriminator for the event. See :data:`AgentEventType`.
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


class AgentExecutorBase(ABC):
    """
    Base class for LLM integrations that manage the full agent loop.

    Unlike :class:`~dapr_agents.llm.chat.ChatClientBase` (stateless
    completion), an ``AgentExecutorBase`` maintains session state and
    yields typed events. Implementations are typically thin wrappers
    around an external runtime (e.g. ``claude_agent_sdk.query``,
    LangGraph ``astream_events``).

    The contract is intentionally narrow so that consumers such as
    :class:`~dapr_agents.agents.durable.DurableAgent` can drive any
    compliant executor without provider-specific branching.
    """

    @abstractmethod
    async def run(
        self,
        prompt: str,
        *,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[AgentEvent]:
        """
        Execute a single agent turn, yielding events in order.

        Args:
            prompt: User/task input for this turn.
            session_id: Optional identifier to resume a prior session.
                If ``None``, the executor should create a fresh session.
            context: Provider-specific extras (MCP server handles,
                tool catalogs, scoped permissions, etc.). Implementations
                may ignore unknown keys.

        Yields:
            :class:`AgentEvent` values. The stream MUST end with either
            a ``complete`` event (success) or an ``error`` event (terminal
            failure). Consumers may treat absence of a terminal event as
            an executor bug.
        """
        ...

    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the persisted state for ``session_id``.

        Args:
            session_id: Identifier previously emitted by :meth:`run`.

        Returns:
            A provider-defined snapshot of the session
            (typically ``{"messages": [...], "metadata": {...}}``),
            or ``None`` if the executor has no record of it.
        """
        ...
