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

"""Streaming types for Dapr Agents.

Wire format shared by every listener, consumer, and transport. Chunks are
best-effort (non-durable); the authoritative final message is always persisted
via the workflow state path.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from dapr_agents.types.message import (
    LLMChatCandidateChunk,
    LLMChatResponseChunk,
    ToolCallChunk,
)

# Pre-release ("-alpha") signals that the streaming chunk schema is still alpha
# and may change in future 1.x releases. Consumers branch on
# AgentStreamChunk.schema_version for stability.
STREAM_SCHEMA_VERSION = "1-alpha"


class StreamChunkType(str, Enum):
    """Discriminator for ``AgentStreamChunk`` wire format.

    Reserved values (``TURN_STARTED``, ``TOOL_RESULT``) are declared for
    forward compatibility but no current code path emits them. Clients
    should not register handlers expecting traffic on those types until a
    future release adds explicit emission.
    """

    START = "start"
    CONTENT_DELTA = "content_delta"
    TOOL_CALL_DELTA = "tool_call_delta"
    #: Reserved — not emitted in v1. Future releases may surface tool-call
    #: results via this type; today tool results land in ``entry.messages``
    #: and are visible on subsequent ``CONTENT_DELTA`` / ``TURN_COMPLETE``.
    TOOL_RESULT = "tool_result"
    #: Reserved — not emitted in v1. Consumers should use the first
    #: ``CONTENT_DELTA`` for a turn as the implicit "turn started" signal.
    TURN_STARTED = "turn_started"
    TURN_PAUSED = "turn_paused"
    TURN_RESUMED = "turn_resumed"
    TURN_COMPLETE = "turn_complete"
    ORCHESTRATION_DECISION = "orchestration_decision"
    SESSION_COMPLETE = "session_complete"
    USER_INPUT_REQUESTED = "user_input_requested"
    USER_INPUT_RECEIVED = "user_input_received"
    USER_INPUT_TIMED_OUT = "user_input_timed_out"
    ERROR = "error"


class StreamDelta(BaseModel):
    """Normalized provider chunk payload for a single delta.

    Mirrors ``LLMChatCandidateChunk`` but lives inside the stream envelope
    so consumers never need provider-specific decoding.
    """

    model_config = ConfigDict(extra="forbid")

    content: Optional[str] = None
    role: Optional[str] = None
    refusal: Optional[str] = None
    tool_calls: Optional[List[ToolCallChunk]] = None
    finish_reason: Optional[str] = None
    index: Optional[int] = 0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentStreamChunk(BaseModel):
    """Wire format for every chunk emitted to a ``StreamListener``.

    Ordering key: ``(workflow_instance_id, sequence)``. Retry dedup key:
    ``chunk_id``. See the streaming design doc for delivery semantics.

    ``event_data`` schemas per ``type``
    ----------------------------------
    Fields not listed are ``None`` for a given ``type``.

    - ``START``                    — no ``event_data``; ``delta`` None.
    - ``CONTENT_DELTA``            — ``delta.content`` populated; role on
                                     first chunk only.
    - ``TOOL_CALL_DELTA``          — ``delta.tool_calls`` populated with
                                     partial ``ToolCallChunk`` fragments.
    - ``TOOL_RESULT``              — reserved, not emitted in v1.
    - ``TURN_STARTED``             — reserved, not emitted in v1.
    - ``TURN_PAUSED``              — ``event_data``:
                                     ``{"reason": "child_agent_dispatch",
                                     "children": [{"child_agent": str,
                                     "child_instance_id": str}, ...]}``
                                     for the agent-as-tool dispatch path,
                                     or ``{"reason": "orchestration_dispatch",
                                     "child_agent": str,
                                     "child_instance_id": str}`` for the
                                     orchestrator dispatch path.
    - ``TURN_RESUMED``             — ``event_data``:
                                     ``{"children": [{"child_agent": str,
                                     "child_instance_id": str}, ...]}``
                                     paired with the fan-out ``TURN_PAUSED``
                                     form, or ``{"child_agent": str,
                                     "child_instance_id": str}`` for
                                     the orchestrator form.
    - ``TURN_COMPLETE``            — ``complete_message`` populated iff the
                                     session opted in via
                                     ``include_complete_message=True``;
                                     ``metadata`` carries ``finish_reason``
                                     and provider usage.
    - ``ORCHESTRATION_DECISION``   — ``event_data``:
                                     ``{"selected_agent": str,
                                     "instruction": str,
                                     "child_instance_id": str}``;
                                     ``phase == "routing"``.
    - ``SESSION_COMPLETE``         — root-only (``depth == 0``).
                                     ``event_data``: ``{"status":
                                     "completed"}``. ``complete_message``
                                     populated iff opted in.
    - ``USER_INPUT_REQUESTED``     — ``event_data``:
                                     ``{"request_id": str, "question": str,
                                     "target_instance_id": str,
                                     "timeout_seconds": int}``.
    - ``USER_INPUT_RECEIVED``      — ``event_data``:
                                     ``{"request_id": str, "answer": str}``
                                     (answer is already length-capped and
                                     passed through neutral framing).
    - ``USER_INPUT_TIMED_OUT``     — ``event_data``:
                                     ``{"request_id": str}``.
    - ``ERROR``                    — ``error``: ``{"type": str,
                                     "message": str}``; ``event_data``
                                     may carry ``{"retryable": bool}`` on
                                     activity-retry cases.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = STREAM_SCHEMA_VERSION
    chunk_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    sequence: int
    type: StreamChunkType

    agent: str
    workflow_instance_id: str
    turn: int

    root_instance_id: str
    parent_agent: Optional[str] = None
    parent_instance_id: Optional[str] = None
    depth: int = 0
    call_path: List[str] = Field(default_factory=list)

    phase: Optional[str] = None
    delta: Optional[StreamDelta] = None
    complete_message: Optional[Dict[str, Any]] = None
    tool_call: Optional[Dict[str, Any]] = None
    event_data: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None

    timestamp: str = Field(default_factory=_utc_now_iso)
    trace_parent: Optional[str] = None


class UserInputResponse(BaseModel):
    """Inbound payload that answers a ``USER_INPUT_REQUESTED`` chunk."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    target_instance_id: str
    answer: str


# ---------------------------------------------------------------------------
# Accumulator — chunk → AssistantMessage reduction
# ---------------------------------------------------------------------------


class _ToolCallBuffer:
    """Accumulates deltas for a single tool call across chunks."""

    __slots__ = ("index", "id", "type", "name", "arguments")

    def __init__(self, index: int) -> None:
        self.index = index
        self.id: Optional[str] = None
        self.type: Optional[str] = None
        self.name: Optional[str] = None
        self.arguments: List[str] = []

    def merge(self, chunk: ToolCallChunk) -> None:
        if chunk.id is not None:
            self.id = chunk.id
        if chunk.type is not None:
            self.type = chunk.type
        fn = chunk.function
        if fn is not None:
            if fn.name is not None:
                self.name = (self.name or "") + fn.name
            if fn.arguments is not None:
                self.arguments.append(fn.arguments)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id or "",
            "type": self.type or "function",
            "function": {
                "name": self.name or "",
                "arguments": "".join(self.arguments),
            },
        }


class AssistantMessageAccumulator:
    """Merges provider chunks into a final ``AssistantMessage`` dict.

    Provider-agnostic: accepts ``LLMChatResponseChunk``,
    ``LLMChatCandidateChunk``, or the dict form of either. Content and
    refusal strings are concatenated in arrival order; tool calls are
    bucketed by ``index`` with name/arguments concatenated.

    This class lives in ``dapr_agents.types`` so both the streaming
    emitter and the observability wrapper can depend on it without
    creating a cross-domain edge between ``observability`` and
    ``streaming``. It is runtime logic rather than a wire schema, but
    its inputs and outputs are all wire-level types, so the placement
    is deliberate.
    """

    def __init__(self) -> None:
        self._content_parts: List[str] = []
        self._refusal_parts: List[str] = []
        self._tool_calls: Dict[int, _ToolCallBuffer] = {}
        self._finish_reason: Optional[str] = None
        self._role: Optional[str] = None
        self._last_metadata: Dict[str, Any] = {}

    # -- ingestion ---------------------------------------------------------

    def ingest(self, chunk: Any) -> LLMChatCandidateChunk:
        """Fold a single packet into the accumulator, returning the candidate."""

        candidate, metadata = self.unwrap(chunk)
        if candidate.content:
            self._content_parts.append(candidate.content)
        if candidate.refusal:
            self._refusal_parts.append(candidate.refusal)
        if candidate.role and not self._role:
            self._role = candidate.role
        if candidate.finish_reason:
            self._finish_reason = candidate.finish_reason
        for tc in candidate.tool_calls or []:
            buf = self._tool_calls.get(tc.index)
            if buf is None:
                buf = _ToolCallBuffer(tc.index)
                self._tool_calls[tc.index] = buf
            buf.merge(tc)
        if metadata:
            self._last_metadata = metadata
        return candidate

    # -- readers -----------------------------------------------------------

    @property
    def finish_reason(self) -> Optional[str]:
        return self._finish_reason

    @property
    def last_metadata(self) -> Dict[str, Any]:
        return self._last_metadata

    def assistant_message(self) -> Dict[str, Any]:
        content = "".join(self._content_parts)
        message: Dict[str, Any] = {
            "role": self._role or "assistant",
            "content": content if content else None,
        }
        if self._refusal_parts:
            message["refusal"] = "".join(self._refusal_parts)
        if self._tool_calls:
            message["tool_calls"] = [
                self._tool_calls[i].to_dict() for i in sorted(self._tool_calls)
            ]
        return message

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def unwrap(chunk: Any) -> Tuple[LLMChatCandidateChunk, Dict[str, Any]]:
        """Normalise provider chunk variants into ``(candidate, metadata)``.

        Public so callers that already accumulated the chunk upstream can
        still get a candidate view without running a second ingest pass.
        """
        if isinstance(chunk, LLMChatResponseChunk):
            return chunk.result, dict(chunk.metadata or {})
        if isinstance(chunk, LLMChatCandidateChunk):
            return chunk, {}
        if isinstance(chunk, dict):
            if "result" in chunk:
                candidate = LLMChatCandidateChunk.model_validate(chunk["result"])
                return candidate, dict(chunk.get("metadata") or {})
            return LLMChatCandidateChunk.model_validate(chunk), {}
        raise TypeError(
            f"AssistantMessageAccumulator cannot ingest object of type "
            f"{type(chunk).__name__}"
        )


__all__ = [
    "STREAM_SCHEMA_VERSION",
    "AgentStreamChunk",
    "AssistantMessageAccumulator",
    "StreamChunkType",
    "StreamDelta",
    "UserInputResponse",
]
