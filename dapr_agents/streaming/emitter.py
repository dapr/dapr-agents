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

"""Streaming emitter — bridges an LLM provider's chunk iterator to a listener.

Instantiated inside the ``call_llm`` workflow activity. Consumes provider
chunks, emits ``AgentStreamChunk`` events through the configured
``StreamListener``, and accumulates the deltas into a final ``AssistantMessage``
that the activity returns (and that gets persisted to durable state).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from dapr_agents.streaming.listeners import StreamListener
from dapr_agents.types.message import LLMChatCandidateChunk
from dapr_agents.types.streaming import (
    AgentStreamChunk,
    AssistantMessageAccumulator,
    StreamChunkType,
    StreamDelta,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------


def _delta_from(candidate: LLMChatCandidateChunk) -> StreamDelta:
    return StreamDelta(
        content=candidate.content,
        role=candidate.role,
        refusal=candidate.refusal,
        tool_calls=list(candidate.tool_calls) if candidate.tool_calls else None,
        finish_reason=candidate.finish_reason,
        index=candidate.index if candidate.index is not None else 0,
    )


class StreamEmitter:
    """Wraps an ``StreamListener`` and translates LLM chunks → ``AgentStreamChunk``.

    One emitter per ``call_llm`` activity invocation. Sequencing is
    monotonic per ``(workflow_instance_id, turn)``; chunk ids are random UUIDs
    so clients can deduplicate across activity retries.
    """

    def __init__(
        self,
        *,
        listener: StreamListener,
        agent_name: str,
        workflow_instance_id: str,
        turn: int,
        root_instance_id: str,
        parent_agent: Optional[str] = None,
        parent_instance_id: Optional[str] = None,
        depth: int = 0,
        call_path: Optional[List[str]] = None,
        phase: Optional[str] = None,
        trace_parent: Optional[str] = None,
        include_complete_message: bool = False,
        owns_listener: bool = True,
    ) -> None:
        if not listener:
            raise ValueError("StreamEmitter requires a listener")
        if not agent_name:
            raise ValueError("StreamEmitter requires an agent_name")
        if not workflow_instance_id:
            raise ValueError("StreamEmitter requires a workflow_instance_id")
        if not root_instance_id:
            raise ValueError("StreamEmitter requires a root_instance_id")
        self._listener = listener
        self._agent = agent_name
        self._wf_id = workflow_instance_id
        self._turn = turn
        self._root = root_instance_id
        self._parent_agent = parent_agent
        self._parent_instance_id = parent_instance_id
        self._depth = depth
        self._call_path = list(call_path) if call_path else [agent_name]
        self._phase = phase
        self._trace_parent = trace_parent
        self._include_complete_message = include_complete_message
        self._owns_listener = owns_listener
        self._sequence = 0
        self._started = False

    # -- public API --------------------------------------------------------

    def close(self) -> None:
        """Flush and close the underlying listener.

        No-op when the emitter does not own the listener (the caller is
        managing the listener's lifecycle via a cache, typically per-session
        on the agent). Safe to call multiple times in either mode.
        """
        if self._owns_listener:
            self._listener.close()

    def emit_event(
        self,
        chunk_type: StreamChunkType,
        *,
        event_data: Optional[Dict[str, Any]] = None,
        complete_message: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a one-off stream event with attribution from this emitter.

        Used for standalone events (orchestration decisions, user-input
        request/response, session completion) that aren't part of an LLM
        chunk stream. Sequence numbering is shared with
        ``consume``/``consume_non_streaming``.
        """
        self._emit(
            chunk_type,
            event_data=event_data,
            complete_message=complete_message,
            metadata=metadata,
        )

    # -- main entrypoints -------------------------------------------------

    def consume(self, iterator: Iterable[Any]) -> Dict[str, Any]:
        """Iterate the provider stream, emit chunks, return the final AssistantMessage.

        Reuses a shared ``AssistantMessageAccumulator`` attached to the
        iterator via ``_dapr_accumulator`` (populated by
        :class:`LLMWrapper`) when present. This halves per-chunk CPU work
        when observability is enabled — without the sharing, both the
        wrapper and this method would ingest every chunk independently.
        """

        shared = getattr(iterator, "_dapr_accumulator", None)
        accumulator = shared if shared is not None else AssistantMessageAccumulator()
        self._emit(StreamChunkType.START)
        try:
            for packet in iterator:
                if shared is None:
                    candidate = accumulator.ingest(packet)
                else:
                    # Accumulation already done upstream; we still need a
                    # normalised candidate object to emit the delta event.
                    candidate = AssistantMessageAccumulator.unwrap(packet)[0]
                self._emit_delta(candidate)
        except Exception as exc:
            self._emit(
                StreamChunkType.ERROR,
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            raise
        final = accumulator.assistant_message()
        metadata = dict(accumulator.last_metadata)
        if accumulator.finish_reason:
            metadata.setdefault("finish_reason", accumulator.finish_reason)
        self._emit_turn_complete(final, metadata=metadata)
        return final

    def consume_non_streaming(
        self,
        assistant_message: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Emit ``START`` + ``TURN_COMPLETE`` for providers that didn't stream.

        Used when ``stream=True`` was requested but the provider fell back
        (Dapr Conversation API pre-streaming) or structured output disabled
        streaming. Consumers see a uniform stream shape (two chunks).

        ``TURN_COMPLETE.complete_message`` is only populated when the caller
        opted in via ``include_complete_message=True`` on the session. Clients
        that did not opt in must fetch the final assistant message from
        workflow state rather than receiving it over the (potentially
        broadcast) stream topic.
        """

        self._emit(StreamChunkType.START)
        self._emit_turn_complete(assistant_message, metadata=metadata or {})
        return assistant_message

    # -- emission helpers -------------------------------------------------

    def _emit_delta(self, candidate: LLMChatCandidateChunk) -> None:
        has_content = bool(candidate.content)
        has_tool = bool(candidate.tool_calls)
        has_refusal = bool(candidate.refusal)
        # Skip packets that only carry role or finish_reason: the role chunk
        # is a pre-roll artifact, and finish_reason flows through the
        # TURN_COMPLETE metadata instead of a separate delta event.
        if not (has_content or has_tool or has_refusal):
            return
        delta = _delta_from(candidate)
        chunk_type = (
            StreamChunkType.TOOL_CALL_DELTA
            if has_tool and not has_content
            else StreamChunkType.CONTENT_DELTA
        )
        self._emit(chunk_type, delta=delta)

    def _emit_turn_complete(
        self,
        assistant_message: Dict[str, Any],
        *,
        metadata: Dict[str, Any],
    ) -> None:
        payload: Dict[str, Any] = {"metadata": metadata} if metadata else {}
        if self._include_complete_message:
            payload["complete_message"] = assistant_message
        self._emit(StreamChunkType.TURN_COMPLETE, **payload)

    def _emit(
        self,
        chunk_type: StreamChunkType,
        *,
        delta: Optional[StreamDelta] = None,
        complete_message: Optional[Dict[str, Any]] = None,
        tool_call: Optional[Dict[str, Any]] = None,
        event_data: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._sequence += 1
        envelope = AgentStreamChunk(
            sequence=self._sequence,
            type=chunk_type,
            agent=self._agent,
            workflow_instance_id=self._wf_id,
            turn=self._turn,
            root_instance_id=self._root,
            parent_agent=self._parent_agent,
            parent_instance_id=self._parent_instance_id,
            depth=self._depth,
            call_path=list(self._call_path),
            phase=self._phase,
            delta=delta,
            complete_message=complete_message,
            tool_call=tool_call,
            event_data=event_data,
            error=error,
            metadata=metadata,
            trace_parent=self._trace_parent,
        )
        try:
            self._listener.emit(envelope)
        except Exception as exc:  # noqa: BLE001 - listeners must not raise
            logger.error(
                "StreamListener.emit raised unexpectedly (chunk=%s): %s",
                envelope.chunk_id,
                exc,
            )


__all__ = [
    "AssistantMessageAccumulator",
    "StreamEmitter",
]
