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

"""Tests for StreamEmitter public API and owns_listener semantics."""

from __future__ import annotations

from typing import List

import pytest

from dapr_agents.streaming.emitter import StreamEmitter
from dapr_agents.types.streaming import AgentStreamChunk, StreamChunkType


class _CapturingListener:
    def __init__(self) -> None:
        self.chunks: List[AgentStreamChunk] = []
        self.closed = False

    def emit(self, chunk: AgentStreamChunk) -> None:
        self.chunks.append(chunk)

    def close(self) -> None:
        self.closed = True


def _make_emitter(listener, *, owns_listener: bool = True) -> StreamEmitter:
    return StreamEmitter(
        listener=listener,
        agent_name="alice",
        workflow_instance_id="wf",
        turn=0,
        root_instance_id="root",
        owns_listener=owns_listener,
    )


class TestEmitEvent:
    def test_emit_event_produces_chunk_with_type_and_data(self) -> None:
        listener = _CapturingListener()
        emitter = _make_emitter(listener)
        emitter.emit_event(
            StreamChunkType.ORCHESTRATION_DECISION,
            event_data={"selected_agent": "bob"},
        )
        assert len(listener.chunks) == 1
        chunk = listener.chunks[0]
        assert chunk.type is StreamChunkType.ORCHESTRATION_DECISION
        assert chunk.event_data == {"selected_agent": "bob"}
        assert chunk.agent == "alice"

    def test_emit_event_shares_sequence_with_consume(self) -> None:
        """Standalone events live on the same sequence counter as the
        content stream so clients can use (instance_id, sequence) as a
        total ordering within one emitter instance."""
        listener = _CapturingListener()
        emitter = _make_emitter(listener)
        emitter.emit_event(StreamChunkType.TURN_PAUSED)
        emitter.emit_event(StreamChunkType.TURN_RESUMED)
        seqs = [c.sequence for c in listener.chunks]
        assert seqs == [1, 2]

    def test_emit_event_with_complete_message(self) -> None:
        listener = _CapturingListener()
        emitter = _make_emitter(listener)
        emitter.emit_event(
            StreamChunkType.SESSION_COMPLETE,
            event_data={"status": "completed"},
            complete_message={"role": "assistant", "content": "done"},
        )
        assert listener.chunks[0].complete_message == {
            "role": "assistant",
            "content": "done",
        }


class TestOwnsListener:
    def test_close_closes_when_owned(self) -> None:
        listener = _CapturingListener()
        emitter = _make_emitter(listener, owns_listener=True)
        emitter.close()
        assert listener.closed is True

    def test_close_noop_when_not_owned(self) -> None:
        """Per-session listener caching depends on emitters created with
        ``owns_listener=False`` leaving the cached listener alive."""
        listener = _CapturingListener()
        emitter = _make_emitter(listener, owns_listener=False)
        emitter.close()
        assert listener.closed is False
        # And a second close is still a no-op.
        emitter.close()
        assert listener.closed is False
