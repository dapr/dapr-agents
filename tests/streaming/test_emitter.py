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

"""Tests for StreamEmitter and the chunk accumulator."""

from __future__ import annotations

import json
from typing import Iterable, List

import pytest

from dapr_agents.streaming.emitter import (
    AssistantMessageAccumulator,
    StreamEmitter,
)
from dapr_agents.types.message import (
    FunctionCallChunk,
    LLMChatCandidateChunk,
    LLMChatResponseChunk,
    ToolCallChunk,
)
from dapr_agents.types.streaming import AgentStreamChunk, StreamChunkType


def _packet(**kwargs) -> LLMChatResponseChunk:
    return LLMChatResponseChunk(
        result=LLMChatCandidateChunk(**kwargs),
        metadata=kwargs.pop("_metadata", None),
    )


def _content_stream(content_parts: Iterable[str]) -> List[LLMChatResponseChunk]:
    packets = [_packet(role="assistant")]
    for part in content_parts:
        packets.append(_packet(content=part))
    packets.append(_packet(finish_reason="stop"))
    return packets


def _tool_call_stream(
    *,
    call_id: str = "call_1",
    name_parts: Iterable[str] = ("get_", "weather"),
    args_parts: Iterable[str] = ('{"city":', ' "NYC"}'),
) -> List[LLMChatResponseChunk]:
    packets = [_packet(role="assistant")]
    name_iter = iter(name_parts)
    args_iter = iter(args_parts)
    # First tool_call chunk carries id + type + first name fragment
    packets.append(
        _packet(
            tool_calls=[
                ToolCallChunk(
                    index=0,
                    id=call_id,
                    type="function",
                    function=FunctionCallChunk(name=next(name_iter), arguments=""),
                )
            ]
        )
    )
    # Subsequent tool_call chunks carry more name/arg fragments
    for name_part in name_iter:
        packets.append(
            _packet(
                tool_calls=[
                    ToolCallChunk(
                        index=0,
                        function=FunctionCallChunk(name=name_part),
                    )
                ]
            )
        )
    for args_part in args_iter:
        packets.append(
            _packet(
                tool_calls=[
                    ToolCallChunk(
                        index=0,
                        function=FunctionCallChunk(arguments=args_part),
                    )
                ]
            )
        )
    packets.append(_packet(finish_reason="tool_calls"))
    return packets


class _CapturingListener:
    def __init__(self) -> None:
        self.chunks: List[AgentStreamChunk] = []
        self.closed = False

    def emit(self, chunk: AgentStreamChunk) -> None:
        self.chunks.append(chunk)

    def close(self) -> None:
        self.closed = True


def _make_emitter(
    listener: _CapturingListener, *, include_complete_message: bool = False
) -> StreamEmitter:
    return StreamEmitter(
        listener=listener,
        agent_name="alice",
        workflow_instance_id="wf-1",
        turn=1,
        root_instance_id="root-1",
        include_complete_message=include_complete_message,
    )


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------


class TestAssistantMessageAccumulator:
    def test_content_concatenation(self) -> None:
        acc = AssistantMessageAccumulator()
        for part in ("Hello ", "world", "!"):
            acc.ingest(_packet(content=part))
        msg = acc.assistant_message()
        assert msg == {"role": "assistant", "content": "Hello world!"}

    def test_role_and_finish_reason_tracked(self) -> None:
        acc = AssistantMessageAccumulator()
        for packet in _content_stream(["hi"]):
            acc.ingest(packet)
        assert acc.finish_reason == "stop"
        msg = acc.assistant_message()
        assert msg["role"] == "assistant"

    def test_tool_call_reconstruction(self) -> None:
        acc = AssistantMessageAccumulator()
        for packet in _tool_call_stream():
            acc.ingest(packet)
        msg = acc.assistant_message()
        assert msg["tool_calls"] == [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "NYC"}',
                },
            }
        ]

    def test_multiple_parallel_tool_calls(self) -> None:
        acc = AssistantMessageAccumulator()
        # Two tool calls at index 0 and 1, interleaved arguments.
        acc.ingest(
            _packet(
                tool_calls=[
                    ToolCallChunk(
                        index=0,
                        id="call_a",
                        type="function",
                        function=FunctionCallChunk(name="search", arguments="{"),
                    ),
                    ToolCallChunk(
                        index=1,
                        id="call_b",
                        type="function",
                        function=FunctionCallChunk(name="translate", arguments="{"),
                    ),
                ]
            )
        )
        acc.ingest(
            _packet(
                tool_calls=[
                    ToolCallChunk(
                        index=0,
                        function=FunctionCallChunk(arguments='"q":"py"'),
                    ),
                    ToolCallChunk(
                        index=1,
                        function=FunctionCallChunk(arguments='"text":"hi"'),
                    ),
                ]
            )
        )
        acc.ingest(
            _packet(
                tool_calls=[
                    ToolCallChunk(index=0, function=FunctionCallChunk(arguments="}")),
                    ToolCallChunk(index=1, function=FunctionCallChunk(arguments="}")),
                ]
            )
        )
        msg = acc.assistant_message()
        assert [tc["id"] for tc in msg["tool_calls"]] == ["call_a", "call_b"]
        assert msg["tool_calls"][0]["function"]["arguments"] == '{"q":"py"}'
        assert msg["tool_calls"][1]["function"]["arguments"] == '{"text":"hi"}'

    def test_empty_tool_call_fragment_produces_no_tool_calls(self) -> None:
        """Finish-reason-only packet without any preceding tool_call fragments
        must not fabricate an empty tool_calls list — that would confuse the
        downstream LLM if an empty array lands in the conversation history."""
        acc = AssistantMessageAccumulator()
        acc.ingest(_packet(role="assistant"))
        acc.ingest(_packet(finish_reason="tool_calls"))
        msg = acc.assistant_message()
        assert "tool_calls" not in msg
        # Content should also be absent rather than an empty string.
        assert msg.get("content") is None

    def test_refusal_accumulates(self) -> None:
        acc = AssistantMessageAccumulator()
        acc.ingest(_packet(refusal="I "))
        acc.ingest(_packet(refusal="cannot"))
        msg = acc.assistant_message()
        assert msg["refusal"] == "I cannot"

    def test_accepts_bare_candidate_and_dict(self) -> None:
        acc = AssistantMessageAccumulator()
        acc.ingest(LLMChatCandidateChunk(content="a"))
        acc.ingest({"result": {"content": "b"}})
        acc.ingest({"content": "c"})
        assert acc.assistant_message()["content"] == "abc"

    def test_rejects_unknown_type(self) -> None:
        acc = AssistantMessageAccumulator()
        with pytest.raises(TypeError):
            acc.ingest(object())

    def test_metadata_tracked(self) -> None:
        acc = AssistantMessageAccumulator()
        acc.ingest(
            LLMChatResponseChunk(
                result=LLMChatCandidateChunk(content="x"),
                metadata={"model": "gpt-4o", "usage": {"total_tokens": 5}},
            )
        )
        assert acc.last_metadata["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------


class TestStreamEmitterConsume:
    def test_emits_start_deltas_complete(self) -> None:
        listener = _CapturingListener()
        emitter = _make_emitter(listener)
        final = emitter.consume(_content_stream(["Hi ", "there!"]))

        types = [c.type for c in listener.chunks]
        assert types[0] is StreamChunkType.START
        assert types[-1] is StreamChunkType.TURN_COMPLETE
        # At least two content deltas (Hi , there!)
        assert types.count(StreamChunkType.CONTENT_DELTA) == 2
        # Final AssistantMessage returned
        assert final == {"role": "assistant", "content": "Hi there!"}
        # TURN_COMPLETE should NOT carry complete_message by default
        final_chunk = listener.chunks[-1]
        assert final_chunk.complete_message is None

    def test_include_complete_message(self) -> None:
        listener = _CapturingListener()
        emitter = _make_emitter(listener, include_complete_message=True)
        emitter.consume(_content_stream(["hello"]))
        assert listener.chunks[-1].complete_message == {
            "role": "assistant",
            "content": "hello",
        }

    def test_monotonic_sequence(self) -> None:
        listener = _CapturingListener()
        emitter = _make_emitter(listener)
        emitter.consume(_content_stream(["a", "b", "c"]))
        seqs = [c.sequence for c in listener.chunks]
        assert seqs == list(range(1, len(seqs) + 1))

    def test_skips_empty_preroll(self) -> None:
        # Role-only chunk should not produce a delta event.
        listener = _CapturingListener()
        emitter = _make_emitter(listener)
        emitter.consume([_packet(role="assistant")])
        # START + TURN_COMPLETE only, no deltas
        types = [c.type for c in listener.chunks]
        assert types == [StreamChunkType.START, StreamChunkType.TURN_COMPLETE]

    def test_tool_call_stream_produces_tool_call_deltas(self) -> None:
        listener = _CapturingListener()
        emitter = _make_emitter(listener)
        final = emitter.consume(_tool_call_stream())

        types = [c.type for c in listener.chunks]
        assert StreamChunkType.TOOL_CALL_DELTA in types
        assert types[0] is StreamChunkType.START
        assert types[-1] is StreamChunkType.TURN_COMPLETE
        assert final["tool_calls"][0]["function"]["name"] == "get_weather"
        assert final["tool_calls"][0]["function"]["arguments"] == '{"city": "NYC"}'

    def test_exception_emits_error_and_raises(self) -> None:
        listener = _CapturingListener()
        emitter = _make_emitter(listener)

        def bad_stream() -> Iterable[LLMChatResponseChunk]:
            yield _packet(content="start")
            raise RuntimeError("llm blew up")

        with pytest.raises(RuntimeError):
            emitter.consume(bad_stream())

        assert listener.chunks[-1].type is StreamChunkType.ERROR
        assert listener.chunks[-1].error["type"] == "RuntimeError"
        assert listener.chunks[-1].error["message"] == "llm blew up"

    def test_listener_emit_error_does_not_abort_stream(self) -> None:
        """A listener that raises in emit must not break the LLM read loop."""

        class BrokenListener:
            def __init__(self) -> None:
                self.count = 0

            def emit(self, chunk: AgentStreamChunk) -> None:
                self.count += 1
                raise RuntimeError("listener broken")

            def close(self) -> None:
                pass

        listener = BrokenListener()
        emitter = StreamEmitter(
            listener=listener,
            agent_name="a",
            workflow_instance_id="w",
            turn=0,
            root_instance_id="r",
        )
        final = emitter.consume(_content_stream(["hi"]))
        assert final["content"] == "hi"
        # START + 1 delta + TURN_COMPLETE = 3 emits
        assert listener.count == 3


class TestStreamEmitterConsumeNonStreaming:
    def test_start_plus_single_complete_default_no_body(self) -> None:
        """Default: complete_message is None (consumers fetch from state).

        This is the opt-in contract — broadcasting the full assistant
        message to every pub/sub subscriber on the session topic without
        the caller asking for it leaks content on shared brokers.
        """
        listener = _CapturingListener()
        emitter = _make_emitter(listener)  # include_complete_message=False
        result = emitter.consume_non_streaming(
            {"role": "assistant", "content": "final"},
        )
        types = [c.type for c in listener.chunks]
        assert types == [StreamChunkType.START, StreamChunkType.TURN_COMPLETE]
        assert listener.chunks[-1].complete_message is None
        assert result == {"role": "assistant", "content": "final"}

    def test_start_plus_single_complete_opt_in_carries_body(self) -> None:
        listener = _CapturingListener()
        emitter = _make_emitter(listener, include_complete_message=True)
        emitter.consume_non_streaming(
            {"role": "assistant", "content": "final"},
        )
        assert listener.chunks[-1].complete_message == {
            "role": "assistant",
            "content": "final",
        }


class TestStreamEmitterAttribution:
    def test_attribution_fields_propagated(self) -> None:
        listener = _CapturingListener()
        emitter = StreamEmitter(
            listener=listener,
            agent_name="bob",
            workflow_instance_id="wf-child",
            turn=2,
            root_instance_id="root-123",
            parent_agent="alice",
            parent_instance_id="wf-parent",
            depth=1,
            call_path=["alice", "bob"],
            phase="routing",
            trace_parent="00-trace-span-01",
        )
        emitter.consume(_content_stream(["x"]))

        first = listener.chunks[0]
        assert first.agent == "bob"
        assert first.workflow_instance_id == "wf-child"
        assert first.turn == 2
        assert first.root_instance_id == "root-123"
        assert first.parent_agent == "alice"
        assert first.parent_instance_id == "wf-parent"
        assert first.depth == 1
        assert first.call_path == ["alice", "bob"]
        assert first.phase == "routing"
        assert first.trace_parent == "00-trace-span-01"
