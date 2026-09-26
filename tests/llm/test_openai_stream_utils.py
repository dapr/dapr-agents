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

"""Regression and compliance tests for OpenAI streaming chunk normalization."""

from unittest.mock import MagicMock
import pytest

from dapr_agents.llm.openai.utils import process_openai_stream
from dapr_agents.llm.utils.stream import StreamHandler
from dapr_agents.types.message import FunctionCall, LLMChatResponseChunk
from tests.llm.streaming_test_harness import StreamingComplianceHarness


def _packet(data: dict) -> MagicMock:
    """Wrap a dict as an OpenAI SDK chunk exposing ``model_dump``."""
    pkt = MagicMock()
    pkt.model_dump.return_value = data
    return pkt


def _content_packet(content: str, *, finish_reason=None, role=None) -> MagicMock:
    delta: dict = {"content": content}
    if role:
        delta["role"] = role
    return _packet(
        {
            "id": "chatcmpl-1",
            "created": 1,
            "model": "gpt-4o-mini",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
    )


def _tool_call_packet(
    tool_calls: list[dict], *, finish_reason=None, role=None
) -> MagicMock:
    delta: dict = {"tool_calls": tool_calls}
    if role:
        delta["role"] = role
    return _packet(
        {
            "id": "chatcmpl-1",
            "created": 1,
            "model": "gpt-4o-mini",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
    )


def _usage_only_packet() -> MagicMock:
    # OpenAI's terminal packet when include_usage is on: empty choices + usage.
    return _packet(
        {
            "id": "chatcmpl-1",
            "created": 1,
            "model": "gpt-4o-mini",
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": {
                "completion_tokens": 238,
                "prompt_tokens": 638,
                "total_tokens": 876,
            },
        }
    )


def test_usage_only_final_packet_yields_valid_chunk():
    """The empty-choices usage packet must not crash chunk construction."""
    raw = [
        _content_packet("Hello", role="assistant"),
        _content_packet(" world", finish_reason="stop"),
        _usage_only_packet(),
    ]

    chunks = list(
        process_openai_stream(
            iter(raw), enrich_metadata={"provider": "openai"}, on_chunk=None
        )
    )

    # Validate against compliance harness contract
    StreamingComplianceHarness.assert_valid_stream_contract(
        chunks, expected_provider="openai", expected_min_chunks=3
    )

    # The terminal usage packet carries an empty candidate (no content/finish)
    # but preserves provider metadata for downstream TURN_COMPLETE attribution.
    final = chunks[-1]
    assert final.result.content is None
    assert final.result.finish_reason is None
    assert final.metadata is not None
    assert final.metadata.get("model") == "gpt-4o-mini"
    assert final.metadata.get("usage") == {
        "completion_tokens": 238,
        "prompt_tokens": 638,
        "total_tokens": 876,
    }


def test_content_chunks_reconstruct_full_message():
    """Content deltas accumulate correctly alongside the usage-only tail."""
    raw = [
        _content_packet("Hel", role="assistant"),
        _content_packet("lo", finish_reason="stop"),
        _usage_only_packet(),
    ]

    chunks = list(
        process_openai_stream(
            iter(raw), enrich_metadata={"provider": "openai"}, on_chunk=None
        )
    )
    StreamingComplianceHarness.assert_valid_stream_contract(
        chunks, expected_provider="openai"
    )
    StreamingComplianceHarness.assert_reconstructs_text(chunks, expected_text="Hello")


def test_tool_call_chunks_reconstruct_tool_call():
    """Tool call deltas reconstruct into complete ToolCall with arguments."""
    raw = [
        _tool_call_packet(
            [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": ""},
                }
            ],
            role="assistant",
        ),
        _tool_call_packet(
            [
                {
                    "index": 0,
                    "function": {"arguments": '{"location": "San Francisco"}'},
                }
            ],
            finish_reason="tool_calls",
        ),
        _usage_only_packet(),
    ]

    chunks = list(
        process_openai_stream(
            iter(raw), enrich_metadata={"provider": "openai"}, on_chunk=None
        )
    )
    StreamingComplianceHarness.assert_valid_stream_contract(
        chunks, expected_provider="openai"
    )
    StreamingComplianceHarness.assert_reconstructs_tool_calls(
        chunks,
        expected_tools=[
            {"name": "get_weather", "arguments": {"location": "San Francisco"}}
        ],
        expected_finish_reason="tool_calls",
    )


def test_openai_stream_on_chunk_callback_lifecycle():
    """Verify on_chunk callback receives every yielded chunk in order."""
    raw = [
        _content_packet("Hi", role="assistant"),
        _content_packet(" there", finish_reason="stop"),
        _usage_only_packet(),
    ]

    StreamingComplianceHarness.assert_on_chunk_callback_lifecycle(
        lambda on_chunk: process_openai_stream(
            iter(raw), enrich_metadata={"provider": "openai"}, on_chunk=on_chunk
        )
    )


def test_openai_stream_snapshot_isolation():
    """Verify earlier chunk metadata dicts are not mutated by later events."""
    raw = [
        _content_packet("One", role="assistant"),
        _content_packet(" Two", finish_reason="stop"),
        _usage_only_packet(),
    ]

    chunks = list(
        process_openai_stream(
            iter(raw), enrich_metadata={"provider": "openai"}, on_chunk=None
        )
    )
    StreamingComplianceHarness.assert_snapshot_isolation(chunks)


@pytest.mark.parametrize(
    "finish_reason", ["length", "content_filter", "function_call", "custom_finish"]
)
def test_openai_stream_arbitrary_finish_reasons(finish_reason: str):
    """Any non-None finish reason marks meta['last_chunk'] = True and sets finish_reason."""
    raw = [
        _content_packet("Truncated", finish_reason=finish_reason),
    ]
    chunks = list(
        process_openai_stream(iter(raw), enrich_metadata={"provider": "openai"})
    )
    assert len(chunks) == 1
    assert chunks[0].result.finish_reason == finish_reason
    assert chunks[0].metadata.get("last_chunk") is True


def test_openai_stream_plain_dictionaries():
    """Stream of plain Python dictionaries is normalized correctly."""
    raw = [
        {
            "id": "chatcmpl-plain-1",
            "model": "gpt-4o",
            "choices": [
                {"index": 0, "delta": {"content": "Hello", "role": "assistant"}}
            ],
        },
        {
            "id": "chatcmpl-plain-1",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " world"},
                    "finish_reason": "stop",
                }
            ],
        },
    ]
    chunks = list(
        process_openai_stream(iter(raw), enrich_metadata={"provider": "openai"})
    )
    StreamingComplianceHarness.assert_valid_stream_contract(
        chunks, expected_provider="openai"
    )
    StreamingComplianceHarness.assert_reconstructs_text(
        chunks, expected_text="Hello world"
    )
    assert chunks[0].metadata["id"] == "chatcmpl-plain-1"


def test_openai_stream_sdk_like_objects():
    """Stream of class instances with attribute access (SDK-like objects) works directly."""

    class _Delta:
        def __init__(self, content=None, role=None):
            self.content = content
            self.role = role
            self.tool_calls = None
            self.function_call = None
            self.refusal = None

    class _Choice:
        def __init__(self, index, delta, finish_reason=None):
            self.index = index
            self.delta = delta
            self.finish_reason = finish_reason
            self.logprobs = None

    class _Chunk:
        def __init__(self, choices, id="chunk-sdk-1", model="gpt-4o"):
            self.id = id
            self.model = model
            self.choices = choices
            self.usage = None

    raw = [
        _Chunk([_Choice(0, _Delta("SDK ", role="assistant"))]),
        _Chunk([_Choice(0, _Delta("stream"), finish_reason="stop")]),
    ]
    chunks = list(
        process_openai_stream(iter(raw), enrich_metadata={"provider": "openai"})
    )
    StreamingComplianceHarness.assert_reconstructs_text(
        chunks, expected_text="SDK stream"
    )
    assert chunks[0].metadata["id"] == "chunk-sdk-1"


def test_openai_stream_to_dict_wrapper_objects():
    """Stream of objects exposing to_dict() method are unpacked correctly."""

    class _ToDictWrapper:
        def __init__(self, data: dict):
            self._data = data

        def to_dict(self):
            return self._data

    raw = [
        _ToDictWrapper(
            {
                "id": "cmpl-td-1",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Wrapped", "role": "assistant"},
                    }
                ],
            }
        ),
        _ToDictWrapper(
            {
                "id": "cmpl-td-1",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": " to_dict"},
                        "finish_reason": "stop",
                    }
                ],
            }
        ),
    ]
    chunks = list(
        process_openai_stream(iter(raw), enrich_metadata={"provider": "openai"})
    )
    StreamingComplianceHarness.assert_reconstructs_text(
        chunks, expected_text="Wrapped to_dict"
    )
    assert chunks[0].metadata["id"] == "cmpl-td-1"


def test_openai_stream_malformed_packets_and_choices():
    """Malformed choices (None, missing delta, invalid tool_call) are handled defensively."""
    raw = [
        # Packet with choices containing None
        {
            "id": "cmpl-malformed-1",
            "model": "gpt-4o",
            "choices": [None, {"index": 0, "delta": {"content": "Good"}}],
        },
        # Packet with choice having delta=None
        {
            "id": "cmpl-malformed-1",
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": None}],
        },
        # Packet with malformed tool call
        {
            "id": "cmpl-malformed-1",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            "invalid_string_not_dict_or_object",
                            {
                                "index": 0,
                                "function": {
                                    "name": "valid_fn",
                                    "arguments": "{}",
                                },
                            },
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
    ]
    chunks = list(
        process_openai_stream(iter(raw), enrich_metadata={"provider": "openai"})
    )
    assert len(chunks) == 3
    assert chunks[0].result.content == "Good"
    assert chunks[1].result.content is None
    # Third chunk dropped the invalid string tool call and kept valid_fn
    assert len(chunks[2].result.tool_calls) == 1
    assert chunks[2].result.tool_calls[0].function.name == "valid_fn"


def test_openai_stream_varied_delta_types():
    """Stream handles role-only, content-only, refusal, and empty/None deltas."""
    raw = [
        # Role-only (content=None)
        {
            "id": "cmpl-var-1",
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"role": "assistant"}}],
        },
        # Content-only
        {
            "id": "cmpl-var-1",
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"content": "Hello"}}],
        },
        # Refusal delta
        {
            "id": "cmpl-var-1",
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"refusal": "I cannot answer this."}}],
        },
        # Empty delta
        {
            "id": "cmpl-var-1",
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    chunks = list(
        process_openai_stream(iter(raw), enrich_metadata={"provider": "openai"})
    )
    assert len(chunks) == 4
    assert chunks[0].result.role == "assistant"
    assert chunks[0].result.content is None
    assert chunks[1].result.content == "Hello"
    assert chunks[2].result.refusal == "I cannot answer this."
    assert chunks[3].result.finish_reason == "stop"


def test_openai_stream_legacy_function_call():
    """Legacy function_call deltas (dict or SDK object) are preserved in candidate chunk."""

    class _SDKFnCall:
        def __init__(self, name, args):
            self.name = name
            self.arguments = args

    raw = [
        {
            "id": "cmpl-fn-1",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "function_call": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    },
                    "finish_reason": "function_call",
                }
            ],
        },
        {
            "id": "cmpl-fn-2",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "function_call": _SDKFnCall("get_stock", '{"ticker": "MSFT"}'),
                    },
                    "finish_reason": "function_call",
                }
            ],
        },
    ]
    chunks = list(
        process_openai_stream(iter(raw), enrich_metadata={"provider": "openai"})
    )
    assert len(chunks) == 2
    fn_call_1 = chunks[0].result.function_call
    assert fn_call_1 == {"name": "get_weather", "arguments": '{"city": "Paris"}'}
    assert chunks[0].result.finish_reason == "function_call"
    assert chunks[0].metadata.get("last_chunk") is True

    fn_call_2 = chunks[1].result.function_call
    assert fn_call_2 == {"name": "get_stock", "arguments": '{"ticker": "MSFT"}'}
    assert chunks[1].result.finish_reason == "function_call"


def test_openai_stream_multiple_interleaved_tool_calls():
    """Multiple tool calls with interleaved indices accumulate correctly."""
    raw = [
        # Call 0 start
        _tool_call_packet(
            [
                {
                    "index": 0,
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "calc", "arguments": ""},
                }
            ],
            role="assistant",
        ),
        # Call 1 start
        _tool_call_packet(
            [
                {
                    "index": 1,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": ""},
                }
            ],
        ),
        # Call 0 args
        _tool_call_packet(
            [{"index": 0, "function": {"arguments": '{"expr": "1+1"}'}}],
        ),
        # Call 1 args
        _tool_call_packet(
            [{"index": 1, "function": {"arguments": '{"q": "python"}'}}],
            finish_reason="tool_calls",
        ),
        _usage_only_packet(),
    ]
    chunks = list(
        process_openai_stream(iter(raw), enrich_metadata={"provider": "openai"})
    )
    StreamingComplianceHarness.assert_valid_stream_contract(
        chunks, expected_provider="openai"
    )
    StreamingComplianceHarness.assert_reconstructs_tool_calls(
        chunks,
        expected_tools=[
            {"name": "calc", "arguments": {"expr": "1+1"}},
            {"name": "search", "arguments": {"q": "python"}},
        ],
        expected_finish_reason="tool_calls",
    )


def test_openai_stream_callback_exception_propagates_and_cleans_up():
    """An exception raised inside on_chunk callback propagates and stream is closed."""
    close_mock = MagicMock()

    class ClosableStream:
        def __iter__(self):
            yield _content_packet("Hello", role="assistant")
            yield _content_packet(" world")

        def close(self):
            close_mock()

    def bad_callback(chunk):
        raise ValueError("Callback exploded")

    with pytest.raises(ValueError, match="Callback exploded"):
        list(
            StreamHandler.process_stream(
                ClosableStream(), llm_provider="openai", on_chunk=bad_callback
            )
        )
    assert close_mock.call_count == 1


def test_openai_stream_iteration_exception_propagates_and_cleans_up():
    """An exception during stream iteration propagates and underlying stream is closed."""
    close_mock = MagicMock()

    class FaultyStream:
        def __iter__(self):
            yield _content_packet("Hello", role="assistant")
            raise ConnectionResetError("Connection lost")

        def close(self):
            close_mock()

    with pytest.raises(ConnectionResetError, match="Connection lost"):
        list(StreamHandler.process_stream(FaultyStream(), llm_provider="openai"))
    assert close_mock.call_count == 1


def test_openai_stream_close_failure_logged_and_reraised():
    """Failure during stream close() propagates."""

    class FailingCloseStream:
        def __iter__(self):
            yield _content_packet("Hello")

        def close(self):
            raise RuntimeError("close failed")

    with pytest.raises(RuntimeError, match="close failed"):
        list(StreamHandler.process_stream(FailingCloseStream(), llm_provider="openai"))


def test_openai_stream_context_manager_enter_returns_different_iterator():
    """StreamHandler supports context-manager streams returning a separate iterator from __enter__()."""
    exit_mock = MagicMock()
    packets = [_content_packet("CM content", role="assistant", finish_reason="stop")]

    class SDKStreamContextManager:
        def __enter__(self):
            return iter(packets)

        def __exit__(self, exc_type, exc_val, exc_tb):
            exit_mock()

    chunks = list(
        StreamHandler.process_stream(SDKStreamContextManager(), llm_provider="openai")
    )
    assert len(chunks) == 1
    assert chunks[0].result.content == "CM content"
    assert exit_mock.call_count == 1


def test_openai_stream_empty_stream():
    """An empty stream yields zero chunks without error."""
    chunks = list(
        process_openai_stream(iter([]), enrich_metadata={"provider": "openai"})
    )
    assert chunks == []


def test_openai_stream_metadata_enrichment_immutability():
    """Metadata enrichment must not mutate the caller's input enrich_metadata dictionary."""
    input_meta = {"provider": "openai", "tenant": "corp", "version": 1}
    raw = [
        _content_packet("Hello", role="assistant", finish_reason="stop"),
    ]
    chunks = list(process_openai_stream(iter(raw), enrich_metadata=input_meta))
    assert len(chunks) == 1
    # Caller input dictionary must be completely untouched
    assert input_meta == {"provider": "openai", "tenant": "corp", "version": 1}
    # Chunk metadata contains enriched keys plus extracted packet keys
    assert chunks[0].metadata["tenant"] == "corp"
    assert chunks[0].metadata["model"] == "gpt-4o-mini"


def test_openai_stream_usage_serialization_failure_preserves_metadata():
    """Failure to serialize usage object preserves id, model, and other metadata fields."""

    class BadUsage:
        def model_dump(self):
            raise RuntimeError("Cannot dump usage")

        def __iter__(self):
            raise RuntimeError("Cannot iter usage")

    bad_usage_pkt = {
        "id": "chatcmpl-preserve-1",
        "created": 9999,
        "model": "gpt-4o",
        "object": "chat.completion.chunk",
        "choices": [],
        "usage": BadUsage(),
    }
    chunks = list(
        process_openai_stream(
            iter([bad_usage_pkt]), enrich_metadata={"provider": "openai"}
        )
    )
    assert len(chunks) == 1
    meta = chunks[0].metadata
    assert meta["id"] == "chatcmpl-preserve-1"
    assert meta["model"] == "gpt-4o"
    assert meta["created"] == 9999
    assert meta["provider"] == "openai"
    assert "usage" not in meta  # usage failed, but non-usage metadata is retained
