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

from dapr_agents.llm.openai.utils import process_openai_stream
from dapr_agents.types.message import LLMChatResponseChunk
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
