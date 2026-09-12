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

"""Compliance tests for Hugging Face streaming chunk normalization."""

from unittest.mock import MagicMock

from dapr_agents.llm.huggingface.utils import process_hf_stream
from tests.llm.streaming_test_harness import StreamingComplianceHarness


def _packet(data: dict) -> MagicMock:
    """Wrap a dict as a Hugging Face stream chunk exposing ``model_dump``."""
    pkt = MagicMock()
    pkt.model_dump.return_value = data
    return pkt


def _content_packet(content: str, *, finish_reason=None, role=None) -> MagicMock:
    delta: dict = {"content": content}
    if role:
        delta["role"] = role
    return _packet(
        {
            "id": "hf-1",
            "created": 1,
            "model": "meta-llama/Llama-3.1-8B-Instruct",
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
            "id": "hf-1",
            "created": 1,
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
    )


def _usage_only_packet() -> MagicMock:
    return _packet(
        {
            "id": "hf-1",
            "created": 1,
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "choices": [],
            "usage": {
                "completion_tokens": 12,
                "prompt_tokens": 30,
                "total_tokens": 42,
            },
        }
    )


def test_hf_stream_content_reconstructs_message():
    """Hugging Face content stream reconstructs full message and meets compliance contract."""
    raw = [
        _content_packet("Hello", role="assistant"),
        _content_packet(" world", finish_reason="stop"),
        _usage_only_packet(),
    ]

    chunks = list(
        process_hf_stream(
            iter(raw), enrich_metadata={"provider": "huggingface"}, on_chunk=None
        )
    )

    StreamingComplianceHarness.assert_valid_stream_contract(
        chunks, expected_provider="huggingface", expected_min_chunks=3
    )
    StreamingComplianceHarness.assert_reconstructs_text(
        chunks, expected_text="Hello world"
    )

    final = chunks[-1]
    assert final.result.content is None
    assert final.result.finish_reason is None
    assert final.metadata is not None
    assert final.metadata.get("model") == "meta-llama/Llama-3.1-8B-Instruct"


def test_hf_stream_tool_calls_reconstruct():
    """Hugging Face tool call chunks reconstruct into full ToolCall."""
    raw = [
        _tool_call_packet(
            [
                {
                    "index": 0,
                    "id": "call_hf_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": ""},
                }
            ],
            role="assistant",
        ),
        _tool_call_packet(
            [
                {
                    "index": 0,
                    "function": {"arguments": '{"query": "dapr"}'},
                }
            ],
            finish_reason="tool_calls",
        ),
        _usage_only_packet(),
    ]

    chunks = list(
        process_hf_stream(
            iter(raw), enrich_metadata={"provider": "huggingface"}, on_chunk=None
        )
    )

    StreamingComplianceHarness.assert_valid_stream_contract(
        chunks, expected_provider="huggingface"
    )
    StreamingComplianceHarness.assert_reconstructs_tool_calls(
        chunks,
        expected_tools=[{"name": "lookup", "arguments": {"query": "dapr"}}],
        expected_finish_reason="tool_calls",
    )


def test_hf_stream_on_chunk_callback_lifecycle():
    """Verify on_chunk callback lifecycle on Hugging Face stream."""
    raw = [
        _content_packet("Chunk1", role="assistant"),
        _content_packet(" Chunk2", finish_reason="stop"),
        _usage_only_packet(),
    ]

    StreamingComplianceHarness.assert_on_chunk_callback_lifecycle(
        lambda on_chunk: process_hf_stream(
            iter(raw), enrich_metadata={"provider": "huggingface"}, on_chunk=on_chunk
        )
    )


def test_hf_stream_snapshot_isolation():
    """Verify snapshot isolation on Hugging Face stream."""
    raw = [
        _content_packet("A", role="assistant"),
        _content_packet(" B", finish_reason="stop"),
        _usage_only_packet(),
    ]

    chunks = list(
        process_hf_stream(
            iter(raw), enrich_metadata={"provider": "huggingface"}, on_chunk=None
        )
    )
    StreamingComplianceHarness.assert_snapshot_isolation(chunks)
