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

"""Tests for centralized stream processing and StreamHandler."""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch
import pytest

from dapr_agents.llm.huggingface.utils import process_hf_stream
from dapr_agents.llm.openai.utils import process_openai_stream
from dapr_agents.llm.utils.providers import PROVIDERS_WITH_STREAMING
from dapr_agents.llm.utils.stream import (
    StreamHandler,
    extract_packet_metadata,
    managed_stream,
    process_choice_delta,
    process_choice_delta_stream,
)
from dapr_agents.types.message import LLMChatResponseChunk
from tests.llm.streaming_test_harness import StreamingComplianceHarness


def _mock_packet(data: dict) -> MagicMock:
    pkt = MagicMock()
    pkt.model_dump.return_value = data
    return pkt


@dataclass
class _DataclassPacket:
    id: str
    model: str
    choices: list


class _ToDictPacket:
    def __init__(self, data: dict):
        self._data = data

    def to_dict(self) -> dict:
        return self._data


def test_managed_stream_with_context_manager():
    """managed_stream enters and exits context managers cleanly."""
    entered = False
    exited = False

    class CM:
        def __enter__(self):
            nonlocal entered
            entered = True
            return ["chunk1", "chunk2"]

        def __exit__(self, exc_type, exc_val, exc_tb):
            nonlocal exited
            exited = True

    cm = CM()
    with managed_stream(cm) as s:
        assert entered is True
        assert exited is False
        assert list(s) == ["chunk1", "chunk2"]
    assert exited is True


def test_managed_stream_with_closable():
    """managed_stream closes streams that have a close() method."""
    close_called = False

    class Closable:
        def __iter__(self):
            yield 1
            yield 2

        def close(self):
            nonlocal close_called
            close_called = True

    closable = Closable()
    with managed_stream(closable) as s:
        assert list(s) == [1, 2]
        assert close_called is False
    assert close_called is True


def test_managed_stream_plain_iterator():
    """managed_stream yields plain iterators unchanged without error."""
    plain = [1, 2, 3]
    with managed_stream(iter(plain)) as s:
        assert list(s) == [1, 2, 3]


def test_managed_stream_reraises_close_failure():
    """managed_stream logs and re-raises exceptions from close()."""

    class BadClose:
        def close(self):
            raise RuntimeError("close failed")

    with pytest.raises(RuntimeError, match="close failed"):
        with managed_stream(BadClose()):
            pass


def test_managed_stream_nested_idempotency():
    """Nested managed_stream contexts avoid double entering or double closing."""
    close_mock = MagicMock()

    class Closable:
        def close(self):
            close_mock()

    closable = Closable()
    with managed_stream(closable) as s1:
        with managed_stream(s1) as s2:
            assert s1 is s2
            assert close_mock.call_count == 0
        # Inner exit did not close early
        assert close_mock.call_count == 0
    # Outer exit closes once
    assert close_mock.call_count == 1


def test_providers_with_streaming_contains_all_supported():
    """PROVIDERS_WITH_STREAMING contains all providers supported by StreamHandler."""
    expected = {
        "openai",
        "nvidia",
        "litellm",
        "anthropic",
        "claude",
        "huggingface",
        "iflytek",
    }
    assert set(PROVIDERS_WITH_STREAMING) == expected


@pytest.mark.parametrize(
    "provider",
    ["openai", "nvidia", "litellm", "iflytek", "huggingface"],
)
def test_stream_handler_choice_delta_providers(provider: str):
    """StreamHandler routes choice-delta providers through consolidated processing."""
    raw = [
        _mock_packet(
            {
                "id": f"{provider}-1",
                "model": "model-abc",
                "choices": [
                    {"index": 0, "delta": {"content": "Hello", "role": "assistant"}}
                ],
            }
        ),
        _mock_packet(
            {
                "id": f"{provider}-1",
                "model": "model-abc",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": " world"},
                        "finish_reason": "stop",
                    }
                ],
            }
        ),
        _mock_packet(
            {
                "id": f"{provider}-1",
                "model": "model-abc",
                "choices": [],
                "usage": {"total_tokens": 10},
            }
        ),
    ]

    chunks = list(StreamHandler.process_stream(iter(raw), llm_provider=provider))

    StreamingComplianceHarness.assert_valid_stream_contract(
        chunks, expected_provider=provider, expected_min_chunks=3
    )
    StreamingComplianceHarness.assert_reconstructs_text(
        chunks, expected_text="Hello world"
    )


def test_stream_handler_anthropic_routing():
    """StreamHandler routes anthropic and claude providers to process_anthropic_stream."""
    dummy_chunk = MagicMock(spec=LLMChatResponseChunk)
    with patch(
        "dapr_agents.llm.anthropic.utils.process_anthropic_stream",
        return_value=iter([dummy_chunk]),
    ) as mock_proc:
        chunks = list(StreamHandler.process_stream(iter([]), llm_provider="anthropic"))
        assert chunks == [dummy_chunk]
        mock_proc.assert_called_once_with(
            raw_stream=pytest.approx(mock_proc.call_args[1]["raw_stream"]),
            enrich_metadata={"provider": "anthropic"},
            on_chunk=None,
        )

    with patch(
        "dapr_agents.llm.anthropic.utils.process_anthropic_stream",
        return_value=iter([dummy_chunk]),
    ) as mock_proc:
        chunks = list(StreamHandler.process_stream(iter([]), llm_provider="claude"))
        assert chunks == [dummy_chunk]
        mock_proc.assert_called_once_with(
            raw_stream=pytest.approx(mock_proc.call_args[1]["raw_stream"]),
            enrich_metadata={"provider": "claude"},
            on_chunk=None,
        )


def test_stream_handler_unsupported_provider():
    """StreamHandler raises ValueError for unsupported providers."""
    with pytest.raises(
        ValueError, match="Streaming not supported for provider: unknown"
    ):
        list(StreamHandler.process_stream(iter([]), llm_provider="unknown"))


def test_process_choice_delta_stream_packet_types():
    """process_choice_delta_stream accepts model_dump, to_dict, and dataclass packets."""
    raw = [
        _ToDictPacket(
            {
                "id": "dict-1",
                "choices": [{"index": 0, "delta": {"content": "Part1"}}],
            }
        ),
        _DataclassPacket(
            id="dc-1",
            model="test-model",
            choices=[
                {"index": 0, "delta": {"content": "Part2"}, "finish_reason": "stop"}
            ],
        ),
    ]

    chunks = list(process_choice_delta_stream(iter(raw)))
    assert len(chunks) == 2
    StreamingComplianceHarness.assert_reconstructs_text(
        chunks, expected_text="Part1Part2"
    )


def test_process_choice_delta_stream_invalid_packet_type():
    """process_choice_delta_stream raises TypeError for unparseable packets."""
    with pytest.raises(TypeError, match="Cannot serialize packet of type"):
        list(process_choice_delta_stream(iter(["invalid_packet_string"])))


def test_extract_packet_metadata_error_handling():
    """extract_packet_metadata safely falls back to empty dict on malformed packets."""
    bad_packet = MagicMock()
    bad_packet.get.side_effect = RuntimeError("Extraction failed")

    res = extract_packet_metadata(bad_packet)
    assert res == {}


def test_extract_packet_metadata_with_usage():
    """extract_packet_metadata preserves usage when present in packet."""
    pkt = {
        "id": "test-usage-1",
        "model": "test-model",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    meta = extract_packet_metadata(pkt, {"provider": "openai"})
    assert meta["id"] == "test-usage-1"
    assert meta["provider"] == "openai"
    assert meta["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }


def test_stream_handler_case_insensitivity():
    """StreamHandler routes providers regardless of string casing."""
    raw = [
        _mock_packet(
            {
                "id": "test-case-1",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Test case"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )
    ]
    chunks = list(StreamHandler.process_stream(iter(raw), llm_provider="OpenAI"))
    assert len(chunks) == 1
    assert chunks[0].metadata.get("provider") == "openai"

    dummy_chunk = MagicMock(spec=LLMChatResponseChunk)
    with patch(
        "dapr_agents.llm.anthropic.utils.process_anthropic_stream",
        return_value=iter([dummy_chunk]),
    ) as mock_proc:
        chunks_anthropic = list(
            StreamHandler.process_stream(iter([]), llm_provider="Anthropic")
        )
        assert chunks_anthropic == [dummy_chunk]
        assert mock_proc.call_args[1]["enrich_metadata"] == {"provider": "anthropic"}


def test_process_choice_delta_stream_multiple_choices_in_packet():
    """process_choice_delta_stream yields chunks for all choices in a packet."""
    raw = [
        _mock_packet(
            {
                "id": "multi-choice-1",
                "choices": [
                    {"index": 0, "delta": {"content": "Candidate 0"}},
                    {"index": 1, "delta": {"content": "Candidate 1"}},
                ],
            }
        )
    ]
    chunks = list(process_choice_delta_stream(iter(raw)))
    assert len(chunks) == 2
    assert chunks[0].result.index == 0
    assert chunks[0].result.content == "Candidate 0"
    assert chunks[0].metadata.get("first_chunk") is True

    assert chunks[1].result.index == 1
    assert chunks[1].result.content == "Candidate 1"
    assert "first_chunk" not in chunks[1].metadata


def test_process_choice_delta_stream_closes_raw_stream():
    """process_choice_delta_stream closes raw_stream on normal exit and early break."""
    close_mock = MagicMock()

    class ClosableStream:
        def __iter__(self):
            yield _mock_packet(
                {"id": "c1", "choices": [{"index": 0, "delta": {"content": "1"}}]}
            )
            yield _mock_packet(
                {"id": "c2", "choices": [{"index": 0, "delta": {"content": "2"}}]}
            )

        def close(self):
            close_mock()

    # Normal completion
    list(process_choice_delta_stream(ClosableStream()))
    assert close_mock.call_count == 1

    # Early break
    close_mock.reset_mock()
    for chunk in process_choice_delta_stream(ClosableStream()):
        if chunk.result.content == "1":
            break
    assert close_mock.call_count == 1


def test_stream_handler_closes_raw_stream():
    """StreamHandler.process_stream closes raw_stream on normal exit and early break."""
    close_mock = MagicMock()

    class ClosableStream:
        def __iter__(self):
            yield _mock_packet(
                {"id": "c1", "choices": [{"index": 0, "delta": {"content": "1"}}]}
            )
            yield _mock_packet(
                {"id": "c2", "choices": [{"index": 0, "delta": {"content": "2"}}]}
            )

        def close(self):
            close_mock()

    # Normal completion
    list(StreamHandler.process_stream(ClosableStream(), llm_provider="openai"))
    assert close_mock.call_count == 1

    # Early break
    close_mock.reset_mock()
    for chunk in StreamHandler.process_stream(ClosableStream(), llm_provider="openai"):
        if chunk.result.content == "1":
            break
    assert close_mock.call_count == 1


def test_stream_handler_exits_context_manager():
    """StreamHandler.process_stream exits context manager stream on normal exit and early break."""
    exit_mock = MagicMock()

    class CMStream:
        def __enter__(self):
            return [
                _mock_packet(
                    {"id": "c1", "choices": [{"index": 0, "delta": {"content": "1"}}]}
                ),
                _mock_packet(
                    {"id": "c2", "choices": [{"index": 0, "delta": {"content": "2"}}]}
                ),
            ]

        def __exit__(self, exc_type, exc_val, exc_tb):
            exit_mock()

    # Normal completion
    list(StreamHandler.process_stream(CMStream(), llm_provider="openai"))
    assert exit_mock.call_count == 1

    # Early break
    exit_mock.reset_mock()
    for chunk in StreamHandler.process_stream(CMStream(), llm_provider="openai"):
        if chunk.result.content == "1":
            break
    assert exit_mock.call_count == 1


def test_stream_handler_closes_anthropic_stream():
    """StreamHandler.process_stream closes raw stream for Anthropic provider."""
    close_mock = MagicMock()

    class ClosableAnthropicStream:
        def __iter__(self):
            yield {
                "type": "message_start",
                "message": {
                    "id": "msg-1",
                    "role": "assistant",
                    "usage": {"input_tokens": 5, "output_tokens": 1},
                },
            }

        def close(self):
            close_mock()

    list(
        StreamHandler.process_stream(
            ClosableAnthropicStream(), llm_provider="anthropic"
        )
    )
    assert close_mock.call_count == 1


def test_stream_handler_unhandled_streaming_provider():
    """StreamHandler raises ValueError defensively if a provider in PROVIDERS_WITH_STREAMING is unhandled."""
    with patch(
        "dapr_agents.llm.utils.stream.PROVIDERS_WITH_STREAMING",
        ("mock_future_provider",),
    ):
        with pytest.raises(
            ValueError,
            match="Streaming not supported for provider: mock_future_provider",
        ):
            list(
                StreamHandler.process_stream(
                    iter([]), llm_provider="mock_future_provider"
                )
            )


def test_process_choice_delta_malformed_tool_call():
    """process_choice_delta safely ignores malformed tool_call entries without crashing."""
    choice = {
        "index": 0,
        "delta": {
            "tool_calls": [
                "not-a-dict-tool-call",
                {"index": 0, "function": {"name": "valid_fn", "arguments": "{}"}},
            ]
        },
    }
    chunks = list(process_choice_delta(choice, overall_meta={}))
    assert len(chunks) == 1
    assert len(chunks[0].result.tool_calls) == 1
    assert chunks[0].result.tool_calls[0].function.name == "valid_fn"
