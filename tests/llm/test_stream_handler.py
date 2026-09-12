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

from dapr_agents.llm.huggingface.utils import (
    _get_packet_metadata as hf_get_packet_metadata,
    _process_choice_delta as hf_process_choice_delta,
    process_hf_stream,
)
from dapr_agents.llm.openai.utils import (
    _get_packet_metadata as openai_get_packet_metadata,
    _process_choice_delta as openai_process_choice_delta,
    process_openai_stream,
)
from dapr_agents.llm.utils.stream import (
    StreamHandler,
    extract_packet_metadata,
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


def test_backward_compatible_shims():
    """Verify backward-compatible shims match centralized helpers."""
    assert openai_get_packet_metadata is extract_packet_metadata
    assert hf_get_packet_metadata is extract_packet_metadata
    assert openai_process_choice_delta is process_choice_delta
    assert hf_process_choice_delta is process_choice_delta

    pkt = {"id": "test-1", "model": "gpt-4o", "created": 123}
    meta = openai_get_packet_metadata(pkt, {"extra": "val"})
    assert meta["id"] == "test-1"
    assert meta["model"] == "gpt-4o"
    assert meta["extra"] == "val"


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
