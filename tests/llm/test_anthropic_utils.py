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
Unit tests for Anthropic utility functions and stream handling.

Covers message conversion, multimodal content block normalization,
response and stream translations, metadata isolation, and tests with
real SDK models from anthropic.types.
"""

import json
from collections.abc import Iterable
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
from anthropic.types import (
    Message,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    SignatureDelta,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
    Usage,
)
from anthropic.types.raw_message_delta_event import Delta, MessageDeltaUsage
from pydantic import BaseModel

from dapr_agents.llm.anthropic.utils import (
    SUPPORTED_IMAGE_MEDIA_TYPES,
    _normalize_content_blocks,
    _snapshot_stream_meta,
    as_assistant_with_tool_use,
    as_tool_result,
    parse_function_call_response,
    parse_json_response,
    process_anthropic_stream,
    resolve_target_model,
    split_messages,
    to_llm_chat_response,
)
from dapr_agents.llm.utils.response import ResponseHandler
from dapr_agents.llm.utils.stream import StreamHandler
from dapr_agents.types.message import (
    AssistantMessage,
    LLMChatResponse,
    LLMChatResponseChunk,
)
from tests.llm.streaming_test_harness import StreamingComplianceHarness


def _stream_cm(events: Iterable[Any]) -> Any:
    """Wrap an event iterable so it behaves like the SDK's Stream context manager."""
    return nullcontext(iter(events))


class TestNormalizeContentBlocks:
    """Edge cases and validations for _normalize_content_blocks."""

    def test_text_type_dict_passes_through(self) -> None:
        block = {"type": "text", "text": "hello"}
        assert _normalize_content_blocks([block]) == [block]

    def test_plain_string_items_wrapped_as_text_blocks(self) -> None:
        result = _normalize_content_blocks(["hello", "world"])
        assert result == [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]

    def test_corrupted_base64_data_uri_passes_through(self) -> None:
        corrupted = {"type": "image_url", "image_url": {"url": "data:not_a_valid_uri"}}
        result = _normalize_content_blocks([corrupted])
        assert result == [corrupted]

    def test_data_uri_missing_base64_marker_passes_through(self) -> None:
        url_encoded_svg = {
            "type": "image_url",
            "image_url": {"url": "data:image/svg+xml,<svg></svg>"},
        }
        result = _normalize_content_blocks([url_encoded_svg])
        assert result == [url_encoded_svg]

    def test_data_uri_unsupported_media_type_passes_through(self) -> None:
        unsupported_media = {
            "type": "image_url",
            "image_url": {"url": "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4="},
        }
        result = _normalize_content_blocks([unsupported_media])
        assert result == [unsupported_media]

    def test_data_uri_supported_media_types_converted(self) -> None:
        for media_type in SUPPORTED_IMAGE_MEDIA_TYPES:
            block = {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,AQIDBA=="},
            }
            result = _normalize_content_blocks([block])
            assert len(result) == 1
            converted = result[0]
            assert converted["type"] == "image"
            assert converted["source"]["type"] == "base64"
            assert converted["source"]["media_type"] == media_type
            assert converted["source"]["data"] == "AQIDBA=="

    def test_non_dict_non_string_items_pass_through(self) -> None:
        items = [123, True, None]
        assert _normalize_content_blocks(items) == items

    def test_unrecognized_dict_type_passes_through(self) -> None:
        unknown = {"type": "custom_block", "payload": "data"}
        assert _normalize_content_blocks([unknown]) == [unknown]

    def test_non_list_content_returned_as_is(self) -> None:
        assert _normalize_content_blocks("just a string") == "just a string"

    def test_image_url_with_unsupported_scheme_passes_through(self) -> None:
        ftp_block = {
            "type": "image_url",
            "image_url": {"url": "ftp://example.com/img.jpg"},
        }
        assert _normalize_content_blocks([ftp_block]) == [ftp_block]


class TestSplitMessages:
    """Tests for split_messages system prompt and cache control handling."""

    def test_string_system_message_extracted(self) -> None:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        system, rest = split_messages(messages)
        assert system == "You are a helpful assistant."
        assert len(rest) == 1
        assert rest[0]["role"] == "user"

    def test_structured_system_blocks_preserved(self) -> None:
        system_blocks = [
            {
                "type": "text",
                "text": "Cached system prompt",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        messages = [
            {"role": "system", "content": system_blocks},
            {"role": "user", "content": "Hello"},
        ]
        system, rest = split_messages(messages)
        assert system == system_blocks
        assert len(rest) == 1


class TestToolHelpers:
    """Tests for as_tool_result and as_assistant_with_tool_use."""

    def test_as_tool_result_valid(self) -> None:
        msg = as_tool_result({"tool_call_id": "call_1", "content": {"status": "ok"}})
        assert msg["role"] == "user"
        content = msg["content"]
        assert len(content) == 1
        assert content[0]["type"] == "tool_result"
        assert content[0]["tool_use_id"] == "call_1"
        assert json.loads(content[0]["content"]) == {"status": "ok"}

    def test_as_tool_result_missing_id_raises(self) -> None:
        with pytest.raises(ValueError, match=r"tool_call_id.*is required"):
            as_tool_result({"content": "result"})

    def test_as_assistant_with_tool_use_valid(self) -> None:
        msg = as_assistant_with_tool_use(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            }
        )
        assert msg["role"] == "assistant"
        content = msg["content"]
        assert len(content) == 1
        assert content[0]["type"] == "tool_use"
        assert content[0]["id"] == "call_1"
        assert content[0]["name"] == "get_weather"
        assert content[0]["input"] == {"city": "Paris"}

    def test_as_assistant_with_tool_use_missing_fields_raise(self) -> None:
        with pytest.raises(
            ValueError, match="both `id` and `function.name` are required"
        ):
            as_assistant_with_tool_use(
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "", "function": {"name": "test"}}],
                }
            )
        with pytest.raises(
            ValueError, match="both `id` and `function.name` are required"
        ):
            as_assistant_with_tool_use(
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "call_1", "function": {"name": ""}}],
                }
            )


class TestResolveTargetModel:
    """Tests for resolve_target_model type resolution."""

    def test_single_model(self) -> None:
        class Model(BaseModel):
            x: int

        fmt, cls = resolve_target_model(Model)
        assert fmt is Model
        assert cls is Model

    def test_list_of_model(self) -> None:
        class Item(BaseModel):
            x: int

        fmt, cls = resolve_target_model(list[Item])
        assert issubclass(cls, BaseModel)
        assert cls.__name__ == "IterableItem"

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(
            TypeError, match="response_format must resolve to a Pydantic model"
        ):
            resolve_target_model(int)  # type: ignore[arg-type]


class TestProcessAnthropicStream:
    """Tests for process_anthropic_stream behavior."""

    def test_stream_metadata_provider_fallback(self) -> None:
        """Provider is set to PROVIDER by default even when enrich_metadata has no provider key."""
        events = [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(id="m1", model="claude-3", usage=None),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text="Hi"),
            ),
        ]
        chunks = list(
            process_anthropic_stream(
                _stream_cm(events), enrich_metadata={"custom_key": "val"}
            )
        )
        assert len(chunks) == 1
        assert chunks[0].metadata["provider"] == "anthropic"
        assert chunks[0].metadata["custom_key"] == "val"

    def test_stream_metadata_provider_override(self) -> None:
        """enrich_metadata provider overrides default provider."""
        events = [
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text="Hi"),
            ),
        ]
        chunks = list(
            process_anthropic_stream(
                _stream_cm(events), enrich_metadata={"provider": "claude"}
            )
        )
        assert len(chunks) == 1
        assert chunks[0].metadata["provider"] == "claude"

    def test_stream_collects_multiple_thinking_signatures(self) -> None:
        """Multiple signature_delta events are collected in thinking_signatures."""
        events = [
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="thinking_delta", thinking="step 1"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="signature_delta", signature="sig_alpha"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=1,
                delta=SimpleNamespace(type="thinking_delta", thinking="step 2"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=1,
                delta=SimpleNamespace(type="signature_delta", signature="sig_beta"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=2,
                delta=SimpleNamespace(type="text_delta", text="Done"),
            ),
        ]
        chunks = list(process_anthropic_stream(_stream_cm(events)))
        assert len(chunks) == 1
        meta = chunks[0].metadata
        assert meta.get("thinking_signature") == "sig_beta"
        assert meta.get("thinking_signatures") == ["sig_alpha", "sig_beta"]
        assert meta.get("thinking_deltas") == ["step 1", "step 2"]

    def test_snapshot_isolation(self) -> None:
        """Yielded chunk metadata snapshots are isolated from subsequent events."""
        meta: dict[str, Any] = {
            "thinking_blocks": ["thought 1"],
            "thinking_deltas": ["d1"],
            "thinking_signatures": ["s1"],
        }
        snap = _snapshot_stream_meta(meta)
        meta["thinking_blocks"].append("thought 2")
        meta["thinking_deltas"].append("d2")
        meta["thinking_signatures"].append("s2")

        assert snap["thinking_blocks"] == ["thought 1"]
        assert snap["thinking_deltas"] == ["d1"]
        assert snap["thinking_signatures"] == ["s1"]


class TestRealSDKModelFixtures:
    """Tests using genuine anthropic.types classes to catch real SDK shape regressions."""

    def test_to_llm_chat_response_with_real_sdk_models(self) -> None:
        real_msg = Message(
            id="msg_real_001",
            model="claude-3-7-sonnet-20250219",
            role="assistant",
            type="message",
            content=[
                TextBlock(type="text", text="Here is your result."),
                ToolUseBlock(
                    type="tool_use",
                    id="tool_call_weather",
                    name="get_weather",
                    input={"location": "San Francisco, CA"},
                ),
                ThinkingBlock(
                    type="thinking",
                    thinking="Considering the current forecast.",
                    signature="sig_real_123",
                ),
            ],
            stop_reason="tool_use",
            stop_sequence=None,
            usage=Usage(input_tokens=42, output_tokens=18),
        )

        response = to_llm_chat_response(real_msg)
        assert isinstance(response, LLMChatResponse)
        assert response.metadata["id"] == "msg_real_001"
        assert response.metadata["model"] == "claude-3-7-sonnet-20250219"
        assert response.metadata["stop_reason"] == "tool_use"
        assert response.metadata["usage"] == {
            "cache_creation": None,
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": None,
            "inference_geo": None,
            "input_tokens": 42,
            "output_tokens": 18,
            "output_tokens_details": None,
            "server_tool_use": None,
            "service_tier": None,
        }
        assert len(response.metadata["raw_content"]) == 3

        assistant = response.get_message()
        assert isinstance(assistant, AssistantMessage)
        assert assistant.content == "Here is your result."
        assert assistant.tool_calls is not None
        assert len(assistant.tool_calls) == 1

        tool_call = assistant.tool_calls[0]
        assert tool_call.id == "tool_call_weather"
        assert tool_call.function.name == "get_weather"
        assert json.loads(tool_call.function.arguments) == {
            "location": "San Francisco, CA"
        }

    def test_process_anthropic_stream_with_real_sdk_events(self) -> None:
        real_events = [
            RawMessageStartEvent(
                type="message_start",
                message=Message(
                    id="msg_stream_real",
                    model="claude-3-7-sonnet-20250219",
                    role="assistant",
                    type="message",
                    content=[],
                    stop_reason=None,
                    stop_sequence=None,
                    usage=Usage(input_tokens=15, output_tokens=0),
                ),
            ),
            RawContentBlockStartEvent(
                type="content_block_start",
                index=0,
                content_block=ThinkingBlock(
                    type="thinking",
                    thinking="Starting real thinking block",
                    signature="sig_start",
                ),
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=ThinkingDelta(type="thinking_delta", thinking=" - delta 1"),
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=SignatureDelta(type="signature_delta", signature="real_sig_abc"),
            ),
            RawContentBlockStartEvent(
                type="content_block_start",
                index=1,
                content_block=ToolUseBlock(
                    type="tool_use",
                    id="tu_stream_1",
                    name="calculate",
                    input={},
                ),
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=2,
                delta=TextDelta(type="text_delta", text="Computing answer..."),
            ),
            RawMessageDeltaEvent(
                type="message_delta",
                delta=Delta(stop_reason="end_turn"),
                usage=MessageDeltaUsage(output_tokens=25),
            ),
        ]

        chunks = list(
            StreamHandler.process_stream(
                stream=_stream_cm(real_events),
                llm_provider="anthropic",
            )
        )
        assert (
            len(chunks) == 3
        )  # ToolUse start chunk, TextDelta chunk, MessageDelta chunk

        tool_chunk = chunks[0]
        assert tool_chunk.result.tool_calls is not None
        assert tool_chunk.result.tool_calls[0].id == "tu_stream_1"
        assert tool_chunk.result.tool_calls[0].function.name == "calculate"

        text_chunk = chunks[1]
        assert text_chunk.result.content == "Computing answer..."
        assert text_chunk.metadata.get("thinking_signature") == "real_sig_abc"
        assert text_chunk.metadata.get("thinking_signatures") == ["real_sig_abc"]
        assert text_chunk.metadata.get("thinking_deltas") == [" - delta 1"]
        assert text_chunk.metadata.get("thinking_blocks") == [
            "Starting real thinking block"
        ]

        final_chunk = chunks[2]
        assert final_chunk.result.finish_reason == "end_turn"
        assert final_chunk.metadata.get("usage", {}).get("output_tokens") == 25

        # Validate with StreamingComplianceHarness
        StreamingComplianceHarness.assert_valid_stream_contract(
            chunks, expected_provider="anthropic", expected_min_chunks=3
        )
        StreamingComplianceHarness.assert_reconstructs_text(
            chunks,
            expected_text="Computing answer...",
            expected_finish_reason="end_turn",
        )
        StreamingComplianceHarness.assert_reconstructs_tool_calls(
            chunks,
            expected_tools=[{"name": "calculate"}],
            expected_finish_reason="end_turn",
        )
        StreamingComplianceHarness.assert_snapshot_isolation(chunks)

    def test_response_handler_with_real_sdk_structured_response(self) -> None:
        class CityWeather(BaseModel):
            city: str
            temp_c: float

        real_msg = Message(
            id="msg_real_struct",
            model="claude-3-7-sonnet-20250219",
            role="assistant",
            type="message",
            content=[
                TextBlock(
                    type="text",
                    text='{"city": "Tokyo", "temp_c": 22.5}',
                )
            ],
            stop_reason="end_turn",
            stop_sequence=None,
            usage=Usage(input_tokens=10, output_tokens=12),
        )

        result = ResponseHandler.process_response(
            response=real_msg,
            llm_provider="anthropic",
            response_format=CityWeather,
            structured_mode="json",
        )
        assert isinstance(result, CityWeather)
        assert result.city == "Tokyo"
        assert result.temp_c == 22.5
