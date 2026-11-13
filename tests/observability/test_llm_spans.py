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

"""Tests for LLM observability spans and token usage extraction."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from dapr_agents.observability.constants import (
    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    LLM_TOKEN_COUNT_COMPLETION,
    LLM_TOKEN_COUNT_PROMPT,
    LLM_TOKEN_COUNT_TOTAL,
)
from dapr_agents.observability.message_processors import (
    extract_token_usage,
    extract_usage_from_dict,
    extract_usage_from_object,
)
from dapr_agents.observability.wrappers.llm import LLMWrapper
from dapr_agents.types.message import (
    AssistantMessage,
    LLMChatCandidate,
    LLMChatCandidateChunk,
    LLMChatResponse,
    LLMChatResponseChunk,
)


@pytest.fixture
def exporter():
    return InMemorySpanExporter()


@pytest.fixture
def tracer(exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test_llm")


@pytest.fixture
def wrapper(tracer):
    return LLMWrapper(tracer)


class TestTokenUsageExtraction:
    def test_extract_usage_from_dict_openai(self):
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        attrs = extract_usage_from_dict(usage)
        assert attrs[LLM_TOKEN_COUNT_PROMPT] == 10
        assert attrs[LLM_TOKEN_COUNT_COMPLETION] == 20
        assert attrs[LLM_TOKEN_COUNT_TOTAL] == 30

    def test_extract_usage_from_dict_anthropic(self):
        usage = {"input_tokens": 15, "output_tokens": 25}
        attrs = extract_usage_from_dict(usage)
        assert attrs[LLM_TOKEN_COUNT_PROMPT] == 15
        assert attrs[LLM_TOKEN_COUNT_COMPLETION] == 25

    def test_extract_usage_from_dict_anthropic_with_cache(self):
        usage = {
            "input_tokens": 15,
            "output_tokens": 25,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 10,
        }
        attrs = extract_usage_from_dict(usage)
        assert attrs[LLM_TOKEN_COUNT_PROMPT] == 30
        assert attrs[LLM_TOKEN_COUNT_COMPLETION] == 25
        assert attrs[LLM_TOKEN_COUNT_TOTAL] == 55

    def test_extract_usage_from_object_openai(self):
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        attrs = extract_usage_from_object(usage)
        assert attrs[LLM_TOKEN_COUNT_PROMPT] == 10
        assert attrs[LLM_TOKEN_COUNT_COMPLETION] == 20
        assert attrs[LLM_TOKEN_COUNT_TOTAL] == 30

    def test_extract_usage_from_object_anthropic(self):
        usage = SimpleNamespace(input_tokens=15, output_tokens=25)
        attrs = extract_usage_from_object(usage)
        assert attrs[LLM_TOKEN_COUNT_PROMPT] == 15
        assert attrs[LLM_TOKEN_COUNT_COMPLETION] == 25

    def test_extract_usage_from_object_anthropic_with_cache(self):
        usage = SimpleNamespace(
            input_tokens=15,
            output_tokens=25,
            cache_creation_input_tokens=5,
            cache_read_input_tokens=10,
        )
        attrs = extract_usage_from_object(usage)
        assert attrs[LLM_TOKEN_COUNT_PROMPT] == 30
        assert attrs[LLM_TOKEN_COUNT_COMPLETION] == 25
        assert attrs[LLM_TOKEN_COUNT_TOTAL] == 55

    def test_extract_token_usage_metadata_formats(self):
        # Dict metadata with Anthropic usage
        meta_dict = {"usage": {"input_tokens": 8, "output_tokens": 16}}
        attrs = extract_token_usage(meta_dict)
        assert attrs[LLM_TOKEN_COUNT_PROMPT] == 8
        assert attrs[LLM_TOKEN_COUNT_COMPLETION] == 16

        # Object metadata with Anthropic usage
        meta_obj = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=9, output_tokens=18)
        )
        attrs = extract_token_usage(meta_obj)
        assert attrs[LLM_TOKEN_COUNT_PROMPT] == 9
        assert attrs[LLM_TOKEN_COUNT_COMPLETION] == 18


class TestLLMWrapperTokenAttributes:
    def test_set_token_attributes_openai(self, tracer, exporter, wrapper):
        metadata = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}
        span = tracer.start_span("test")
        wrapper._set_token_attributes(span, metadata)
        span.end()

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 10
        assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 20
        assert attrs[LLM_TOKEN_COUNT_PROMPT] == 10
        assert attrs[LLM_TOKEN_COUNT_COMPLETION] == 20

    def test_set_token_attributes_anthropic_dict(self, tracer, exporter, wrapper):
        metadata = {
            "usage": {
                "input_tokens": 42,
                "output_tokens": 84,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 20,
            }
        }
        span = tracer.start_span("test")
        wrapper._set_token_attributes(span, metadata)
        span.end()

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 72
        assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 84
        assert attrs[GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] == 10
        assert attrs[GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 20
        assert attrs[LLM_TOKEN_COUNT_PROMPT] == 72
        assert attrs[LLM_TOKEN_COUNT_COMPLETION] == 84

    def test_set_token_attributes_anthropic_object(self, tracer, exporter, wrapper):
        metadata = {
            "usage": SimpleNamespace(
                input_tokens=42,
                output_tokens=84,
                cache_creation_input_tokens=10,
                cache_read_input_tokens=20,
            )
        }
        span = tracer.start_span("test")
        wrapper._set_token_attributes(span, metadata)
        span.end()

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 72
        assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 84
        assert attrs[GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] == 10
        assert attrs[GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 20
        assert attrs[LLM_TOKEN_COUNT_PROMPT] == 72
        assert attrs[LLM_TOKEN_COUNT_COMPLETION] == 84

    def test_set_token_attributes_precedence(self, tracer, exporter, wrapper):
        """Verify prompt_tokens takes precedence over input_tokens and cache fields are not double-counted."""
        metadata = {
            "usage": {
                "prompt_tokens": 100,
                "input_tokens": 999,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 20,
                "completion_tokens": 50,
                "output_tokens": 999,
            }
        }
        span = tracer.start_span("test")
        wrapper._set_token_attributes(span, metadata)
        span.end()

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 100
        assert attrs[LLM_TOKEN_COUNT_PROMPT] == 100
        assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 50
        assert attrs[LLM_TOKEN_COUNT_COMPLETION] == 50

    def test_set_token_attributes_anthropic_no_cache(self, tracer, exporter, wrapper):
        metadata = {"usage": {"input_tokens": 42, "output_tokens": 84}}
        span = tracer.start_span("test")
        wrapper._set_token_attributes(span, metadata)
        span.end()

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 42
        assert attrs[LLM_TOKEN_COUNT_PROMPT] == 42
        assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 84
        assert attrs[LLM_TOKEN_COUNT_COMPLETION] == 84
        assert GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS not in attrs
        assert GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS not in attrs

    def test_set_token_attributes_cache_read_only(self, tracer, exporter, wrapper):
        metadata = {"usage": {"input_tokens": 40, "cache_read_input_tokens": 10}}
        span = tracer.start_span("test")
        wrapper._set_token_attributes(span, metadata)
        span.end()

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 50
        assert attrs[LLM_TOKEN_COUNT_PROMPT] == 50
        assert attrs[GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 10
        assert GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS not in attrs

    def test_set_token_attributes_cache_creation_only(self, tracer, exporter, wrapper):
        metadata = {"usage": {"input_tokens": 40, "cache_creation_input_tokens": 15}}
        span = tracer.start_span("test")
        wrapper._set_token_attributes(span, metadata)
        span.end()

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 55
        assert attrs[LLM_TOKEN_COUNT_PROMPT] == 55
        assert attrs[GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] == 15
        assert GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS not in attrs

    def test_set_token_attributes_zero_cache(self, tracer, exporter, wrapper):
        metadata = {
            "usage": {
                "input_tokens": 40,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            }
        }
        span = tracer.start_span("test")
        wrapper._set_token_attributes(span, metadata)
        span.end()

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 40
        assert attrs[LLM_TOKEN_COUNT_PROMPT] == 40
        assert attrs[GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] == 0
        assert attrs[GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 0

    def test_set_token_attributes_none_cache(self, tracer, exporter, wrapper):
        metadata = {
            "usage": {
                "input_tokens": 40,
                "cache_creation_input_tokens": None,
                "cache_read_input_tokens": None,
            }
        }
        span = tracer.start_span("test")
        wrapper._set_token_attributes(span, metadata)
        span.end()

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 40
        assert attrs[LLM_TOKEN_COUNT_PROMPT] == 40
        assert GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS not in attrs
        assert GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS not in attrs

    def test_set_token_attributes_invalid_values(self, tracer, exporter, wrapper):
        metadata = {
            "usage": {
                "input_tokens": 40,
                "output_tokens": 20,
                "cache_creation_input_tokens": -5,
                "cache_read_input_tokens": "invalid",
            }
        }
        span = tracer.start_span("test")
        wrapper._set_token_attributes(span, metadata)
        span.end()

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 40
        assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 20
        assert GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS not in attrs
        assert GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS not in attrs

    def test_set_token_attributes_openai_cached_tokens(self, tracer, exporter, wrapper):
        metadata = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 30},
            }
        }
        span = tracer.start_span("test")
        wrapper._set_token_attributes(span, metadata)
        span.end()

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 100
        assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 50
        assert attrs[GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 30

    def test_set_token_attributes_openai_cached_tokens_object(
        self, tracer, exporter, wrapper
    ):
        metadata = {
            "usage": SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=50,
                prompt_tokens_details=SimpleNamespace(cached_tokens=30),
            )
        }
        span = tracer.start_span("test")
        wrapper._set_token_attributes(span, metadata)
        span.end()

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 100
        assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 50
        assert attrs[GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 30


class TestLLMStreamingTokenAttributes:
    def test_streaming_output_attributes_openai(self, tracer, exporter, wrapper):
        metadata = {"usage": {"prompt_tokens": 12, "completion_tokens": 24}}
        span = tracer.start_span("test_stream")
        wrapper._set_streaming_output_attributes(
            span,
            assistant_message={"role": "assistant", "content": "hi"},
            metadata=metadata,
            finish_reason="stop",
        )
        span.end()

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 12
        assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 24

    def test_streaming_output_attributes_anthropic(self, tracer, exporter, wrapper):
        metadata = {
            "usage": {
                "input_tokens": 55,
                "output_tokens": 77,
                "cache_creation_input_tokens": 8,
                "cache_read_input_tokens": 16,
            }
        }
        span = tracer.start_span("test_stream")
        wrapper._set_streaming_output_attributes(
            span,
            assistant_message={"role": "assistant", "content": "hi"},
            metadata=metadata,
            finish_reason="end_turn",
        )
        span.end()

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 79
        assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 77
        assert attrs[GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] == 8
        assert attrs[GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 16

    def test_streaming_output_attributes_precedence(self, tracer, exporter, wrapper):
        metadata = {
            "usage": {
                "prompt_tokens": 100,
                "input_tokens": 999,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 20,
                "completion_tokens": 50,
                "output_tokens": 999,
            }
        }
        span = tracer.start_span("test_stream")
        wrapper._set_streaming_output_attributes(
            span,
            assistant_message={"role": "assistant", "content": "hi"},
            metadata=metadata,
            finish_reason="stop",
        )
        span.end()

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 100
        assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 50

    def test_streaming_output_attributes_invalid_values(
        self, tracer, exporter, wrapper
    ):
        metadata = {
            "usage": {
                "input_tokens": 50,
                "output_tokens": 25,
                "cache_creation_input_tokens": -10,
                "cache_read_input_tokens": "invalid",
            }
        }
        span = tracer.start_span("test_stream")
        wrapper._set_streaming_output_attributes(
            span,
            assistant_message={"role": "assistant", "content": "hi"},
            metadata=metadata,
            finish_reason="end_turn",
        )
        span.end()

        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 50
        assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 25
        assert GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS not in attrs
        assert GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS not in attrs


class TestEndToEndLLMWrapper:
    def test_non_streaming_anthropic_call(self, tracer, exporter, wrapper):
        client = SimpleNamespace(model="claude-3-7-sonnet")
        candidate = LLMChatCandidate(
            message=AssistantMessage(content="Hello!"),
            finish_reason="end_turn",
        )
        fake_response = LLMChatResponse(
            results=[candidate],
            metadata={
                "usage": {
                    "input_tokens": 30,
                    "output_tokens": 60,
                    "cache_creation_input_tokens": 5,
                    "cache_read_input_tokens": 15,
                }
            },
        )
        mock_generate = MagicMock(return_value=fake_response)

        res = wrapper(
            mock_generate,
            client,
            ([{"role": "user", "content": "Hello"}],),
            {"model": "claude-3-7-sonnet"},
        )
        assert res == fake_response

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = spans[0].attributes
        assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 50
        assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 60
        assert attrs[GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] == 5
        assert attrs[GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 15
        assert attrs[LLM_TOKEN_COUNT_PROMPT] == 50
        assert attrs[LLM_TOKEN_COUNT_COMPLETION] == 60

    def test_streaming_anthropic_call(self, tracer, exporter, wrapper):
        client = SimpleNamespace(model="claude-3-7-sonnet")
        chunks = [
            LLMChatResponseChunk(
                result=LLMChatCandidateChunk(content="Hello"),
                metadata={"usage": {"input_tokens": 25}},
            ),
            LLMChatResponseChunk(
                result=LLMChatCandidateChunk(
                    content=" world", finish_reason="end_turn"
                ),
                metadata={
                    "usage": {
                        "input_tokens": 25,
                        "output_tokens": 50,
                        "cache_read_input_tokens": 10,
                    }
                },
            ),
        ]
        mock_generate = MagicMock(return_value=iter(chunks))

        stream_gen = wrapper(
            mock_generate,
            client,
            ([{"role": "user", "content": "Hello"}],),
            {"model": "claude-3-7-sonnet", "stream": True},
        )
        consumed = list(stream_gen)
        assert len(consumed) == 2

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = spans[0].attributes
        assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 35
        assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 50
        assert attrs[GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 10

    def test_streaming_split_usage_chunks(self, tracer, exporter, wrapper):
        """Verify usage split across streaming chunks (e.g. Anthropic message_start + message_delta) is merged."""
        client = SimpleNamespace(model="claude-3-7-sonnet")
        chunks = [
            LLMChatResponseChunk(
                result=LLMChatCandidateChunk(content="Hello"),
                metadata={
                    "usage": {
                        "input_tokens": 40,
                        "cache_read_input_tokens": 10,
                        "cache_creation_input_tokens": 5,
                    }
                },
            ),
            LLMChatResponseChunk(
                result=LLMChatCandidateChunk(
                    content=" world", finish_reason="end_turn"
                ),
                metadata={
                    "usage": {
                        "output_tokens": 20,
                    }
                },
            ),
        ]
        mock_generate = MagicMock(return_value=iter(chunks))

        stream_gen = wrapper(
            mock_generate,
            client,
            ([{"role": "user", "content": "Hello"}],),
            {"model": "claude-3-7-sonnet", "stream": True},
        )
        consumed = list(stream_gen)
        assert len(consumed) == 2

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = spans[0].attributes
        assert attrs[GEN_AI_USAGE_INPUT_TOKENS] == 55
        assert attrs[GEN_AI_USAGE_OUTPUT_TOKENS] == 20
        assert attrs[GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 10
        assert attrs[GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] == 5
