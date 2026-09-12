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

"""Tests for provider-independent LLM observability behavior."""

from types import SimpleNamespace
from typing import Any, Dict

import pytest

from dapr_agents.observability.constants import (
    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    LLM_TOKEN_COUNT_COMPLETION,
    LLM_TOKEN_COUNT_PROMPT,
    LLM_TOKEN_COUNT_TOTAL,
)
from dapr_agents.observability.wrappers.llm import LLMWrapper


class _RecordingSpan:
    """Minimal span test double that records attributes set by the wrapper."""

    def __init__(self) -> None:
        self.attributes: Dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value


@pytest.fixture
def wrapper() -> LLMWrapper:
    return LLMWrapper(tracer=None)


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
        SimpleNamespace(prompt_tokens=12, completion_tokens=7, total_tokens=19),
    ],
    ids=["dictionary-usage", "typed-usage"],
)
def test_non_streaming_token_attributes_accept_provider_usage_shapes(
    wrapper: LLMWrapper, usage: Any
) -> None:
    span = _RecordingSpan()

    wrapper._set_token_attributes(span, {"usage": usage})

    assert span.attributes[LLM_TOKEN_COUNT_PROMPT] == 12
    assert span.attributes[LLM_TOKEN_COUNT_COMPLETION] == 7
    assert span.attributes[LLM_TOKEN_COUNT_TOTAL] == 19
    assert span.attributes[GEN_AI_USAGE_INPUT_TOKENS] == 12
    assert span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS] == 7


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 12, "completion_tokens": 7},
        SimpleNamespace(prompt_tokens=12, completion_tokens=7),
    ],
    ids=["dictionary-usage", "typed-usage"],
)
def test_streaming_output_attributes_accept_provider_usage_shapes(
    wrapper: LLMWrapper, usage: Any
) -> None:
    span = _RecordingSpan()

    wrapper._set_streaming_output_attributes(
        span,
        assistant_message={"role": "assistant", "content": "done"},
        metadata={"usage": usage},
        finish_reason="stop",
    )

    assert span.attributes[GEN_AI_USAGE_INPUT_TOKENS] == 12
    assert span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS] == 7


def test_missing_streaming_usage_does_not_set_token_attributes(
    wrapper: LLMWrapper,
) -> None:
    span = _RecordingSpan()

    wrapper._set_streaming_output_attributes(
        span,
        assistant_message={"role": "assistant", "content": "done"},
        metadata={},
        finish_reason="stop",
    )

    assert GEN_AI_USAGE_INPUT_TOKENS not in span.attributes
    assert GEN_AI_USAGE_OUTPUT_TOKENS not in span.attributes


def test_non_streaming_cache_usage_supports_typed_objects(
    wrapper: LLMWrapper,
) -> None:
    usage = SimpleNamespace(
        prompt_tokens=12,
        completion_tokens=7,
        prompt_tokens_details=SimpleNamespace(cached_tokens=5),
        cache_creation_input_tokens=3,
        cache_read_input_tokens=2,
    )
    span = _RecordingSpan()

    wrapper._set_token_attributes(span, {"usage": usage})

    assert span.attributes[GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 2
    assert span.attributes[GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] == 3


def test_malformed_usage_is_ignored_without_breaking_streaming_output(
    wrapper: LLMWrapper,
) -> None:
    span = _RecordingSpan()

    wrapper._set_streaming_output_attributes(
        span,
        assistant_message={"role": "assistant", "content": "done"},
        metadata={"usage": object()},
        finish_reason="stop",
    )

    assert GEN_AI_USAGE_INPUT_TOKENS not in span.attributes
    assert GEN_AI_USAGE_OUTPUT_TOKENS not in span.attributes
