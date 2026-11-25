"""Regression tests for provider-specific LLM usage field names."""

from dapr_agents.observability.constants import (
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from dapr_agents.observability.wrappers.llm import LLMWrapper
from dapr_agents.observability.wrappers.llm_streaming import LLMStreamingMixin


class RecordingSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


class StreamingWrapper(LLMStreamingMixin):
    def _set_serialized_output(self, span, payload):
        pass


def test_sync_usage_supports_anthropic_field_names():
    span = RecordingSpan()
    LLMWrapper(None)._set_token_attributes(
        span, {"usage": {"input_tokens": 11, "output_tokens": 7}}
    )

    assert span.attributes[GEN_AI_USAGE_INPUT_TOKENS] == 11
    assert span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS] == 7


def test_sync_usage_prefers_openai_field_names():
    span = RecordingSpan()
    LLMWrapper(None)._set_token_attributes(
        span,
        {
            "usage": {
                "prompt_tokens": 13,
                "completion_tokens": 5,
                "input_tokens": 99,
                "output_tokens": 99,
            }
        },
    )

    assert span.attributes[GEN_AI_USAGE_INPUT_TOKENS] == 13
    assert span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS] == 5


def test_streaming_usage_supports_anthropic_field_names():
    span = RecordingSpan()
    StreamingWrapper()._set_streaming_output_attributes(
        span,
        assistant_message={"role": "assistant", "content": "ok"},
        metadata={"usage": {"input_tokens": 11, "output_tokens": 7}},
        finish_reason=None,
    )

    assert span.attributes[GEN_AI_USAGE_INPUT_TOKENS] == 11
    assert span.attributes[GEN_AI_USAGE_OUTPUT_TOKENS] == 7
