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

"""Internal token usage parsing and normalization utilities."""

from typing import Any

from dapr_agents.utils._config import coerce_integral, coerce_token_count


def get_usage_value(usage: Any, key: str) -> Any:
    """Extract a value from a usage dict or object."""
    if isinstance(usage, dict):
        return usage.get(key)
    return getattr(usage, key, None)


def get_input_token_count(usage: Any) -> int | None:
    """Return aggregate input tokens including cache-related tokens, or None when unavailable."""
    # Prefer the OpenAI-style aggregate input-token field when present.
    prompt_tokens = coerce_token_count(get_usage_value(usage, "prompt_tokens"))
    if prompt_tokens is not None:
        return prompt_tokens

    # Fall back to Anthropic's input-token field, which excludes cache-related tokens.
    input_tokens = coerce_token_count(get_usage_value(usage, "input_tokens"))
    if input_tokens is None:
        return None

    # Anthropic reports cache-read and cache-creation tokens separately from
    # input_tokens; include them in the aggregate input count.
    cache_creation_input_tokens = (
        coerce_token_count(get_usage_value(usage, "cache_creation_input_tokens")) or 0
    )
    cache_read_input_tokens = (
        coerce_token_count(get_usage_value(usage, "cache_read_input_tokens")) or 0
    )

    return input_tokens + cache_creation_input_tokens + cache_read_input_tokens


def get_output_token_count(usage: Any) -> int | None:
    """Return output token count, or None when unavailable."""
    completion_tokens = coerce_token_count(get_usage_value(usage, "completion_tokens"))
    if completion_tokens is not None:
        return completion_tokens
    return coerce_token_count(get_usage_value(usage, "output_tokens"))


def get_total_token_count(usage: Any) -> int | None:
    """Return total token count, or None when unavailable."""
    total_tokens = coerce_token_count(get_usage_value(usage, "total_tokens"))
    if total_tokens is not None:
        return total_tokens
    input_cnt = get_input_token_count(usage)
    output_cnt = get_output_token_count(usage)
    if input_cnt is not None and output_cnt is not None:
        return input_cnt + output_cnt
    return None


def get_cache_creation_token_count(usage: Any) -> int | None:
    """Return cache creation token count, or None when unavailable."""
    return coerce_token_count(get_usage_value(usage, "cache_creation_input_tokens"))


def get_cache_read_token_count(usage: Any) -> int | None:
    """Return cache read token count, or None when unavailable."""
    # Prefer OpenAI-style cached tokens in prompt_tokens_details when present.
    prompt_details = get_usage_value(usage, "prompt_tokens_details")
    if prompt_details is not None:
        cached = coerce_token_count(get_usage_value(prompt_details, "cached_tokens"))
        if cached is not None:
            return cached

    # Fall back to Anthropic's cache_read_input_tokens field.
    return coerce_token_count(get_usage_value(usage, "cache_read_input_tokens"))
