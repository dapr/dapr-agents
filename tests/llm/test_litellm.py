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

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dapr_agents.llm.litellm.chat import LiteLLMChatClient
from dapr_agents.types.message import (
    AssistantMessage,
    LLMChatResponse,
    LLMChatResponseChunk,
)


# ---------------------------------------------------------------------------
# Helpers - fake LiteLLM responses (OpenAI-compatible format)
# ---------------------------------------------------------------------------


def _fake_completion_response(
    content="Hello from LiteLLM",
    model="anthropic/claude-sonnet-4-6",
    tool_calls=None,
):
    msg = {"role": "assistant", "content": content, "refusal": None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
        msg["content"] = None
    choice = {
        "index": 0,
        "message": msg,
        "finish_reason": "tool_calls" if tool_calls else "stop",
        "logprobs": None,
    }
    return SimpleNamespace(
        model_dump=lambda: {
            "id": "chatcmpl-test-123",
            "model": model,
            "object": "chat.completion",
            "choices": [choice],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "created": 1700000000,
        }
    )


def _fake_stream_chunks(texts=("Hel", "lo")):
    for i, text in enumerate(texts):
        yield SimpleNamespace(
            model_dump=lambda t=text, idx=i: {
                "id": "chatcmpl-stream-123",
                "model": "openai/gpt-4o",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant" if idx == 0 else None,
                            "content": t,
                        },
                        "finish_reason": None,
                        "logprobs": None,
                    }
                ],
                "created": 1700000000,
            }
        )
    yield SimpleNamespace(
        model_dump=lambda: {
            "id": "chatcmpl-stream-123",
            "model": "openai/gpt-4o",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "created": 1700000000,
        }
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_litellm_client_default_model():
    client = LiteLLMChatClient()
    assert client.model == "openai/gpt-4o"
    assert client.provider == "litellm"
    assert client.api == "chat"
    assert client.client is None


def test_litellm_client_explicit_model():
    client = LiteLLMChatClient(model="anthropic/claude-sonnet-4-6")
    assert client.model == "anthropic/claude-sonnet-4-6"


def test_litellm_client_env_model():
    with patch.dict(os.environ, {"LITELLM_MODEL": "google/gemini-pro"}, clear=False):
        client = LiteLLMChatClient()
        assert client.model == "google/gemini-pro"


def test_litellm_client_config_from_params():
    client = LiteLLMChatClient(
        api_key="sk-test-key",
        api_base="http://localhost:4000",
    )
    assert client.config.api_key == "sk-test-key"
    assert client.config.api_base == "http://localhost:4000"


def test_litellm_client_config_from_env():
    env = {
        "LITELLM_API_KEY": "env-key",
        "LITELLM_API_BASE": "http://proxy.example.com",
    }
    with patch.dict(os.environ, env, clear=False):
        client = LiteLLMChatClient()
        assert client.config.api_key == "env-key"
        assert client.config.api_base == "http://proxy.example.com"


# ---------------------------------------------------------------------------
# generate(): one-shot
# ---------------------------------------------------------------------------


@patch("litellm.completion")
def test_litellm_generate_basic(mock_completion):
    mock_completion.return_value = _fake_completion_response()

    client = LiteLLMChatClient(model="anthropic/claude-sonnet-4-6")
    resp = client.generate("Say hello")

    assert isinstance(resp, LLMChatResponse)
    msg = resp.get_message()
    assert isinstance(msg, AssistantMessage)
    assert msg.content == "Hello from LiteLLM"

    mock_completion.assert_called_once()
    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["model"] == "anthropic/claude-sonnet-4-6"
    assert call_kwargs["drop_params"] is True
    assert call_kwargs["messages"] == [{"role": "user", "content": "Say hello"}]


@patch("litellm.completion")
def test_litellm_generate_with_api_key_and_base(mock_completion):
    mock_completion.return_value = _fake_completion_response()

    client = LiteLLMChatClient(
        model="openai/gpt-4o",
        api_key="sk-proxy",
        api_base="http://localhost:4000",
    )
    client.generate("hi")

    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["api_key"] == "sk-proxy"
    assert call_kwargs["api_base"] == "http://localhost:4000"


@patch("litellm.completion")
def test_litellm_generate_model_override(mock_completion):
    mock_completion.return_value = _fake_completion_response(model="openai/gpt-4o-mini")

    client = LiteLLMChatClient(model="openai/gpt-4o")
    client.generate("hi", model="openai/gpt-4o-mini")

    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["model"] == "openai/gpt-4o-mini"


@patch("litellm.completion")
def test_litellm_generate_forwards_kwargs(mock_completion):
    mock_completion.return_value = _fake_completion_response()

    client = LiteLLMChatClient(model="openai/gpt-4o")
    client.generate("hi", temperature=0.3, max_tokens=100)

    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["temperature"] == 0.3
    assert call_kwargs["max_tokens"] == 100


@patch("litellm.completion")
def test_litellm_generate_tool_call_response(mock_completion):
    tool_calls = [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "San Francisco"}',
            },
        }
    ]
    mock_completion.return_value = _fake_completion_response(tool_calls=tool_calls)

    client = LiteLLMChatClient(model="openai/gpt-4o")
    resp = client.generate("What's the weather?")

    msg = resp.get_message()
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].function.name == "get_weather"
    assert msg.tool_calls[0].function.arguments == '{"location": "San Francisco"}'
    assert resp.results[0].finish_reason == "tool_calls"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


@patch("litellm.completion")
def test_litellm_generate_streaming(mock_completion):
    mock_completion.return_value = _fake_stream_chunks()

    client = LiteLLMChatClient(model="openai/gpt-4o")
    chunks = list(client.generate("hi", stream=True))

    assert len(chunks) > 0
    assert all(isinstance(c, LLMChatResponseChunk) for c in chunks)
    text = "".join(
        c.result.content or "" for c in chunks if c.result and c.result.content
    )
    assert text == "Hello"

    call_kwargs = mock_completion.call_args.kwargs
    assert call_kwargs["stream"] is True


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@patch("litellm.completion")
def test_litellm_generate_api_error(mock_completion):
    mock_completion.side_effect = RuntimeError("API down")

    client = LiteLLMChatClient(model="openai/gpt-4o")
    with pytest.raises(ValueError, match="LiteLLM API error"):
        client.generate("hi")


def test_litellm_generate_requires_messages():
    client = LiteLLMChatClient()
    with pytest.raises(ValueError):
        client.generate()


def test_litellm_from_prompty_not_implemented():
    with pytest.raises(NotImplementedError, match="not yet supported"):
        LiteLLMChatClient.from_prompty("some_source.prompty")


def test_litellm_generate_rejects_unknown_structured_mode():
    client = LiteLLMChatClient()
    with pytest.raises(ValueError):
        client.generate("hi", structured_mode="grammar")
