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

import json
import os
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Iterable
from unittest.mock import MagicMock, patch

import pytest

from dapr_agents.llm.anthropic.chat import AnthropicChatClient
from dapr_agents.types.message import (
    AssistantMessage,
    LLMChatResponse,
    LLMChatResponseChunk,
    ToolCall,
    UserMessage,
)


def _stream_cm(events: Iterable):
    """Wrap an event iterable so it behaves like the SDK's `Stream` context manager."""
    return nullcontext(iter(events))


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_client_initialization(mock_anthropic_class):
    """Defaults + ANTHROPIC_MODEL/API_KEY/BASE_URL env fallbacks."""
    mock_anthropic_class.return_value = MagicMock()

    client = AnthropicChatClient(api_key="fake-key")
    assert client.model == "claude-sonnet-4-6"
    assert client.provider == "anthropic"

    env = {
        "ANTHROPIC_MODEL": "claude-opus-4-5",
        "ANTHROPIC_API_KEY": "env-secret",
        "ANTHROPIC_BASE_URL": "https://proxy.example/v1",
    }
    with patch.dict(os.environ, env, clear=False):
        client_env = AnthropicChatClient()
        assert client_env.model == "claude-opus-4-5"
        assert client_env.config.api_key == "env-secret"
        assert client_env.config.base_url == "https://proxy.example/v1"


# ---------------------------------------------------------------------------
# generate(): one-shot
# ---------------------------------------------------------------------------


def _fake_anthropic_response(text: str = "Hello from Claude", tool_use=None):
    blocks = [SimpleNamespace(type="text", text=text)]
    if tool_use:
        blocks.append(
            SimpleNamespace(
                type="tool_use",
                id=tool_use["id"],
                name=tool_use["name"],
                input=tool_use["input"],
            )
        )
    return SimpleNamespace(
        id="msg_test",
        model="claude-opus-4-5",
        content=blocks,
        stop_reason="end_turn",
        stop_sequence=None,
        usage=SimpleNamespace(
            model_dump=lambda: {"input_tokens": 7, "output_tokens": 3}
        ),
    )


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_generate_basic(mock_anthropic_class):
    """generate() returns an LLMChatResponse with the assistant content."""
    sdk = MagicMock()
    sdk.messages.create.return_value = _fake_anthropic_response("Hello from Claude")
    mock_anthropic_class.return_value = sdk

    client = AnthropicChatClient(api_key="fake-key")
    resp = client.generate("Say hello")

    assert isinstance(resp, LLMChatResponse)
    msg = resp.get_message()
    assert isinstance(msg, AssistantMessage)
    assert msg.content == "Hello from Claude"
    assert resp.metadata["usage"] == {"input_tokens": 7, "output_tokens": 3}
    assert resp.metadata["stop_reason"] == "end_turn"
    sdk.messages.create.assert_called_once()

    # Default model + max_tokens and the user message both made it into the request.
    kwargs = sdk.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["max_tokens"] == 4096
    assert kwargs["messages"] == [{"role": "user", "content": "Say hello"}]
    assert "system" not in kwargs


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_generate_extracts_system(mock_anthropic_class):
    """system messages are pulled into the top-level `system` param."""
    sdk = MagicMock()
    sdk.messages.create.return_value = _fake_anthropic_response("OK")
    mock_anthropic_class.return_value = sdk

    client = AnthropicChatClient(api_key="fake-key")
    client.generate(
        messages=[
            {"role": "system", "content": "you are terse"},
            {"role": "system", "content": "answer in haiku"},
            UserMessage("hi"),
        ]
    )

    kwargs = sdk.messages.create.call_args.kwargs
    assert kwargs["system"] == "you are terse\n\nanswer in haiku"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_generate_preserves_structured_system_blocks(mock_anthropic_class):
    """List-of-blocks system content (with cache_control) flows through verbatim."""
    sdk = MagicMock()
    sdk.messages.create.return_value = _fake_anthropic_response("ok")
    mock_anthropic_class.return_value = sdk

    cached_block = {
        "type": "text",
        "text": "long cached system prompt",
        "cache_control": {"type": "ephemeral"},
    }
    client = AnthropicChatClient(api_key="fake-key")
    client.generate(
        messages=[
            {"role": "system", "content": [cached_block]},
            UserMessage("hi"),
        ]
    )

    kwargs = sdk.messages.create.call_args.kwargs
    assert kwargs["system"] == [cached_block]


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_generate_mixed_system_promotes_to_blocks(mock_anthropic_class):
    """Mixing a plain-string system with a structured one yields a block list,
    so cache markers from the structured message aren't lost to stringification."""
    sdk = MagicMock()
    sdk.messages.create.return_value = _fake_anthropic_response("ok")
    mock_anthropic_class.return_value = sdk

    cached_block = {
        "type": "text",
        "text": "cached",
        "cache_control": {"type": "ephemeral"},
    }
    client = AnthropicChatClient(api_key="fake-key")
    client.generate(
        messages=[
            {"role": "system", "content": "preamble"},
            {"role": "system", "content": [cached_block]},
            UserMessage("hi"),
        ]
    )

    kwargs = sdk.messages.create.call_args.kwargs
    assert kwargs["system"] == [
        {"type": "text", "text": "preamble"},
        cached_block,
    ]


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_generate_tool_use_response(mock_anthropic_class):
    """tool_use content blocks are surfaced as ToolCall on the assistant message."""
    sdk = MagicMock()
    sdk.messages.create.return_value = _fake_anthropic_response(
        text="",
        tool_use={"id": "tu_1", "name": "lookup", "input": {"q": "weather"}},
    )
    mock_anthropic_class.return_value = sdk

    client = AnthropicChatClient(api_key="fake-key")
    resp = client.generate("ask")

    msg = resp.get_message()
    calls = msg.get_tool_calls()
    assert calls and len(calls) == 1
    tc = calls[0]
    assert isinstance(tc, ToolCall)
    assert tc.id == "tu_1"
    assert tc.function.name == "lookup"
    assert json.loads(tc.function.arguments) == {"q": "weather"}


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_generate_translates_tool_messages(mock_anthropic_class):
    """OpenAI-style tool messages convert to Anthropic tool_result blocks."""
    sdk = MagicMock()
    sdk.messages.create.return_value = _fake_anthropic_response("done")
    mock_anthropic_class.return_value = sdk

    client = AnthropicChatClient(api_key="fake-key")
    client.generate(
        messages=[
            {"role": "user", "content": "what's the weather?"},
            {
                "role": "assistant",
                "content": "let me check",
                "tool_calls": [
                    {
                        "id": "tu_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"q": "weather"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tu_1", "content": "sunny"},
        ]
    )

    msgs = sdk.messages.create.call_args.kwargs["messages"]
    # 3 turns: user, assistant-with-tool_use, user-with-tool_result
    assert len(msgs) == 3
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"][0] == {"type": "text", "text": "let me check"}
    assert msgs[1]["content"][1] == {
        "type": "tool_use",
        "id": "tu_1",
        "name": "lookup",
        "input": {"q": "weather"},
    }
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "tu_1",
        "content": "sunny",
    }


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_rejects_assistant_tool_call_with_invalid_json_args(
    mock_anthropic_class,
):
    """An assistant `tool_calls[].arguments` that isn't valid JSON raises before the API call."""
    mock_anthropic_class.return_value = MagicMock()
    client = AnthropicChatClient(api_key="fake-key")
    with pytest.raises(ValueError, match="not valid JSON"):
        client.generate(
            messages=[
                {"role": "user", "content": "?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tu_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": "{not json",
                            },
                        }
                    ],
                },
            ]
        )


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_generate_forwards_native_kwargs(mock_anthropic_class):
    """`thinking`, `top_k`, etc. flow straight through to messages.create."""
    sdk = MagicMock()
    sdk.messages.create.return_value = _fake_anthropic_response("ok")
    mock_anthropic_class.return_value = sdk

    client = AnthropicChatClient(api_key="fake-key")
    client.generate(
        "compute",
        thinking={"type": "enabled", "budget_tokens": 1024},
        top_k=40,
        max_tokens=2048,
    )

    kwargs = sdk.messages.create.call_args.kwargs
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert kwargs["top_k"] == 40
    assert kwargs["max_tokens"] == 2048


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_generate_formats_tools_for_claude(mock_anthropic_class):
    """Tools are translated to Anthropic's `input_schema` shape."""
    sdk = MagicMock()
    sdk.messages.create.return_value = _fake_anthropic_response("ok")
    mock_anthropic_class.return_value = sdk

    client = AnthropicChatClient(api_key="fake-key")
    client.generate(
        "what is the weather?",
        tools=[
            {
                "name": "get_weather",
                "description": "Return the current weather.",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
    )

    tools = sdk.messages.create.call_args.kwargs["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "get_weather"
    # Anthropic uses `input_schema`, not the OpenAI `parameters`/`function` wrapper.
    assert "input_schema" in tools[0]
    assert "function" not in tools[0]


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_generate_streaming(mock_anthropic_class):
    """Streamed SSE events are translated into LLMChatResponseChunk objects."""
    events = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(id="msg_s", model="claude-opus-4-5"),
        ),
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="text"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text="Hel"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="text_delta", text="lo"),
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(
                model_dump=lambda: {"input_tokens": 2, "output_tokens": 2}
            ),
        ),
    ]
    sdk = MagicMock()
    sdk.messages.create.return_value = _stream_cm(events)
    mock_anthropic_class.return_value = sdk

    client = AnthropicChatClient(api_key="fake-key")
    chunks = list(client.generate("hi", stream=True))

    assert chunks and all(isinstance(c, LLMChatResponseChunk) for c in chunks)
    text = "".join(c.result.content or "" for c in chunks if c.result.content)
    assert text == "Hello"
    finish = [c.result.finish_reason for c in chunks if c.result.finish_reason]
    assert finish == ["end_turn"]
    # metadata is enriched mid-stream from message_start + message_delta.usage
    last_meta = chunks[-1].metadata
    assert last_meta["provider"] == "anthropic"
    assert last_meta["id"] == "msg_s"
    assert last_meta["model"] == "claude-opus-4-5"
    assert last_meta["usage"] == {"input_tokens": 2, "output_tokens": 2}


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_generate_streaming_tool_use(mock_anthropic_class):
    """tool_use streaming: content_block_start carries id/name, input_json_delta
    streams the argument JSON, and the final message_delta surfaces stop_reason."""
    events = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(id="msg_t", model="claude-opus-4-5"),
        ),
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="tool_use", id="tu_1", name="lookup"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"q":'),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="input_json_delta", partial_json=' "weather"}'),
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="tool_use"),
            usage=SimpleNamespace(
                model_dump=lambda: {"input_tokens": 5, "output_tokens": 8}
            ),
        ),
    ]
    sdk = MagicMock()
    sdk.messages.create.return_value = _stream_cm(events)
    mock_anthropic_class.return_value = sdk

    client = AnthropicChatClient(api_key="fake-key")
    chunks = list(client.generate("ask", stream=True))

    tool_chunks = [tc for c in chunks for tc in (c.result.tool_calls or [])]
    assert len(tool_chunks) == 3  # block_start + 2 json deltas

    start_chunk = tool_chunks[0]
    assert start_chunk.id == "tu_1"
    assert start_chunk.function.name == "lookup"
    assert start_chunk.function.arguments == ""

    args_concat = "".join(tc.function.arguments for tc in tool_chunks[1:])
    assert json.loads(args_concat) == {"q": "weather"}

    assert chunks[-1].result.finish_reason == "tool_use"
    assert chunks[-1].metadata["usage"] == {"input_tokens": 5, "output_tokens": 8}


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_streaming_propagates_sdk_errors(mock_anthropic_class):
    """SDK errors raised on stream open propagate with their original type
    (no wrapping in ValueError)."""

    class FakeAnthropicError(RuntimeError):
        pass

    sdk = MagicMock()
    sdk.messages.create.side_effect = FakeAnthropicError("upstream 503")
    mock_anthropic_class.return_value = sdk

    client = AnthropicChatClient(api_key="fake-key")
    stream = client.generate("hi", stream=True)
    with pytest.raises(FakeAnthropicError, match="upstream 503"):
        list(stream)


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_non_streaming_propagates_sdk_errors(mock_anthropic_class):
    """Non-stream path also surfaces the original exception type."""

    class FakeAnthropicError(RuntimeError):
        pass

    sdk = MagicMock()
    sdk.messages.create.side_effect = FakeAnthropicError("upstream 503")
    mock_anthropic_class.return_value = sdk

    client = AnthropicChatClient(api_key="fake-key")
    with pytest.raises(FakeAnthropicError, match="upstream 503"):
        client.generate("hi")


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_response_format_function_call(mock_anthropic_class):
    """structured_mode='function_call' forces a single tool and parses tool_use.input back."""
    from pydantic import BaseModel

    class Out(BaseModel):
        answer: str

    sdk = MagicMock()
    sdk.messages.create.return_value = SimpleNamespace(
        id="msg_struct",
        model="claude-opus-4-5",
        content=[
            SimpleNamespace(
                type="tool_use",
                id="tu_struct",
                name="Out",
                input={"answer": "42"},
            ),
        ],
        stop_reason="tool_use",
        stop_sequence=None,
        usage=SimpleNamespace(
            model_dump=lambda: {"input_tokens": 1, "output_tokens": 1}
        ),
    )
    mock_anthropic_class.return_value = sdk

    client = AnthropicChatClient(api_key="fake-key")
    result = client.generate(
        "the meaning?", response_format=Out, structured_mode="function_call"
    )

    assert isinstance(result, Out)
    assert result.answer == "42"

    kwargs = sdk.messages.create.call_args.kwargs
    assert kwargs["tool_choice"] == {"type": "tool", "name": "Out"}
    tool_names = [t.get("name") for t in kwargs["tools"]]
    assert "Out" in tool_names
    assert "output_config" not in kwargs


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_response_format_json_native(mock_anthropic_class):
    """structured_mode='json' (default) uses Anthropic's native output_config and
    parses the JSON text block back into the Pydantic model."""
    from pydantic import BaseModel

    class Out(BaseModel):
        answer: str
        confidence: float

    sdk = MagicMock()
    sdk.messages.create.return_value = SimpleNamespace(
        id="msg_json",
        model="claude-opus-4-7",
        content=[
            SimpleNamespace(
                type="text",
                text='{"answer": "42", "confidence": 0.99}',
            ),
        ],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=SimpleNamespace(
            model_dump=lambda: {"input_tokens": 1, "output_tokens": 1}
        ),
    )
    mock_anthropic_class.return_value = sdk

    client = AnthropicChatClient(api_key="fake-key")
    # No structured_mode kwarg — exercises the default.
    result = client.generate("the meaning?", response_format=Out)

    assert isinstance(result, Out)
    assert result.answer == "42"
    assert result.confidence == 0.99

    kwargs = sdk.messages.create.call_args.kwargs
    assert "tool_choice" not in kwargs  # native path, not the tool-call fallback
    output_format = kwargs["output_config"]["format"]
    assert output_format["type"] == "json_schema"
    schema = output_format["schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"answer", "confidence"}
    assert set(schema["required"]) == {"answer", "confidence"}


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_function_call_preserves_user_tools(mock_anthropic_class):
    """function_call structured mode appends the structured tool alongside user-provided
    tools and forces tool_choice to the structured one."""
    from pydantic import BaseModel

    class Out(BaseModel):
        answer: str

    sdk = MagicMock()
    sdk.messages.create.return_value = SimpleNamespace(
        id="msg_x",
        model="claude-opus-4-5",
        content=[
            SimpleNamespace(
                type="tool_use", id="tu_x", name="Out", input={"answer": "42"}
            ),
        ],
        stop_reason="tool_use",
        stop_sequence=None,
        usage=SimpleNamespace(
            model_dump=lambda: {"input_tokens": 1, "output_tokens": 1}
        ),
    )
    mock_anthropic_class.return_value = sdk

    user_tool = {
        "name": "get_weather",
        "description": "Return the current weather.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
    client = AnthropicChatClient(api_key="fake-key")
    client.generate(
        "?", tools=[user_tool], response_format=Out, structured_mode="function_call"
    )

    kwargs = sdk.messages.create.call_args.kwargs
    tool_names = [t["name"] for t in kwargs["tools"]]
    assert tool_names == ["get_weather", "Out"]
    assert kwargs["tool_choice"] == {"type": "tool", "name": "Out"}


@pytest.mark.parametrize(
    "structured_mode, content, match",
    [
        (
            "function_call",
            [SimpleNamespace(type="text", text="I refuse")],
            "No tool_use block",
        ),
        ("json", [], "No text block"),
    ],
    ids=["function_call-missing-tool_use", "json-missing-text-block"],
)
@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_structured_parse_raises_when_block_missing(
    mock_anthropic_class, structured_mode, content, match
):
    """Each structured-output mode raises a clear error when the expected block is absent."""
    from pydantic import BaseModel

    class Out(BaseModel):
        answer: str

    sdk = MagicMock()
    sdk.messages.create.return_value = SimpleNamespace(
        id="m",
        model="claude-opus-4-5",
        content=content,
        stop_reason="end_turn",
        stop_sequence=None,
        usage=SimpleNamespace(
            model_dump=lambda: {"input_tokens": 1, "output_tokens": 1}
        ),
    )
    mock_anthropic_class.return_value = sdk

    client = AnthropicChatClient(api_key="fake-key")
    with pytest.raises(ValueError, match=match):
        client.generate("?", response_format=Out, structured_mode=structured_mode)


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_response_format_list_of_model(mock_anthropic_class):
    """response_format=list[Item] wraps into `IterableItem` and parses the JSON into it."""
    from pydantic import BaseModel

    class Item(BaseModel):
        n: int

    sdk = MagicMock()
    sdk.messages.create.return_value = SimpleNamespace(
        id="m",
        model="claude-opus-4-7",
        content=[
            SimpleNamespace(type="text", text='{"objects": [{"n": 1}, {"n": 2}]}'),
        ],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=SimpleNamespace(
            model_dump=lambda: {"input_tokens": 1, "output_tokens": 1}
        ),
    )
    mock_anthropic_class.return_value = sdk

    client = AnthropicChatClient(api_key="fake-key")
    result = client.generate("?", response_format=list[Item])

    assert type(result).__name__ == "IterableItem"
    items = result.objects
    assert [it.n for it in items] == [1, 2]
    assert all(isinstance(it, Item) for it in items)

    schema = sdk.messages.create.call_args.kwargs["output_config"]["format"]["schema"]
    assert "objects" in schema["properties"]


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_response_format_rejects_unknown_mode(mock_anthropic_class):
    """structured_mode must be one of the two supported values."""
    from pydantic import BaseModel

    mock_anthropic_class.return_value = MagicMock()
    client = AnthropicChatClient(api_key="fake-key")

    class Out(BaseModel):
        answer: str

    with pytest.raises(ValueError, match="structured_mode"):
        client.generate("hi", response_format=Out, structured_mode="grammar")


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_json_mode_blocked_for_unsupported_model(mock_anthropic_class):
    """structured_mode='json' raises when the SDK reports structured_outputs unsupported."""
    from pydantic import BaseModel

    sdk = MagicMock()
    sdk.models.retrieve.return_value = SimpleNamespace(
        capabilities=SimpleNamespace(
            structured_outputs=SimpleNamespace(supported=False),
        ),
    )
    mock_anthropic_class.return_value = sdk

    client = AnthropicChatClient(api_key="fake-key", model="claude-3-opus-20240229")

    class Out(BaseModel):
        answer: str

    with pytest.raises(ValueError, match="function_call"):
        client.generate("hi", response_format=Out)

    sdk.messages.create.assert_not_called()


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_json_mode_allows_function_call_on_unsupported_model(
    mock_anthropic_class,
):
    """The gate only blocks json mode; function_call mode goes through regardless."""
    from pydantic import BaseModel

    sdk = MagicMock()
    sdk.models.retrieve.return_value = SimpleNamespace(
        capabilities=SimpleNamespace(
            structured_outputs=SimpleNamespace(supported=False),
        ),
    )
    sdk.messages.create.return_value = SimpleNamespace(
        id="msg_x",
        model="claude-3-opus-20240229",
        content=[
            SimpleNamespace(
                type="tool_use", id="tu_x", name="Out", input={"answer": "42"}
            ),
        ],
        stop_reason="tool_use",
        stop_sequence=None,
        usage=SimpleNamespace(
            model_dump=lambda: {"input_tokens": 1, "output_tokens": 1}
        ),
    )
    mock_anthropic_class.return_value = sdk

    client = AnthropicChatClient(api_key="fake-key", model="claude-3-opus-20240229")

    class Out(BaseModel):
        answer: str

    result = client.generate("hi", response_format=Out, structured_mode="function_call")
    assert isinstance(result, Out)
    sdk.models.retrieve.assert_not_called()


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_json_mode_fails_open_on_sdk_error(mock_anthropic_class):
    """If the capability lookup raises, we log and let the request through."""
    from pydantic import BaseModel

    sdk = MagicMock()
    sdk.models.retrieve.side_effect = RuntimeError("network down")
    sdk.messages.create.return_value = SimpleNamespace(
        id="msg_j",
        model="claude-some-future-1",
        content=[SimpleNamespace(type="text", text='{"answer": "ok"}')],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=SimpleNamespace(
            model_dump=lambda: {"input_tokens": 1, "output_tokens": 1}
        ),
    )
    mock_anthropic_class.return_value = sdk

    client = AnthropicChatClient(api_key="fake-key", model="claude-some-future-1")

    class Out(BaseModel):
        answer: str

    result = client.generate("hi", response_format=Out)
    assert isinstance(result, Out)
    assert result.answer == "ok"
    sdk.messages.create.assert_called_once()


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_json_mode_caches_capability_lookup(mock_anthropic_class):
    """Repeat generate() calls with the same client+model only hit models.retrieve once."""
    from pydantic import BaseModel

    sdk = MagicMock()
    sdk.models.retrieve.return_value = SimpleNamespace(
        capabilities=SimpleNamespace(
            structured_outputs=SimpleNamespace(supported=True),
        ),
    )
    sdk.messages.create.return_value = SimpleNamespace(
        id="msg_c",
        model="claude-sonnet-4-6",
        content=[SimpleNamespace(type="text", text='{"answer": "ok"}')],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=SimpleNamespace(
            model_dump=lambda: {"input_tokens": 1, "output_tokens": 1}
        ),
    )
    mock_anthropic_class.return_value = sdk

    client = AnthropicChatClient(api_key="fake-key", model="claude-sonnet-4-6")

    class Out(BaseModel):
        answer: str

    client.generate("hi", response_format=Out)
    client.generate("hi again", response_format=Out)

    assert sdk.models.retrieve.call_count == 1


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_from_prompty(mock_anthropic_class):
    """from_prompty loads model/config/parameters from an Anthropic Prompty source."""
    mock_anthropic_class.return_value = MagicMock()
    prompty_content = """---
name: Anthropic Test
model:
  api: chat
  configuration:
    type: anthropic
    name: claude-opus-4-5
    base_url: https://api.anthropic.com
    api_key: dummy_key
  parameters:
    temperature: 0.5
    max_tokens: 256
    top_k: 40
---
system:
You are a helpful assistant.
"""
    client = AnthropicChatClient.from_prompty(prompty_content)

    assert client.model == "claude-opus-4-5"
    assert client.api_key == "dummy_key"
    assert client.base_url == "https://api.anthropic.com"

    assert client.prompty is not None
    assert client.prompty.model.parameters.temperature == 0.5
    assert client.prompty.model.parameters.max_tokens == 256
    assert client.prompty.model.parameters.top_k == 40


PROMPTY_MISSING_MODEL_NAME = """---
name: Missing Model
model:
  api: chat
  configuration:
    type: anthropic
    api_key: dummy
  parameters:
    temperature: 0.5
---
system: hi
"""

PROMPTY_WRONG_PROVIDER_TYPE = """---
name: Wrong Provider
model:
  api: chat
  configuration:
    type: openai
    name: gpt-4o
    api_key: dummy
  parameters:
    temperature: 0.5
---
system: hi
"""


@pytest.mark.parametrize(
    "prompty_yaml, match",
    [
        (PROMPTY_MISSING_MODEL_NAME, "must specify a model name"),
        (
            PROMPTY_WRONG_PROVIDER_TYPE,
            "Expected Prompty model configuration type to be 'anthropic'",
        ),
    ],
    ids=["missing-model-name", "wrong-provider-type"],
)
@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_from_prompty_rejects_bad_config(
    mock_anthropic_class, prompty_yaml, match
):
    """from_prompty rejects Prompty configs with missing model name or wrong provider."""
    mock_anthropic_class.return_value = MagicMock()
    with pytest.raises(ValueError, match=match):
        AnthropicChatClient.from_prompty(prompty_yaml)


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_generate_merges_prompty_parameters(mock_anthropic_class):
    """Prompty-declared parameters flow into messages.create as defaults."""
    sdk = MagicMock()
    sdk.messages.create.return_value = _fake_anthropic_response("ok")
    mock_anthropic_class.return_value = sdk

    prompty_content = """---
name: Anthropic Test
model:
  api: chat
  configuration:
    type: anthropic
    name: claude-opus-4-5
    api_key: dummy_key
  parameters:
    temperature: 0.3
    max_tokens: 128
    top_k: 40
---
system:
be terse

user:
{{question}}
"""
    client = AnthropicChatClient.from_prompty(prompty_content)
    client.generate(messages=[UserMessage("hi")])

    kwargs = sdk.messages.create.call_args.kwargs
    assert kwargs["temperature"] == 0.3
    assert kwargs["max_tokens"] == 128
    assert kwargs["top_k"] == 40
    # Explicit call kwargs override Prompty defaults.
    sdk.messages.create.reset_mock()
    client.generate(messages=[UserMessage("hi")], temperature=0.9)
    assert sdk.messages.create.call_args.kwargs["temperature"] == 0.9


@patch("dapr_agents.llm.anthropic.client.Anthropic")
def test_anthropic_generate_requires_messages_or_input_data(mock_anthropic_class):
    mock_anthropic_class.return_value = MagicMock()
    client = AnthropicChatClient(api_key="fake-key")
    with pytest.raises(ValueError, match="Either messages or input_data"):
        client.generate()
