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
from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel

from dapr_agents.llm.openai.chat import OpenAIChatClient
from dapr_agents.llm.openai.client.base import OpenAIClientBase
from dapr_agents.types.message import UserMessage


class Answer(BaseModel):
    answer: str


def _completion(content: str = "Hello") -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-test",
        created=1,
        model="gpt-test",
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    )


def _stream() -> list[ChatCompletionChunk]:
    return [
        ChatCompletionChunk(
            id="chatcmpl-stream",
            created=1,
            model="gpt-test",
            object="chat.completion.chunk",
            choices=[
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hi"},
                    "finish_reason": None,
                }
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-stream",
            created=1,
            model="gpt-test",
            object="chat.completion.chunk",
            choices=[
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        ),
    ]


def _client(api: MagicMock) -> OpenAIChatClient:
    with patch.object(OpenAIClientBase, "get_client", return_value=api):
        return OpenAIChatClient(api_key="test-key", model="gpt-test")


def test_initialization_uses_environment_model() -> None:
    api = MagicMock()
    with patch.object(OpenAIClientBase, "get_client", return_value=api):
        with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-from-env"}):
            client = OpenAIChatClient(api_key="test-key")

    assert client.model == "gpt-from-env"
    assert client.provider == "openai"
    assert client.api == "chat"


def test_generate_normalizes_supported_message_inputs_and_forwards_options() -> None:
    api = MagicMock()
    api.chat.completions.create.return_value = _completion()
    client = _client(api)

    response = client.generate(
        [UserMessage(content="Hello")],
        model="gpt-override",
        temperature=0.2,
        top_p=0.9,
    )

    assert response.get_message().content == "Hello"
    call = api.chat.completions.create.call_args.kwargs
    assert call["messages"] == [{"role": "user", "content": "Hello"}]
    assert call["model"] == "gpt-override"
    assert call["temperature"] == 0.2
    assert call["top_p"] == 0.9
    assert call["stream"] is False
    assert call["timeout"] == client.timeout


def test_generate_structured_json_and_streaming_paths() -> None:
    api = MagicMock()
    api.chat.completions.create.side_effect = [
        _completion('{"answer":"ok"}'),
        iter(_stream()),
    ]
    client = _client(api)

    structured = client.generate("Return an answer", response_format=Answer)
    streamed = list(client.generate("Say hi", stream=True))

    assert structured == Answer(answer="ok")
    assert "Hi" == "".join(chunk.result.content or "" for chunk in streamed)
    structured_call = api.chat.completions.create.call_args_list[0].kwargs
    assert structured_call["response_format"]["type"] == "json_schema"
    stream_call = api.chat.completions.create.call_args_list[1].kwargs
    # Usage is requested on the terminal stream packet for observability.
    assert stream_call["stream_options"]["include_usage"] is True


def test_generate_accepts_tools_and_rejects_invalid_inputs() -> None:
    api = MagicMock()
    api.chat.completions.create.return_value = _completion()
    client = _client(api)

    client.generate(
        {"role": "user", "content": "Use a tool"},
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a value.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    assert api.chat.completions.create.call_args.kwargs["tools"]

    with pytest.raises(ValueError, match="structured_mode"):
        client.generate("hello", structured_mode="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Either messages"):
        client.generate()


def test_generate_wraps_api_errors() -> None:
    api = MagicMock()
    api.chat.completions.create.side_effect = RuntimeError("service unavailable")
    client = _client(api)

    with pytest.raises(ValueError, match=r"OpenAI API error \(RuntimeError\)"):
        client.generate("hello")


def test_from_prompty_builds_openai_client() -> None:
    source = """---
name: OpenAI Test
model:
  api: chat
  configuration:
    type: openai
    name: gpt-prompty
    api_key: test-key
  parameters:
    temperature: 0.1
---
system:
You are helpful.
"""
    with patch.object(OpenAIClientBase, "get_client", return_value=MagicMock()):
        client = OpenAIChatClient.from_prompty(source)

    assert client.model == "gpt-prompty"
    assert client.api_key == "test-key"
    assert client.prompty is not None
    assert client.prompt_template is not None
