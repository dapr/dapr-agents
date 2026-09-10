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
from huggingface_hub import ChatCompletionOutput, ChatCompletionStreamOutput
from huggingface_hub.inference._generated.types.chat_completion import (
    ChatCompletionOutputComplete,
    ChatCompletionOutputMessage,
    ChatCompletionOutputUsage,
    ChatCompletionStreamOutputChoice,
    ChatCompletionStreamOutputDelta,
    ChatCompletionStreamOutputUsage,
)
from pydantic import BaseModel

from dapr_agents.llm.huggingface.chat import HFHubChatClient
from dapr_agents.llm.huggingface.client import HFHubInferenceClientBase


class Answer(BaseModel):
    answer: str


def _output(content: str = "Hello") -> ChatCompletionOutput:
    return ChatCompletionOutput(
        choices=[
            ChatCompletionOutputComplete(
                finish_reason="stop",
                index=0,
                message=ChatCompletionOutputMessage(role="assistant", content=content),
            )
        ],
        created=1,
        id="hf-test",
        model="hf-test-model",
        system_fingerprint="test-fingerprint",
        usage=ChatCompletionOutputUsage(
            completion_tokens=1, prompt_tokens=1, total_tokens=2
        ),
    )


def _stream() -> list[ChatCompletionStreamOutput]:
    metadata = {
        "created": 1,
        "id": "hf-stream",
        "model": "hf-test-model",
        "system_fingerprint": "test-fingerprint",
    }
    return [
        ChatCompletionStreamOutput(
            **metadata,
            choices=[
                ChatCompletionStreamOutputChoice(
                    index=0,
                    delta=ChatCompletionStreamOutputDelta(
                        role="assistant", content="Hi"
                    ),
                    finish_reason=None,
                )
            ],
        ),
        ChatCompletionStreamOutput(
            **metadata,
            choices=[],
            usage=ChatCompletionStreamOutputUsage(
                completion_tokens=1, prompt_tokens=1, total_tokens=2
            ),
        ),
    ]


def _client(api: MagicMock) -> HFHubChatClient:
    with patch.object(HFHubInferenceClientBase, "get_client", return_value=api):
        return HFHubChatClient(api_key="test-key", model="hf-test-model")


def test_initialization_accepts_token_alias_and_derives_model() -> None:
    api = MagicMock()
    with patch.object(HFHubInferenceClientBase, "get_client", return_value=api):
        client = HFHubChatClient(
            token="test-token", base_url="https://hf.test/models/demo"
        )

    assert client.api_key == "test-token"
    assert client.model == "demo"
    assert client.provider == "huggingface"
    assert client.api == "chat"


def test_initialization_rejects_conflicting_or_missing_configuration() -> None:
    with pytest.raises(ValueError, match="only one"):
        HFHubChatClient(
            api_key="key",
            token="token",
            model="hf-test-model",
        )
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="API key is required"):
            HFHubChatClient(model="hf-test-model")


def test_generate_forwards_messages_model_and_kwargs() -> None:
    api = MagicMock()
    api.chat.completions.create.return_value = _output()
    client = _client(api)

    response = client.generate(
        [{"role": "user", "content": "Hello"}],
        model="hf-override",
        temperature=0.3,
    )

    assert response.get_message().content == "Hello"
    call = api.chat.completions.create.call_args.kwargs
    assert call["messages"] == [{"role": "user", "content": "Hello"}]
    assert call["model"] == "hf-override"
    assert call["temperature"] == 0.3
    assert call["stream"] is False


def test_generate_structured_and_streaming_paths() -> None:
    api = MagicMock()
    api.chat.completions.create.side_effect = [
        _output('{"answer":"ok"}'),
        iter(_stream()),
    ]
    client = _client(api)

    structured = client.generate(
        "Return an answer",
        response_format=Answer,
        structured_mode="function_call",
    )
    streamed = list(client.generate("Say hi", stream=True))

    assert structured == Answer(answer="ok")
    assert "Hi" == "".join(chunk.result.content or "" for chunk in streamed)
    stream_call = api.chat.completions.create.call_args_list[1].kwargs
    assert stream_call["stream"] is True


def test_generate_processes_hf_error_responses_and_api_errors() -> None:
    api = MagicMock()
    error_response = MagicMock(code=503, message="model unavailable")
    api.chat.completions.create.return_value = error_response
    client = _client(api)

    with pytest.raises(ValueError, match="HuggingFace error 503"):
        client.generate("hello")

    api.chat.completions.create.side_effect = RuntimeError("network down")
    with pytest.raises(ValueError, match="Failed to process HF chat completion"):
        client.generate("hello")


def test_generate_rejects_invalid_structured_mode_and_missing_messages() -> None:
    client = _client(MagicMock())

    with pytest.raises(ValueError, match="structured_mode"):
        client.generate("hello", structured_mode="json")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Either messages"):
        client.generate()


def test_from_prompty_builds_huggingface_client() -> None:
    source = """---
name: Hugging Face Test
model:
  api: chat
  configuration:
    type: huggingface
    name: hf-prompty
    api_key: test-key
    endpoint: https://hf.test
  parameters:
    temperature: 0.2
---
system:
You are helpful.
"""
    with patch.object(HFHubInferenceClientBase, "get_client", return_value=MagicMock()):
        client = HFHubChatClient.from_prompty(source)

    assert client.model == "hf-prompty"
    assert client.api_key == "test-key"
    assert client.prompty is not None
    assert client.prompt_template is not None
