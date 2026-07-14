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
from unittest.mock import patch, MagicMock

import pytest

from dapr_agents.llm.iflytek.chat import IFlytekChatClient


def test_iflytek_client_initialization():
    """Test default model, base URL, provider and API key env fallback."""
    with patch.dict(os.environ, {"IFLYTEK_API_KEY": "env-key"}, clear=False):
        client = IFlytekChatClient()
    assert client.model == "generalv3.5"
    assert client.base_url == "https://spark-api-open.xf-yun.com/v1"
    assert client.provider == "iflytek"
    assert client.api == "chat"
    assert client.api_key == "env-key"


def test_iflytek_requires_api_key():
    """Test that a missing API key raises a clear error."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="API key is required"):
            IFlytekChatClient()


@patch("dapr_agents.llm.iflytek.client.OpenAI")
def test_iflytek_generate(mock_openai_class):
    """Test basic text generation using a mocked OpenAI-compatible client."""
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice

    mock_instance = MagicMock()
    mock_openai_class.return_value = mock_instance

    completion = ChatCompletion(
        id="test-iflytek-id",
        object="chat.completion",
        created=0,
        model="generalv3.5",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant", content="Hello from Spark"
                ),
            )
        ],
    )
    mock_instance.chat.completions.create.return_value = completion

    client = IFlytekChatClient(api_key="fake-key")
    response = client.generate("Say hello")

    assert response.get_message().content == "Hello from Spark"
    mock_instance.chat.completions.create.assert_called_once()


def test_iflytek_from_prompty():
    """Test initializing IFlytekChatClient from a Prompty template."""
    prompty_content = """---
name: iFlytek Test
model:
  api: chat
  configuration:
    type: iflytek
    name: 4.0Ultra
    base_url: https://spark-api-open.xf-yun.com/v1
    api_key: dummy_key
  parameters:
    temperature: 0.5
    max_tokens: 100
---
system:
You are a helpful assistant.
"""
    client = IFlytekChatClient.from_prompty(prompty_content)

    assert client.model == "4.0Ultra"
    assert client.api_key == "dummy_key"
    assert client.base_url == "https://spark-api-open.xf-yun.com/v1"

    assert client.prompty is not None
    assert client.prompty.model.parameters.temperature == 0.5
    assert client.prompty.model.parameters.max_tokens == 100
