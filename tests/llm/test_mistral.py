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
from dapr_agents.llm.mistral.chat import MistralChatClient


def test_mistral_client_initialization():
    """Test default model and environment variable fallback."""
    client = MistralChatClient()
    assert client.model == "mistral-large-latest"

    with patch.dict(os.environ, {"MISTRAL_MODEL": "mistral-small"}, clear=False):
        client_env = MistralChatClient()
        assert client_env.model == "mistral-small"


@patch("dapr_agents.llm.mistral.client.Mistral")
def test_mistral_generate(mock_mistral_class):
    """Test basic text generation using mocked SDK."""
    mock_instance = MagicMock()
    mock_mistral_class.return_value = mock_instance

    mock_message = MagicMock()
    mock_message.role = "assistant"
    mock_message.content = "Hello from Mistral"

    mock_choice = MagicMock()
    mock_choice.index = 0
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_resp = MagicMock()
    mock_resp.id = "test-mistral-id"
    mock_resp.model = "mistral-large-latest"
    mock_resp.choices = [mock_choice]
    mock_resp.usage = None

    mock_instance.chat.complete.return_value = mock_resp

    client = MistralChatClient(api_key="fake-key")
    response = client.generate("Say hello")

    assert response.get_message().content == "Hello from Mistral"
    mock_instance.chat.complete.assert_called_once()
