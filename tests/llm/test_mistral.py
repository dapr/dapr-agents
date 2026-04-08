import os
from unittest.mock import patch, MagicMock
from dapr_agents.llm.mistral.chat import MistralChatClient

def test_mistral_client_initialization():
    """Test default model and environment variable fallback."""
    client = MistralChatClient()
    assert client.model == "mistral-large-latest"

    os.environ["MISTRAL_MODEL"] = "mistral-small"
    client_env = MistralChatClient()
    assert client_env.model == "mistral-small"
    del os.environ["MISTRAL_MODEL"]

@patch("dapr_agents.llm.mistral.chat.ResponseHandler.process_response")
@patch("dapr_agents.llm.mistral.client.Mistral")
def test_mistral_generate(mock_mistral_class, mock_process_response):
    """Test basic text generation using mocked SDK."""
    mock_instance = MagicMock()
    mock_mistral_class.return_value = mock_instance
    
    mock_chat_response = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Hello from Mistral"
    mock_chat_response.get_message.return_value = mock_message
    mock_process_response.return_value = mock_chat_response

    client = MistralChatClient(api_key="fake-key")
    response = client.generate("Say hello")
    
    assert response.get_message().content == "Hello from Mistral"
    mock_instance.chat.complete.assert_called_once()