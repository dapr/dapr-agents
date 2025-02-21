from dapr_agents.types.llm import DaprInferenceClientConfig
from dapr_agents.llm.base import LLMClientBase
from typing import Optional, Dict, Any, List
from pydantic import Field, model_validator
import os
import logging
import requests
import json

logger = logging.getLogger(__name__)

class DaprClient:
    def __init__(self):
        self._dapr_endpoint = os.getenv('DAPR_BASE_URL', 'http://localhost') + ':' + os.getenv(
                    'DAPR_HTTP_PORT', '3500')

    def chat_completion(self, llm: str, request: List[Dict]) -> Any:
        # Invoke Dapr
        result = requests.post(
            url='%s/v1.0-alpha1/conversation/%s/converse' % (self._dapr_endpoint, llm),
            data=json.dumps(request)
        )
        
        return result.json()

class DaprInferenceClientBase(LLMClientBase):
    """
    Base class for managing Dapr Inference API clients.
    Handles client initialization, configuration, and shared logic.
    """
    @model_validator(mode="before")
    def validate_and_initialize(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        return values

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes private attributes after validation.
        """
        self._provider = "dapr"

        # Set up the private config and client attributes
        self._config = self.get_config()
        self._client = self.get_client()
        return super().model_post_init(__context)
    
    def get_config(self) -> DaprInferenceClientConfig:
        """
        Returns the appropriate configuration for the Dapr Conversation API.
        """
        return DaprInferenceClientConfig()

    def get_client(self) -> DaprClient:
        """
        Initializes and returns the Dapr Inference client.
        """
        config: DaprInferenceClientConfig = self.config
        return DaprClient()
    
    @classmethod
    def from_config(cls, client_options: DaprInferenceClientConfig, timeout: float = 1500):
        """
        Initializes the DaprInferenceClientBase using DaprInferenceClientConfig.

        Args:
            client_options: The configuration options for the client.
            timeout: Timeout for requests (default is 1500 seconds).

        Returns:
            DaprInferenceClientBase: The initialized client instance.
        """
        return cls()

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @property
    def client(self) -> DaprClient:
        return self._client
