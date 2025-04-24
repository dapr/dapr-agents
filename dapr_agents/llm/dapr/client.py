from dapr_agents.types.llm import DaprInferenceClientConfig
from dapr_agents.llm.base import LLMClientBase
from dapr.clients import DaprClient
from dapr.clients.grpc._request import ConversationInput
from dapr.clients.grpc._response import ConversationResponse
from typing import Dict, Any, List
from pydantic import model_validator

import logging

logger = logging.getLogger(__name__)


class DaprInferenceClient:
    def __init__(self):
        self.dapr_client = DaprClient()

    def translate_to_json(self, response: ConversationResponse) -> dict:
        response_dict = {
            "outputs": [
                {
                    "result": output.result,
                }
                for output in response.outputs
            ]
        }

        return response_dict

    def chat_completion(
        self,
        llm: str,
        conversation_inputs: List[ConversationInput],
        scrub_pii: bool | None = None,
        temperature: float | None = None,
    ) -> Any:
        response = self.dapr_client.converse_alpha1(
            name=llm,
            inputs=conversation_inputs,
            scrub_pii=scrub_pii,
            temperature=temperature,
        )
        output = self.translate_to_json(response)

        return output


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

    def get_client(self) -> DaprInferenceClient:
        """
        Initializes and returns the Dapr Inference client.
        """
        return DaprInferenceClient()

    @classmethod
    def from_config(
        cls, client_options: DaprInferenceClientConfig, timeout: float = 1500
    ):
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
    def client(self) -> DaprInferenceClient:
        return self._client
