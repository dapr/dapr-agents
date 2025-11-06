from pydantic import BaseModel, PrivateAttr, Field
from abc import ABC, abstractmethod
from typing import Any, Optional
from dapr_agents.prompt.base import PromptTemplateBase


class LLMClientBase(BaseModel, ABC):
    """
    Abstract base class for LLM models.
    """

    _provider: str = PrivateAttr()
    _api: str = PrivateAttr()
    _config: Any = PrivateAttr()
    _client: Any = PrivateAttr()
    prompt_template: Optional[PromptTemplateBase] = Field(
        default=None, description="Prompt template for rendering (optional)."
    )

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def api(self) -> str:
        return self._api

    @property
    def config(self) -> Any:
        return self._config

    @property
    def client(self) -> Any:
        return self._client

    @abstractmethod
    def get_client(self) -> Any:
        """Abstract method to get the client for the LLM model."""
        pass

    @abstractmethod
    def get_config(self) -> Any:
        """Abstract method to get the configuration for the LLM model."""
        pass

    def refresh_client(self) -> None:
        """
        Public method to refresh the client by regenerating the config and client.
        """
        # Refresh config and client using the current state
        self._config = self.get_config()
        self._client = self.get_client()
