from pydantic import BaseModel, PrivateAttr
from abc import ABC, abstractmethod
from typing import Any

class LLMClientBase(BaseModel, ABC):
    """
    Abstract base class for LLM models.
    """
    # Private attributes for provider and api
    _provider: str = PrivateAttr()
    _api: str = PrivateAttr()

    # Private attributes for config and client
    _config: Any = PrivateAttr()
    _client: Any = PrivateAttr()
    
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