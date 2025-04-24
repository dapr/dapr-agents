from dapr_agents.types.llm import NVIDIAClientConfig
from dapr_agents.llm.base import LLMClientBase
from typing import Any, Optional
from pydantic import Field
from openai import OpenAI
import os
import logging

logger = logging.getLogger(__name__)


class NVIDIAClientBase(LLMClientBase):
    """
    Base class for managing NVIDIA LLM clients.
    Handles client initialization, configuration, and shared logic specific to NVIDIA's API.
    """

    api_key: Optional[str] = Field(
        default=None,
        description="API key for authenticating with the NVIDIA LLM API. If not provided, it will be sourced from the 'NVIDIA_API_KEY' environment variable.",
    )
    base_url: Optional[str] = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="Base URL for the NVIDIA LLM API endpoints.",
    )

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes private attributes and performs any post-validation setup.

        This includes setting up provider-specific attributes such as configuration and client instances.

        Args:
            __context (Any): Additional context for post-initialization (not used here).
        """
        self._provider = "nvidia"

        # Use environment variable if `api_key` is not explicitly provided
        if self.api_key is None:
            self.api_key = os.environ.get("NVIDIA_API_KEY")

        if self.api_key is None:
            raise ValueError(
                "API key is required. Set it explicitly or in the 'NVIDIA_API_KEY' environment variable."
            )

        # Set up the private config and client attributes
        self._config: NVIDIAClientConfig = self.get_config()
        self._client: OpenAI = self.get_client()
        return super().model_post_init(__context)

    def get_config(self) -> NVIDIAClientConfig:
        """
        Returns the configuration object for the NVIDIA LLM API client.

        This configuration includes the API key and base URL, ensuring the client can communicate with the API.

        Returns:
            NVIDIAClientConfig: Configuration object containing API credentials and endpoint details.
        """
        return NVIDIAClientConfig(api_key=self.api_key, base_url=self.base_url)

    def get_client(self) -> OpenAI:
        """
        Initializes and returns the NVIDIA LLM API client.

        This method sets up the client using the provided configuration.

        Returns:
            OpenAI: The initialized NVIDIA API client instance.
        """
        config = self.config

        logger.info("Initializing NVIDIA API client...")
        return OpenAI(api_key=config.api_key, base_url=config.base_url)

    @property
    def config(self) -> NVIDIAClientConfig:
        """
        Provides access to the NVIDIA API client configuration.

        Returns:
            NVIDIAClientConfig: Configuration object for the NVIDIA API client.
        """
        return self._config

    @property
    def client(self) -> OpenAI:
        """
        Provides access to the NVIDIA API client instance.

        Returns:
            OpenAI: The NVIDIA API client instance.
        """
        return self._client
