from dapr_agents.types.llm import ElevenLabsClientConfig
from dapr_agents.llm.base import LLMClientBase
from typing import Any, Optional
from pydantic import Field
import os
import logging

logger = logging.getLogger(__name__)


class ElevenLabsClientBase(LLMClientBase):
    """
    Base class for managing ElevenLabs LLM clients.
    Handles client initialization, configuration, and shared logic specific to the ElevenLabs API.
    """

    api_key: Optional[str] = Field(
        default=None,
        description="API key for authenticating with the ElevenLabs API. Defaults to environment variables 'ELEVENLABS_API_KEY' or 'ELEVEN_API_KEY'.",
    )
    base_url: Optional[str] = Field(
        default="https://api.elevenlabs.io",
        description="Base URL for the ElevenLabs API endpoints.",
    )

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes private attributes and performs any post-validation setup.

        This includes setting up provider-specific attributes such as configuration and client instances.
        """
        self._provider = "elevenlabs"

        # Use environment variable if `api_key` is not explicitly provided
        if self.api_key is None:
            self.api_key = os.getenv("ELEVENLABS_API_KEY") or os.getenv(
                "ELEVEN_API_KEY"
            )

        if self.api_key is None:
            raise ValueError(
                "API key is required. Set it explicitly or in the 'ELEVENLABS_API_KEY' or 'ELEVEN_API_KEY' environment variable."
            )

        # Initialize configuration and client
        self._config = self.get_config()
        self._client = self.get_client()
        logger.info("ElevenLabs client initialized successfully.")

        return super().model_post_init(__context)

    def get_config(self) -> ElevenLabsClientConfig:
        """
        Returns the configuration object for the ElevenLabs API client.
        """
        return ElevenLabsClientConfig(api_key=self.api_key, base_url=self.base_url)

    def get_client(self) -> Any:
        """
        Initializes and returns the ElevenLabs API client.

        This method sets up the client using the provided configuration.
        """
        try:
            from elevenlabs import ElevenLabs
        except ImportError as e:
            raise ImportError(
                "The 'elevenlabs' package is required but not installed. Install it with 'pip install elevenlabs'."
            ) from e

        config = self.config

        logger.info("Initializing ElevenLabs API client...")
        return ElevenLabs(api_key=config.api_key, base_url=config.base_url)

    @property
    def config(self) -> ElevenLabsClientConfig:
        """
        Provides access to the ElevenLabs API client configuration.
        """
        return self._config

    @property
    def client(self) -> Any:
        """
        Provides access to the ElevenLabs API client instance.
        """
        return self._client
