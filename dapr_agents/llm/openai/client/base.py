from dapr_agents.types.llm import OpenAIClientConfig, AzureOpenAIClientConfig
from dapr_agents.llm.openai.client import AzureOpenAIClient, OpenAIClient
from dapr_agents.llm.base import LLMClientBase
from openai import OpenAI, AzureOpenAI
from typing import Any, Optional, Union, Dict
from pydantic import Field
import logging

logger = logging.getLogger(__name__)


class OpenAIClientBase(LLMClientBase):
    """
    Base class for managing OpenAI and Azure OpenAI clients.
    Handles client initialization, configuration, and shared logic.
    """

    api_key: Optional[str] = Field(
        default=None, description="API key for OpenAI or Azure OpenAI."
    )
    base_url: Optional[str] = Field(
        default=None, description="Base URL for OpenAI API (OpenAI-specific)."
    )
    azure_endpoint: Optional[str] = Field(
        default=None, description="Azure endpoint URL (Azure OpenAI-specific)."
    )
    azure_deployment: Optional[str] = Field(
        default=None, description="Azure deployment name (Azure OpenAI-specific)."
    )
    api_version: Optional[str] = Field(
        default=None, description="Azure API version (Azure OpenAI-specific)."
    )
    organization: Optional[str] = Field(
        default=None, description="Organization for OpenAI or Azure OpenAI."
    )
    project: Optional[str] = Field(
        default=None, description="Project for OpenAI or Azure OpenAI."
    )
    azure_ad_token: Optional[str] = Field(
        default=None, description="Azure AD token for authentication (Azure-specific)."
    )
    azure_client_id: Optional[str] = Field(
        default=None, description="Client ID for Managed Identity (Azure-specific)."
    )
    timeout: Union[int, float, Dict[str, Any]] = Field(
        default=1500, description="Timeout for requests in seconds."
    )

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes private attributes after validation.
        """
        self._provider = "openai"

        # Set up the private config and client attributes
        self._config: Union[
            AzureOpenAIClientConfig, OpenAIClientConfig
        ] = self.get_config()
        self._client: Union[AzureOpenAI, OpenAI] = self.get_client()
        return super().model_post_init(__context)

    def get_config(self) -> Union[OpenAIClientConfig, AzureOpenAIClientConfig]:
        """
        Returns the appropriate configuration for OpenAI or Azure OpenAI.
        """
        is_azure = self.azure_endpoint or self.azure_deployment

        if is_azure:
            return AzureOpenAIClientConfig(
                api_key=self.api_key,
                organization=self.organization,
                project=self.project,
                azure_ad_token=self.azure_ad_token,
                azure_endpoint=self.azure_endpoint,
                azure_deployment=self.azure_deployment,
                api_version=self.api_version,
            )
        else:
            return OpenAIClientConfig(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
                project=self.project,
            )

    def get_client(self) -> Union[AzureOpenAI, OpenAI]:
        """
        Initialize and return the appropriate client (OpenAI or Azure OpenAI).
        """
        config = self.config
        timeout = self.timeout

        if isinstance(config, AzureOpenAIClientConfig):
            logger.info("Initializing Azure OpenAI client...")
            return AzureOpenAIClient(
                api_key=config.api_key,
                azure_ad_token=config.azure_ad_token,
                azure_endpoint=config.azure_endpoint,
                azure_deployment=config.azure_deployment,
                api_version=config.api_version,
                organization=config.organization,
                project=config.project,
                azure_client_id=self.azure_client_id,
                timeout=timeout,
            ).get_client()

        logger.info("Initializing OpenAI client...")
        return OpenAIClient(
            api_key=config.api_key,
            base_url=config.base_url,
            organization=config.organization,
            project=config.project,
            timeout=timeout,
        ).get_client()

    @property
    def config(self) -> Union[AzureOpenAIClientConfig, OpenAIClientConfig]:
        return self._config

    @property
    def client(self) -> Union[OpenAI, AzureOpenAI]:
        return self._client
