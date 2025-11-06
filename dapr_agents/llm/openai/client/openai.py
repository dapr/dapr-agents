import logging
from typing import Optional, Union

from openai import OpenAI

from dapr_agents.llm.utils import HTTPHelper
from dapr_agents.types.llm import OpenAIClientConfig

logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    Client for interfacing with OpenAI's language models.
    This client handles API communication, including sending requests and processing responses.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        timeout: Union[int, float, dict] = 1500,
    ):
        """
        Initializes the OpenAI client with API key, base URL, and organization.

        Args:
            api_key: The OpenAI API key (will fall back to OPENAI_API_KEY env var if not provided).
            base_url: The base URL for OpenAI API (defaults to https://api.openai.com/v1).
            organization: The OpenAI organization (optional).
            project: The OpenAI Project name (optional).
            timeout: Timeout for requests (default is 1500 seconds).
        """
        self.api_key = api_key  # Will be None if not provided - OpenAI SDK will handle env var fallback
        self.base_url = base_url
        self.organization = organization
        self.project = project
        self.timeout = HTTPHelper.configure_timeout(timeout)

    def get_client(self) -> OpenAI:
        """
        Returns the OpenAI client.

        Returns:
            OpenAI: The initialized OpenAI client.
        """
        # Build kwargs, only including non-None values so SDK can fall back to env vars
        kwargs = {}
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        if self.organization is not None:
            kwargs["organization"] = self.organization
        if self.project is not None:
            kwargs["project"] = self.project
        kwargs["timeout"] = self.timeout

        return OpenAI(**kwargs)

    @classmethod
    def from_config(
        cls, client_options: OpenAIClientConfig, timeout: Union[int, float, dict] = 1500
    ):
        """
        Initialize OpenAIBaseClient using OpenAIClientConfig.

        Args:
            client_options: The client options containing the configuration.
            timeout: Timeout for requests (default is 1500 seconds).

        Returns:
            OpenAIBaseClient: An initialized instance.
        """
        return cls(
            api_key=client_options.api_key,
            base_url=client_options.base_url,
            organization=client_options.organization,
            project=client_options.project,
            timeout=timeout,
        )
