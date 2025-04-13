from dapr_agents.types.llm import OpenAIClientConfig
from dapr_agents.llm.utils import HTTPHelper
from typing import Union, Optional
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    Client for interfacing with OpenAI's language models.
    This client handles API communication, including sending requests and processing responses.
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        timeout: Union[int, float, dict] = 1500,
    ):
        """
        Initializes the OpenAI client with API key, base URL, and organization.

        Args:
            api_key: The OpenAI API key.
            base_url: The base URL for OpenAI API (defaults to https://api.openai.com/v1).
            organization: The OpenAI organization (optional).
            project: The OpenAI Project name (optional).
            timeout: Timeout for requests (default is 1500 seconds).
        """
        self.api_key = api_key  # or inferred from OPENAI_API_KEY env variable.
        self.base_url = base_url  # or set to "https://api.openai.com/v1" by default.
        self.organization = organization  # or inferred from OPENAI_ORG_ID env variable.
        self.project = project  # or inferred from OPENAI_PROJECT_ID env variable.
        self.timeout = HTTPHelper.configure_timeout(timeout)

    def get_client(self) -> OpenAI:
        """
        Returns the OpenAI client.

        Returns:
            OpenAI: The initialized OpenAI client.
        """
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization,
            project=self.project,
            timeout=self.timeout,
        )

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
