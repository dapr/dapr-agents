#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from dapr_agents.types.llm import IFlytekClientConfig
from dapr_agents.llm.base import LLMClientBase
from typing import Any, Optional
from pydantic import Field
from openai import OpenAI
import os
import logging

logger = logging.getLogger(__name__)


class IFlytekClientBase(LLMClientBase):
    """
    Base class for managing iFlytek Spark LLM clients.

    iFlytek Spark exposes an OpenAI-compatible chat-completions endpoint, so the
    OpenAI SDK is reused for transport, mirroring the other OpenAI-compatible
    providers (e.g. NVIDIA). Handles client initialization, configuration, and
    shared logic specific to the iFlytek Spark API.
    """

    api_key: Optional[str] = Field(
        default=None,
        description=(
            "API key (the HTTP API password from the iFlytek open-platform console) "
            "for authenticating with the iFlytek Spark API. If not provided, it will "
            "be sourced from the 'IFLYTEK_API_KEY' environment variable."
        ),
    )
    base_url: Optional[str] = Field(
        default="https://spark-api-open.xf-yun.com/v1",
        description="Base URL for the iFlytek Spark OpenAI-compatible API endpoints.",
    )

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes private attributes and performs any post-validation setup.

        This includes setting up provider-specific attributes such as configuration
        and client instances.

        Args:
            __context (Any): Additional context for post-initialization (not used here).
        """
        self._provider = "iflytek"

        # Use environment variable if `api_key` is not explicitly provided
        if self.api_key is None:
            self.api_key = os.environ.get("IFLYTEK_API_KEY")

        if self.api_key is None:
            raise ValueError(
                "API key is required. Set it explicitly or in the 'IFLYTEK_API_KEY' environment variable."
            )

        # Set up the private config and client attributes
        self._config: IFlytekClientConfig = self.get_config()
        self._client: OpenAI = self.get_client()
        return super().model_post_init(__context)

    def get_config(self) -> IFlytekClientConfig:
        """
        Returns the configuration object for the iFlytek Spark LLM API client.

        Returns:
            IFlytekClientConfig: Configuration object containing API credentials and endpoint details.
        """
        return IFlytekClientConfig(api_key=self.api_key, base_url=self.base_url)

    def get_client(self) -> OpenAI:
        """
        Initializes and returns the iFlytek Spark LLM API client.

        Returns:
            OpenAI: The initialized iFlytek Spark API client instance.
        """
        config = self.config

        logger.info("Initializing iFlytek Spark API client...")
        return OpenAI(api_key=config.api_key, base_url=config.base_url)

    @property
    def config(self) -> IFlytekClientConfig:
        """
        Provides access to the iFlytek Spark API client configuration.

        Returns:
            IFlytekClientConfig: Configuration object for the iFlytek Spark API client.
        """
        return self._config

    @property
    def client(self) -> OpenAI:
        """
        Provides access to the iFlytek Spark API client instance.

        Returns:
            OpenAI: The iFlytek Spark API client instance.
        """
        return self._client
