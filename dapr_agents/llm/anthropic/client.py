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


import logging
import os
from typing import Any

from anthropic import Anthropic
from pydantic import Field

from dapr_agents.llm.base import LLMClientBase
from dapr_agents.llm.utils import HTTPHelper
from dapr_agents.types.llm import AnthropicClientConfig

logger = logging.getLogger(__name__)

PROVIDER = "anthropic"


class AnthropicClientBase(LLMClientBase):
    api_key: str | None = Field(
        default=None,
        description="API key for Anthropic. Falls back to ANTHROPIC_API_KEY env var.",
    )
    base_url: str | None = Field(
        default=None,
        description="Base URL override for the Anthropic API (proxy or compatible endpoint).",
    )
    timeout: int | float | dict[str, Any] = Field(
        default=1500,
        description="Default request timeout in seconds, or an httpx.Timeout-style kwarg dict.",
    )

    def model_post_init(self, __context: Any) -> None:
        self._provider = PROVIDER
        self._config: AnthropicClientConfig = self.get_config()
        self._client = self.get_client()
        return super().model_post_init(__context)

    def get_config(self) -> AnthropicClientConfig:
        return AnthropicClientConfig(
            api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY"),
            base_url=self.base_url or os.environ.get("ANTHROPIC_BASE_URL"),
        )

    def get_client(self) -> Anthropic:
        config = self.config
        kwargs: dict[str, Any] = {"timeout": HTTPHelper.configure_timeout(self.timeout)}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url

        logger.info("Initializing Anthropic API client...")
        return Anthropic(**kwargs)
