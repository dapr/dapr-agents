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
from typing import Any, Optional

from pydantic import Field

from dapr_agents.llm.base import LLMClientBase
from dapr_agents.types.llm import LiteLLMClientConfig

logger = logging.getLogger(__name__)

PROVIDER = "litellm"


class LiteLLMClientBase(LLMClientBase):
    api_key: Optional[str] = Field(
        default=None,
        description="API key for LiteLLM proxy or the underlying provider.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="Base URL for a LiteLLM proxy or compatible endpoint.",
    )

    def model_post_init(self, __context: Any) -> None:
        self._provider = PROVIDER
        self._config: LiteLLMClientConfig = self.get_config()
        self._client = self.get_client()
        return super().model_post_init(__context)

    def get_config(self) -> LiteLLMClientConfig:
        return LiteLLMClientConfig(
            api_key=self.api_key or os.environ.get("LITELLM_API_KEY"),
            api_base=self.api_base or os.environ.get("LITELLM_API_BASE"),
        )

    def get_client(self) -> None:
        # LiteLLM uses a functional API (litellm.completion());
        # no persistent client object is needed.
        return None
