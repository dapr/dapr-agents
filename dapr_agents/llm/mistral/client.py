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
from typing import Any, Optional

from mistralai.client import Mistral

from dapr_agents.llm.base import LLMClientBase
from dapr_agents.types.llm import MistralClientConfig

logger = logging.getLogger(__name__)


class MistralClientBase(LLMClientBase):
    api_key: Optional[str] = None
    endpoint: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        self._provider = "mistral"
        super().model_post_init(__context)
        self.refresh_client()

    def get_config(self) -> MistralClientConfig:
        return MistralClientConfig(api_key=self.api_key, endpoint=self.endpoint)

    def get_client(self) -> Mistral:
        kwargs = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.endpoint:
            kwargs["server_url"] = self.endpoint

        logger.info("Initializing Mistral API client...")
        return Mistral(**kwargs)
