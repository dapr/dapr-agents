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

import httpx
from anthropic import Anthropic, Timeout
from pydantic import ConfigDict, Field

from dapr_agents.llm.base import LLMClientBase
from dapr_agents.types.llm import AnthropicClientConfig

logger = logging.getLogger(__name__)

PROVIDER = "anthropic"


class AnthropicClientBase(LLMClientBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: str | None = Field(
        default=None,
        description="API key for Anthropic. Falls back to ANTHROPIC_API_KEY env var.",
    )
    base_url: str | None = Field(
        default=None,
        description="Base URL override for the Anthropic API (proxy or compatible endpoint).",
    )
    timeout: int | float | dict[str, Any] | Timeout | httpx.Timeout | None = Field(
        default=1500,
        description="Default request timeout in seconds, or an SDK-appropriate Timeout / kwarg dict.",
    )

    @staticmethod
    def _parse_timeout_value(val: Any) -> float | None:
        if val is None:
            return None
        if isinstance(val, bool):
            raise ValueError(
                f"Invalid timeout: booleans are not valid timeouts ({val})"
            )
        if isinstance(val, str):
            try:
                val = float(val)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid timeout configuration: {val!r}") from e
        if isinstance(val, (int, float)):
            val = float(val)
            if val < 0:
                raise ValueError(f"Timeout cannot be negative: {val}")
            return val
        raise ValueError(f"Invalid timeout configuration: {val!r}")

    @staticmethod
    def configure_timeout(
        timeout: int | float | dict[str, Any] | Timeout | httpx.Timeout | None,
    ) -> float | Timeout | None:
        """
        Configure the timeout setting for the Anthropic client.

        :param timeout: Timeout in seconds, dictionary of timeout configurations,
            or an SDK Timeout instance.
        :return: A float, Timeout instance, or None.
        """
        if timeout is None:
            return None
        if isinstance(timeout, Timeout):
            return timeout
        # Handle httpx.Timeout or similar objects duck-typed with timeout fields
        if (
            hasattr(timeout, "connect")
            and hasattr(timeout, "read")
            and hasattr(timeout, "write")
            and hasattr(timeout, "pool")
        ):
            return Timeout(
                connect=AnthropicClientBase._parse_timeout_value(timeout.connect),
                read=AnthropicClientBase._parse_timeout_value(timeout.read),
                write=AnthropicClientBase._parse_timeout_value(timeout.write),
                pool=AnthropicClientBase._parse_timeout_value(timeout.pool),
            )
        if isinstance(timeout, dict):
            timeout_dict = {
                k: AnthropicClientBase._parse_timeout_value(v)
                for k, v in timeout.items()
            }
            if "total" in timeout_dict and "timeout" not in timeout_dict:
                timeout_dict["timeout"] = timeout_dict.pop("total")
            try:
                return Timeout(**timeout_dict)
            except ValueError:
                # If neither a default timeout nor all four parameters were provided,
                # supply a default timeout fallback
                if "timeout" not in timeout_dict and not all(
                    k in timeout_dict for k in ("connect", "read", "write", "pool")
                ):
                    return Timeout(1500.0, **timeout_dict)
                raise
        return AnthropicClientBase._parse_timeout_value(timeout)

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
        kwargs: dict[str, Any] = {"timeout": self.configure_timeout(self.timeout)}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url

        logger.info("Initializing Anthropic API client...")
        return Anthropic(**kwargs)
