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
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from dapr_agents.llm.anthropic.client import PROVIDER, AnthropicClientBase
from dapr_agents.llm.anthropic.utils import (
    STRUCTURED_INJECTORS,
    STRUCTURED_PARSERS,
    assert_json_output_supported,
    iter_stream,
    split_messages,
    to_llm_chat_response,
)
from dapr_agents.llm.chat import ChatClientBase
from dapr_agents.llm.utils import RequestHandler
from dapr_agents.prompt.base import PromptTemplateBase
from dapr_agents.prompt.prompty import Prompty
from dapr_agents.tool import AgentTool
from dapr_agents.tool.utils.tool import ToolHelper
from dapr_agents.types.llm import AnthropicModelConfig
from dapr_agents.types.message import (
    BaseMessage,
    LLMChatResponse,
    LLMChatResponseChunk,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096


class AnthropicChatClient(AnthropicClientBase, ChatClientBase):
    model: str | None = Field(default=None, description="Anthropic model name.")
    prompty: Prompty | None = Field(default=None)
    prompt_template: PromptTemplateBase | None = Field(default=None)

    @model_validator(mode="before")
    def validate_and_initialize(cls, values: dict[str, Any]) -> dict[str, Any]:
        if not values.get("model"):
            values["model"] = os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL
        return values

    def model_post_init(self, __context: Any) -> None:
        self._api = "chat"
        super().model_post_init(__context)

    @classmethod
    def from_prompty(
        cls,
        prompty_source: str | Path,
        timeout: int | float | dict[str, Any] = 1500,
    ) -> "AnthropicChatClient":
        prompty = Prompty.load(prompty_source)
        config = prompty.model.configuration
        parameters = prompty.model.parameters
        if not isinstance(config, AnthropicModelConfig):
            raise ValueError(
                f"Expected Prompty model configuration type to be {PROVIDER!r}, "
                f"but got {config.type!r}."
            )
        model_name = parameters.model or config.name
        if not model_name:
            raise ValueError(
                "Anthropic Prompty must specify a model name via "
                "`parameters.model` or `configuration.name`."
            )
        return cls(
            model=model_name,
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=timeout,
            prompty=prompty,
            prompt_template=prompty.to_prompt_template(),
        )

    def generate(
        self,
        messages: str
        | dict[str, Any]
        | BaseMessage
        | Iterable[dict[str, Any] | BaseMessage]
        | None = None,
        *,
        input_data: dict[str, Any] | None = None,
        model: str | None = None,
        tools: list[AgentTool | dict[str, Any]] | None = None,
        response_format: type[BaseModel] | None = None,
        structured_mode: Literal["json", "function_call"] = "json",
        stream: bool = False,
        **kwargs: Any,
    ) -> Iterator[LLMChatResponseChunk] | LLMChatResponse | BaseModel | list[BaseModel]:
        """Run one chat turn against the Anthropic Messages API.

        Returns:
             An `LLMChatResponse` by default, a streaming iterator when
            `stream=True`, or a validated Pydantic instance when `response_format`
            is given.
        """
        if structured_mode not in STRUCTURED_INJECTORS:
            raise ValueError(
                f"structured_mode must be one of {sorted(STRUCTURED_INJECTORS)}; "
                f"got {structured_mode!r}."
            )

        if input_data:
            if not self.prompt_template:
                raise ValueError("No prompt_template set for input_data usage.")
            messages = self.prompt_template.format_prompt(**input_data)
        if not messages:
            raise ValueError("Either messages or input_data must be provided.")

        messages_normalized = RequestHandler.normalize_chat_messages(messages)
        system, messages_anthropic = split_messages(messages_normalized)

        prompty_params = (
            self.prompty.model.parameters.model_dump(exclude_none=True)
            if self.prompty
            else {}
        )
        extras = prompty_params | kwargs
        tools_effective = tools if tools is not None else extras.get("tools")
        tools_formatted = (
            [ToolHelper.format_tool(t, tool_format="claude") for t in tools_effective]
            if tools_effective
            else None
        )

        params: dict[str, Any] = {
            **extras,
            "model": model or extras.get("model") or self.model,
            "messages": messages_anthropic,
            "max_tokens": extras.get("max_tokens", DEFAULT_MAX_TOKENS),
        }
        if system:
            params["system"] = system
        if tools_formatted:
            params["tools"] = tools_formatted
        if response_format is not None:
            if structured_mode == "json":
                assert_json_output_supported(self.client, params["model"])
            inject_structured_request = STRUCTURED_INJECTORS[structured_mode]
            inject_structured_request(params, response_format)
        params = RequestHandler.make_params_json_serializable(params)

        logger.info("Calling Anthropic Messages API...")
        logger.debug(f"Anthropic request params: {params}")
        if stream:
            return iter_stream(self.client, params)
        try:
            raw_resp = self.client.messages.create(**params)
        except Exception:
            logger.exception("Anthropic Messages API call failed")
            raise

        if response_format is not None:
            parse_structured_response = STRUCTURED_PARSERS[structured_mode]
            return parse_structured_response(raw_resp, response_format)
        return to_llm_chat_response(raw_resp)
