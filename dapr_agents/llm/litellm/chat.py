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
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Type,
    Union,
)

import litellm
from pydantic import BaseModel, Field, model_validator

from dapr_agents.llm.chat import ChatClientBase
from dapr_agents.llm.litellm.client import LiteLLMClientBase
from dapr_agents.llm.utils import RequestHandler, ResponseHandler
from dapr_agents.prompt.base import PromptTemplateBase
from dapr_agents.prompt.prompty import Prompty
from dapr_agents.tool import AgentTool
from dapr_agents.types.message import (
    BaseMessage,
    LLMChatCandidateChunk,
    LLMChatResponse,
)

logger = logging.getLogger(__name__)


class LiteLLMChatClient(LiteLLMClientBase, ChatClientBase):
    """
    LiteLLM Chat Client for Dapr Agents.

    LiteLLM is a unified gateway that routes to 100+ LLM providers
    (OpenAI, Anthropic, Google, Azure, AWS Bedrock, etc.) through a
    single interface.

    Configure the model with the ``provider/model`` naming convention
    (e.g. ``anthropic/claude-sonnet-4-6``, ``openai/gpt-4o``).

    Model resolution order:
    1. Explicit ``model`` parameter during initialization.
    2. ``LITELLM_MODEL`` environment variable.
    3. Default fallback: ``openai/gpt-4o``.
    """

    model: Optional[str] = Field(
        default=None,
        description="LiteLLM model identifier (e.g. 'openai/gpt-4o').",
    )
    prompty: Optional[Prompty] = Field(default=None)
    prompt_template: Optional[PromptTemplateBase] = Field(default=None)

    SUPPORTED_STRUCTURED_MODES: ClassVar[set[str]] = {"json", "function_call"}

    @model_validator(mode="before")
    def validate_and_initialize(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        env_model = os.environ.get("LITELLM_MODEL")
        if env_model:
            values["model"] = env_model
        elif not values.get("model"):
            values["model"] = "openai/gpt-4o"
        return values

    def model_post_init(self, __context: Any) -> None:
        self._api = "chat"
        super().model_post_init(__context)

    @classmethod
    def from_prompty(
        cls,
        prompty_source: Union[str, Path],
        timeout: Union[int, float, Dict[str, Any]] = 1500,
    ) -> "LiteLLMChatClient":
        raise NotImplementedError(
            "Prompty-based initialization is not yet supported for the LiteLLM provider."
        )

    def generate(
        self,
        messages: Union[
            str,
            Dict[str, Any],
            BaseMessage,
            Iterable[Union[Dict[str, Any], BaseMessage]],
        ] = None,
        *,
        input_data: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        tools: Optional[List[Union[AgentTool, Dict[str, Any]]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        structured_mode: Literal["json", "function_call"] = "json",
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[
        Iterator[LLMChatCandidateChunk],
        LLMChatResponse,
        BaseModel,
        List[BaseModel],
    ]:
        if structured_mode not in self.SUPPORTED_STRUCTURED_MODES:
            raise ValueError(
                f"structured_mode must be one of {self.SUPPORTED_STRUCTURED_MODES}"
            )

        if input_data:
            if not self.prompt_template:
                raise ValueError("No prompt_template set for input_data usage.")
            logger.info("Formatting messages via prompt_template.")
            messages = self.prompt_template.format_prompt(**input_data)

        if not messages:
            raise ValueError("Either messages or input_data must be provided.")

        params = {"messages": RequestHandler.normalize_chat_messages(messages)}
        if self.prompty:
            params = {**self.prompty.model.parameters.model_dump(), **params, **kwargs}
        else:
            params.update(kwargs)

        params["stream"] = stream
        params["model"] = model or self.model

        params = RequestHandler.process_params(
            params,
            llm_provider=self.provider,
            tools=tools,
            response_format=response_format,
            structured_mode=structured_mode,
        )

        params = RequestHandler.make_params_json_serializable(params)

        # LiteLLM-specific: silently drop provider-unsupported kwargs
        params["drop_params"] = True
        if self.config.api_key:
            params["api_key"] = self.config.api_key
        if self.config.api_base:
            params["api_base"] = self.config.api_base

        try:
            logger.info("Calling LiteLLM completion...")
            logger.debug(f"LiteLLM request params: {params}")
            resp = litellm.completion(**params)
            logger.info("LiteLLM response received.")
            return ResponseHandler.process_response(
                response=resp,
                llm_provider=self.provider,
                response_format=response_format,
                structured_mode=structured_mode,
                stream=stream,
            )
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            logger.error(f"LiteLLM API error: {error_type} - {error_msg}")
            logger.error("Full error details:", exc_info=True)
            raise ValueError(f"LiteLLM API error ({error_type}): {error_msg}") from e
