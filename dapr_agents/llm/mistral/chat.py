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

from pydantic import BaseModel, Field, model_validator

from dapr_agents.llm.chat import ChatClientBase
from dapr_agents.llm.mistral.client import MistralClientBase
from dapr_agents.llm.utils import RequestHandler
from dapr_agents.prompt.base import PromptTemplateBase
from dapr_agents.prompt.prompty import Prompty
from dapr_agents.tool import AgentTool
from dapr_agents.types.message import (
    BaseMessage,
    LLMChatCandidateChunk,
    LLMChatResponse,
)

logger = logging.getLogger(__name__)


class MistralChatClient(MistralClientBase, ChatClientBase):
    """
    Mistral Chat Client for Dapr Agents.

    Consumers can configure the model used by this client in three ways:
    1. Passing it explicitly during initialization.
    2. Setting the `MISTRAL_MODEL` environment variable.
    3. Relying on the default fallback: `mistral-large-latest`.
    """

    model: Optional[str] = Field(default=None, description="Mistral model name.")
    prompty: Optional[Prompty] = Field(default=None)
    prompt_template: Optional[PromptTemplateBase] = Field(default=None)

    SUPPORTED_STRUCTURED_MODES: ClassVar[set[str]] = {"json", "function_call"}

    @model_validator(mode="before")
    def validate_and_initialize(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure `.model` is set.

        Falls back to the `MISTRAL_MODEL` environment variable if not provided.
        Defaults to `mistral-large-latest` if the environment variable is missing.
        """
        env_model = os.environ.get("MISTRAL_MODEL")
        if env_model:
            values["model"] = env_model
        elif not values.get("model"):
            values["model"] = "mistral-large-latest"
        return values

    def model_post_init(self, __context: Any) -> None:
        self._api = "chat"
        super().model_post_init(__context)

    @classmethod
    def from_prompty(
        cls,
        prompty_source: Union[str, Path],
        timeout: Union[int, float, Dict[str, Any]] = 1500,
    ) -> "MistralChatClient":
        raise NotImplementedError(
            "Prompty configuration is not yet supported for the Mistral provider."
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
        Iterator[LLMChatCandidateChunk], LLMChatResponse, BaseModel, List[BaseModel]
    ]:

        if stream:
            raise NotImplementedError(
                "Streaming is not yet supported for the Mistral provider."
            )
        if tools or response_format:
            raise NotImplementedError(
                "Tools and structured output are not yet supported for the Mistral provider."
            )

        if input_data:
            if not self.prompt_template:
                raise ValueError("No prompt_template set for input_data usage.")
            messages = self.prompt_template.format_prompt(**input_data)

        if not messages:
            raise ValueError("Either messages or input_data must be provided.")

        params = {"messages": RequestHandler.normalize_chat_messages(messages)}
        if self.prompty:
            params = {**self.prompty.model.parameters.model_dump(), **params, **kwargs}
        else:
            params.update(kwargs)

        params["model"] = model or self.model
        params = RequestHandler.make_params_json_serializable(params)

        try:
            logger.info("Calling Mistral ChatCompletion...")
            resp = self.client.chat.complete(**params)

            return LLMChatResponse(
                id=resp.id,
                model=resp.model,
                results=[
                    {
                        "index": choice.index,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content,
                        },
                        "finish_reason": choice.finish_reason,
                    }
                    for choice in resp.choices
                ],
                usage=resp.usage.model_dump() if getattr(resp, "usage", None) else None,
            )

        except Exception as e:
            logger.error("Mistral API error", exc_info=True)
            raise ValueError(f"Mistral API error: {e}") from e
