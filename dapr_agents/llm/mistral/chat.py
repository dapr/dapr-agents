import logging
import os
from pathlib import Path
from typing import (
    Any, ClassVar, Dict, Iterable, Iterator, List, Literal, Optional, Type, Union
)

from pydantic import BaseModel, Field, model_validator

from dapr_agents.llm.chat import ChatClientBase
from dapr_agents.llm.mistral.client import MistralClientBase
from dapr_agents.llm.utils import RequestHandler, ResponseHandler
from dapr_agents.prompt.base import PromptTemplateBase
from dapr_agents.prompt.prompty import Prompty
from dapr_agents.tool import AgentTool
from dapr_agents.types.message import BaseMessage, LLMChatCandidateChunk, LLMChatResponse

logger = logging.getLogger(__name__)

class MistralChatClient(MistralClientBase, ChatClientBase):
    model: Optional[str] = Field(default=None, description="Mistral model name.")
    prompty: Optional[Prompty] = Field(default=None)
    prompt_template: Optional[PromptTemplateBase] = Field(default=None)

    SUPPORTED_STRUCTURED_MODES: ClassVar[set[str]] = {"json", "function_call"}

    @model_validator(mode="before")
    def validate_and_initialize(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure `.model` is set, falling back to env var or a default."""
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
        prompty_instance = Prompty.load(prompty_source)
        prompt_template = Prompty.to_prompt_template(prompty_instance)
        cfg = prompty_instance.model.configuration

        return cls.model_validate({
            "model": cfg.name,
            "api_key": getattr(cfg, "api_key", None),
            "endpoint": getattr(cfg, "base_url", None),
            "prompty": prompty_instance,
            "prompt_template": prompt_template,
        })

    def generate(
        self,
        messages: Union[str, Dict[str, Any], BaseMessage, Iterable[Union[Dict[str, Any], BaseMessage]]] = None,
        *,
        input_data: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        tools: Optional[List[Union[AgentTool, Dict[str, Any]]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        structured_mode: Literal["json", "function_call"] = "json",
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[Iterator[LLMChatCandidateChunk], LLMChatResponse, BaseModel, List[BaseModel]]:
        
        if structured_mode not in self.SUPPORTED_STRUCTURED_MODES:
            raise ValueError(f"structured_mode must be one of {self.SUPPORTED_STRUCTURED_MODES}")

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

        params = RequestHandler.process_params(
            params,
            llm_provider=self.provider,
            tools=tools,
            response_format=response_format,
            structured_mode=structured_mode,
        )
        params = RequestHandler.make_params_json_serializable(params)

        try:
            logger.info("Calling Mistral ChatCompletion...")
            if stream:
                resp = self.client.chat.stream(**params)
            else:
                resp = self.client.chat.complete(**params)

            return ResponseHandler.process_response(
                response=resp,
                llm_provider=self.provider,
                response_format=response_format,
                structured_mode=structured_mode,
                stream=stream,
            )
        except Exception as e:
            logger.error("Mistral API error", exc_info=True)
            raise ValueError(f"Mistral API error: {e}") from e