import logging
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
from dapr_agents.llm.openai.client.base import OpenAIClientBase
from dapr_agents.llm.utils import RequestHandler, ResponseHandler
from dapr_agents.prompt.base import PromptTemplateBase
from dapr_agents.prompt.prompty import Prompty
from dapr_agents.tool import AgentTool
from dapr_agents.types.llm import AzureOpenAIModelConfig, OpenAIModelConfig
from dapr_agents.types.message import (
    BaseMessage,
    LLMChatCandidateChunk,
    LLMChatResponse,
)

logger = logging.getLogger(__name__)


class OpenAIChatClient(OpenAIClientBase, ChatClientBase):
    """
    Chat client for OpenAI models, layering in Prompty-driven prompt templates
    and unified request/response handling.

    Inherits:
      - OpenAIClientBase: manages API key, base_url, retries, etc.
      - ChatClientBase: provides chat-specific abstractions.
    """

    model: Optional[str] = Field(
        default=None, description="Model name or Azure deployment ID."
    )
    prompty: Optional[Prompty] = Field(
        default=None, description="Optional Prompty instance for templating."
    )
    prompt_template: Optional[PromptTemplateBase] = Field(
        default=None, description="Optional prompt-template to format inputs."
    )

    SUPPORTED_STRUCTURED_MODES: ClassVar[set[str]] = {"json", "function_call"}

    @model_validator(mode="before")
    def validate_and_initialize(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure `.model` is always set.  If unset, fall back to `azure_deployment`
        or default to `"gpt-4o"`.
        """
        if not values.get("model"):
            values["model"] = values.get("azure_deployment", "gpt-4o")
        return values

    def model_post_init(self, __context: Any) -> None:
        """After Pydantic init, ensure we're in the “chat” API mode."""
        self._api = "chat"
        super().model_post_init(__context)

    @classmethod
    def from_prompty(
        cls,
        prompty_source: Union[str, Path],
        timeout: Union[int, float, Dict[str, Any]] = 1500,
    ) -> "OpenAIChatClient":
        """
        Load a Prompty file (or inline YAML/JSON string), extract its
        model configuration and prompt template, and return a fully-wired client.

        Args:
            prompty_source: path or inline text for a Prompty spec.
            timeout:        seconds or HTTPX-style timeout, defaults to 1500.

        Returns:
            Configured OpenAIChatClient.
        """
        prompty_instance = Prompty.load(prompty_source)
        prompt_template = Prompty.to_prompt_template(prompty_instance)
        cfg = prompty_instance.model.configuration

        common = {
            "timeout": timeout,
            "prompty": prompty_instance,
            "prompt_template": prompt_template,
        }

        if isinstance(cfg, OpenAIModelConfig):
            return cls.model_validate(
                {
                    **common,
                    "model": cfg.name,
                    "api_key": cfg.api_key,
                    "base_url": cfg.base_url,
                    "organization": cfg.organization,
                    "project": cfg.project,
                }
            )
        elif isinstance(cfg, AzureOpenAIModelConfig):
            return cls.model_validate(
                {
                    **common,
                    "model": cfg.azure_deployment,
                    "api_key": cfg.api_key,
                    "azure_endpoint": cfg.azure_endpoint,
                    "azure_deployment": cfg.azure_deployment,
                    "api_version": cfg.api_version,
                    "organization": cfg.organization,
                    "project": cfg.project,
                    "azure_ad_token": cfg.azure_ad_token,
                    "azure_client_id": cfg.azure_client_id,
                }
            )
        else:
            raise ValueError(f"Unsupported model config: {type(cfg)}")

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
        """
        Issue a chat completion.

        - If `stream=True` in params, returns an iterator of `LLMChatCandidateChunk`.
        - Otherwise returns either:
            • raw `AssistantMessage` wrapped in `LLMChatResponse`, or
            • validated Pydantic model(s) per `response_format`.

        Args:
            messages:        pre-built messages or None to use `input_data`.
            input_data:      variables for the Prompty template.
            model:           override client's default model.
            tools:           list of AgentTool or dict specs.
            response_format: Pydantic model (or list thereof) for structured output.
            structured_mode: “json” or “function_call” (non-stream only).
            stream:          if True, return an iterator of `LLMChatCandidateChunk`.
            **kwargs:        any other LLM params (temperature, top_p, stream, etc.).

        Returns:
            • `Iterator[LLMChatCandidateChunk]` if streaming
            • `LLMChatResponse` or Pydantic instance(s) if non-streaming

        Raises:
            ValueError: on invalid `structured_mode`, missing prompts, etc.
        """
        # 1) Validate structured_mode
        if structured_mode not in self.SUPPORTED_STRUCTURED_MODES:
            raise ValueError(
                f"structured_mode must be one of {self.SUPPORTED_STRUCTURED_MODES}"
            )

        # 2) If using a prompt template, build messages
        if input_data:
            if not self.prompt_template:
                raise ValueError("No prompt_template set for input_data usage.")
            logger.info("Formatting messages via prompt_template.")
            messages = self.prompt_template.format_prompt(**input_data)

        if not messages:
            raise ValueError("Either messages or input_data must be provided.")

        # 3) Normalize messages + merge client/prompty defaults
        params = {"messages": RequestHandler.normalize_chat_messages(messages)}
        if self.prompty:
            params = {**self.prompty.model.parameters.model_dump(), **params, **kwargs}
        else:
            params.update(kwargs)

        # 4) Add the stream parameter explicitly to params
        params["stream"] = stream

        # 5) Override model if given
        params["model"] = model or self.model

        # 6) Let RequestHandler inject tools / response_format / structured_mode
        params = RequestHandler.process_params(
            params,
            llm_provider=self.provider,
            tools=tools,
            response_format=response_format,
            structured_mode=structured_mode,
        )

        # 7) Call API + hand off to ResponseHandler
        try:
            logger.info("Calling OpenAI ChatCompletion...")
            logger.debug(f"ChatCompletion params: {params}")
            resp = self.client.chat.completions.create(**params, timeout=self.timeout)
            logger.info("ChatCompletion response received.")
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

            logger.error(f"OpenAI ChatCompletion API error: {error_type} - {error_msg}")
            logger.error("Full error details:", exc_info=True)

            raise ValueError(f"OpenAI API error ({error_type}): {error_msg}") from e
