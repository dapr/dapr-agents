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

from pydantic import BaseModel, Field

from dapr_agents.llm.chat import ChatClientBase
from dapr_agents.llm.nvidia.client import NVIDIAClientBase
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


class NVIDIAChatClient(NVIDIAClientBase, ChatClientBase):
    """
    Chat client for NVIDIA chat models, combining NVIDIA client management
    with Prompty-specific prompt templates and unified request/response handling.

    Inherits:
      - NVIDIAClientBase: manages API key, base_url, retries, etc. for NVIDIA endpoints.
      - ChatClientBase: provides chat-specific abstractions.

    Attributes:
        model: the model name to use (e.g. "meta/llama-3.1-8b-instruct").
        max_tokens: maximum number of tokens to generate per call.
    """

    model: str = Field(
        default="meta/llama-3.1-8b-instruct",
        description="Model name to use. Defaults to 'meta/llama-3.1-8b-instruct'.",
    )
    max_tokens: Optional[int] = Field(
        default=1024,
        description=(
            "Maximum number of tokens to generate in a single call. "
            "Must be ≥1; defaults to 1024."
        ),
    )
    prompty: Optional[Prompty] = Field(
        default=None, description="Optional Prompty instance for templating."
    )
    prompt_template: Optional[PromptTemplateBase] = Field(
        default=None, description="Optional prompt-template to format inputs."
    )

    # NVIDIA currently only supports function_call structured output
    SUPPORTED_STRUCTURED_MODES: ClassVar[set[str]] = {"function_call"}

    def model_post_init(self, __context: Any) -> None:
        """
        After Pydantic init, configure the client for 'chat' API.
        """
        self._api = "chat"
        super().model_post_init(__context)

    @classmethod
    def from_prompty(cls, prompty_source: Union[str, Path]) -> "NVIDIAChatClient":
        """
        Build an NVIDIAChatClient from a Prompty spec (file path or inline YAML/JSON).

        Args:
            prompty_source: Path or inline content of a Prompty specification.

        Returns:
            Configured NVIDIAChatClient instance.
        """
        prompty_instance = Prompty.load(prompty_source)
        prompt_template = Prompty.to_prompt_template(prompty_instance)
        cfg = prompty_instance.model.configuration

        return cls.model_validate(
            {
                "model": cfg.name,
                "api_key": cfg.api_key,
                "base_url": cfg.base_url,
                "prompty": prompty_instance,
                "prompt_template": prompt_template,
            }
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
        max_tokens: Optional[int] = None,
        structured_mode: Literal["function_call"] = "function_call",
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[
        Iterator[LLMChatCandidateChunk],  # streaming
        LLMChatResponse,  # non‑stream + no format
        BaseModel,  # non‑stream + single structured format
        List[BaseModel],  # non‑stream + list structured format
    ]:
        """
        Issue a chat completion to NVIDIA.

        - If `stream=True` in kwargs, returns an iterator of
          `LLMChatCandidateChunk` via ResponseHandler.
        - Otherwise returns either:
            • a raw `LLMChatResponse`, or
            • validated Pydantic model(s) per `response_format`.

        Args:
            messages:        Pre-built messages or None to use `input_data`.
            input_data:      Variables for the Prompty template.
            model:           Override default model name.
            tools:           List of AgentTool or dict specs.
            response_format: Pydantic model (or list thereof) for structured output.
            max_tokens:      Override default max_tokens.
            structured_mode: Must be "function_call" (only supported mode).
            stream:          If True, return an iterator of `LLMChatCandidateChunk`.
            **kwargs:        Other LLM params (temperature, stream, etc.).

        Raises:
            ValueError: for invalid structured_mode or missing inputs.
        """
        # 1) Validate structured_mode
        if structured_mode not in self.SUPPORTED_STRUCTURED_MODES:
            raise ValueError(
                f"structured_mode must be one of {self.SUPPORTED_STRUCTURED_MODES}"
            )

        # 2) If input_data is provided, format messages via Prompty
        if input_data:
            if not self.prompt_template:
                raise ValueError("input_data provided but no prompt_template is set.")
            logger.info("Formatting messages via prompt_template.")
            messages = self.prompt_template.format_prompt(**input_data)

        if not messages:
            raise ValueError("Either 'messages' or 'input_data' must be provided.")

        # 3) Normalize messages + merge client/prompty defaults
        params: Dict[str, Any] = {
            "messages": RequestHandler.normalize_chat_messages(messages)
        }
        if self.prompty:
            params = {**self.prompty.model.parameters.model_dump(), **params, **kwargs}
        else:
            params.update(kwargs)

        # 4) Add the stream parameter explicitly to params
        params["stream"] = stream

        # 5) Override model & max_tokens if provided
        params["model"] = model or self.model
        params["max_tokens"] = max_tokens or self.max_tokens

        # 6) Inject tools / response_format / structured_mode
        params = RequestHandler.process_params(
            params,
            llm_provider=self.provider,
            tools=tools,
            response_format=response_format,
            structured_mode=structured_mode,
        )

        # 7) Call NVIDIA API + dispatch to ResponseHandler
        try:
            logger.info("Calling NVIDIA ChatCompletion API.")
            logger.debug(f"Parameters: {params}")
            resp = self.client.chat.completions.create(**params)
            return ResponseHandler.process_response(
                response=resp,
                llm_provider=self.provider,
                response_format=response_format,
                structured_mode=structured_mode,
                stream=stream,
            )
        except Exception:
            logger.error("NVIDIA ChatCompletion error", exc_info=True)
            raise
