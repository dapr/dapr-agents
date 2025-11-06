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
from dapr_agents.llm.huggingface.client import HFHubInferenceClientBase
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


class HFHubChatClient(HFHubInferenceClientBase, ChatClientBase):
    """
    Chat client for Hugging Face Hub's Inference API.

    Extends:
      - HFHubInferenceClientBase: manages HF-specific auth, endpoints, retries.
      - ChatClientBase: provides the `.from_prompty()` and `.generate()` contract.

    Supports only function_call-style structured responses.
    """

    prompty: Optional[Prompty] = Field(
        default=None, description="Optional Prompty instance for templating."
    )
    prompt_template: Optional[PromptTemplateBase] = Field(
        default=None, description="Optional prompt-template to format inputs."
    )

    SUPPORTED_STRUCTURED_MODES: ClassVar[set[str]] = {"function_call"}

    def model_post_init(self, __context: Any) -> None:
        """
        After Pydantic __init__, set the internal API type to "chat".
        """
        self._api = "chat"
        super().model_post_init(__context)

    @classmethod
    def from_prompty(
        cls,
        prompty_source: Union[str, Path],
        timeout: Union[int, float, Dict[str, Any]] = 1500,
    ) -> "HFHubChatClient":
        """
        Load a Prompty spec (file or inline), extract model config & prompt template,
        and return a configured HFHubChatClient.

        Args:
            prompty_source: Path or inline text of a Prompty YAML/JSON.
            timeout:        Request timeout (seconds or HTTPX timeout dict).

        Returns:
            HFHubChatClient: client ready for .generate() calls.
        """
        prompty_instance = Prompty.load(prompty_source)
        prompt_template = Prompty.to_prompt_template(prompty_instance)
        cfg = prompty_instance.model.configuration

        return cls.model_validate(
            {
                "model": cfg.name,
                "api_key": cfg.api_key,
                "base_url": cfg.base_url,
                "headers": cfg.headers,
                "cookies": cfg.cookies,
                "proxies": cfg.proxies,
                "timeout": timeout,
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
        structured_mode: Literal["function_call"] = "function_call",
        stream: bool = False,
        **kwargs: Any,  # accept any extra params, even if unused
    ) -> Union[
        Iterator[LLMChatCandidateChunk],
        LLMChatResponse,
        BaseModel,
        List[BaseModel],
    ]:
        """
        Issue a chat completion via Hugging Face's Inference API.

        - If `stream=True` in **kwargs**, returns an iterator of `LLMChatCandidateChunk`.
        - Otherwise returns either:
            • raw `AssistantMessage` wrapped in `LLMChatResponse`, or
            • validated Pydantic model(s) per `response_format`.

        Args:
            messages:        Pre-built messages or None to use `input_data`.
            input_data:      Variables for the Prompty template.
            model:           Override the client's default model name.
            tools:           List of AgentTool or dict specs.
            response_format: Pydantic model (or List[Model]) for structured output.
            structured_mode: Must be `"function_call"` (only supported mode here).
            stream:          If True, return an iterator of `LLMChatCandidateChunk`.
            **kwargs:        Any other LLM params (temperature, top_p, stream, etc.).

        Returns:
            • `Iterator[LLMChatCandidateChunk]` if streaming
            • `LLMChatResponse` or Pydantic instance(s) if non-streaming

        Raises:
            ValueError: on invalid `structured_mode`, missing prompts, or API errors.
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

        # 6) Inject tools / response_format via RequestHandler
        params = RequestHandler.process_params(
            params,
            llm_provider=self.provider,
            tools=tools,
            response_format=response_format,
            structured_mode=structured_mode,
        )

        # 7) Call HF API + delegate parsing to ResponseHandler
        try:
            logger.info("Calling HF ChatCompletion Inference API...")
            logger.debug(f"HF params: {params}")
            response = self.client.chat.completions.create(**params)
            logger.info("HF ChatCompletion response received.")

            # HF-specific error‐code handling
            code = getattr(response, "code", 200)
            if code != 200:
                msg = getattr(response, "message", response)
                logger.error(f"❌ HF error {code}: {msg}")
                raise RuntimeError(f"HuggingFace error {code}: {msg}")

            return ResponseHandler.process_response(
                response=response,
                llm_provider=self.provider,
                response_format=response_format,
                structured_mode=structured_mode,
                stream=stream,
            )

        except Exception as e:
            logger.error("Hugging Face ChatCompletion API error", exc_info=True)
            raise ValueError(f"Failed to process HF chat completion: {e}") from e
