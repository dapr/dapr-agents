import logging
import os
import time
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Type,
    Union,
)

# Alpha2 uses message variants; keep alpha1 import removed
from dapr.clients.grpc import conversation as dapr_conversation
from pydantic import BaseModel, Field

from dapr_agents.llm.chat import ChatClientBase
from dapr_agents.llm.dapr.client import DaprInferenceClientBase
from dapr_agents.llm.utils import RequestHandler, ResponseHandler
from dapr_agents.prompt.base import PromptTemplateBase
from dapr_agents.prompt.prompty import Prompty
from dapr_agents.tool import AgentTool
from dapr_agents.types.message import (
    BaseMessage,
    LLMChatResponse,
)

logger = logging.getLogger(__name__)


class DaprChatClient(DaprInferenceClientBase, ChatClientBase):
    """
    Chat client for Dapr's Inference API.

    Integrates Prompty-driven prompt templates, tool injection,
    PII scrubbing, and normalizes the Dapr output into our unified
    LLMChatResponse schema.  **Streaming is not supported.**
    """

    prompty: Optional[Prompty] = Field(
        default=None, description="Optional Prompty instance for templating."
    )
    prompt_template: Optional[PromptTemplateBase] = Field(
        default=None, description="Optional prompt-template to format inputs."
    )

    # Only function_call–style structured output is supported
    SUPPORTED_STRUCTURED_MODES: ClassVar[set[str]] = {"function_call"}

    def model_post_init(self, __context: Any) -> None:
        """
        After Pydantic init, set up API/type and default LLM component from env.
        """
        self._api = "chat"
        self._llm_component = os.environ["DAPR_LLM_COMPONENT_DEFAULT"]
        super().model_post_init(__context)

    @classmethod
    def from_prompty(
        cls,
        prompty_source: Union[str, Path],
        timeout: Union[int, float, Dict[str, Any]] = 1500,
    ) -> "DaprChatClient":
        """
        Build a DaprChatClient from a Prompty spec.

        Args:
            prompty_source: Path or inline Prompty YAML/JSON.
            timeout:       Request timeout in seconds or HTTPX-style dict.

        Returns:
            Configured DaprChatClient.
        """
        prompty_instance = Prompty.load(prompty_source)
        prompt_template = Prompty.to_prompt_template(prompty_instance)
        return cls.model_validate(
            {
                "timeout": timeout,
                "prompty": prompty_instance,
                "prompt_template": prompt_template,
            }
        )

    def translate_response(self, response: dict, model: str) -> dict:
        """
        Convert Dapr Alpha2 response into OpenAI-style ChatCompletion dict.
        """
        choices = []
        outputs = response.get("outputs", []) or []
        for out in outputs:
            for ch in out.get("choices", []) or []:
                msg = ch.get("message", {})
                finish = ch.get("finish_reason", "stop")
                choices.append(
                    {
                        "finish_reason": finish,
                        "index": len(choices),
                        "message": msg,
                        "logprobs": None,
                    }
                )
        return {
            "choices": choices,
            "created": int(time.time()),
            "model": model,
            "object": "chat.completion",
            "usage": {"total_tokens": "-1"},
        }

    def _to_alpha2_messages(
        self, inputs: List[Dict[str, Any]]
    ) -> List[dapr_conversation.ConversationInputAlpha2]:
        """
        Convert normalized messages into a single Alpha2 conversation input containing
        a sequence of message variants (system/user/assistant/tool).
        """
        alpha2_messages: List[Any] = []
        for item in inputs:
            role = item.get("role")
            content = item.get("content")
            if role == "system":
                alpha2_messages.append(dapr_conversation.create_system_message(content))
            elif role == "user":
                alpha2_messages.append(dapr_conversation.create_user_message(content))
            elif role == "assistant":
                # Support assistant with tool_calls when present (Alpha2 requires this for tool responses)
                tool_calls_src = item.get("tool_calls") or []
                if tool_calls_src:
                    tool_calls = []
                    for tc in tool_calls_src:
                        fn = (tc or {}).get("function", {})
                        tool_calls.append(
                            dapr_conversation.ConversationToolCalls(
                                id=(tc or {}).get("id", ""),
                                function=dapr_conversation.ConversationToolCallsOfFunction(
                                    name=fn.get("name", ""),
                                    arguments=fn.get("arguments", ""),
                                ),
                            )
                        )
                    alpha2_messages.append(
                        dapr_conversation.ConversationMessage(
                            of_assistant=dapr_conversation.ConversationMessageOfAssistant(
                                content=[
                                    dapr_conversation.ConversationMessageContent(
                                        text=content
                                    )
                                ]
                                if content
                                else [],
                                tool_calls=tool_calls,
                            )
                        )
                    )
                else:
                    alpha2_messages.append(
                        dapr_conversation.create_assistant_message(content)
                    )
            elif role == "tool":
                tool_call_id = item.get("tool_call_id") or ""
                tool_name = item.get("name") or "tool"
                alpha2_messages.append(
                    dapr_conversation.create_tool_message(
                        tool_id=tool_call_id,
                        name=tool_name,
                        content=content,
                    )
                )
            else:
                # default to user if unknown
                alpha2_messages.append(dapr_conversation.create_user_message(content))

        return [dapr_conversation.ConversationInputAlpha2(messages=alpha2_messages)]

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
        llm_component: Optional[str] = None,
        tools: Optional[List[Union[AgentTool, Dict[str, Any]]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        structured_mode: Literal["function_call"] = "function_call",
        scrubPII: bool = False,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Union[
        LLMChatResponse,
        BaseModel,
        List[BaseModel],
    ]:
        """
        Issue a non-streaming chat completion via Dapr.

        - **Streaming is not supported** and setting `stream=True` will raise.
        - Returns a unified `LLMChatResponse` (if no `response_format`), or
          validated Pydantic model(s) when `response_format` is provided.

        Args:
            messages:        Prebuilt messages or None to use `input_data`.
            input_data:      Variables for Prompty template rendering.
            llm_component:   Dapr component name (defaults from env).
            tools:           AgentTool or dict specifications.
            response_format: Pydantic model for structured output.
            structured_mode: Must be "function_call".
            scrubPII:        Obfuscate sensitive output if True.
            temperature:     Sampling temperature.
            **kwargs:        Other Dapr API parameters.

        Returns:
            • `LLMChatResponse` if no `response_format`
            • Pydantic model (or `List[...]`) when `response_format` is set

        Raises:
            ValueError: on invalid `structured_mode`, missing inputs, or if `stream=True`.
        """
        # 1) Validate structured_mode
        if structured_mode not in self.SUPPORTED_STRUCTURED_MODES:
            raise ValueError(
                f"structured_mode must be one of {self.SUPPORTED_STRUCTURED_MODES}"
            )
        # 2) Disallow response_format + streaming
        if response_format is not None:
            raise ValueError("`response_format` is not supported by DaprChatClient.")
        if kwargs.get("stream"):
            raise ValueError("Streaming is not supported by DaprChatClient.")

        # 3) Build messages via Prompty
        if input_data:
            if not self.prompt_template:
                raise ValueError("input_data provided but no prompt_template is set.")
            messages = self.prompt_template.format_prompt(**input_data)

        if not messages:
            raise ValueError("Either 'messages' or 'input_data' must be provided.")

        # 4) Normalize + merge defaults
        params: Dict[str, Any] = {
            "inputs": RequestHandler.normalize_chat_messages(messages)
        }
        if self.prompty:
            params = {**self.prompty.model.parameters.model_dump(), **params, **kwargs}
        else:
            params.update(kwargs)

        # 5) Inject tools + structured directives
        params = RequestHandler.process_params(
            params,
            llm_provider=self.provider,
            tools=tools,
            response_format=response_format,
            structured_mode=structured_mode,
        )

        # 6) Build Alpha2 inputs and call the Alpha2 API with tools
        alpha2_inputs = self._to_alpha2_messages(params["inputs"])
        try:
            logger.info("Invoking the Dapr Conversation API (Alpha2).")
            raw = self.client.chat_completion_alpha2(
                llm=llm_component or self._llm_component,
                inputs=alpha2_inputs,
                scrub_pii=scrubPII,
                temperature=temperature,
                tools=params.get("tools"),
                tool_choice=params.get("tool_choice"),
            )
            normalized = self.translate_response(
                raw, llm_component or self._llm_component
            )
            logger.info("Chat completion (Alpha2) retrieved successfully.")
        except Exception as e:
            logger.error(
                f"An error occurred during the Dapr Conversation Alpha2 API call: {e}"
            )
            raise

        # 7) Hand off to our unified handler (always non‐stream)
        return ResponseHandler.process_response(
            response=normalized,
            llm_provider=self.provider,
            response_format=response_format,
            structured_mode=structured_mode,
            stream=False,
        )
