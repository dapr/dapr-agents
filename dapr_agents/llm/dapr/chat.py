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

from pydantic import BaseModel, Field
from dapr_agents.llm.chat import ChatClientBase
from dapr_agents.llm.dapr.client import DaprInferenceClientBase
from dapr_agents.llm.utils import RequestHandler, ResponseHandler
from dapr_agents.prompt.base import PromptTemplateBase
from dapr_agents.prompt.prompty import Prompty
from dapr_agents.tool import AgentTool
from dapr_agents.types.exceptions import DaprRuntimeVersionNotSupportedError
from dapr_agents.types.message import (
    BaseMessage,
    LLMChatResponse,
)
from dapr_agents.utils import is_version_supported


# Lazy import to avoid import issues during test collection
def _import_conversation_types():
    from dapr.clients.grpc.conversation import (
        ConversationInputAlpha2,
        ConversationMessage,
        ConversationMessageOfAssistant,
        ConversationMessageContent,
        ConversationToolCalls,
        ConversationToolCallsOfFunction,
        create_user_message,
        create_system_message,
        create_assistant_message,
        create_tool_message,
    )

    return (
        ConversationInputAlpha2,
        ConversationMessage,
        ConversationMessageOfAssistant,
        ConversationMessageContent,
        ConversationToolCalls,
        ConversationToolCallsOfFunction,
        create_user_message,
        create_system_message,
        create_assistant_message,
        create_tool_message,
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

    component_name: Optional[str] = None

    # Support both function_call and json structured output modes
    SUPPORTED_STRUCTURED_MODES: ClassVar[set[str]] = {"function_call", "json"}

    def model_post_init(self, __context: Any) -> None:
        """
        After Pydantic init, set up API/type and default LLM component from env.
        """
        self._api = "chat"
        self._llm_component = self.component_name
        if not self._llm_component:
            self._llm_component = os.environ.get("DAPR_LLM_COMPONENT_DEFAULT")
        if not self._llm_component:
            logger.debug(
                "No LLM component provided and no default component found in the environment. Will try to get it from the metadata at runtime."
            )
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
        if not isinstance(response, dict):
            logger.error(f"Invalid response type: {type(response)}")
            raise ValueError(f"Response must be a dictionary, got {type(response)}")

        # Flatten all output choices from Alpha2 envelope
        choices: List[Dict[str, Any]] = []
        outputs = response.get("outputs", []) or []
        if not isinstance(outputs, list):
            logger.error(f"Invalid outputs type: {type(outputs)}")
            raise ValueError(f"Outputs must be a list, got {type(outputs)}")

        for output in outputs:
            if not isinstance(output, dict):
                logger.error(f"Invalid output type: {type(output)}")
                continue
            output_choices = output.get("choices", []) or []
            if not isinstance(output_choices, list):
                logger.error(f"Invalid choices type: {type(output_choices)}")
                continue
            for choice in output_choices:
                if not isinstance(choice, dict):
                    logger.error(f"Invalid choice type: {type(choice)}")
                    continue
                # Ensure message is present and has required fields
                message = choice.get("message", {})
                if not isinstance(message, dict):
                    logger.error(f"Invalid message type: {type(message)}")
                    continue
                # Add required fields if missing
                if "content" not in message:
                    message["content"] = ""
                if "role" not in message:
                    message["role"] = "assistant"
                choice["message"] = message
                choices.append(choice)

        return {
            "choices": choices,
            "created": int(time.time()),
            "model": model,
            "object": "chat.completion",
            "usage": {"total_tokens": "-1"},
        }

    def convert_to_conversation_inputs(self, inputs: List[Dict[str, Any]]) -> List[Any]:
        """
        Map normalized messages into a single Alpha2 ConversationInput that preserves history.

        Alpha2 expects a list of ConversationMessage entries inside one ConversationInputAlpha2
        for a turn. If there are tool results, they must reference prior assistant tool_calls by id.
        """
        # Lazy import conversation types
        (
            ConversationInputAlpha2,
            ConversationMessage,
            ConversationMessageOfAssistant,
            ConversationMessageContent,
            ConversationToolCalls,
            ConversationToolCallsOfFunction,
            create_user_message,
            create_system_message,
            create_assistant_message,
            create_tool_message,
        ) = _import_conversation_types()

        history_messages: List[ConversationMessage] = []
        scrub_flags: List[bool] = []

        for item in inputs:
            role = item.get("role")
            content = item.get("content", "")

            if role == "user":
                msg = create_user_message(content)
            elif role == "system":
                msg = create_system_message(content)
            elif role == "assistant":
                # Preserve assistant tool_calls if present (OpenAI-like schema)
                tool_calls_data = item.get("tool_calls") or []
                if tool_calls_data:
                    converted_calls: List[ConversationToolCalls] = []
                    for tc in tool_calls_data:
                        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                        name = fn.get("name", "")
                        arguments = fn.get("arguments", "")
                        # Ensure arguments is a string
                        if not isinstance(arguments, str):
                            try:
                                import json as _json

                                arguments = _json.dumps(arguments)
                            except Exception:
                                arguments = str(arguments)
                        converted_calls.append(
                            ConversationToolCalls(
                                id=tc.get("id", None),
                                function=ConversationToolCallsOfFunction(
                                    name=name, arguments=arguments
                                ),
                            )
                        )
                    msg = ConversationMessage(
                        of_assistant=ConversationMessageOfAssistant(
                            content=[ConversationMessageContent(text=content)],
                            tool_calls=converted_calls,
                        )
                    )
                else:
                    msg = create_assistant_message(content)
            elif role == "tool":
                tool_id = item.get("tool_call_id") or item.get("id") or ""
                name = item.get("name", "")
                msg = create_tool_message(tool_id, name, content)
            else:
                raise ValueError(f"Unsupported role for Alpha2 conversion: {role}")

            history_messages.append(msg)
            scrub_flags.append(bool(item.get("scrubPII")))

        # Use scrub_pii if any message requested it
        scrub_any = any(scrub_flags) if scrub_flags else None

        return [ConversationInputAlpha2(messages=history_messages, scrub_pii=scrub_any)]

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
        structured_mode: Literal["function_call", "json"] = "function_call",
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
            structured_mode: Must be "function_call" or "json".
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
        # 2) Disallow streaming
        # Note: response_format is now supported for structured output
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

        logger.debug(f"Processed parameters for Dapr: {params}")
        if response_format:
            logger.debug(f"Response format: {response_format}")
            logger.debug(f"Structured mode: {structured_mode}")

        # 6) Convert to Dapr inputs & call
        conv_inputs = self.convert_to_conversation_inputs(params["inputs"])
        try:
            logger.debug("Invoking the Dapr Conversation API.")
            # Log tools/tool_choice/parameters for debugging
            if params.get("tools"):
                try:
                    logger.debug(
                        f"Alpha2 tools payload: {[t.get('function', {}).get('name', '') for t in params['tools'] if isinstance(t, dict)]}"
                    )
                except Exception:
                    logger.warning(
                        "Alpha2 tools payload present (could not render names)."
                    )
            if params.get("tool_choice") is not None:
                logger.debug(f"Alpha2 tool_choice: {params.get('tool_choice')}")
            if params.get("parameters") is not None:
                logger.debug(
                    f"Alpha2 parameters keys: {list(params.get('parameters', {}).keys())}"
                )
            # get metadata information from the dapr client
            metadata = self.client.dapr_client.get_metadata()
            _check_dapr_runtime_support(metadata)

            llm_component = llm_component or self._llm_component
            if not llm_component:
                llm_component = _get_llm_component(metadata)

            # Extract and serialize response format parameters
            api_params = {}
            if "response_format" in params:
                try:
                    import json

                    api_params["response_format"] = json.dumps(
                        params["response_format"]
                    )
                except Exception as e:
                    logger.warning(f"Failed to serialize response_format: {e}")
            if "structured_mode" in params:
                api_params["structured_mode"] = str(params["structured_mode"])

            raw = self.client.chat_completion_alpha2(
                llm=llm_component or self._llm_component,
                inputs=conv_inputs,
                scrub_pii=scrubPII,
                temperature=temperature,
                tools=params.get("tools"),
                tool_choice=params.get("tool_choice"),
                parameters=api_params or None,
            )
            normalized = self.translate_response(
                raw, llm_component or self._llm_component
            )
            logger.debug(f"Dapr Conversation API response: {raw}")
            logger.debug(f"Normalized response: {normalized}")
        except Exception as e:
            logger.error(
                f"An error occurred during the Dapr Conversation API call: {e}"
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


def _check_dapr_runtime_support(metadata: "GetMetadataResponse"):  # noqa: F821
    """Check if the Dapr runtime version is supported for Alpha2 Chat Client."""
    extended_metadata = metadata.extended_metadata
    dapr_runtime_version = extended_metadata.get("daprRuntimeVersion", None)
    if dapr_runtime_version is not None:
        # Allow only versions >=1.16.0, edge, and <2.0.0 for Alpha2 Chat Client
        if not is_version_supported(str(dapr_runtime_version), ">=1.16.0, <2.0.0"):
            raise DaprRuntimeVersionNotSupportedError(
                f"!!!!! Dapr Runtime Version {dapr_runtime_version} is not supported with Alpha2 Dapr Chat Client. Only Dapr runtime versions >=1.16.0 and <2.0.0 are supported."
            )


def _get_llm_component(metadata: "GetMetadataResponse") -> str:  # noqa: F821
    """Get the LLM component from the metadata."""
    conversation_components = [
        component
        for component in metadata.registered_components
        if component.type.startswith("conversation.")
    ]
    if len(conversation_components) == 1:
        return conversation_components[0].name
    elif len(conversation_components) > 1:
        raise ValueError(
            "Multiple LLM components found in the metadata. Please provide the component name explicitly (e.g. llm = DaprChatClient(component_name='openai')) or environment variable DAPR_LLM_COMPONENT_DEFAULT."
        )
    else:
        raise ValueError(
            "No LLM component provided and no default component found in the metadata."
        )
