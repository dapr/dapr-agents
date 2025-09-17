from dapr_agents.types.llm import DaprInferenceClientConfig
from dapr_agents.llm.base import LLMClientBase
from dapr.clients import DaprClient
from dapr.clients.grpc import conversation as dapr_conversation
from typing import Dict, Any, List, Optional
from pydantic import model_validator

import json
import logging

logger = logging.getLogger(__name__)


class DaprInferenceClient:
    def __init__(self):
        self.dapr_client = DaprClient()

    # ──────────────────────────────────────────────────────────────────────────
    # Alpha2 (Tool Calling) support
    # ──────────────────────────────────────────────────────────────────────────
    def _convert_openai_tools_to_conversation_tools(
        self, tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[dapr_conversation.ConversationTools]]:
        """
        Convert OpenAI-style tools (type=function, function={name, description, parameters})
        into Dapr ConversationTools objects for Alpha2.
        """
        if not tools:
            return None
        converted: List[dapr_conversation.ConversationTools] = []
        for tool in tools:
            fn = tool.get("function", {}) if isinstance(tool, dict) else {}
            name = fn.get("name")
            description = fn.get("description")
            parameters = fn.get("parameters")
            function_spec = dapr_conversation.ConversationToolsFunction(
                name=name or "",
                description=description or "",
                parameters=parameters or {},
            )
            conv_tool = dapr_conversation.ConversationTools(function=function_spec)
            converted.append(conv_tool)
        return converted

    def chat_completion_alpha2(
        self,
        *,
        llm: str,
        inputs: List[dapr_conversation.ConversationInputAlpha2],
        scrub_pii: Optional[bool] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        context_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Invoke Dapr Conversation API Alpha2 with optional tool-calling support and
        convert the response into a simplified OpenAI-like JSON envelope.
        """
        conv_tools = self._convert_openai_tools_to_conversation_tools(tools)

        # TODO: Remove when langchaningo is updated in contrib to latest version with a fix for openai-like temperature
        if not temperature:
            temperature = 1

        response_alpha2 = self.dapr_client.converse_alpha2(
            name=llm,
            inputs=inputs,
            context_id=context_id,
            parameters=parameters,
            scrub_pii=scrub_pii,
            temperature=temperature,
            tools=conv_tools,
            tool_choice=tool_choice,
        )

        outputs: List[Dict[str, Any]] = []
        for output in getattr(response_alpha2, "outputs", []) or []:
            choices_list: List[Dict[str, Any]] = []
            for choice in getattr(output, "choices", []) or []:
                msg = getattr(choice, "message", None)
                content = getattr(msg, "content", None) if msg else None

                # Convert tool calls if present
                tool_calls_json: Optional[List[Dict[str, Any]]] = None
                if msg and getattr(msg, "tool_calls", None):
                    tool_calls_json = []
                    for tc in msg.tool_calls:
                        fn = getattr(tc, "function", None)
                        arguments = getattr(fn, "arguments", None) if fn else None
                        if isinstance(arguments, (dict, list)):
                            try:
                                arguments = json.dumps(arguments)
                            except Exception:
                                arguments = str(arguments)
                        elif arguments is None:
                            arguments = ""

                        tool_calls_json.append(
                            {
                                "id": getattr(tc, "id", ""),
                                "type": "function",
                                "function": {
                                    "name": getattr(fn, "name", "") if fn else "",
                                    "arguments": arguments,
                                },
                            }
                        )

                choices_list.append(
                    {
                        "message": {
                            "role": "assistant",
                            "content": content,
                            **(
                                {"tool_calls": tool_calls_json}
                                if tool_calls_json
                                else {}
                            ),
                        },
                        "finish_reason": getattr(choice, "finish_reason", "stop"),
                    }
                )
            outputs.append({"choices": choices_list})

        return {
            "context_id": getattr(response_alpha2, "context_id", None),
            "outputs": outputs,
        }


class DaprInferenceClientBase(LLMClientBase):
    """
    Base class for managing Dapr Inference API clients.
    Handles client initialization, configuration, and shared logic.
    """

    @model_validator(mode="before")
    def validate_and_initialize(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        return values

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes private attributes after validation.
        """
        self._provider = "dapr"

        # Set up the private config and client attributes
        self._config = self.get_config()
        self._client = self.get_client()
        return super().model_post_init(__context)

    def get_config(self) -> DaprInferenceClientConfig:
        """
        Returns the appropriate configuration for the Dapr Conversation API.
        """
        return DaprInferenceClientConfig()

    def get_client(self) -> DaprInferenceClient:
        """
        Initializes and returns the Dapr Inference client.
        """
        return DaprInferenceClient()

    @classmethod
    def from_config(
        cls, client_options: DaprInferenceClientConfig, timeout: float = 1500
    ):
        """
        Initializes the DaprInferenceClientBase using DaprInferenceClientConfig.

        Args:
            client_options: The configuration options for the client.
            timeout: Timeout for requests (default is 1500 seconds).

        Returns:
            DaprInferenceClientBase: The initialized client instance.
        """
        return cls()

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @property
    def client(self) -> DaprInferenceClient:
        return self._client
