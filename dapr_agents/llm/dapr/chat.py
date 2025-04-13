from dapr_agents.llm.dapr.client import DaprInferenceClientBase
from dapr_agents.llm.utils import RequestHandler, ResponseHandler
from dapr_agents.prompt.prompty import Prompty
from dapr_agents.types.message import BaseMessage
from dapr_agents.llm.chat import ChatClientBase
from dapr_agents.tool import AgentTool
from dapr.clients.grpc._request import ConversationInput
from typing import (
    Union,
    Optional,
    Iterable,
    Dict,
    Any,
    List,
    Iterator,
    Type,
    Literal,
    ClassVar,
)
from pydantic import BaseModel
from pathlib import Path
import logging
import os
import time

logger = logging.getLogger(__name__)


class DaprChatClient(DaprInferenceClientBase, ChatClientBase):
    """
    Concrete class for Dapr's chat completion API using the Inference API.
    This class extends the ChatClientBase.
    """

    SUPPORTED_STRUCTURED_MODES: ClassVar[set] = {"function_call"}

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes private attributes for provider, api, config, and client after validation.
        """
        # Set the private provider and api attributes
        self._api = "chat"
        self._llm_component = os.environ["DAPR_LLM_COMPONENT_DEFAULT"]

        return super().model_post_init(__context)

    @classmethod
    def from_prompty(
        cls,
        prompty_source: Union[str, Path],
        timeout: Union[int, float, Dict[str, Any]] = 1500,
    ) -> "DaprChatClient":
        """
        Initializes an DaprChatClient client using a Prompty source, which can be a file path or inline content.

        Args:
            prompty_source (Union[str, Path]): The source of the Prompty file, which can be a path to a file
                or inline Prompty content as a string.
            timeout (Union[int, float, Dict[str, Any]], optional): Timeout for requests, defaults to 1500 seconds.

        Returns:
            DaprChatClient: An instance of DaprChatClient configured with the model settings from the Prompty source.
        """
        # Load the Prompty instance from the provided source
        prompty_instance = Prompty.load(prompty_source)

        # Generate the prompt template from the Prompty instance
        prompt_template = Prompty.to_prompt_template(prompty_instance)

        # Initialize the DaprChatClient based on the Prompty model configuration
        return cls.model_validate(
            {
                "timeout": timeout,
                "prompty": prompty_instance,
                "prompt_template": prompt_template,
            }
        )

    def translate_response(self, response: dict, model: str) -> dict:
        """Converts a Dapr response dict into a structure compatible with Choice and ChatCompletion."""
        choices = [
            {
                "finish_reason": "stop",
                "index": i,
                "message": {"content": output["result"], "role": "assistant"},
                "logprobs": None,
            }
            for i, output in enumerate(response.get("outputs", []))
        ]

        return {
            "choices": choices,
            "created": int(time.time()),
            "model": model,
            "object": "chat.completion",
            "usage": {"total_tokens": "-1"},
        }

    def convert_to_conversation_inputs(
        self, inputs: List[Dict[str, Any]]
    ) -> List[ConversationInput]:
        return [
            ConversationInput(
                content=item["content"],
                role=item.get("role"),
                scrub_pii=item.get("scrubPII") == "true",
            )
            for item in inputs
        ]

    def generate(
        self,
        messages: Union[
            str,
            Dict[str, Any],
            BaseMessage,
            Iterable[Union[Dict[str, Any], BaseMessage]],
        ] = None,
        input_data: Optional[Dict[str, Any]] = None,
        llm_component: Optional[str] = None,
        tools: Optional[List[Union[AgentTool, Dict[str, Any]]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        structured_mode: Literal["function_call"] = "function_call",
        scrubPII: Optional[bool] = False,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Union[Iterator[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate chat completions based on provided messages or input_data for prompt templates.

        Args:
            messages (Optional): Either pre-set messages or None if using input_data.
            input_data (Optional[Dict[str, Any]]): Input variables for prompt templates.
            llm_component (str): Name of the LLM component to use for the request.
            tools (List[Union[AgentTool, Dict[str, Any]]]): List of tools for the request.
            response_format (Type[BaseModel]): Optional Pydantic model for structured response parsing.
            structured_mode (Literal["function_call"]): Mode for structured output: "function_call" (Limited Support).
            scrubPII (Type[bool]): Optional flag to obfuscate any sensitive information coming back from the LLM.
            **kwargs: Additional parameters for the language model.

        Returns:
            Union[Iterator[Dict[str, Any]], Dict[str, Any]]: The chat completion response(s).
        """
        if structured_mode not in self.SUPPORTED_STRUCTURED_MODES:
            raise ValueError(
                f"Invalid structured_mode '{structured_mode}'. Must be one of {self.SUPPORTED_STRUCTURED_MODES}."
            )

        # If input_data is provided, check for a prompt_template
        if input_data:
            if not self.prompt_template:
                raise ValueError(
                    "Inputs are provided but no 'prompt_template' is set. Please set a 'prompt_template' to use the input_data."
                )

            logger.info("Using prompt template to generate messages.")
            messages = self.prompt_template.format_prompt(**input_data)

        # Ensure we have messages at this point
        if not messages:
            raise ValueError("Either 'messages' or 'input_data' must be provided.")

        # Process and normalize the messages
        params = {"inputs": RequestHandler.normalize_chat_messages(messages)}
        # Merge Prompty parameters if available, then override with any explicit kwargs
        if self.prompty:
            params = {**self.prompty.model.parameters.model_dump(), **params, **kwargs}
        else:
            params.update(kwargs)

        # Prepare request parameters
        params = RequestHandler.process_params(
            params,
            llm_provider=self.provider,
            tools=tools,
            response_format=response_format,
            structured_mode=structured_mode,
        )
        inputs = self.convert_to_conversation_inputs(params["inputs"])

        try:
            logger.info("Invoking the Dapr Conversation API.")
            response = self.client.chat_completion(
                llm=llm_component or self._llm_component,
                conversation_inputs=inputs,
                scrub_pii=scrubPII,
                temperature=temperature,
            )
            transposed_response = self.translate_response(response, self._llm_component)
            logger.info("Chat completion retrieved successfully.")

            return ResponseHandler.process_response(
                transposed_response,
                llm_provider=self.provider,
                response_format=response_format,
                structured_mode=structured_mode,
                stream=params.get("stream", False),
            )
        except Exception as e:
            logger.error(
                f"An error occurred during the Dapr Conversation API call: {e}"
            )
            raise
