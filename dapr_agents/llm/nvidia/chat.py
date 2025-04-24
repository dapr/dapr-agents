from dapr_agents.llm.utils import RequestHandler, ResponseHandler
from dapr_agents.llm.nvidia.client import NVIDIAClientBase
from dapr_agents.types.message import BaseMessage
from dapr_agents.llm.chat import ChatClientBase
from dapr_agents.prompt.prompty import Prompty
from dapr_agents.tool import AgentTool
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
from openai.types.chat import ChatCompletionMessage
from pydantic import BaseModel, Field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class NVIDIAChatClient(NVIDIAClientBase, ChatClientBase):
    """
    Chat client for NVIDIA chat models.
    Combines NVIDIA client management with Prompty-specific functionality for handling chat completions.
    """

    model: str = Field(
        default="meta/llama3-8b-instruct",
        description="Model name to use. Defaults to 'meta/llama3-8b-instruct'.",
    )
    max_tokens: Optional[int] = Field(
        default=1024,
        description=(
            "The maximum number of tokens to generate in any given call. Must be an integer ≥ 1. Defaults to 1024."
        ),
    )

    SUPPORTED_STRUCTURED_MODES: ClassVar[set] = {"function_call"}

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes chat-specific attributes after validation.

        Args:
            __context (Any): Additional context for post-initialization (not used here).
        """
        self._api = "chat"
        super().model_post_init(__context)

    @classmethod
    def from_prompty(cls, prompty_source: Union[str, Path]) -> "NVIDIAChatClient":
        """
        Initializes an NVIDIAChatClient client using a Prompty source, which can be a file path or inline content.

        Args:
            prompty_source (Union[str, Path]): The source of the Prompty file, which can be a path to a file
                or inline Prompty content as a string.

        Returns:
            NVIDIAChatClient: An instance of NVIDIAChatClient configured with the model settings from the Prompty source.
        """
        # Load the Prompty instance from the provided source
        prompty_instance = Prompty.load(prompty_source)

        # Generate the prompt template from the Prompty instance
        prompt_template = Prompty.to_prompt_template(prompty_instance)

        # Extract the model configuration from Prompty
        model_config = prompty_instance.model

        # Initialize the NVIDIAChatClient instance using model_validate
        return cls.model_validate(
            {
                "model": model_config.configuration.name,
                "api_key": model_config.configuration.api_key,
                "base_url": model_config.configuration.base_url,
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
        input_data: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        tools: Optional[List[Union[AgentTool, Dict[str, Any]]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        max_tokens: Optional[int] = None,
        structured_mode: Literal["function_call"] = "function_call",
        **kwargs,
    ) -> Union[Iterator[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate chat completions based on provided messages or input_data for prompt templates.

        Args:
            messages (Optional): Either pre-set messages or None if using input_data.
            input_data (Optional[Dict[str, Any]]): Input variables for prompt templates.
            model (str): Specific model to use for the request, overriding the default.
            tools (List[Union[AgentTool, Dict[str, Any]]]): List of tools for the request.
            response_format (Type[BaseModel]): Optional Pydantic model for structured response parsing.
            max_tokens (Optional[int]): The maximum number of tokens to generate. Defaults to the instance setting.
            structured_mode (Literal["function_call"]): Mode for structured output: "function_call" (Limited Support).
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
        params = {"messages": RequestHandler.normalize_chat_messages(messages)}

        # Merge prompty parameters if available, then override with any explicit kwargs
        if self.prompty:
            params = {**self.prompty.model.parameters.model_dump(), **params, **kwargs}
        else:
            params.update(kwargs)

        # If a model is provided, override the default model
        params["model"] = model or self.model

        # Apply max_tokens if provided
        params["max_tokens"] = max_tokens or self.max_tokens

        # Prepare request parameters
        params = RequestHandler.process_params(
            params,
            llm_provider=self.provider,
            tools=tools,
            response_format=response_format,
            structured_mode=structured_mode,
        )

        try:
            logger.info("Invoking ChatCompletion API.")
            logger.debug(f"ChatCompletion API Parameters:{params}")
            response: ChatCompletionMessage = self.client.chat.completions.create(
                **params
            )
            logger.info("Chat completion retrieved successfully.")

            return ResponseHandler.process_response(
                response,
                llm_provider=self.provider,
                response_format=response_format,
                structured_mode=structured_mode,
                stream=params.get("stream", False),
            )
        except Exception as e:
            logger.error(f"An error occurred during the ChatCompletion API call: {e}")
            raise
