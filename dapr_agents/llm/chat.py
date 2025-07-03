from typing import Union, Dict, Any, Optional, Iterable, List, Iterator, Type
from dapr_agents.prompt.base import PromptTemplateBase
from dapr_agents.prompt.prompty import Prompty
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from pathlib import Path
from dapr_agents.tool.base import AgentTool


class ChatClientBase(BaseModel, ABC):
    """
    Base class for chat-specific functionality.
    Handles Prompty integration and provides abstract methods for chat client configuration.
    """

    prompty: Optional[Prompty] = Field(
        default=None, description="Instance of the Prompty object (optional)."
    )
    prompt_template: Optional[PromptTemplateBase] = Field(
        default=None, description="Prompt template for rendering (optional)."
    )

    @classmethod
    @abstractmethod
    def from_prompty(
        cls,
        prompty_source: Union[str, Path],
        timeout: Union[int, float, Dict[str, Any]] = 1500,
    ) -> "ChatClientBase":
        """
        Abstract method to load a Prompty source and configure the chat client.

        Args:
            prompty_source (Union[str, Path]): Source of the Prompty, either a file path or inline Prompty content.
            timeout (Union[int, float, Dict[str, Any]]): Timeout for requests.

        Returns:
            ChatClientBase: Configured chat client instance.
        """
        pass

    @abstractmethod
    def generate(
        self,
        messages: Union[
            str, Dict[str, Any], BaseModel, Iterable[Union[Dict[str, Any], BaseModel]]
        ] = None,
        input_data: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        tools: Optional[List[Union[AgentTool, Dict[str, Any]]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        structured_mode: Optional[str] = None,
        **kwargs,
    ) -> Union[Iterator[Dict[str, Any]], Dict[str, Any]]:
        """
        Abstract method to generate chat completions.

        Args:
            messages (Optional): Either pre-set messages or None if using input_data.
            input_data (Optional[Dict[str, Any]]): Input variables for prompt templates.
            model (Optional[str]): Specific model to use for the request, overriding the default.
            tools (Optional[List[Union[AgentTool, Dict[str, Any]]]]): List of tools for the request.
            response_format (Optional[Type[BaseModel]]): Optional Pydantic model for structured response parsing.
            structured_mode (Optional[str]): Mode for structured output.
            **kwargs: Additional parameters for the chat completion API.

        Returns:
            Union[Iterator[Dict[str, Any]], Dict[str, Any]]: The chat completion response(s).
        """
        pass
