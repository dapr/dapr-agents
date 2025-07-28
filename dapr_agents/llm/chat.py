from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    overload,
)

from pydantic import BaseModel

from dapr_agents.prompt.base import PromptTemplateBase
from dapr_agents.prompt.prompty import Prompty
from dapr_agents.tool.base import AgentTool
from dapr_agents.types.message import LLMChatCandidateChunk, LLMChatResponse

T = TypeVar("T", bound=BaseModel)


class ChatClientBase(ABC):
    """
    Base class for chat-specific functionality.
    Handles Prompty integration and provides abstract methods for chat client configuration.

    Attributes:
        prompty: Optional Prompty spec used to render `input_data` into messages.
        prompt_template: Optional prompt template object for rendering.
    """

    prompty: Optional[Prompty]
    prompt_template: Optional[PromptTemplateBase]

    @classmethod
    @abstractmethod
    def from_prompty(
        cls,
        prompty_source: Union[str, Path],
        timeout: Union[int, float, Dict[str, Any]] = 1500,
    ) -> "ChatClientBase":
        """
        Load a Prompty spec (path or inline), extract its model config and
        prompt template, and return a configured chat client.

        Args:
            prompty_source: Path or inline YAML/JSON for a Prompty spec.
            timeout: HTTP timeout (seconds or HTTPX-style dict).

        Returns:
            A ready-to-use ChatClientBase subclass instance.
        """
        ...

    @overload
    def generate(
        self,
        messages: Union[
            str, Dict[str, Any], Any, Iterable[Union[Dict[str, Any], Any]]
        ] = None,
        *,
        input_data: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        tools: Optional[List[Union[AgentTool, Dict[str, Any]]]] = None,
        response_format: None = None,
        structured_mode: Optional[str] = None,
        stream: Literal[False] = False,
        **kwargs: Any,
    ) -> LLMChatResponse:
        ...

    """If `stream=False` and no `response_format`, returns raw LLMChatResponse."""

    @overload
    def generate(
        self,
        messages: Union[
            str, Dict[str, Any], Any, Iterable[Union[Dict[str, Any], Any]]
        ] = None,
        *,
        input_data: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        tools: Optional[List[Union[AgentTool, Dict[str, Any]]]] = None,
        response_format: Type[T],
        structured_mode: Optional[str] = None,
        stream: Literal[False] = False,
        **kwargs: Any,
    ) -> Union[T, List[T]]:
        ...

    """If `stream=False` and `response_format=SomeModel`, returns that model or a list thereof."""

    @overload
    def generate(
        self,
        messages: Union[
            str, Dict[str, Any], Any, Iterable[Union[Dict[str, Any], Any]]
        ] = None,
        *,
        input_data: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        tools: Optional[List[Union[AgentTool, Dict[str, Any]]]] = None,
        response_format: Optional[Type[T]] = None,
        structured_mode: Optional[str] = None,
        stream: Literal[True],
        **kwargs: Any,
    ) -> Iterator[LLMChatCandidateChunk]:
        ...

    """If `stream=True`, returns a streaming iterator of chunks."""

    @abstractmethod
    def generate(
        self,
        messages: Union[
            str, Dict[str, Any], Any, Iterable[Union[Dict[str, Any], Any]]
        ] = None,
        *,
        input_data: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        tools: Optional[List[Union[AgentTool, Dict[str, Any]]]] = None,
        response_format: Optional[Type[T]] = None,
        structured_mode: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[
        Iterator[LLMChatCandidateChunk],
        LLMChatResponse,
        T,
        List[T],
    ]:
        """
        The implementation must accept the full set of kwargs and return
        the union of all possible overload returns.
        """
        ...
