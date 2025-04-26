from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Callable
from pydantic import BaseModel, Field, ConfigDict
import logging

logger = logging.getLogger(__name__)


class PromptTemplateBase(ABC, BaseModel):
    """
    Abstract base class for creating prompt templates. This class provides common attributes and methods for handling
    input variables and placeholders.
    """

    input_variables: List[str] = Field(
        ...,
        description="A list of undeclared input variables expected in the prompt template.",
    )
    pre_filled_variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="A dictionary of variables used to pre-fill the template.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def format_prompt(self, **kwargs: Any) -> Union[str, List[Any]]:
        """
        Abstract method for formatting the prompt. Must be implemented by subclasses.

        Args:
            **kwargs: Keyword arguments to be used for formatting the prompt.

        Returns:
            Union[str, List[Any]]: The formatted prompt.
        """
        pass

    def pre_fill_variables(
        self, **kwargs: Union[str, Callable[[], str]]
    ) -> "PromptTemplateBase":
        """
        Create a new instance of the prompt template with some input variables already pre-filled.

        Args:
            **kwargs: Variables to pre-fill the template. Can be strings or callables that return strings.

        Returns:
            PromptTemplateBase: A new instance of the prompt template with the pre-filled variables set.
        """
        # Directly access the attributes without calling model_dump()
        input_variables = list(set(self.input_variables) - set(kwargs.keys()))
        pre_filled_variables = {**self.pre_filled_variables, **kwargs}

        # Directly construct a new instance to retain MessagePlaceHolder objects
        return self.__class__(
            input_variables=input_variables,
            pre_filled_variables=pre_filled_variables,
            messages=self.messages,  # Preserve MessagePlaceHolder without transforming it
            template_format=self.template_format,
        )

    def prepare_variables_for_formatting(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Merge user-provided variables with pre-filled variables, resolving any callable pre-fills.

        Args:
            **kwargs: User-provided variables.

        Returns:
            Dict[str, Any]: The merged variables.
        """
        pre_filled_kwargs = {
            k: v() if callable(v) else v for k, v in self.pre_filled_variables.items()
        }
        merged_kwargs = {**pre_filled_kwargs, **kwargs}
        logger.debug(f"Merged variables for formatting: {merged_kwargs}")
        return merged_kwargs
