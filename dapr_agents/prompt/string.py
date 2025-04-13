from dapr_agents.prompt.base import PromptTemplateBase
from dapr_agents.prompt.utils.string import StringPromptHelper
from typing import Any, Union, Literal


class StringPromptTemplate(PromptTemplateBase):
    """
    A template class designed to handle string-based prompts. This class can format a string template
    by replacing variables and merging additional context or keywords.

    Attributes:
        template (str): The template string that defines the prompt structure.
        template_format (Literal["f-string", "jinja2"]): The format used for rendering the template, either f-string or Jinja2.
    """

    template: str
    template_format: Literal["f-string", "jinja2"] = "f-string"

    def format_prompt(self, **kwargs: Any) -> str:
        """
        Format the prompt by replacing variables in the template, using any provided keyword arguments.
        Validates the template structure before formatting.

        Args:
            **kwargs: Additional keyword arguments to replace variables in the template.

        Returns:
            str: The formatted prompt with all variables replaced.

        Raises:
            ValueError: If required variables are missing or extra undeclared variables are passed.
        """
        # Extract the required input variables from the template
        input_variables = StringPromptHelper.extract_variables(
            self.template, self.template_format
        )

        # Check for missing variables
        missing_vars = [var for var in input_variables if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables in template: {missing_vars}")

        # Check for extra variables that were not expected
        extra_variables = [var for var in kwargs if var not in input_variables]
        if extra_variables:
            raise ValueError(f"Undeclared variables were passed: {extra_variables}")

        # Prepare the variables for formatting
        kwargs = self.prepare_variables_for_formatting(**kwargs)

        # Use the helper to format the content
        return StringPromptHelper.format_content(
            self.template, self.template_format, **kwargs
        )

    @classmethod
    def from_template(
        cls, template: str, template_format: str = "f-string", **kwargs: Any
    ) -> "StringPromptTemplate":
        """
        Create a StringPromptTemplate from a template string.

        Args:
            template (str): The template string that defines the structure of the prompt.
            template_format (str): The format of the template, either "f-string" or "jinja2". Default is "f-string".
            **kwargs: Additional keyword arguments for the constructor.

        Returns:
            StringPromptTemplate: A new instance of the template with extracted input variables.
        """
        input_variables = StringPromptHelper.extract_variables(
            template, template_format
        )
        return cls(
            template=template,
            template_format=template_format,
            input_variables=input_variables,
            **kwargs,
        )

    def __add__(
        self, other: Union[str, "StringPromptTemplate"]
    ) -> "StringPromptTemplate":
        """
        Override the + operator to allow for combining prompt templates.

        Args:
            other: Another prompt template or string to be combined with the current one.

        Returns:
            StringPromptTemplate: A new instance of the combined templates with merged variables.
        """
        if isinstance(other, StringPromptTemplate):
            # Ensure both templates use the same format
            if self.template_format != other.template_format:
                raise ValueError(
                    "Adding prompt templates only supported for the same template format."
                )

            # Combine input variables
            input_variables = list(
                set(self.input_variables) | set(other.input_variables)
            )

            # Combine template strings
            template = self.template + other.template

            # Combine pre-filled variables
            pre_filled_variables = {
                **self.pre_filled_variables,
                **other.pre_filled_variables,
            }

            # Create and return a new combined StringPromptTemplate
            return StringPromptTemplate(
                template=template,
                template_format=self.template_format,
                input_variables=input_variables,
                pre_filled_variables=pre_filled_variables,
            )
        elif isinstance(other, str):
            # If other is a string, convert it to a StringPromptTemplate and combine
            return self + StringPromptTemplate.from_template(
                other, template_format=self.template_format
            )
        else:
            # Raise error for unsupported types
            raise NotImplementedError(f"Unsupported operand type for +: {type(other)}")
