from dapr_agents.types.llm import (
    PromptyModelConfig,
    OpenAIModelConfig,
    AzureOpenAIModelConfig,
    PromptyDefinition,
)
from dapr_agents.prompt.base import PromptTemplateBase
from dapr_agents.prompt.chat import ChatPromptTemplate
from dapr_agents.prompt.string import StringPromptTemplate
from dapr_agents.prompt.utils.prompty import PromptyHelper
from typing import Dict, Any, Union, Optional, Literal, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Prompty(PromptyDefinition):
    """
    A class to handle loading and formatting of Prompty templates for language models workflows.
    """

    def extract_input_variables(
        self, template_format: Literal["f-string", "jinja2"] = "jinja2"
    ) -> Tuple[List[str], List[str]]:
        """
        Extract all input variables from the Prompty instance, including placeholders from content,
        predefined inputs, and sample inputs. This method returns both regular input variables and
        more complex placeholders that may require additional processing.

        Args:
            template_format (Literal["f-string", "jinja2"]): Template format for content parsing. Default is 'jinja2'.

        Returns:
            Tuple[List[str], List[str]]:
                - A list of regular input variables.
                - A list of placeholders that may require extra processing (e.g., loops or attributes).
        """
        # Extract undeclared variables and placeholders from the content
        undeclared_variables = PromptyHelper.extract_placeholders_from_content(
            self.content, template_format
        )

        # Gather predefined inputs and sample inputs, default to empty dict if they are None
        predefined_inputs = list(self.inputs.keys()) if self.inputs else []
        sample_inputs = list(self.sample.keys()) if self.sample else []

        # Combine undeclared variables (regular inputs), filtered predefined inputs, and sample inputs
        regular_variables = list(
            set(undeclared_variables + predefined_inputs + sample_inputs)
        )

        return regular_variables

    def to_prompt_template(
        self, template_format: Literal["f-string", "jinja2"] = "jinja2"
    ) -> PromptTemplateBase:
        """
        Convert this Prompty instance into a PromptTemplateBase instance by pre-processing the content
        into a list of messages (for chat) or a string (for completion).
        No inputs are provided at this stage, so placeholders and dynamic parts remain in the template.

        Args:
            template_format (Literal["f-string", "jinja2"]): Template format for content parsing. Default is 'jinja2'.

        Returns:
            PromptTemplateBase: An instance of ChatPromptTemplate or StringPromptTemplate.
        """
        # Ensure that content is present in the Prompty instance
        if not self.content:
            raise ValueError(
                "Prompty instance is missing 'content'. Cannot convert to prompt template."
            )

        # Extract input variables and placeholders using the updated method
        regular_variables = self.extract_input_variables(template_format)

        # Pre-process the content into a list of messages for chat-based prompts
        if self.model.api == "chat":
            # Process the content into messages using PromptyHelper
            messages = PromptyHelper.to_prompt(self.content, api_type="chat")
            return ChatPromptTemplate(
                input_variables=regular_variables,
                messages=messages,
                template_format=template_format,
            )
        else:
            return StringPromptTemplate(
                input_variables=regular_variables,
                template=self.content,
                template_format=template_format,
            )

    @classmethod
    def load(
        cls,
        prompty_source: Union[str, Path],
        model: Optional[
            Union[OpenAIModelConfig, AzureOpenAIModelConfig, Dict[str, Any]]
        ] = None,
    ) -> "Prompty":
        """
        Load a Prompty template from a file or inline content and configure the Prompty object.

        Args:
            prompty_source (Union[str, Path]): Path to the Prompty file or inline content as a string.
            model (Optional[Union[OpenAIModelConfig, AzureOpenAIModelConfig, Dict[str, Any]]]): Optional model configuration to override.

        Returns:
            Prompty: A validated Prompty object.
        """
        # Convert prompty_source to Path if it seems like a file path
        if isinstance(prompty_source, str) and prompty_source.endswith(".prompty"):
            path_object = Path(prompty_source).resolve()
        elif isinstance(prompty_source, Path):
            path_object = prompty_source.resolve()
        else:
            path_object = None

        # Determine if we're dealing with a file path or inline content
        if path_object and path_object.exists():
            # Use file path, resolving any path-based references
            metadata, content = PromptyHelper.parse_prompty_content(path_object)
            metadata = PromptyHelper.normalize(metadata, parent=path_object.parent)
        else:
            # Treat prompty_source as inline content
            metadata, content = PromptyHelper.parse_prompty_content(prompty_source)
            metadata = PromptyHelper.normalize(metadata, parent=Path().resolve())

        # Override the model if provided
        if model:
            if isinstance(model, dict):
                metadata["model"] = PromptyModelConfig(**model)
            else:
                metadata["model"] = model.model_dump()

        # Validate and construct the Prompty object
        metadata["content"] = content
        return cls.model_validate(metadata)
