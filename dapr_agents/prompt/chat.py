from dapr_agents.prompt.utils.jinja import (
    render_jinja_template,
    extract_jinja_variables,
)
from dapr_agents.prompt.utils.fstring import (
    render_fstring_template,
    extract_fstring_variables,
)
from dapr_agents.prompt.utils.chat import ChatPromptHelper
from dapr_agents.prompt.base import PromptTemplateBase
from dapr_agents.types.message import (
    BaseMessage,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    MessagePlaceHolder,
)
from typing import Any, Dict, List, Tuple, Union, Literal, Optional
from pydantic import Field
import logging

logger = logging.getLogger(__name__)

DEFAULT_FORMATTER_MAPPING = {
    "f-string": render_fstring_template,
    "jinja2": render_jinja_template,
}

DEFAULT_VARIABLE_EXTRACTOR_MAPPING = {
    "f-string": extract_fstring_variables,
    "jinja2": extract_jinja_variables,
}


class ChatPromptTemplate(PromptTemplateBase):
    """
    A template class designed to handle chat-based prompts. This class can format a sequence of chat messages
    and merge chat history with provided variables or placeholders.

    Attributes:
        messages (List[Union[Tuple[str, str], Dict[str, Any], BaseMessage, MessagePlaceHolder]]): A list of messages that make up the prompt template.
        template_format (Literal["f-string", "jinja2"]): The format used for rendering the template.
    """

    messages: List[
        Union[Tuple[str, str], Dict[str, Any], BaseMessage, MessagePlaceHolder]
    ] = Field(default_factory=list)
    template_format: Literal["f-string", "jinja2"] = "f-string"

    _ROLE_MAP = {
        "system": SystemMessage,
        "user": UserMessage,
        "assistant": AssistantMessage,
        "tool": ToolMessage,
    }

    def format_prompt(
        self, template_format: Optional[str] = None, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Format the prompt by processing placeholders, rendering the template with variables,
        and then incorporating both plain text and role-based messages, if applicable.

        Returns:
            List[Dict[str, Any]]: The list of formatted messages as dictionaries.
        """
        template_format = template_format or self.template_format
        all_variables = self.prepare_variables_for_formatting(**kwargs)

        # Check for undeclared or missing variables
        extra_variables = [
            var
            for var in all_variables
            if var not in self.input_variables and var not in self.pre_filled_variables
        ]
        if extra_variables:
            raise ValueError(f"Undeclared variables were passed: {extra_variables}")

        missing_variables = [
            var for var in self.input_variables if var not in all_variables
        ]
        if missing_variables:
            logger.info(f"Some input variables were not provided: {missing_variables}")

        rendered_messages = []

        for item in self.messages:
            # Process MessagePlaceHolder with dynamic messages from all_variables
            if isinstance(item, MessagePlaceHolder):
                variable_name = item.variable_name
                if variable_name in all_variables:
                    normalized_messages = ChatPromptHelper.normalize_chat_messages(
                        all_variables[variable_name]
                    )
                    rendered_messages.extend(
                        [msg.model_dump() for msg in normalized_messages]
                    )
                else:
                    logger.info(
                        f"MessagePlaceHolder variable '{variable_name}' was not provided."
                    )

            # Process BaseMessage, Tuple, and Dict with parse_as_messages
            else:
                role, content = ChatPromptHelper.extract_role_and_content(item)
                formatted_content = ChatPromptHelper.format_content(
                    content, template_format, **all_variables
                )
                parsed_messages, plain_text = ChatPromptHelper.parse_as_messages(
                    formatted_content
                )

                # Add the plain text only if parsed messages are also returned
                if plain_text and parsed_messages:
                    rendered_messages.append(
                        ChatPromptHelper.create_message(
                            role, plain_text, {}
                        ).model_dump()
                    )

                # Add parsed role-based messages if they exist
                if parsed_messages:
                    rendered_messages.extend(
                        [msg.model_dump() for msg in parsed_messages]
                    )
                else:
                    # If only plain text is present (no parsed messages), add it as a single message
                    rendered_messages.append(
                        ChatPromptHelper.create_message(
                            role, plain_text or formatted_content, {}
                        ).model_dump()
                    )

        return rendered_messages

    @classmethod
    def from_messages(
        cls,
        messages: List[
            Union[Tuple[str, str], Dict[str, Any], BaseMessage, MessagePlaceHolder]
        ],
        template_format: str = "f-string",
    ) -> "ChatPromptTemplate":
        """
        Create a ChatPromptTemplate from a list of messages, including placeholders.

        Args:
            messages (List[Union[Tuple[str, str], Dict[str, Any], BaseMessage, MessagePlaceHolder]]):
                The list of messages that define the template.
            template_format (str): The format of the template, either "f-string" or "jinja2". Default is "f-string".

        Returns:
            ChatPromptTemplate: A new instance of the template with extracted input variables.
        """
        input_vars = set()

        for msg in messages:
            content = None  # Initialize content to None

            # Handle MessagePlaceHolder by adding its variable directly to input_vars
            if isinstance(msg, MessagePlaceHolder):
                input_vars.add(msg.variable_name)

            elif isinstance(msg, tuple) and len(msg) == 2:
                content = msg[1]

            elif isinstance(msg, dict) and "content" in msg:
                content = msg["content"]

            elif isinstance(msg, BaseMessage):
                content = msg.content

            if isinstance(content, str):
                input_vars.update(
                    DEFAULT_VARIABLE_EXTRACTOR_MAPPING[template_format](content)
                )

        return cls(
            input_variables=list(input_vars),
            messages=messages,
            template_format=template_format,
        )
