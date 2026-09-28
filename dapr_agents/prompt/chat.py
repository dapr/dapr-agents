#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
from pydantic import Field

from dapr_agents.prompt.base import PromptTemplateBase
from dapr_agents.prompt.engine import (
    DEFAULT_FORMATTER_MAPPING,
    DEFAULT_VARIABLE_EXTRACTOR_MAPPING,
    TemplateEngine,
    extract_fstring_variables,
    extract_jinja_variables,
    render_fstring_template,
    render_jinja_template,
)
from dapr_agents.types.message import (
    AssistantMessage,
    BaseMessage,
    MessagePlaceHolder,
    SystemMessage,
    ToolMessage,
    UserMessage,
)

logger = logging.getLogger(__name__)


class ChatPromptHelper:
    """
    Utility class for handling operations on chat prompt messages, such as
    formatting, normalizing, and extracting variables.
    """

    _ROLE_MAP: Dict[str, Type[BaseMessage]] = {
        "system": SystemMessage,
        "user": UserMessage,
        "assistant": AssistantMessage,
        "tool": ToolMessage,
    }

    @classmethod
    def normalize_chat_messages(cls, variable_value: Any) -> List[BaseMessage]:
        """
        Normalize the variable value into a list of BaseMessages, handling strings, dictionaries, and lists.

        Args:
            variable_value (Any): The value associated with a placeholder variable to normalize.

        Returns:
            List[BaseMessage]: A list of normalized BaseMessage instances.

        Raises:
            ValueError: If an unsupported type is encountered within the list or variable.
        """
        normalized_messages: List[BaseMessage] = []

        def validate_and_create_message(
            role: str, content: str, message_data: dict
        ) -> BaseMessage:
            if role not in cls._ROLE_MAP:
                raise ValueError(
                    f"Unrecognized role '{role}' in message: {message_data}"
                )
            return cls.create_message(role, content, message_data)

        if isinstance(variable_value, str):
            normalized_messages.append(cls.create_message("user", variable_value, {}))
        elif isinstance(variable_value, list):
            for item in variable_value:
                if isinstance(item, str):
                    normalized_messages.append(cls.create_message("user", item, {}))
                elif isinstance(item, BaseMessage):
                    normalized_messages.append(item)
                elif isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content", "")
                    if role == "tool" or (
                        role == "assistant" and item.get("tool_calls")
                    ):
                        normalized_messages.append(
                            validate_and_create_message(role, content, item)
                        )
                    else:
                        normalized_messages.append(
                            validate_and_create_message(role, content, {})
                        )
                else:
                    raise ValueError(
                        f"Unsupported type in list for variable: {type(item)}"
                    )
        elif isinstance(variable_value, dict):
            role = variable_value.get("role", "user")
            content = variable_value.get("content", "")
            if role == "tool" or (
                role == "assistant" and variable_value.get("tool_calls")
            ):
                normalized_messages.append(
                    validate_and_create_message(role, content, variable_value)
                )
            else:
                normalized_messages.append(
                    validate_and_create_message(role, content, {})
                )
        else:
            raise ValueError(f"Unsupported type for variable: {type(variable_value)}")

        return normalized_messages

    @classmethod
    def format_message(
        cls,
        message: Union[Tuple[str, str], Dict[str, Any], BaseMessage],
        template_format: str,
        **kwargs: Any,
    ) -> BaseMessage:
        """
        Format a single message by replacing template variables based on the specified format.

        Args:
            message (Union[Tuple[str, str], Dict[str, Any], BaseMessage]): The message to format.
            template_format (str): The format for rendering ('f-string' or 'jinja2').
            **kwargs: Variables used to populate placeholders within the message.

        Returns:
            BaseMessage: The message with variables replaced as per the template format.
        """
        role, content = cls.extract_role_and_content(message)
        content = cls.format_content(content, template_format=template_format, **kwargs)
        if isinstance(message, BaseMessage):
            message_data = message.model_dump()
        elif isinstance(message, dict):
            message_data = message
        else:
            message_data = {}
        return cls.create_message(role, content, message_data)

    @staticmethod
    def format_content(content: str, template_format: str, **kwargs: Any) -> str:
        """
        Apply template formatting to the content string using the specified format.

        Args:
            content (str): The content string to format.
            template_format (str): Template format ('f-string' or 'jinja2').
            **kwargs: Variables for populating placeholders within the content.

        Returns:
            str: The formatted content.
        """
        return TemplateEngine.render(content, template_format=template_format, **kwargs)

    @classmethod
    def extract_role_and_content(
        cls, message: Union[Tuple[str, str], Dict[str, Any], BaseMessage]
    ) -> Tuple[str, str]:
        """
        Extract role and content from a message.

        Args:
            message (Union[Tuple[str, str], Dict[str, Any], BaseMessage]): A message object.

        Returns:
            Tuple[str, str]: Extracted role and content.

        Raises:
            ValueError: If the message is not in a supported format.
        """
        if isinstance(message, tuple) and len(message) == 2:
            return message[0], message[1]
        elif isinstance(message, dict):
            return message.get("role", ""), message.get("content", "")
        elif isinstance(message, BaseMessage):
            return message.role, message.content
        else:
            raise ValueError(
                "Message must be a tuple (role, content), a dict with 'role' and 'content', or a BaseMessage instance."
            )

    @classmethod
    def create_message(
        cls, role: str, content: str, message_data: Dict[str, Any]
    ) -> BaseMessage:
        """
        Create a BaseMessage instance based on role.

        Args:
            role (str): Role of the message (system, user, assistant, tool).
            content (str): Message content.
            message_data (Dict[str, Any]): Additional data.

        Returns:
            BaseMessage: Formatted message instance.

        Raises:
            ValueError: If the role is not recognized.
        """
        if role not in cls._ROLE_MAP:
            raise ValueError(f"Invalid message role: {role}")

        message_class = cls._ROLE_MAP[role]
        if role == "tool":
            return message_class(
                content=content, tool_call_id=message_data.get("tool_call_id")
            )
        elif role == "assistant" and message_data.get("tool_calls") is not None:
            return message_class(
                content=content, tool_calls=message_data.get("tool_calls")
            )
        return message_class(content=content)

    @classmethod
    def get_message_class(cls, role: str) -> Optional[Type[BaseMessage]]:
        """Get the message class for a given role."""
        return cls._ROLE_MAP.get(role.lower(), None)

    @classmethod
    def parse_role_content(cls, content: str) -> Tuple[List[str], Optional[str]]:
        """
        Parse the formatted content into role-based chunks and any remaining plain text.

        Returns:
            Tuple[List[str], Optional[str]]: Role-based chunks and any remaining plain text.
        """
        role_pattern = (
            r"(?i)^\s*#?\s*(" + "|".join(cls._ROLE_MAP.keys()) + r")\s*:\s*\n"
        )

        if not re.search(role_pattern, content, flags=re.MULTILINE):
            return [], content.strip()

        chunks = re.split(role_pattern, content, flags=re.MULTILINE)

        plain_text = None
        if chunks and chunks[0].strip() and chunks[0].lower() not in cls._ROLE_MAP:
            plain_text = chunks.pop(0).strip()

        role_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return role_chunks, plain_text

    @classmethod
    def to_message(cls, role: str, content: str) -> BaseMessage:
        """Parse a single chunk of content into a message object."""
        role = role.strip().lower()
        content = content.strip()

        logger.debug(f"Parsing role: '{role}', content: '{content[:30]}...'")

        message_class = cls.get_message_class(role)
        if not message_class:
            raise ValueError(f"Invalid message role: '{role}'")

        if not content:
            raise ValueError(f"Content missing for role: '{role}'")

        return message_class(content=content)

    @classmethod
    def parse_as_messages(
        cls,
        content: str,
    ) -> Tuple[Optional[List[BaseMessage]], Optional[str]]:
        """
        Parse the content into a list of role-based messages and any unstructured plain text.

        Returns:
            Tuple[List[BaseMessage], Optional[str]]: Parsed messages if role-based chunks are found,
            and any remaining plain text if detected.
        """
        role_chunks, plain_text = cls.parse_role_content(content)

        if not role_chunks:
            logger.debug("No role-based content found; returning plain text.")
            return [], plain_text

        messages: List[BaseMessage] = []
        role: Optional[str] = None

        for chunk in role_chunks:
            if chunk.lower() in cls._ROLE_MAP:
                role = chunk
            elif role:
                messages.append(cls.to_message(role, chunk))
                role = None
            else:
                raise ValueError(f"Unexpected content without a role: {chunk}")

        return messages, plain_text


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

    _ROLE_MAP = ChatPromptHelper._ROLE_MAP

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

        rendered_messages: List[Dict[str, Any]] = []

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
                formatted_content = TemplateEngine.render(
                    content, template_format=template_format, **all_variables
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
        input_vars: set = set()

        for msg in messages:
            content = None

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
                    TemplateEngine.extract_variables(content, template_format)
                )

        return cls(
            input_variables=list(input_vars),
            messages=messages,
            template_format=template_format,
        )


__all__ = [
    "ChatPromptTemplate",
    "ChatPromptHelper",
    "DEFAULT_FORMATTER_MAPPING",
    "DEFAULT_VARIABLE_EXTRACTOR_MAPPING",
    "render_fstring_template",
    "extract_fstring_variables",
    "render_jinja_template",
    "extract_jinja_variables",
]
