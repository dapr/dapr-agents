from dapr_agents.types.message import (
    BaseMessage,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
)
from dapr_agents.prompt.utils.jinja import (
    render_jinja_template,
    extract_jinja_variables,
)
from dapr_agents.prompt.utils.fstring import (
    render_fstring_template,
    extract_fstring_variables,
)
from typing import Any, Dict, List, Tuple, Union, Optional
import re
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


class ChatPromptHelper:
    """
    Utility class for handling various operations on chat prompt messages, such as
    formatting, normalizing, and extracting variables.

    Attributes:
        _ROLE_MAP (Dict[str, Type[BaseMessage]]): A mapping of role names to message classes.
    """

    _ROLE_MAP = {
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
        normalized_messages = []

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
                    normalized_messages.append(
                        validate_and_create_message(role, content, item)
                    )
                else:
                    raise ValueError(
                        f"Unsupported type in list for variable: {type(item)}"
                    )
        elif isinstance(variable_value, dict):
            role = variable_value.get("role", "user")
            content = variable_value.get("content", "")
            normalized_messages.append(
                validate_and_create_message(role, content, variable_value)
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
        return cls.create_message(role, content, message)

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
        formatter = DEFAULT_FORMATTER_MAPPING.get(template_format)
        if not formatter:
            raise ValueError(f"Unsupported template format: {template_format}")
        return formatter(content, **kwargs)

    @classmethod
    def extract_role_and_content(
        cls, message: Union[Tuple[str, str], Dict[str, Any], BaseMessage]
    ) -> Tuple[str, str]:
        """
        Extract role and content from a message.

        Args:
            message (Union[Tuple[str, str], Dict[str, Any], BaseMessage]): A message object in the form of a tuple, dictionary, or BaseMessage.

        Returns:
            Tuple[str, str]: Extracted role and content.

        Raises:
            ValueError: If the message is not in a supported format.
        """
        if isinstance(message, tuple) and len(message) == 2:
            return message[0], message[1]
        elif isinstance(message, dict):
            return message.get("role"), message.get("content", "")
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
            message_data (Dict[str, Any]): Additional data (e.g., tool_call_id for tool messages).

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
        return message_class(content=content)

    @classmethod
    def get_message_class(cls, role: str) -> Union[BaseMessage, None]:
        """Get the message class for a given role."""
        role = role.lower()
        return cls._ROLE_MAP.get(role, None)

    @staticmethod
    def parse_role_content(content: str) -> Tuple[List[str], Optional[str]]:
        """
        Parse the formatted content into role-based chunks and any remaining plain text.
        - `role_chunks`: List of chunks containing roles and their messages.
        - `plain_text`: Text that does not match any role pattern.

        Returns:
            Tuple[List[str], Optional[str]]: Role-based chunks and any remaining plain text.
        """
        # Regex to split on role definitions, capturing role-based content.
        role_pattern = (
            r"(?i)^\s*#?\s*("
            + "|".join(ChatPromptHelper._ROLE_MAP.keys())
            + r")\s*:\s*\n"
        )

        # First, check for role-based patterns. If none, return the entire content as plain_text.
        if not re.search(role_pattern, content, flags=re.MULTILINE):
            return [], content.strip()

        # Split the content on the role pattern.
        chunks = re.split(role_pattern, content, flags=re.MULTILINE)

        # Extract plain text that is not part of role-based content, if any.
        plain_text = None
        if (
            chunks
            and chunks[0].strip()
            and chunks[0].lower() not in ChatPromptHelper._ROLE_MAP
        ):
            plain_text = chunks.pop(0).strip()

        # Filter out empty or whitespace-only chunks from role-based content.
        role_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

        return role_chunks, plain_text

    @staticmethod
    def to_message(role: str, content: str) -> BaseMessage:
        """
        Parse a single chunk of content into a message object.
        """
        role = role.strip().lower()
        content = content.strip()

        logger.debug(f"Parsing role: '{role}', content: '{content[:30]}...'")

        # Map role to a message class
        message_class = ChatPromptHelper.get_message_class(role)
        if not message_class:
            raise ValueError(f"Invalid message role: '{role}'")

        if not content:
            raise ValueError(f"Content missing for role: '{role}'")

        return message_class(content=content)

    @staticmethod
    def parse_as_messages(
        content: str,
    ) -> Tuple[Optional[List[BaseMessage]], Optional[str]]:
        """
        Parse the content into a list of role-based messages and any unstructured plain text.

        Returns:
            Tuple[List[BaseMessage], Optional[str]]: Parsed messages if role-based chunks are found,
            and any remaining plain text if detected.
        """
        role_chunks, plain_text = ChatPromptHelper.parse_role_content(content)

        # If there are no role-based chunks, return the plain_text directly.
        if not role_chunks:
            logger.debug("No role-based content found; returning plain text.")
            return [], plain_text

        # Parse each role-based chunk to extract messages.
        messages = []
        role = None

        for chunk in role_chunks:
            if chunk.lower() in ChatPromptHelper._ROLE_MAP.keys():
                role = chunk  # Assign role if chunk matches a defined role.
            elif (
                role
            ):  # If a role is set, treat this chunk as the content for that role.
                messages.append(ChatPromptHelper.to_message(role, chunk))
                role = None
            else:
                raise ValueError(f"Unexpected content without a role: {chunk}")

        return messages, plain_text
