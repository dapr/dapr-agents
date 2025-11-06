from dapr_agents.types import BaseMessage
from typing import List
from pydantic import ValidationError


def messages_to_string(messages: List[BaseMessage]) -> str:
    """
    Converts messages into a single string with roles and content.

    Args:
        messages (List[BaseMessage]): List of message objects to convert.

    Returns:
        str: A formatted string representing the chat history.

    Raises:
        ValueError: If a message has an unknown role or is missing required fields.
    """
    valid_roles = {"user", "assistant", "system", "tool"}

    def format_message(message):
        if isinstance(message, dict):
            message = BaseMessage(**message)
        elif not isinstance(message, BaseMessage):
            raise ValueError(f"Invalid message type: {type(message)}")

        role = message.role
        content = message.content

        if role not in valid_roles:
            raise ValueError(f"Unknown role in chat history: {role}")

        return f"{role.capitalize()}: {content}"

    try:
        formatted_history = [format_message(message) for message in messages]
    except (ValidationError, ValueError) as e:
        raise ValueError(f"Invalid message in chat history. Error: {e}")

    return "\n".join(formatted_history)
