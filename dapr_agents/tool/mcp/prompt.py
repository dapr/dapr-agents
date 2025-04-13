from typing import Optional, Any, List, Dict
import logging

from mcp import ClientSession
from mcp.types import PromptMessage

from dapr_agents.types import UserMessage, AssistantMessage, BaseMessage

logger = logging.getLogger(__name__)


def convert_prompt_message(message: PromptMessage) -> BaseMessage:
    """
    Convert an MCP PromptMessage to a compatible internal BaseMessage.

    Args:
        message: The MCP PromptMessage instance

    Returns:
        A compatible BaseMessage subclass (UserMessage or AssistantMessage)

    Raises:
        ValueError: If the message contains unsupported content type or role
    """
    # Verify text content type is supported
    if message.content.type != "text":
        error_msg = f"Unsupported content type: {message.content.type}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Convert based on role
    if message.role == "user":
        return UserMessage(content=message.content.text)
    elif message.role == "assistant":
        return AssistantMessage(content=message.content.text)
    else:
        # Fall back to generic message with role preserved
        logger.warning(f"Converting message with non-standard role: {message.role}")
        return BaseMessage(content=message.content.text, role=message.role)


async def load_prompt(
    session: ClientSession, prompt_name: str, arguments: Optional[Dict[str, Any]] = None
) -> List[BaseMessage]:
    """
    Fetch and convert a prompt from the MCP server to internal message format.

    Args:
        session: An initialized MCP client session
        prompt_name: The registered prompt name
        arguments: Optional dictionary of arguments to format the prompt

    Returns:
        A list of internal BaseMessage-compatible messages

    Raises:
        Exception: If prompt retrieval fails
    """
    logger.info(f"Loading prompt '{prompt_name}' from MCP server")

    try:
        # Get prompt from server
        response = await session.get_prompt(prompt_name, arguments or {})

        # Convert all messages
        converted_messages = [convert_prompt_message(m) for m in response.messages]
        logger.info(
            f"Loaded prompt '{prompt_name}' with {len(converted_messages)} messages"
        )

        return converted_messages
    except Exception as e:
        logger.error(f"Failed to load prompt '{prompt_name}': {str(e)}")
        raise
