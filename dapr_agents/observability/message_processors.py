"""
Message processing utilities for OpenTelemetry instrumentation.

This module provides message format conversion and processing functions
for proper OpenInference compatibility and Phoenix UI display.

The processors handle conversion between various message formats (dict, object, string)
to the standardized OpenInference Message format that Phoenix UI expects.

Key Functions:
- convert_messages_to_openinference: Main message conversion pipeline
- extract_tool_schemas: Tool schema extraction from AgentTool instances
- process_llm_response: LLM response processing and attribute extraction
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .constants import (
    LLM_OUTPUT_MESSAGES,
    LLM_TOKEN_COUNT_COMPLETION,
    LLM_TOKEN_COUNT_PROMPT,
    LLM_TOKEN_COUNT_TOTAL,
    MESSAGE_CONTENT,
    MESSAGE_ROLE,
    OPENINFERENCE_AVAILABLE,
    safe_json_dumps,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Input Message Processing
# ============================================================================


def convert_messages_to_openinference(messages: Any) -> List[Dict[str, Any]]:
    """
    Convert various message formats to OpenInference Message format.

    Handles string messages, lists of dictionaries, and object-based messages,
    converting them to the standardized OpenInference Message format that
    Phoenix UI can properly interpret and display.

    Args:
        messages: Input messages in various formats (str, list of dicts/objects)

    Returns:
        List[Dict[str, Any]]: Messages in OpenInference format

    Example:
        >>> convert_messages_to_openinference("Hello")
        [{"role": "user", "content": "Hello"}]

        >>> convert_messages_to_openinference([{"role": "assistant", "content": "Hi", "tool_calls": [...]}])
        [{"role": "assistant", "content": "Hi", "tool_calls": [...]}]
    """
    logger.debug(f"Converting messages of type: {type(messages)}")

    oi_messages = []

    try:
        if isinstance(messages, str):
            # Single string message - convert to user message
            oi_messages.append({"role": "user", "content": messages})
            logger.debug("Converted string message to user message")

        elif isinstance(messages, list):
            for i, message in enumerate(messages):
                oi_message = convert_single_message_to_openinference(message, i)
                if oi_message:
                    oi_messages.append(oi_message)

    except Exception as e:
        logger.warning(f"Error converting messages: {e}")
        # Fallback: add a default user message
        oi_messages = [{"role": "user", "content": ""}]

    return oi_messages


def convert_single_message_to_openinference(
    message: Any, index: int
) -> Optional[Dict[str, Any]]:
    """
    Convert a single message to OpenInference Message format.

    Handles both dictionary and object-based messages, extracting role,
    content, tool_calls, and other relevant fields for proper OpenInference
    structure.

    Args:
        message: Single message (dict or object)
        index: Message index for logging

    Returns:
        Optional[Dict[str, Any]]: OpenInference Message format or None if invalid
    """
    oi_message = {}

    if isinstance(message, dict):
        # Handle dictionary-style messages
        role = message.get("role", "user")
        content = message.get("content")
        tool_calls = message.get("tool_calls")

        oi_message["role"] = role
        oi_message["content"] = content if content is not None else ""

        # Handle tool messages with special attributes
        if role == "tool":
            if tool_call_id := message.get("tool_call_id"):
                oi_message["tool_call_id"] = tool_call_id

        # Convert tool_calls to OpenInference format for assistant messages
        if tool_calls and role == "assistant":
            oi_tool_calls = convert_tool_calls_to_openinference(tool_calls)
            if oi_tool_calls:
                oi_message["tool_calls"] = oi_tool_calls
                logger.debug(
                    f"Converted {len(oi_tool_calls)} tool calls for message {index}"
                )

    elif hasattr(message, "role") and hasattr(message, "content"):
        # Handle object-style messages (Pydantic models, etc.)
        role = message.role
        content = message.content

        oi_message["role"] = role
        oi_message["content"] = content if content is not None else ""

        # Handle tool messages
        if role == "tool" and hasattr(message, "tool_call_id"):
            oi_message["tool_call_id"] = message.tool_call_id

        # Handle tool calls for assistant messages
        if tool_calls := getattr(message, "tool_calls", None):
            if role == "assistant":
                oi_tool_calls = convert_tool_calls_to_openinference(tool_calls)
                if oi_tool_calls:
                    oi_message["tool_calls"] = oi_tool_calls
                    logger.debug(
                        f"Converted {len(oi_tool_calls)} tool calls for message {index}"
                    )

    return oi_message if oi_message else None


def convert_tool_calls_to_openinference(tool_calls: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert tool calls to OpenInference format.

    Handles both dictionary and object-based tool calls, extracting id,
    function name, and arguments for proper OpenInference structure.

    Args:
        tool_calls: List of tool calls (dicts or objects)

    Returns:
        List[Dict[str, Any]]: Tool calls in OpenInference format
    """
    oi_tool_calls = []

    for tool_call in tool_calls:
        oi_tool_call = {}

        if isinstance(tool_call, dict):
            # Handle dictionary tool calls
            if tool_call.get("id"):
                oi_tool_call["id"] = tool_call["id"]
            if function := tool_call.get("function"):
                oi_tool_call["function"] = {
                    "name": function.get("name", ""),
                    "arguments": function.get("arguments", ""),
                }
        else:
            # Handle object-style tool calls (Pydantic models, etc.)
            if hasattr(tool_call, "model_dump"):
                # Pydantic model - convert to dict
                tool_call_dict = tool_call.model_dump()
                if tool_call_dict.get("id"):
                    oi_tool_call["id"] = tool_call_dict["id"]
                if function := tool_call_dict.get("function"):
                    oi_tool_call["function"] = {
                        "name": function.get("name", ""),
                        "arguments": function.get("arguments", ""),
                    }
            else:
                # Handle object attributes directly
                if hasattr(tool_call, "id"):
                    oi_tool_call["id"] = tool_call.id
                if hasattr(tool_call, "function"):
                    function = tool_call.function
                    oi_tool_call["function"] = {
                        "name": getattr(function, "name", ""),
                        "arguments": getattr(function, "arguments", ""),
                    }

        if oi_tool_call:
            oi_tool_calls.append(oi_tool_call)

    return oi_tool_calls


# ============================================================================
# Input Message Attributes
# ============================================================================


def get_input_message_attributes(oi_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get OpenInference input message attributes from converted messages.

    Uses OpenInference helper functions when available, with fallback
    to basic attribute structure for compatibility.

    Args:
        oi_messages: List of messages in OpenInference format

    Returns:
        Dict[str, Any]: Span attributes for input messages
    """
    if OPENINFERENCE_AVAILABLE:
        try:
            from openinference.instrumentation import get_llm_input_message_attributes

            attributes = get_llm_input_message_attributes(oi_messages)
            logger.debug(
                f"Generated {len(attributes)} input message attributes using OpenInference"
            )
            return attributes
        except Exception as e:
            logger.warning(
                f"OpenInference get_llm_input_message_attributes failed: {e}, using fallback"
            )
            return {}
    else:
        logger.debug("OpenInference not available, skipping input message attributes")
        return {}


# ============================================================================
# Tool Schema Processing
# ============================================================================


def extract_tool_schemas(tools: List[Any]) -> Dict[str, Any]:
    """
    Extract tool information using AgentTool.to_function_call() method.

    Converts AgentTool objects to proper JSON schemas for OpenTelemetry tracing,
    enabling Phoenix UI to display available tools and their schemas.

    Args:
        tools: List of AgentTool instances

    Returns:
        Dict[str, Any]: Tool schema attributes
    """
    logger.debug(f"Extracting {len(tools) if tools else 0} tool schemas")

    attributes: Dict[str, Any] = {}

    if not tools or not isinstance(tools, list):
        return attributes

    for i, tool in enumerate(tools):
        try:
            # Use AgentTool's built-in function call format
            if hasattr(tool, "to_function_call"):
                function_call = tool.to_function_call(format_type="openai")
                tool_schema = safe_json_dumps(function_call)
                attributes[f"llm.tools.{i}.tool.json_schema"] = tool_schema
                logger.debug(
                    f"Extracted schema for tool {i}: {function_call.get('name', 'unknown')}"
                )

            elif hasattr(tool, "name"):
                # Fallback: create basic schema from tool attributes
                tool_schema = create_fallback_tool_schema(tool)
                attributes[f"llm.tools.{i}.tool.json_schema"] = safe_json_dumps(
                    tool_schema
                )
                logger.debug(f"Created fallback schema for tool {i}: {tool.name}")
            else:
                logger.warning(f"Tool {i} has no extractable schema - skipping")

        except Exception as e:
            logger.warning(f"Error extracting tool {i}: {e}")
            continue

    return attributes


def create_fallback_tool_schema(tool: Any) -> Dict[str, Any]:
    """
    Create a fallback tool schema from tool attributes.

    Args:
        tool: Tool object with name/description attributes

    Returns:
        Dict[str, Any]: Basic tool schema
    """
    tool_schema = {
        "name": tool.name,
        "description": getattr(tool, "description", ""),
    }

    # Try to extract parameters schema
    if hasattr(tool, "args_schema"):
        try:
            if hasattr(tool.args_schema, "model_json_schema"):
                tool_schema["parameters"] = tool.args_schema.model_json_schema()
            elif hasattr(tool.args_schema, "schema"):
                tool_schema["parameters"] = tool.args_schema.schema()
        except Exception as e:
            logger.debug(f"Could not extract parameters for tool {tool.name}: {e}")

    return tool_schema


# ============================================================================
# Output Message Processing
# ============================================================================


def process_llm_response(result: Any) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Extract AssistantMessage and metadata from LLM response.

    Handles various LLM response formats to extract the message and usage
    metadata for proper span attribute setting.

    Args:
        result: LLM response object

    Returns:
        Tuple[Optional[Any], Optional[Any]]: (message, metadata)
    """
    message = None
    metadata = None

    # Handle Dapr Agents LLMChatResponse structure
    if hasattr(result, "results") and result.results:
        candidate = result.results[0]
        if hasattr(candidate, "message"):
            message = candidate.message
        if hasattr(result, "metadata"):
            metadata = result.metadata

    # Fallback for other LLM response structures
    elif hasattr(result, "candidates") and result.candidates:
        candidate = result.candidates[0]
        if hasattr(candidate, "message"):
            message = candidate.message
        if hasattr(result, "usage"):
            metadata = {"usage": result.usage}

    return message, metadata


def get_output_message_attributes(oi_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get OpenInference output message attributes from converted messages.

    Uses OpenInference helper functions when available, with fallback
    to basic attribute structure for compatibility.

    Args:
        oi_messages: List of messages in OpenInference format

    Returns:
        Dict[str, Any]: Span attributes for output messages
    """
    if OPENINFERENCE_AVAILABLE:
        try:
            from openinference.instrumentation import get_llm_output_message_attributes

            attributes = get_llm_output_message_attributes(oi_messages)
            logger.debug(
                f"Generated {len(attributes)} output message attributes using OpenInference"
            )
            return attributes
        except Exception as e:
            logger.warning(
                f"OpenInference get_llm_output_message_attributes failed: {e}, using fallback"
            )
            return get_fallback_output_attributes(oi_messages[0] if oi_messages else {})
    else:
        logger.debug("OpenInference not available, using fallback output attributes")
        return get_fallback_output_attributes(oi_messages[0] if oi_messages else {})


def get_fallback_output_attributes(oi_message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get basic output message attributes as fallback.

    Args:
        oi_message: Single message in OpenInference format

    Returns:
        Dict[str, Any]: Basic output message attributes
    """
    attributes = {}
    if oi_message:
        attributes[f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}"] = oi_message.get(
            "role", "assistant"
        )
        attributes[f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"] = oi_message.get(
            "content", ""
        )
        logger.debug("Set basic output message attributes (fallback)")
    return attributes


# ============================================================================
# Token Usage Processing
# ============================================================================


def extract_token_usage(metadata: Any) -> Dict[str, Any]:
    """
    Extract token usage attributes from metadata.

    Handles various metadata formats to extract token counts for
    proper span attribute setting.

    Args:
        metadata: Response metadata containing usage information

    Returns:
        Dict[str, Any]: Token usage attributes
    """
    attributes = {}

    try:
        if isinstance(metadata, dict) and "usage" in metadata:
            usage = metadata["usage"]
            attributes.update(extract_usage_from_dict(usage))
        elif hasattr(metadata, "usage"):
            usage = metadata.usage
            attributes.update(extract_usage_from_object(usage))
    except Exception as e:
        logger.debug(f"Could not extract token counts: {e}")

    return attributes


def extract_usage_from_dict(usage: Dict[str, Any]) -> Dict[str, Any]:
    """Extract usage attributes from dictionary."""
    attributes = {}
    if "prompt_tokens" in usage:
        attributes[LLM_TOKEN_COUNT_PROMPT] = usage["prompt_tokens"]
    if "completion_tokens" in usage:
        attributes[LLM_TOKEN_COUNT_COMPLETION] = usage["completion_tokens"]
    if "total_tokens" in usage:
        attributes[LLM_TOKEN_COUNT_TOTAL] = usage["total_tokens"]
    return attributes


def extract_usage_from_object(usage: Any) -> Dict[str, Any]:
    """Extract usage attributes from object."""
    attributes = {}
    if hasattr(usage, "prompt_tokens"):
        attributes[LLM_TOKEN_COUNT_PROMPT] = usage.prompt_tokens
    if hasattr(usage, "completion_tokens"):
        attributes[LLM_TOKEN_COUNT_COMPLETION] = usage.completion_tokens
    if hasattr(usage, "total_tokens"):
        attributes[LLM_TOKEN_COUNT_TOTAL] = usage.total_tokens
    return attributes


# ============================================================================
# Exported Functions
# ============================================================================

__all__ = [
    "convert_messages_to_openinference",
    "convert_single_message_to_openinference",
    "convert_tool_calls_to_openinference",
    "get_input_message_attributes",
    "extract_tool_schemas",
    "create_fallback_tool_schema",
    "process_llm_response",
    "get_output_message_attributes",
    "get_fallback_output_attributes",
    "extract_token_usage",
    "extract_usage_from_dict",
    "extract_usage_from_object",
]
