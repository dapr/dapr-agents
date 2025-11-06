"""
Utility functions for Dapr Agents OpenTelemetry instrumentation.

This module provides essential helper functions for the observability system,
including argument binding, method introspection, data serialization, and
span attribute manipulation used throughout the instrumentation wrappers.

The utilities follow the SmolAgents pattern for compatibility while adding
robust error handling, type annotations, and improved serialization for
better tracing data quality in Phoenix UI.

Key Components:
- Argument Processing: bind_arguments, strip_method_args, get_input_value
- Span Attributes: flatten_attributes for nested dictionary handling
- Message Processing: extract_content_from_result, serialize_tools_for_tracing
- SmolAgents Compatibility: Maintains API compatibility while enhancing functionality

Technical Features:
- Function signature inspection for automatic parameter binding
- Recursive dictionary flattening with dot-notation keys
- AgentTool schema extraction using to_function_call() method
- Comprehensive error handling with graceful fallbacks
- Type-safe operations with proper type annotations
"""

import logging
from inspect import signature
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

from .constants import (
    AttributeValue,
    safe_json_dumps,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Argument Binding and Processing
# ============================================================================


def bind_arguments(method: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Bind method arguments using function signature inspection.

    This utility extracts method parameters and their values using Python's
    signature inspection, applying defaults where necessary. Follows the
    SmolAgents pattern for consistent argument processing across all
    instrumentation wrappers, ensuring reliable parameter capture.

    Args:
        method: Method or function to inspect (callable with signature)
        *args: Positional arguments passed to the method
        **kwargs: Keyword arguments passed to the method

    Returns:
        Dict[str, Any]: Bound arguments with parameter names as keys

    Error Handling:
        - Signature inspection failures: Falls back to raw args/kwargs dict
        - Missing parameters: Apply defaults from function signature
        - Type inspection errors: Graceful degradation with debug logging

    Example:
        >>> def example(a, b=10, c=None): pass
        >>> bind_arguments(example, 1, c=20)
        {'a': 1, 'b': 10, 'c': 20}

        >>> bind_arguments(lambda x: x, "hello")  # Lambda functions
        {'x': 'hello'}
    """
    try:
        method_signature = signature(method)
        bound_args = method_signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return bound_args.arguments
    except Exception as e:
        logger.debug(f"Could not bind arguments for {method}: {e}")
        return {"args": args, "kwargs": kwargs}


def strip_method_args(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Remove self/cls arguments from method parameters.

    Filters out 'self' and 'cls' parameters from bound arguments to avoid
    including instance/class references in span attributes, following the
    SmolAgents pattern for cleaner tracing data.

    Args:
        arguments: Dictionary of bound method arguments

    Returns:
        Dict[str, Any]: Filtered arguments without self/cls

    Example:
        >>> strip_method_args({'self': obj, 'param': 'value', 'cls': MyClass})
        {'param': 'value'}
    """
    return {
        key: value for key, value in arguments.items() if key not in ("self", "cls")
    }


def get_input_value(method: Any, *args: Any, **kwargs: Any) -> str:
    """
    Extract and serialize input value for span attributes.

    Processes method arguments into a JSON string suitable for OpenTelemetry
    span attributes, using proper serialization and fallback handling.
    Combines argument binding with serialization for complete input capture.

    Args:
        method: Method being instrumented (used for signature inspection)
        *args: Method positional arguments to serialize
        **kwargs: Method keyword arguments to serialize

    Returns:
        str: JSON-serialized input arguments without self/cls parameters

    Processing Flow:
        1. Bind arguments using signature inspection
        2. Strip self/cls parameters for cleaner output
        3. JSON serialize with safe_json_dumps for reliability
        4. Fallback to string representation on serialization failure

    Example:
        >>> def my_method(self, query, limit=10): pass
        >>> get_input_value(my_method, obj, "search", limit=5)
        '{"query": "search", "limit": 5}'  # self parameter stripped
    """
    try:
        arguments = bind_arguments(method, *args, **kwargs)
        arguments = strip_method_args(arguments)
        return safe_json_dumps(arguments)
    except Exception as e:
        logger.debug(f"Could not serialize input for {method}: {e}")
        # Fallback to simple string representation
        return str(args[0] if args else kwargs)


# ============================================================================
# Span Attribute Utilities
# ============================================================================


def flatten_attributes(
    mapping: Optional[Mapping[str, Any]],
) -> Iterator[Tuple[str, AttributeValue]]:
    """
    Flatten nested dictionaries for OpenTelemetry span attributes.

    Recursively flattens nested dictionaries and lists into dot-notation
    keys suitable for OpenTelemetry span attributes, following the SmolAgents
    pattern for consistent attribute structure across the instrumentation system.

    Args:
        mapping: Dictionary to flatten (can be None for safety)

    Yields:
        Tuple[str, AttributeValue]: Flattened key-value pairs with dot-notation keys

    Flattening Rules:
        - Nested dicts: {'user': {'name': 'John'}} → ('user.name', 'John')
        - Lists of dicts: [{'id': 1}, {'id': 2}] → ('0.id', 1), ('1.id', 2)
        - None values: Automatically skipped to avoid empty attributes
        - Non-dict lists: Preserved as indexed values

    Example:
        >>> data = {'user': {'name': 'John', 'age': 30}, 'tags': [{'type': 'admin'}]}
        >>> list(flatten_attributes(data))
        [('user.name', 'John'), ('user.age', 30), ('tags.0.type', 'admin')]
    """
    if not mapping:
        return

    for key, value in mapping.items():
        if value is None:
            continue

        if isinstance(value, Mapping):
            # Recursively flatten nested dictionaries
            for sub_key, sub_value in flatten_attributes(value):
                yield f"{key}.{sub_key}", sub_value

        elif isinstance(value, list) and any(
            isinstance(item, Mapping) for item in value
        ):
            # Handle lists of dictionaries with indexed keys
            for index, sub_mapping in enumerate(value):
                if isinstance(sub_mapping, Mapping):
                    for sub_key, sub_value in flatten_attributes(sub_mapping):
                        yield f"{key}.{index}.{sub_key}", sub_value
                else:
                    yield f"{key}.{index}", sub_value
        else:
            # Direct value assignment
            yield key, value


# ============================================================================
# Message Processing Utilities
# ============================================================================


def extract_content_from_result(result: Any) -> str:
    """
    Extract content string from various result object types.

    Handles different response formats including Pydantic models, dictionaries,
    and objects with content attributes to extract the main content for span
    output attributes. Provides fallback serialization for complex objects.

    Args:
        result: Result object from agent/LLM execution (Pydantic model, dict, or object)

    Returns:
        str: Extracted content string or JSON-serialized representation

    Processing Priority:
        1. Direct content attribute access
        2. Pydantic model_dump() with content extraction
        3. Safe JSON serialization of model data
        4. String representation fallback

    Example:
        >>> class Response:
        ...     def __init__(self, content): self.content = content
        >>> extract_content_from_result(Response("Hello world"))
        "Hello world"
    """
    try:
        # Handle objects with content attribute
        if hasattr(result, "content"):
            return str(result.content)

        # Handle Pydantic models with model_dump
        if hasattr(result, "model_dump") and callable(result.model_dump):
            output_dict = result.model_dump()
            if isinstance(output_dict, dict) and "content" in output_dict:
                return str(output_dict["content"])
            else:
                return safe_json_dumps(output_dict)

        # Fallback to string representation
        return str(result)

    except Exception as e:
        logger.debug(f"Could not extract content from result: {e}")
        return str(result)


def serialize_tools_for_tracing(tools: Any) -> Any:
    """
    Serialize tools for tracing with proper schema extraction.

    Converts AgentTool objects to JSON schemas using their built-in
    to_function_call method for proper OpenInference compatibility,
    with fallback to string representation for unsupported tool types.

    Args:
        tools: List of AgentTool instances or other tool objects

    Returns:
        Any: Serialized tools suitable for span attributes

    Example:
        >>> tools = [MyAgentTool(name="search", description="Search tool")]
        >>> serialize_tools_for_tracing(tools)
        [{"name": "search", "description": "Search tool", "parameters": {...}}]
    """
    if not tools:
        return tools

    if not isinstance(tools, list):
        return str(tools)

    serialized_tools = []
    for i, tool in enumerate(tools):
        try:
            if hasattr(tool, "to_function_call"):
                # Use AgentTool's built-in function call format
                tool_schema = tool.to_function_call(format_type="openai")
                serialized_tools.append(tool_schema)
                logger.debug(
                    f"Extracted schema for tool {i}: {tool_schema.get('name', 'unknown')}"
                )
            else:
                # Fallback to string representation
                serialized_tools.append(str(tool))
                logger.debug(f"Used string representation for tool {i}")
        except Exception as e:
            logger.debug(f"Error serializing tool {i}: {e}")
            serialized_tools.append(str(tool))

    return serialized_tools


# ============================================================================
# Exported Functions
# ============================================================================

__all__ = [
    "bind_arguments",
    "strip_method_args",
    "get_input_value",
    "flatten_attributes",
    "extract_content_from_result",
    "serialize_tools_for_tracing",
]
