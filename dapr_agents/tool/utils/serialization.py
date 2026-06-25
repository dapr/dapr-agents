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

"""Utility functions for serializing tool execution results."""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _extract_text_from_content_block(item: Any) -> str:
    """Extract text from a single MCP content block.

    Dapr workflow output uses the flat MCP-spec content shape
    (``{"type": "text", "text": "..."}``).
    """
    if isinstance(item, dict):
        text = item.get("text")
        return text if isinstance(text, str) else ""
    return ""


def _unwrap_mcp_call_tool_result(result: Any) -> Any:
    """Unwrap an MCP CallToolResult into a plain text string.

    The Dapr ``CallTool`` workflow returns a JSON-encoded MCP
    `CallToolResult <https://modelcontextprotocol.io/specification/2024-11-05/server/tools#tool-result>`_::

        {"isError": false, "content": [{"type": "text", "text": "..."}]}

    Input may be a pre-parsed dict, a JSON string, or even double-JSON-encoded
    (string within a string). Returns *result* unchanged when it doesn't
    match the CallToolResult shape.

    The presence of ``isError`` is the distinguishing marker for an MCP
    envelope — a plain ``{"content": [...]}`` dict is too ambiguous, since
    legitimate CallTool workflow results can have that exact shape.
    """
    # Try to parse JSON strings (possibly double-encoded).
    parsed = result
    for _ in range(2):  # at most 2 levels of JSON decoding
        if isinstance(parsed, str):
            try:
                decoded = json.loads(parsed)
                if isinstance(decoded, dict):
                    parsed = decoded
                else:
                    break
            except (json.JSONDecodeError, TypeError):
                break
        else:
            break

    if not isinstance(parsed, dict) or "isError" not in parsed:
        return result

    is_error = parsed["isError"]
    content = parsed.get("content")
    if not isinstance(content, list):
        return "MCP tool call failed: no content blocks" if is_error else result

    parts = [
        text for item in content if (text := _extract_text_from_content_block(item))
    ]

    if is_error:
        return (
            "Error: " + " ".join(parts)
            if parts
            else "MCP tool call failed: no extractable text in content blocks"
        )
    if parts:
        return "\n".join(parts)
    # Content blocks exist but no text extracted — return raw JSON
    return json.dumps(content)


def serialize_tool_result(result: Any) -> str:
    """
    Serialize a tool execution result to a JSON string.

    Handles various data types including:
    - MCP CallToolResult envelopes (unwrapped to text)
    - Strings (returned as-is)
    - Pydantic models (via model_dump)
    - Lists of Pydantic models
    - Objects with __dict__
    - JSON-serializable primitives

    Args:
        result: The tool execution result to serialize.

    Returns:
        str: JSON-serialized string representation of the result.

    Examples:
        >>> from pydantic import BaseModel
        >>> class Flight(BaseModel):
        ...     airline: str
        ...     price: float
        >>>
        >>> flights = [Flight(airline="SkyHigh", price=450.0)]
        >>> serialize_tool_result(flights)
        '[{"airline": "SkyHigh", "price": 450.0}]'
    """
    # Unwrap MCP CallToolResult envelope if present.
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "serialize_tool_result input: type=%s, repr=%.500s",
            type(result).__name__,
            repr(result),
        )
    result = _unwrap_mcp_call_tool_result(result)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "serialize_tool_result after unwrap: type=%s, repr=%.500s",
            type(result).__name__,
            repr(result),
        )

    # String results are already serialized
    if isinstance(result, str):
        return result

    try:
        # Handle lists of objects (most common case for collections)
        if isinstance(result, list):
            serialized_list = []
            for item in result:
                if hasattr(item, "model_dump"):
                    # Pydantic v2 models
                    serialized_list.append(item.model_dump())
                elif hasattr(item, "dict") and callable(item.dict):
                    # Pydantic v1 models (fallback)
                    serialized_list.append(item.dict())
                elif hasattr(item, "__dict__"):
                    # Regular objects with __dict__
                    serialized_list.append(item.__dict__)
                else:
                    # Primitive types or already serializable
                    serialized_list.append(item)
            return json.dumps(serialized_list)

        # Handle single Pydantic models
        if hasattr(result, "model_dump"):
            return json.dumps(result.model_dump())

        # Fallback for Pydantic v1
        if hasattr(result, "dict") and callable(result.dict):
            return json.dumps(result.dict())

        # Handle objects with __dict__
        if hasattr(result, "__dict__"):
            return json.dumps(result.__dict__)

        # Try direct JSON serialization for primitives, dicts, lists, etc.
        return json.dumps(result)

    except (TypeError, ValueError):
        # Final fallback: convert to string
        # This handles non-JSON-serializable objects gracefully
        return str(result)
