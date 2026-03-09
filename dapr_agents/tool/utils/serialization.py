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
from typing import Any


def serialize_tool_result(result: Any) -> str:
    """
    Serialize a tool execution result to a JSON string.

    Handles various data types including:
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
