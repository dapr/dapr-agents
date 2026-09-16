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

from typing import Any
import datetime
import logging
import re

LIST_LIMIT = 100  # Maximum number of elements in a list to be processed

logger = logging.getLogger(__name__)

# Neo4j does not support parameterizing labels, relationship types, or
# property names, so callers must interpolate them into the Cypher query
# text directly. This matches a safe, unquoted Cypher identifier.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_cypher_identifier(name: str, kind: str) -> str:
    """Validate a label/relationship-type/property name before it is
    interpolated into a Cypher query string.

    Without this check, a value containing a backtick or Cypher syntax
    (plausible when labels/types come from LLM-extracted entities rather
    than a fixed schema) could break out of the intended query structure.

    Args:
        name: The identifier to validate.
        kind: Human-readable identifier kind, used in the error message
            (e.g. "label", "relationship type", "property").

    Returns:
        The validated name, unchanged.

    Raises:
        ValueError: If `name` is not a safe Cypher identifier.
    """
    if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Invalid Neo4j {kind} {name!r}: must match {_IDENTIFIER_RE.pattern}."
        )
    return name


def value_sanitize(data: Any) -> Any:
    """
    Sanitizes the input data (dictionary or list) for use in a language model or database context.
    This function filters out large lists, simplifies nested structures, and ensures Neo4j-specific
    data types are handled efficiently.

    Args:
        data (Any): The data to sanitize, which can be a dictionary, list, or other types.

    Returns:
        Any: The sanitized data. Lists exceeding the size limit are truncated to their first `LIST_LIMIT` elements, and `None` is returned for unsupported types.
    """
    if isinstance(data, dict):
        # Sanitize each key-value pair in the dictionary.
        sanitized_dict = {}
        for key, value in data.items():
            # Preserve essential metadata keys starting with "_" (e.g., Neo4j system keys).
            if key.startswith("_"):
                sanitized_dict[key] = value
                continue

            # Recursively sanitize the value.
            sanitized_value = value_sanitize(value)
            if sanitized_value is not None:
                sanitized_dict[key] = sanitized_value

        return sanitized_dict

    elif isinstance(data, list):
        # Truncate or sample large lists to avoid exceeding size limits.
        if len(data) > LIST_LIMIT:
            return data[
                :LIST_LIMIT
            ]  # Return the first `LIST_LIMIT` elements instead of discarding the list.

        # Recursively sanitize each element in the list.
        sanitized_list = [
            sanitized_item
            for item in data
            if (sanitized_item := value_sanitize(item)) is not None
        ]
        return sanitized_list

    elif isinstance(data, tuple):
        # Sanitize tuples (e.g., Neo4j relationships)
        return tuple(value_sanitize(item) for item in data)

    elif isinstance(data, datetime.datetime):
        # Convert datetime objects to ISO 8601 string for consistency.
        return data.isoformat()

    elif isinstance(data, (int, float, bool, str)):
        # Primitive types are returned as-is.
        return data

    else:
        logger.warning(
            f"Unsupported data type encountered: {type(data)}. Value: {repr(data)}"
        )
        return None  # Exclude the data entirely.


def get_current_time():
    """Get current time in UTC for creation and modification of nodes and relationships"""
    return (
        datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    )
