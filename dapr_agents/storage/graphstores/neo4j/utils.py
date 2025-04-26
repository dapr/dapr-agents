from typing import Any
import datetime
import logging

LIST_LIMIT = 100  # Maximum number of elements in a list to be processed

logger = logging.getLogger(__name__)


def value_sanitize(data: Any) -> Any:
    """
    Sanitizes the input data (dictionary or list) for use in a language model or database context.
    This function filters out large lists, simplifies nested structures, and ensures Neo4j-specific
    data types are handled efficiently.

    Args:
        data (Any): The data to sanitize, which can be a dictionary, list, or other types.

    Returns:
        Any: The sanitized data. Returns `None` for lists exceeding the size limit or unsupported types.
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
