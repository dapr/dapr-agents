from typing import Any, Dict, Optional, Type, List
import logging

from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)

# Mapping from JSON Schema types to Python types
TYPE_MAPPING = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
    "null": type(None),
}


def create_pydantic_model_from_schema(
    schema: Dict[str, Any], model_name: str
) -> Type[BaseModel]:
    """
    Create a Pydantic model from a JSON schema definition.

    This function converts a JSON Schema object (commonly used in MCP tool definitions)
    to a Pydantic model that can be used for validation in the Dapr agent framework.

    Args:
        schema: JSON Schema dictionary containing type information
        model_name: Name for the generated model class

    Returns:
        A dynamically created Pydantic model class

    Raises:
        ValueError: If the schema is invalid or cannot be converted
    """
    logger.debug(f"Creating Pydantic model '{model_name}' from schema")

    try:
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        fields = {}

        # Process each property in the schema
        for field_name, field_props in properties.items():
            # Get field type information
            json_type = field_props.get("type", "string")

            # Handle complex type definitions (arrays, unions, etc.)
            if isinstance(json_type, list):
                # Process union types (e.g., ["string", "null"])
                has_null = "null" in json_type
                non_null_types = [t for t in json_type if t != "null"]

                if not non_null_types:
                    # Only null type specified
                    field_type = Optional[str]
                else:
                    # Use the first non-null type
                    # TODO: Proper union type handling would be better but more complex
                    primary_type = non_null_types[0]
                    field_type = TYPE_MAPPING.get(primary_type, str)

                    # Make optional if null is included
                    if has_null:
                        field_type = Optional[field_type]
            else:
                # Simple type
                field_type = TYPE_MAPPING.get(json_type, str)

            # Handle arrays with item type information
            if json_type == "array" or (
                isinstance(json_type, list) and "array" in json_type
            ):
                # Get the items type if specified
                if "items" in field_props:
                    items_type = field_props["items"].get("type", "string")
                    if isinstance(items_type, str):
                        item_py_type = TYPE_MAPPING.get(items_type, str)
                        field_type = List[item_py_type]

            # Set default value based on required status
            if field_name in required:
                default = ...  # Required field
            else:
                default = None
                # Make optional if not already
                if not isinstance(field_type, type(Optional)):
                    field_type = Optional[field_type]

            # Add field with description and default value
            field_description = field_props.get("description", "")
            fields[field_name] = (
                field_type,
                Field(default, description=field_description),
            )

        # Create and return the model class
        return create_model(model_name, **fields)

    except Exception as e:
        logger.error(f"Failed to create model from schema: {str(e)}")
        raise ValueError(f"Invalid schema: {str(e)}")
