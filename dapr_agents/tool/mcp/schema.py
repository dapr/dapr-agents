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
            # --- Handle anyOf/oneOf for nullable/union fields ---
            if "anyOf" in field_props or "oneOf" in field_props:
                variants = field_props.get("anyOf") or field_props.get("oneOf")
                types = [v.get("type", "string") for v in variants]
                has_null = "null" in types
                non_null_variants = [v for v in variants if v.get("type") != "null"]
                if non_null_variants:
                    primary_type = non_null_variants[0].get("type", "string")
                    field_type = TYPE_MAPPING.get(primary_type, str)
                    # Handle array/object with items/properties
                    if primary_type == "array" and "items" in non_null_variants[0]:
                        item_type = non_null_variants[0]["items"].get("type", "string")
                        field_type = List[TYPE_MAPPING.get(item_type, str)]
                    elif primary_type == "object":
                        field_type = dict
                else:
                    field_type = str
                if has_null:
                    field_type = Optional[field_type]
            else:
                # --- Fallback to "type" ---
                json_type = field_props.get("type", "string")
                field_type = TYPE_MAPPING.get(json_type, str)
                if json_type == "array" and "items" in field_props:
                    item_type = field_props["items"].get("type", "string")
                    field_type = List[TYPE_MAPPING.get(item_type, str)]

            # Set default value based on required status
            if field_name in required:
                default = ...
            else:
                default = None
                # Make optional if not already
                if not (
                    hasattr(field_type, "__origin__")
                    and field_type.__origin__ is Optional
                ):
                    field_type = Optional[field_type]

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
