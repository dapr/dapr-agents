from typing import Any, Dict, Optional
from typing_extensions import Literal
from pydantic import BaseModel, Field


class OAIJSONSchema(BaseModel):
    """
    Defines the content for a JSON schema used in OpenAI's response_format.
    """

    name: str = Field(
        ...,
        description="The name of the response format.",
        json_schema_extra={
            "maxLength": 64,
            "pattern": "^[a-zA-Z0-9_-]+$",
            "examples": ["example_schema_name"],
        },
    )
    description: Optional[str] = Field(
        None,
        description="Explains the purpose of this response format so the model knows how to respond.",
    )
    schema_: Optional[Dict[str, Any]] = Field(
        None,
        serialization_alias="schema",
        description="The underlying JSON Schema object describing the response format structure.",
    )
    strict: bool = Field(
        True,
        description=(
            "Whether to enforce strict schema adherence when generating the output. "
            "If set to True, only a subset of JSON Schema features is supported."
        ),
    )


class OAIResponseFormatSchema(BaseModel):
    """
    Represents the top-level structure for OpenAI's 'json_schema' response format.
    """

    type: Literal["json_schema"] = Field(
        "json_schema",
        description="Specifies that this response format is a JSON schema definition.",
    )
    json_schema: OAIJSONSchema = Field(
        ...,
        description="Contains metadata and the actual JSON schema for the structured output.",
    )
