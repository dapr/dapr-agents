from dapr_agents.types import (
    OAIFunctionDefinition,
    OAIToolDefinition,
    ClaudeToolDefinition,
)
from dapr_agents.types.exceptions import FunCallBuilderError
from pydantic import BaseModel, ValidationError
from typing import Dict, Any, Optional

import logging

logger = logging.getLogger(__name__)


def custom_function_schema(model: BaseModel) -> Dict:
    """
    Generates a JSON schema for the provided Pydantic model but filters out the 'title' key
    from both the main schema and from each property in the schema.

    Args:
        model (BaseModel): The Pydantic model from which to generate the JSON schema.

    Returns:
        Dict: The JSON schema of the model, excluding any 'title' keys.
    """
    schema = model.model_json_schema()
    schema.pop("title", None)

    # Remove the 'title' key from each property in the schema
    for property_details in schema.get("properties", {}).values():
        property_details.pop("title", None)

    return schema


def to_openai_function_call_definition(
    name: str,
    description: str,
    args_schema: BaseModel,
    use_deprecated: Optional[bool] = False,
) -> Dict[str, Any]:
    """
    Generates a dictionary representing either a deprecated function or a tool specification of type function
    in the OpenAI API format. It utilizes a Pydantic schema (`args_schema`) to extract parameters and types,
    which are then structured according to the OpenAI specification requirements.

    Args:
        name (str): The name of the function.
        description (str): A brief description of what the function does.
        args_schema (BaseModel): The Pydantic schema representing the function's parameters.
        use_deprecated (bool, optional): A flag to determine if the deprecated function format should be used.
                                         Defaults to False, using the tool format.

    Returns:
        Dict[str, Any]: A dictionary containing the function's specification. If 'use_deprecated' is False,
                        it includes its type as 'function' under a tool specification; otherwise, it returns
                        the function specification alone.
    """
    base_function = OAIFunctionDefinition(
        name=name,
        description=description,
        strict=True,
        parameters=custom_function_schema(args_schema),
    )

    if use_deprecated:
        # Return the function definition directly for deprecated use
        return base_function.model_dump()
    else:
        # Wrap in a tool definition for current API usage
        function_tool = OAIToolDefinition(type="function", function=base_function)
        return function_tool.model_dump()


def to_claude_function_call_definition(
    name: str, description: str, args_schema: BaseModel
) -> Dict[str, Any]:
    """
    Generates a dictionary representing a function call specification in the Claude API format. Similar to the
    OpenAI function definition, it structures the function's details such as name, description, and input parameters
    according to the Claude API specification requirements.

    Args:
        name (str): The name of the function.
        description (str): A brief description of what the function does.
        args_schema (BaseModel): The Pydantic schema representing the function's parameters.

    Returns:
        Dict[str, Any]: A dictionary containing the function's specification, including its name,
                        description, and a JSON schema of parameters formatted for Claude's API.
    """
    function_definition = ClaudeToolDefinition(
        name=name,
        description=description,
        input_schema=custom_function_schema(args_schema),
    )

    return function_definition.model_dump()


def to_gemini_function_call_definition(
    name: str, description: str, args_schema: BaseModel
) -> Dict[str, Any]:
    """
    Generates a dictionary representing a function call specification in the OpenAI API format. It utilizes
    a Pydantic schema (`args_schema`) to extract parameters and types, which are then structured according
    to the OpenAI specification requirements.

    Args:
        name (str): The name of the function.
        description (str): A brief description of what the function does.
        args_schema (BaseModel): The Pydantic schema representing the function's parameters.

    Returns:
        Dict[str, Any]: A dictionary containing the function's specification, including its name,
                        description, and a JSON schema of parameters formatted for the OpenAI API.
    """
    function_definition = OAIFunctionDefinition(
        name=name,
        description=description,
        parameters=custom_function_schema(args_schema),
    )

    return function_definition.model_dump()


def to_function_call_definition(
    name: str,
    description: str,
    args_schema: BaseModel,
    format_type: str = "openai",
    use_deprecated: bool = False,
) -> Dict:
    """
    Generates a dictionary representing a function call specification, supporting various API formats.
    The 'use_deprecated' flag is applicable only for the 'openai' format and is ignored for others.

    Args:
        name (str): The name of the function.
        description (str): A brief description of what the function does.
        args_schema (BaseModel): The Pydantic schema representing the function's parameters.
        format_type (str, optional): The API format to convert to ('openai', 'claude', or 'gemini'). Defaults to 'openai'.
        use_deprecated (bool): Flag to use the deprecated function format, only effective for 'openai'.

    Returns:
        Dict: A dictionary containing the function definition in the specified format.

    Raises:
        FunCallBuilderError: If an unsupported format type is specified.
    """
    if format_type.lower() in ("openai", "nvidia"):
        return to_openai_function_call_definition(
            name, description, args_schema, use_deprecated
        )
    elif format_type.lower() == "claude":
        if use_deprecated:
            logger.warning(
                f"'use_deprecated' flag is ignored for the '{format_type}' format."
            )
        return to_claude_function_call_definition(name, description, args_schema)
    else:
        logger.error(f"Unsupported format type: {format_type}")
        raise FunCallBuilderError(f"Unsupported format type: {format_type}")


def validate_and_format_tool(
    tool: Dict[str, Any], tool_format: str = "openai", use_deprecated: bool = False
) -> dict:
    """
    Validates and formats a tool (provided as a dictionary) based on the specified API request format.

    Args:
        tool: The tool to validate and format.
        tool_format: The API format to convert to ('openai', 'azure_openai', 'claude', 'llama').
        use_deprecated: Whether to use deprecated functions format for OpenAI. Defaults to False.

    Returns:
        dict: The formatted tool dictionary.

    Raises:
        ValueError: If the tool definition format is invalid.
        ValidationError: If the tool doesn't pass validation.
    """
    try:
        if tool_format in ["openai", "azure_openai", "nvidia"]:
            validated_tool = (
                OAIFunctionDefinition(**tool)
                if use_deprecated
                else OAIToolDefinition(**tool)
            )
        elif tool_format == "claude":
            validated_tool = ClaudeToolDefinition(**tool)
        elif tool_format == "llama":
            validated_tool = OAIFunctionDefinition(**tool)
        else:
            logger.error(f"Unsupported tool format: {tool_format}")
            raise ValueError(f"Unsupported tool format: {tool_format}")
        return validated_tool.model_dump()
    except ValidationError as e:
        logger.error(f"Validation error for {tool_format} tool definition: {e}")
        raise ValueError(f"Invalid tool definition format: {tool}")
