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

import logging
import re
from typing import Any, Dict, Optional

from pydantic import BaseModel, ValidationError

from dapr_agents.types import (
    ClaudeToolDefinition,
    OAIFunctionDefinition,
    OAIToolDefinition,
)
from dapr_agents.types.exceptions import FunCallBuilderError

logger = logging.getLogger(__name__)

# OpenAI and Anthropic both require tool names to match ^[a-zA-Z0-9_-]{1,64}$.
# Only letters, digits, underscores, and hyphens are allowed.
# Everything else (spaces, dots, slashes, special symbols, non-ASCII) is stripped.
_VALID_TOOL_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_MAX_TOOL_NAME_LENGTH = 64


def sanitize_openai_tool_name(name: str) -> str:
    """
    Sanitize a name to comply with OpenAI/Anthropic tool-name requirements.

    Provider APIs require tool names to match ``^[a-zA-Z0-9_-]{1,64}$``.
    Any character outside that set is stripped. Hyphens, underscores, and
    the original casing are preserved so that a valid name such as
    ``get-xyz-count`` reaches the LLM unchanged — matching what developers
    write in YAML/config and what ``toolbox_core.ToolboxTool`` exposes as
    the tool's name.

    Args:
        name: The original name (e.g., ``get-items``, ``My Agent``,
            ``agent<name>``).

    Returns:
        A sanitized name with only invalid characters removed. Returns
        ``"unnamed_tool"`` if the input is empty or becomes empty after
        sanitization.

    Raises:
        ValueError: If the sanitized name exceeds 64 characters.

    Examples:
        ``get-xyz-count`` -> ``get-xyz-count``
        ``get_user`` -> ``get_user``
        ``My Agent`` -> ``MyAgent``
        ``agent<name>`` -> ``agentname``
        ``café_finder`` -> ``caf_finder``
    """
    if not name:
        return "unnamed_tool"

    sanitized = _VALID_TOOL_NAME_CHARS.sub("", name)

    if not sanitized:
        return "unnamed_tool"

    if len(sanitized) > _MAX_TOOL_NAME_LENGTH:
        raise ValueError(
            f"Tool/agent name '{name}' is {len(sanitized)} characters after "
            f"sanitization (max {_MAX_TOOL_NAME_LENGTH}). Shorten the name "
            f"to fit within the limit."
        )

    if sanitized != name:
        logger.warning(
            "Tool/agent name '%s' contained characters rejected by the "
            "OpenAI/Anthropic tool-name spec (^[a-zA-Z0-9_-]{1,64}$); "
            "sanitized to '%s'. Update any instruction strings that "
            "reference the original name.",
            name,
            sanitized,
        )

    return sanitized


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
        name (str): The name of the function. Will be sanitized to comply with OpenAI's requirements
                   (no spaces, <, |, \\, /, or > characters).
        description (str): A brief description of what the function does.
        args_schema (BaseModel): The Pydantic schema representing the function's parameters.
        use_deprecated (bool, optional): A flag to determine if the deprecated function format should be used.
                                         Defaults to False, using the tool format.

    Returns:
        Dict[str, Any]: A dictionary containing the function's specification. If 'use_deprecated' is False,
                        it includes its type as 'function' under a tool specification; otherwise, it returns
                        the function specification alone.
    """
    # Sanitize tool name to comply with OpenAI's requirements
    sanitized_name = sanitize_openai_tool_name(name)
    if sanitized_name != name:
        logger.debug(
            "Sanitized tool name '%s' to '%s' to comply with OpenAI requirements",
            name,
            sanitized_name,
        )

    base_function = OAIFunctionDefinition(
        name=sanitized_name,
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
) -> Dict[str, Any]:
    """
    Generates a dictionary representing a function call specification, supporting various API formats.

    - For format_type in ("openai", "nvidia", "huggingface"), produces an OpenAI-style
      tool definition (type="function", function={…}).
    - For "claude", produces a Claude-style {name, description, input_schema}.
    - (Gemini omitted here—call to_gemini_function_call_definition if you need it.)

    The 'use_deprecated' flag is only applicable for OpenAI-style definitions.

    Args:
        name (str): The name of the function.
        description (str): A brief description of what the function does.
        args_schema (BaseModel): The Pydantic model describing the function's parameters.
        format_type (str, optional): Which API flavor to target:
            - "openai", "nvidia", or "huggingface" all share the same OpenAI-style schema.
            - "claude" uses Anthropic's format.
          Defaults to "openai".
        use_deprecated (bool): If True and format_type is OpenAI,
            returns the old function-only schema rather than a tool wrapper.

    Returns:
        Dict[str, Any]: The serialized function/tool definition.

    Raises:
        FunCallBuilderError: If an unsupported format_type is provided.
    """
    fmt = format_type.lower()

    # OpenAI‑style wrapper schema:
    if fmt in ("openai", "nvidia", "huggingface", "dapr"):
        return to_openai_function_call_definition(
            name, description, args_schema, use_deprecated
        )

    # Anthropic Claude needs its own input_schema property
    if fmt == "claude":
        if use_deprecated:
            logger.warning(
                f"'use_deprecated' flag is ignored for the '{format_type}' format."
            )
        return to_claude_function_call_definition(name, description, args_schema)

    # Unsupported provider
    logger.error(f"Unsupported format type: {format_type}")
    raise FunCallBuilderError(f"Unsupported format type: {format_type}")


def validate_and_format_tool(
    tool: Dict[str, Any], tool_format: str = "openai", use_deprecated: bool = False
) -> Dict[str, Any]:
    """
    Validates and formats a tool definition dict for the specified API style.

    - For tool_format in ("openai", "azure_openai", "nvidia", "huggingface"),
      uses OAIToolDefinition (or OAIFunctionDefinition if use_deprecated=True).
    - For "claude", uses ClaudeToolDefinition.
    - For "llama", treats as an OAIFunctionDefinition.

    Args:
        tool (Dict[str, Any]): The raw tool definition.
        tool_format (str): Which API schema to validate against:
            "openai", "azure_openai", "nvidia", "huggingface", "claude", "llama".
        use_deprecated (bool): If True and using OpenAI-style, expects an OAIFunctionDefinition.

    Returns:
        Dict[str, Any]: The validated, serialized tool definition.

    Raises:
        ValueError: If the format is unsupported or validation fails.
    """
    fmt = tool_format.lower()

    try:
        if fmt in ("openai", "azure_openai", "nvidia", "huggingface"):
            validated = (
                OAIFunctionDefinition(**tool)
                if use_deprecated
                else OAIToolDefinition(**tool)
            )
        elif fmt == "claude":
            validated = ClaudeToolDefinition(**tool)
        elif fmt == "llama":
            validated = OAIFunctionDefinition(**tool)
        else:
            logger.error(f"Unsupported tool format: {tool_format}")
            raise ValueError(f"Unsupported tool format: {tool_format}")

        return validated.model_dump()

    except ValidationError as e:
        logger.error(f"Validation error for {tool_format} tool definition: {e}")
        raise ValueError(f"Invalid tool definition format: {tool}")
