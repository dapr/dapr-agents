from pydantic import BaseModel, field_validator, ValidationInfo
from typing import Dict, Literal, Optional, List


class OAIFunctionDefinition(BaseModel):
    """
    Represents a callable function in the OpenAI API format.

    Attributes:
        name (str): The name of the function.
        description (str): A detailed description of what the function does.
        parameters (Dict): A dictionary describing the parameters that the function accepts.
    """

    name: str
    description: str
    parameters: Dict


class OAIToolDefinition(BaseModel):
    """
    Represents a tool (callable function) in the OpenAI API format. This can be a function, code interpreter, or file search tool.

    Attributes:
        type (Literal["function", "code_interpreter", "file_search"]): The type of the tool.
        function (Optional[OAIBaseFunctionDefinition]): The function definition, required if type is 'function'.
    """

    type: Literal["function", "code_interpreter", "file_search"]
    function: Optional[OAIFunctionDefinition] = None

    @field_validator("function")
    def check_function_requirements(cls, v, info: ValidationInfo):
        if info.data.get("type") == "function" and not v:
            raise ValueError(
                "Function definition must be provided for function type tools."
            )
        return v


class ClaudeToolDefinition(BaseModel):
    """
    Represents a tool (callable function) in the Anthropic's Claude API format, suitable for integration with Claude's API services.

    Attributes:
        name (str): The name of the function.
        description (str): A description of the function's purpose and usage.
        input_schema (Dict): A dictionary defining the input schema for the function.
    """

    name: str
    description: str
    input_schema: Dict


class GeminiFunctionDefinition(BaseModel):
    """
    Represents a callable function in the Google's Gemini API format.

    Attributes:
        name (str): The name of the function to call. Must start with a letter or an underscore. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
        description (str): The description and purpose of the function. The model uses this to decide how and whether to call the function. For the best results, we recommend that you include a description.
        parameters (Dict): Describes the parameters of the function in the OpenAPI JSON Schema Object format: OpenAPI 3.0 specification.
    """

    name: str
    description: str
    parameters: Dict


class GeminiToolDefinition(BaseModel):
    """
    Represents a tool (callable function) in the Google's Gemini API format, suitable for integration with Gemini's API services.

    Attributes:
        function_declarations (List): A structured representation of a function declaration as defined by the OpenAPI 3.0 specification that represents a function the model may generate JSON inputs for.
    """

    function_declarations: List[GeminiFunctionDefinition]
