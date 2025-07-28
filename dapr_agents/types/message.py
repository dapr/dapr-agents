from pydantic import (
    BaseModel,
    field_validator,
    ValidationError,
    model_validator,
    ConfigDict,
)
from typing import List, Optional, Dict, Any
import json


class BaseMessage(BaseModel):
    """
    Base class for creating and processing message objects. This class provides common attributes that are shared across different types of messages.

    Attributes:
        content (Optional[str]): The main text content of the message. If provided, it initializes the message with this content.
        role (str): The role associated with the message (e.g., 'user', 'system', 'assistant'). This needs to be set by derived classes.
        name (Optional[str]): An optional name identifier for the message.

    Args:
        text (Optional[str]): An alternate way to provide text content during initialization.
        **data: Additional keyword arguments that are passed directly to the Pydantic model's constructor.
    """

    content: Optional[str]
    role: str
    name: Optional[str] = None

    def __init__(self, text: Optional[str] = None, **data):
        """
        Initializes a new BaseMessage instance. If 'text' is provided, it initializes the 'content' attribute with this value.

        Args:
            text (Optional[str]): Text content for the 'content' attribute.
            **data: Additional fields that can be set during initialization, passed as keyword arguments.
        """
        super().__init__(
            content=text, **data
        ) if text is not None else super().__init__(**data)

    @model_validator(mode="after")
    def remove_empty_name(self):
        attrList = []
        for attribute in self.__dict__:
            if attribute == "name":
                if self.__dict__[attribute] is None:
                    attrList.append(attribute)

        for item in attrList:
            delattr(self, item)

        return self


class FunctionCall(BaseModel):
    """
    Represents a function call with its name and arguments, which are stored as a JSON string.

    Attributes:
        name (str): Name of the function.
        arguments (str): A JSON string containing arguments for the function.
    """

    name: str
    arguments: str

    @field_validator("arguments", mode="before")
    @classmethod
    def validate_json(cls, v):
        """
        Ensures that the arguments are stored as a JSON string. If a dictionary is provided,
        it converts it to a JSON string. If a string is provided, it validates whether it's a proper JSON string.

        Args:
            v (Union[str, dict]): The JSON string or dictionary of arguments to validate and convert.

        Raises:
            ValueError: If the provided string is not valid JSON or if a type other than str or dict is provided.

        Returns:
            str: The JSON string representation of the arguments.
        """
        if isinstance(v, dict):
            try:
                return json.dumps(v)
            except TypeError as e:
                raise ValidationError(f"Invalid data type in dictionary: {e}")
        elif isinstance(v, str):
            try:
                json.loads(v)  # This is to check if it's valid JSON
                return v
            except json.JSONDecodeError as e:
                raise ValidationError(f"Invalid JSON format: {e}")
        else:
            raise TypeError(f"Unsupported type for field: {type(v)}")

    @property
    def arguments_dict(self):
        """
        Property to safely return arguments as a dictionary.
        """
        return json.loads(self.arguments) if self.arguments else {}


class ToolCall(BaseModel):
    """
    Represents a tool call within a message, detailing the tool that should be called.

    Attributes:
        id (str): Unique identifier of the tool call.
        type (str): Type of tool being called.
        function (Function): The function that should be called as part of the tool call.
    """

    id: str
    type: str
    function: FunctionCall


class FunctionCallChunk(BaseModel):
    """
    Represents a function call chunk in a streaming response, containing the function name and arguments.

    Attributes:
        name (str): The name of the function being called.
        arguments (str): The JSON string representation of the function's arguments.
    """

    name: Optional[str] = None
    arguments: Optional[str] = None


class ToolCallChunk(BaseModel):
    """
    Represents a tool call chunk in a streaming response, containing the index, ID, type, and function call details.

    Attributes:
        index (int): The index of the tool call in the response.
        id (str): Unique identifier for the tool call.
        type (str): The type of the tool call.
        function (FunctionCallChunk): The function call details associated with the tool call.
    """

    index: int
    id: Optional[str] = None
    type: Optional[str] = None
    function: FunctionCallChunk


class MessageContent(BaseMessage):
    """
    Extends BaseMessage to include dynamic optional fields for tool calls, function calls, and tool call IDs.

    Utilizes post-initialization validation to dynamically manage the inclusion of `tool_calls`, `function_call`, and `tool_call_id` fields based on their presence in the initialization data. Fields are only retained if they contain data, thus preventing serialization or display of `None` values, which helps maintain clean and concise object representations.

    Attributes:
        tool_calls (List[ToolCall], optional): A list of tool calls added dynamically if provided in the initialization data.
        function_call (FunctionCall, optional): A function call added dynamically if provided in the initialization data.
        tool_call_id (str, optional): Identifier for the specific tool call associated with the message, added dynamically if provided in the initialization data.
    """

    tool_calls: Optional[List[ToolCall]] = None
    function_call: Optional[FunctionCall] = None
    tool_call_id: Optional[str] = None

    @model_validator(mode="after")
    def remove_empty_calls(self):
        attrList = []
        for attribute in self.__dict__:
            if attribute in ("tool_calls", "function_call", "tool_call_id"):
                if self.__dict__[attribute] is None:
                    attrList.append(attribute)

        for item in attrList:
            delattr(self, item)

        return self


class SystemMessage(BaseMessage):
    """
    Represents a system message, automatically assigning the role to 'system'.

    Attributes:
        role (str): The role of the message, set to 'system' by default.
    """

    role: str = "system"


class UserMessage(BaseMessage):
    """
    Represents a user message, automatically assigning the role to 'user'.

    Attributes:
        role (str): The role of the message, set to 'user' by default.
    """

    role: str = "user"


class AssistantMessage(BaseMessage):
    """
    Represents an assistant message, potentially including tool calls, automatically assigning the role to 'assistant'.
    This message type is commonly used for responses generated by an assistant.

    Attributes:
        role (str): The role of the message, set to 'assistant' by default.
        tool_calls (List[ToolCall], optional): A list of tool calls added dynamically if provided in the initialization data.
        function_call (FunctionCall, optional): A function call added dynamically if provided in the initialization data.
    """

    role: str = "assistant"
    refusal: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    function_call: Optional[FunctionCall] = None

    @model_validator(mode="after")
    def remove_empty_calls(self):
        attrList = []
        for attribute in self.__dict__:
            if attribute in ("tool_calls", "function_call", "refusal"):
                if self.__dict__[attribute] is None:
                    attrList.append(attribute)

        for item in attrList:
            delattr(self, item)

        return self

    def get_tool_calls(self) -> Optional[List[ToolCall]]:
        """
        Retrieve tool calls from the message if available.
        """
        if getattr(self, "tool_calls", None) is None:
            return None
        if isinstance(self.tool_calls, list):
            return self.tool_calls
        if isinstance(self.tool_calls, ToolCall):
            return [self.tool_calls]

    def has_tool_calls(self) -> bool:
        """
        Check if the message has tool calls.
        """
        if not hasattr(self, "tool_calls"):
            return False
        if self.tool_calls is not None:
            return True
        if isinstance(self.tool_calls, ToolCall):
            return True


class ToolMessage(BaseMessage):
    """
    Represents a message specifically used for carrying tool interaction information, automatically assigning the role to 'tool'.

    Attributes:
        role (str): The role of the message, set to 'tool' by default.
        tool_call_id (str): Identifier for the specific tool call associated with the message.
    """

    role: str = "tool"
    tool_call_id: str


class LLMChatCandidate(BaseModel):
    """
    Represents a single candidate (output) from an LLM chat response.
    Allows provider-specific extra fields (e.g., index, logprobs, etc.).

    Attributes:
        message (AssistantMessage): The assistant's message for this candidate.
        finish_reason (Optional[str]): Why the model stopped generating text.
        [Any other provider-specific fields, e.g., index, logprobs, etc.]
    """

    message: AssistantMessage
    finish_reason: Optional[str] = None

    class Config:
        extra = "allow"


class LLMChatResponse(BaseModel):
    """
    Unified response for LLM chat completions, supporting multiple providers.

    Attributes:
        results (List[LLMChatCandidate]): List of candidate outputs.
        metadata (dict): Provider/model metadata (id, model, usage, etc.).
    """

    results: List[LLMChatCandidate]
    metadata: dict = {}

    def get_message(self) -> Optional[AssistantMessage]:
        """
        Retrieves the first message from the results if available.
        """
        return self.results[0].message if self.results else None


class LLMChatCandidateChunk(BaseModel):
    """
    Represents a partial (streamed) candidate from an LLM provider, for real-time streaming.
    """

    content: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    refusal: Optional[str] = None
    role: Optional[str] = None
    tool_calls: Optional[List["ToolCallChunk"]] = None
    finish_reason: Optional[str] = None
    index: Optional[int] = None
    logprobs: Optional[dict] = None


class LLMChatResponseChunk(BaseModel):
    """
    Represents a partial (streamed) response from an LLM provider, for real-time streaming.
    """

    result: LLMChatCandidateChunk
    metadata: Optional[dict] = None


class AssistantFinalMessage(BaseModel):
    """
    Represents a custom final message from the assistant, encapsulating a conclusive response to the user.

    Attributes:
        prompt (str): The initial prompt that led to the final answer.
        final_answer (str): The definitive answer or conclusion provided by the assistant.
    """

    prompt: str
    final_answer: str


class MessagePlaceHolder(BaseModel):
    """
    A placeholder for a list of messages in the prompt template.

    This allows dynamic insertion of message lists into the prompt, such as chat history or
    other sequences of messages.
    """

    variable_name: str
    model_config = ConfigDict(frozen=True)

    def __repr__(self):
        return f"MessagePlaceHolder(variable_name={self.variable_name})"


class EventMessageMetadata(BaseModel):
    """
    Represents CloudEvent metadata for describing event context and attributes.

    This class encapsulates core attributes as defined by the CloudEvents specification.
    Each field corresponds to a CloudEvent context attribute, providing additional metadata
    about the event.

    Attributes:
        id (Optional[str]):
            Identifies the event. Producers MUST ensure that source + id is unique for each
            distinct event. Required and must be a non-empty string.
        datacontenttype (Optional[str]):
            Content type of the event data value, e.g., 'application/json'.
            Optional and must adhere to RFC 2046.
        pubsubname (Optional[str]):
            Name of the Pub/Sub system delivering the event. Optional and specific to implementation.
        source (Optional[str]):
            Identifies the context in which an event happened. Required and must be a non-empty URI-reference.
        specversion (Optional[str]):
            The version of the CloudEvents specification used by this event. Required and must be non-empty.
        time (Optional[str]):
            The timestamp of when the occurrence happened in RFC 3339 format. Optional.
        topic (Optional[str]):
            The topic name that categorizes the event within the Pub/Sub system. Optional and specific to implementation.
        traceid (Optional[str]):
            The identifier for tracing systems to correlate events. Optional.
        traceparent (Optional[str]):
            Parent identifier in the tracing system. Optional and adheres to the W3C Trace Context standard.
        type (Optional[str]):
            Describes the type of event related to the originating occurrence. Required and must be a non-empty string.
        tracestate (Optional[str]):
            Vendor-specific tracing information. Optional and adheres to the W3C Trace Context standard.
        headers (Optional[Dict[str, str]]):
            HTTP headers or transport metadata. Optional and contains key-value pairs.
    """

    id: Optional[str]
    datacontenttype: Optional[str]
    pubsubname: Optional[str]
    source: Optional[str]
    specversion: Optional[str]
    time: Optional[str]
    topic: Optional[str]
    traceid: Optional[str]
    traceparent: Optional[str]
    type: Optional[str]
    tracestate: Optional[str]
    headers: Optional[Dict[str, str]]
