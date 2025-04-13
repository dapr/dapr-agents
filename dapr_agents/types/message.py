from pydantic import (
    BaseModel,
    field_validator,
    ValidationError,
    model_validator,
    ConfigDict,
)
from typing import List, Optional, Dict
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


class MessageContent(BaseMessage):
    """
    Extends BaseMessage to include dynamic optional fields for tool and function calls.

    Utilizes post-initialization validation to dynamically manage the inclusion of `tool_calls` and `function_call` fields based on their presence in the initialization data. Fields are only retained if they contain data, thus preventing serialization or display of `None` values, which helps maintain clean and concise object representations.

    Attributes:
        tool_calls (List[ToolCall], optional): A list of tool calls added dynamically if provided in the initialization data.
        function_call (FunctionCall, optional): A function call added dynamically if provided in the initialization data.
    """

    tool_calls: Optional[List[ToolCall]] = None
    function_call: Optional[FunctionCall] = None

    @model_validator(mode="after")
    def remove_empty_calls(self):
        attrList = []
        for attribute in self.__dict__:
            if attribute in ("tool_calls", "function_call"):
                if self.__dict__[attribute] is None:
                    attrList.append(attribute)

        for item in attrList:
            delattr(self, item)

        return self


class Choice(BaseModel):
    """
    Represents a choice made by the model, detailing the reason for completion, its index, and the message content.

    Attributes:
        finish_reason (str): Reason why the model stopped generating text.
        index (int): Index of the choice in a list of potential choices.
        message (MessageContent): Content of the message chosen by the model.
        logprobs (Optional[dict]): Log probabilities associated with the choice.
    """

    finish_reason: str
    index: int
    message: MessageContent
    logprobs: Optional[dict]


class ChatCompletion(BaseModel):
    """
    Represents the full response from the chat API, including all choices, metadata, and usage information.

    Attributes:
        choices (List[Choice]): List of choices provided by the model.
        created (int): Timestamp when the response was created.
        id (str): Unique identifier for the response.
        model (str): Model used for generating the response.
        object (str): Type of object returned.
        usage (dict): Information about API usage for this request.
    """

    choices: List[Choice]
    created: int
    id: Optional[str] = None
    model: str
    object: Optional[str] = None
    usage: dict

    def get_message(self) -> Optional[str]:
        """
        Retrieve the main message content from the first choice.
        """
        return self.choices[0].message.model_dump() if self.choices else None

    def get_reason(self) -> Optional[str]:
        """
        Retrieve the reason for completion from the first choice.
        """
        return self.choices[0].finish_reason if self.choices else None

    def get_tool_calls(self) -> Optional[List[ToolCall]]:
        """
        Retrieve tool calls from the first choice, if available.
        """
        return (
            self.choices[0].message.tool_calls
            if self.choices and self.choices[0].message.tool_calls
            else None
        )

    def get_content(self) -> Optional[str]:
        """
        Retrieve the content from the first choice's message.
        """
        message = self.get_message()
        return message.get("content") if message else None


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
    tool_calls: Optional[List[ToolCall]] = None
    function_call: Optional[FunctionCall] = None

    @model_validator(mode="after")
    def remove_empty_calls(self):
        attrList = []
        for attribute in self.__dict__:
            if attribute in ("tool_calls", "function_call"):
                if self.__dict__[attribute] is None:
                    attrList.append(attribute)

        for item in attrList:
            delattr(self, item)

        return self


class ToolMessage(BaseMessage):
    """
    Represents a message specifically used for carrying tool interaction information, automatically assigning the role to 'tool'.

    Attributes:
        role (str): The role of the message, set to 'tool' by default.
        tool_call_id (str): Identifier for the specific tool call associated with the message.
    """

    role: str = "tool"
    tool_call_id: str


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
