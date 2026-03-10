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

from typing import Optional, List, Dict, Literal, Any
from pydantic import BaseModel, field_validator, ValidationInfo, Field, model_validator
from datetime import datetime, timezone
import uuid
from datetime import timedelta
from enum import Enum


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


class SseServerParameters(BaseModel):
    """
    Configuration for Server-Sent Events (SSE) transport.

    Attributes:
        url (str): The SSE endpoint URL.
        headers (Optional[Dict[str, str]]): Optional HTTP headers.
        timeout (float): Connection timeout in seconds.
        sse_read_timeout (float): Timeout for SSE read operations.
    """

    url: str
    headers: Optional[Dict[str, str]] = None
    timeout: float = 5.0
    sse_read_timeout: float = 300.0


class StreamableHTTPServerParameters(BaseModel):
    """
    Configuration for streamable HTTP transport.

    Attributes:
        url (str): The streamable HTTP endpoint URL.
        headers (Optional[Dict[str, str]]): Optional HTTP headers.
        timeout (timedelta): Connection timeout as a timedelta.
        sse_read_timeout (timedelta): Timeout for SSE read operations as a timedelta.
        terminate_on_close (bool): Whether to terminate the connection on close.
    """

    url: str
    headers: Optional[Dict[str, str]] = None
    timeout: timedelta = timedelta(seconds=30)
    sse_read_timeout: timedelta = timedelta(seconds=300)
    terminate_on_close: bool = True


class WebSocketServerParameters(BaseModel):
    """
    Configuration for websocket transport.
    """

    url: str = Field(
        ...,
        description="The websocket endpoint URL.",
    )


class ToolExecutionStatus(str, Enum):
    """
    Tool execution lifecycle status, aligned with DaprWorkflowStatus values.
    TIMEOUT is a tool-specific addition for calls that exceed a deadline.
    """

    PENDING = "pending"  # Dispatched but not yet started
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Finished successfully
    FAILED = "failed"  # Finished with an error
    TIMEOUT = "timeout"  # Exceeded execution deadline


class ToolExecutionRecord(BaseModel):
    """
    Represents a record of a tool execution, capturing identity, timing, status,
    and result data useful for workflow observability and debugging.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this execution record",
    )
    tool_call_id: str = Field(
        ...,
        description="LLM-assigned identifier for the tool call (matches the message tool_call_id)",
    )
    tool_name: str = Field(
        ...,
        description="Name of the tool or agent invoked",
    )
    tool_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments passed to the tool",
    )

    # Status
    status: ToolExecutionStatus = Field(
        default=ToolExecutionStatus.PENDING,
        description="Execution lifecycle status",
    )
    is_agent_call: bool = Field(
        default=False,
        description="True when this tool call invoked another agent rather than a local function",
    )

    # Timing
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the tool execution was dispatched",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Timestamp when the tool execution returned",
    )
    duration_ms: Optional[float] = Field(
        None,
        description="Wall-clock execution time in milliseconds (auto-computed when both started_at and completed_at are set)",
    )

    # Result
    execution_result: Optional[str] = Field(
        None,
        description="Text representation of the tool result",
    )
    structured_result: Optional[Dict[str, Any]] = Field(
        None,
        description="Structured (JSON-serializable) result for tools that return rich data",
    )
    error_message: Optional[str] = Field(
        None,
        description="Error detail when status is 'failed' or 'timeout'",
    )

    # Provenance
    executing_agent: Optional[str] = Field(
        None,
        description="Name of the agent that dispatched this tool call",
    )
    agent_workflow_instance_id: Optional[str] = Field(
        None,
        description="Workflow instance ID of the invoked agent (populated for agent-as-tool calls)",
    )
    attempt: int = Field(
        default=1,
        description="Attempt number starting at 1; incremented on retries",
    )

    @model_validator(mode="after")
    def _compute_duration(self) -> "ToolExecutionRecord":
        if self.duration_ms is None and self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            self.duration_ms = delta.total_seconds() * 1000
        return self


class TextContent(BaseModel):
    """Simple text content block."""

    type: str = "text"
    text: str


class ToolResult(BaseModel):
    """
    Standardized result from tool execution.
    """

    content: List[TextContent] = Field(default_factory=list)
    structuredContent: Optional[dict[str, Any]] = None
    isError: bool = False

    @classmethod
    def success(cls, result: Any, text: Optional[str] = None) -> "ToolResult":
        """Create a successful result."""
        return cls(
            content=[TextContent(text=text or str(result))],
            structuredContent={"result": result} if result is not None else None,
            isError=False,
        )

    @classmethod
    def error(cls, message: str) -> "ToolResult":
        """Create an error result."""
        return cls(
            content=[TextContent(text=message)],
            isError=True,
        )
