import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from dapr_agents.prompt.prompty import Prompty
from dapr_agents.types import MessageContent, ToolExecutionRecord
from dapr_agents.types.message import BaseMessage
from dapr_agents.types.workflow import DaprWorkflowStatus


def utcnow() -> datetime:
    """Return current time as timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class BroadcastMessage(BaseMessage):
    """
    Represents a broadcast message from an agent.
    """


class AgentTaskResponse(BaseMessage):
    """
    Represents a response message from an agent after completing a task.
    """

    workflow_instance_id: Optional[str] = Field(
        default=None, description="Dapr workflow instance id from source if available"
    )


class TriggerAction(BaseModel):
    """
    Represents a message used to trigger an agent's activity within the workflow.
    """

    task: Optional[str] = Field(
        None,
        description="The specific task to execute. If not provided, the agent will act based on its memory or predefined behavior.",
    )
    workflow_instance_id: Optional[str] = Field(
        default=None, description="Dapr workflow instance id from source if available"
    )


class AgentWorkflowMessage(MessageContent):
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the message",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="Timestamp when the message was created",
    )


class AgentWorkflowEntry(BaseModel):
    """Represents a workflow and its associated data, including metadata on the source of the task request."""

    input_value: str = Field(
        ..., description="The input or description of the Workflow to be performed"
    )
    output: Optional[str] = Field(
        default=None, description="The output or result of the Workflow, if completed"
    )
    start_time: datetime = Field(
        default_factory=utcnow,
        description="Timestamp when the workflow was started",
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the workflow was completed or failed",
    )
    messages: List[AgentWorkflowMessage] = Field(
        default_factory=list,
        description="Messages exchanged during the workflow (user, assistant, or tool messages).",
    )
    system_messages: List[AgentWorkflowMessage] = Field(
        default_factory=list,
        description="Rendered system prompt messages included when invoking the LLM.",
    )
    last_message: Optional[AgentWorkflowMessage] = Field(
        default=None, description="Last processed message in the workflow"
    )
    tool_history: List[ToolExecutionRecord] = Field(
        default_factory=list, description="Tool message exchanged during the workflow"
    )
    source: Optional[str] = Field(None, description="Entity that initiated the task.")
    workflow_instance_id: Optional[str] = Field(
        default=None,
        description="The agent's own workflow instance ID.",
    )
    triggering_workflow_instance_id: Optional[str] = Field(
        default=None,
        description="The workflow instance ID of the entity that triggered this agent (for multi-agent communication).",
    )
    workflow_name: Optional[str] = Field(
        default=None,
        description="The name of the workflow.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Conversation memory session identifier, when available.",
    )
    trace_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="OpenTelemetry trace context for workflow resumption.",
    )
    status: str = Field(
        default=DaprWorkflowStatus.RUNNING.value,
        description="Current status of the workflow.",
    )


class AgentWorkflowState(BaseModel):
    """Represents the state of multiple Agent workflows."""

    instances: Dict[str, AgentWorkflowEntry] = Field(
        default_factory=dict,
        description="Workflow entries indexed by their instance_id.",
    )


class AgentMetadata(BaseModel):
    """Metadata about an agent's configuration and capabilities."""

    appid: str = Field(..., description="Dapr application ID of the agent")
    type: str = Field(..., description="Type of the agent (e.g., standalone, durable)")
    orchestrator: bool = Field(
        False, description="Indicates if the agent is an orchestrator"
    )
    role: str = Field(default="", description="Role of the agent")
    goal: str = Field(default="", description="High-level objective of the agent")
    name: str = Field(default="", description="Namememory of the agent")
    instructions: Optional[List[str]] = Field(
        default=None, description="Instructions for the agent"
    )
    statestore: Optional[str] = Field(
        default=None, description="Dapr state store component name used by the agent"
    )
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt guiding the agent's behavior"
    )


class PubSubMetadata(BaseModel):
    """Pub/Sub configuration information."""

    agent_name: str = Field(..., description="Pub/Sub topic the agent subscribes to")
    name: str = Field(..., description="Pub/Sub component name")
    broadcast_topic: Optional[str] = Field(
        default=None, description="Pub/Sub topic for broadcasting messages"
    )
    agent_topic: Optional[str] = Field(
        default=None, description="Pub/Sub topic for direct agent messages"
    )


class MemoryMetadata(BaseModel):
    """Memory configuration information."""

    type: str = Field(..., description="Type of memory used by the agent")
    statestore: Optional[str] = Field(
        default=None, description="Dapr state store component name for memory"
    )
    session_id: Optional[str] = Field(
        default=None, description="Default session ID for the agent's memory"
    )


class LLMMetadata(BaseModel):
    """LLM configuration information."""

    client: str = Field(..., description="LLM client used by the agent")
    provider: str = Field(..., description="LLM provider used by the agent")
    api: str = Field(default="unknown", description="API type used by the LLM client")
    model: str = Field(default="unknown", description="Model name or identifier")
    component_name: Optional[str] = Field(
        default=None, description="Dapr component name for the LLM client"
    )
    base_url: Optional[str] = Field(
        default=None, description="Base URL for the LLM API if applicable"
    )
    azure_endpoint: Optional[str] = Field(
        default=None, description="Azure endpoint if using Azure OpenAI"
    )
    azure_deployment: Optional[str] = Field(
        default=None, description="Azure deployment name if using Azure OpenAI"
    )
    prompt_template: Optional[str] = Field(
        default=None, description="Prompt template used by the agent"
    )
    prompty: Optional[Prompty] = Field(
        default=None, description="Prompty template name if used"
    )


class ToolMetadata(BaseModel):
    """Metadata about a tool available to the agent."""

    tool_name: str = Field(..., description="Name of the tool")
    tool_description: str = Field(
        ..., description="Description of the tool's functionality"
    )
    tool_args: str = Field(..., description="Arguments for the tool")


class RegistryMetadata(BaseModel):
    """Registry configuration information."""

    statestore: Optional[str] = Field(
        None, description="Name of the statestore component for the registry"
    )
    name: Optional[str] = Field(default=None, description="Name of the team registry")


class AgentMetadataSchema(BaseModel):
    """Schema for agent metadata including schema version."""

    schema_version: str = Field(
        ...,
        description="Version of the schema used for the agent metadata.",
    )
    agent: AgentMetadata = Field(
        ..., description="Agent configuration and capabilities"
    )
    name: str = Field(..., description="Name of the agent")
    registered_at: str = Field(..., description="ISO 8601 timestamp of registration")
    pubsub: Optional[PubSubMetadata] = Field(
        None, description="Pub/sub configuration if enabled"
    )
    memory: Optional[MemoryMetadata] = Field(
        None, description="Memory configuration if enabled"
    )
    llm: Optional[LLMMetadata] = Field(None, description="LLM configuration")
    registry: Optional[RegistryMetadata] = Field(
        None, description="Registry configuration"
    )
    tools: Optional[List[ToolMetadata]] = Field(None, description="Available tools")
    max_iterations: Optional[int] = Field(
        None, description="Maximum iterations for agent execution"
    )
    tool_choice: Optional[str] = Field(None, description="Tool choice strategy")
    agent_metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata about the agent"
    )

    @classmethod
    def export_json_schema(cls, version: str) -> Dict[str, Any]:
        """
        Export the JSON schema with version information.

        Args:
            version: The dapr-agents version for this schema

        Returns:
            JSON schema dictionary with metadata
        """
        schema = cls.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["version"] = version
        return schema
