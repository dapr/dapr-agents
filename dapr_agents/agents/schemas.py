import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

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
