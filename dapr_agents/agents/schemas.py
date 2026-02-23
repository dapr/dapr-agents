import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from dapr_agents.prompt.prompty import Prompty
from dapr_agents.types import MessageContent, ToolExecutionRecord
from dapr_agents.types.message import BaseMessage


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


class ConversationSummary(BaseModel):
    """
    Structured summary of a conversation for long-term memory storage.
    Used as response_format when generating summaries so the LLM returns
    parseable, consistent output that is easier to store and reuse.
    """

    summary: str = Field(
        ...,
        description="Concise summary of the conversation and tool usage for long-term memory: key facts, decisions, and outcomes.",
    )


class AgentWorkflowEntry(BaseModel):
    """
    Workflow entry data stored in the agent state store.
    Excludes fields provided by Dapr get_workflow.
    """

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
    triggering_workflow_instance_id: Optional[str] = Field(
        default=None,
        description="The workflow instance ID of the entity that triggered this agent (for multi-agent communication).",
    )
    trace_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="OpenTelemetry trace context for workflow resumption.",
    )
