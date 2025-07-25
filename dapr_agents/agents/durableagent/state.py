from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from dapr_agents.types import MessageContent, ToolExecutionRecord
from datetime import datetime
import uuid


class DurableAgentMessage(MessageContent):
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the message",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the message was created",
    )


class DurableAgentWorkflowEntry(BaseModel):
    """Represents a workflow and its associated data, including metadata on the source of the task request."""

    input: str = Field(
        ..., description="The input or description of the Workflow to be performed"
    )
    output: Optional[str] = Field(
        default=None, description="The output or result of the Workflow, if completed"
    )
    start_time: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the workflow was started",
    )
    end_time: Optional[datetime] = Field(
        default_factory=datetime.now,
        description="Timestamp when the workflow was completed or failed",
    )
    messages: List[DurableAgentMessage] = Field(
        default_factory=list,
        description="Messages exchanged during the workflow (user, assistant, or tool messages).",
    )
    last_message: Optional[DurableAgentMessage] = Field(
        default=None, description="Last processed message in the workflow"
    )
    tool_history: List[ToolExecutionRecord] = Field(
        default_factory=list, description="Tool message exchanged during the workflow"
    )
    source: Optional[str] = Field(None, description="Entity that initiated the task.")
    source_workflow_instance_id: Optional[str] = Field(
        default=None,
        description="The workflow instance ID associated with the original request.",
    )


class DurableAgentWorkflowState(BaseModel):
    """Represents the state of multiple Agent workflows."""

    instances: Dict[str, DurableAgentWorkflowEntry] = Field(
        default_factory=dict,
        description="Workflow entries indexed by their instance_id.",
    )
    chat_history: List[DurableAgentMessage] = Field(
        default_factory=list,
        description="Chat history of messages exchanged during the workflow.",
    )
