from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from dapr_agents.types import ToolMessage
from datetime import datetime
import uuid


class AssistantWorkflowMessage(BaseModel):
    """Represents a message exchanged within the workflow."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the message",
    )
    role: str = Field(
        ..., description="The role of the message sender, e.g., 'user' or 'assistant'"
    )
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the message was created",
    )
    name: Optional[str] = Field(
        default=None,
        description="Optional name of the assistant or user sending the message",
    )


class AssistantWorkflowToolMessage(ToolMessage):
    """Represents a Tool message exchanged within the workflow."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the message",
    )
    function_name: str = Field(
        ...,
        description="Name of tool suggested by the model to run for a specific task.",
    )
    function_args: Optional[str] = Field(
        None,
        description="Tool arguments suggested by the model to run for a specific task.",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the message was created",
    )


class AssistantWorkflowEntry(BaseModel):
    """Represents a workflow and its associated data, including metadata on the source of the task request."""

    input: str = Field(
        ..., description="The input or description of the Workflow to be performed"
    )
    output: Optional[str] = Field(
        None, description="The output or result of the Workflow, if completed"
    )
    start_time: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the workflow was started",
    )
    end_time: Optional[datetime] = Field(
        None, description="Timestamp when the workflow was completed or failed"
    )
    messages: List[AssistantWorkflowMessage] = Field(
        default_factory=list, description="Messages exchanged during the workflow"
    )
    last_message: Optional[AssistantWorkflowMessage] = Field(
        default=None, description="Last processed message in the workflow"
    )
    tool_history: List[AssistantWorkflowToolMessage] = Field(
        default_factory=list, description="Tool message exchanged during the workflow"
    )
    source: Optional[str] = Field(None, description="Entity that initiated the task.")
    source_workflow_instance_id: Optional[str] = Field(
        None,
        description="The workflow instance ID associated with the original request.",
    )


class AssistantWorkflowState(BaseModel):
    """Represents the state of multiple Assistant workflows."""

    instances: Dict[str, AssistantWorkflowEntry] = Field(
        default_factory=dict,
        description="Workflow entries indexed by their instance_id.",
    )
