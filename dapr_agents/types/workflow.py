from dapr.ext.workflow import DaprWorkflowContext
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum
from datetime import datetime
import uuid

class DaprWorkflowStatus(str, Enum):
    """Enumeration of possible workflow statuses for standardized tracking."""
    UNKNOWN = "unknown"              # Workflow is in an undefined state
    RUNNING = "running"              # Workflow is actively running
    COMPLETED = "completed"          # Workflow has completed
    FAILED = "failed"                # Workflow encountered an error
    TERMINATED = "terminated"        # Workflow was canceled or forcefully terminated
    SUSPENDED = "suspended"          # Workflow was temporarily paused
    PENDING = "pending"              # Workflow is waiting to start

class WorkflowMessage(BaseModel):
    """Represents a message exchanged within the workflow."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the message")
    role: str = Field(..., description="The role of the message sender, e.g., 'user' or 'assistant'")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp when the message was created")
    name: Optional[str] = Field(default=None, description="Optional name of the assistant or user sending the message")

class WorkflowEntry(BaseModel):
    """Represents a workflow and its associated data."""
    input: str = Field(..., description="The input or description of the Workflow to be performed")
    output: Optional[str] = Field(None, description="The output or result of the Workflow, if completed")
    status: WorkflowStatus = Field(..., description="Current status of the Workflow")
    start_time: datetime = Field(default_factory=datetime.now, description="Timestamp when the workflow was started")
    end_time: Optional[datetime] = Field(None, description="Timestamp when the workflow was completed or failed")
    messages: List[WorkflowMessage] = Field(default_factory=list, description="Messages exchanged during the workflow")

class WorkflowStateMap(BaseModel):
    """Represents the state of multiple workflows."""
    instances: Dict[str, WorkflowEntry] = Field(default_factory=dict, description="Workflow entries indexed by their instance_id.")