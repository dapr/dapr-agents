from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum
import uuid


class AgentStatus(str, Enum):
    """Enumeration of possible agent statuses for standardized tracking."""

    ACTIVE = "active"  # The agent is actively working on tasks
    IDLE = "idle"  # The agent is idle and waiting for tasks
    PAUSED = "paused"  # The agent is temporarily paused
    COMPLETE = "complete"  # The agent has completed all assigned tasks
    ERROR = "error"  # The agent encountered an error and needs attention


class AgentTaskStatus(str, Enum):
    """Enumeration of possible task statuses for standardizing task tracking."""

    IN_PROGRESS = "in-progress"  # Task is currently in progress
    COMPLETE = "complete"  # Task has been completed successfully
    FAILED = "failed"  # Task has failed to complete as expected
    PENDING = "pending"  # Task is awaiting to be started
    CANCELED = "canceled"  # Task was canceled and will not be completed


class AgentTaskEntry(BaseModel):
    """Represents a task handled by the agent, including its input, output, and status."""

    task_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the task",
    )
    input: str = Field(
        ..., description="The input or description of the task to be performed"
    )
    output: Optional[str] = Field(
        None, description="The output or result of the task, if completed"
    )
    status: AgentTaskStatus = Field(..., description="Current status of the task")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of task initiation or update",
    )
