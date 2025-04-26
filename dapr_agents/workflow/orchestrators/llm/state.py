from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal
from datetime import datetime
import uuid


class TaskResult(BaseModel):
    """
    Represents the result of an agent's task execution.
    """

    agent: str = Field(..., description="The agent who executed the task.")
    step: int = Field(..., description="The step number associated with the task.")
    substep: Optional[float] = Field(
        None, description="The substep number (if applicable)."
    )
    result: str = Field(
        ..., description="The response or outcome of the task execution."
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of when the result was recorded.",
    )


class SubStep(BaseModel):
    substep: float = Field(
        ..., description="Substep identifier (float, e.g., 1.1, 2.3)."
    )
    description: str = Field(..., description="Detailed action to be performed.")
    status: Literal["not_started", "in_progress", "blocked", "completed"] = Field(
        ..., description="Current state of the sub-step."
    )


class PlanStep(BaseModel):
    step: int = Field(..., description="Step identifier (integer).")
    description: str = Field(..., description="Detailed action to be performed.")
    status: Literal["not_started", "in_progress", "blocked", "completed"] = Field(
        ..., description="Current state of the step."
    )
    substeps: Optional[List[SubStep]] = Field(
        None, description="Optional list of sub-steps."
    )


class LLMWorkflowMessage(BaseModel):
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


class LLMWorkflowEntry(BaseModel):
    """Represents a workflow and its associated data."""

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
    messages: List[LLMWorkflowMessage] = Field(
        default_factory=list, description="Messages exchanged during the workflow"
    )
    last_message: Optional[LLMWorkflowMessage] = Field(
        default=None, description="Last processed message in the workflow"
    )
    plan: Optional[List[PlanStep]] = Field(
        None, description="Structured execution plan for the workflow."
    )
    task_history: List[TaskResult] = Field(
        default_factory=list,
        description="A history of task executions and their results.",
    )


class LLMWorkflowState(BaseModel):
    """Represents the state of multiple LLM workflows."""

    instances: Dict[str, LLMWorkflowEntry] = Field(
        default_factory=dict,
        description="Workflow entries indexed by their instance_id.",
    )
