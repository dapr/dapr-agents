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

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import StrEnum
import uuid


class ToolChoice(StrEnum):
    """
    Enumeration of supported tool choice strategies for durable agents.

    AUTO: The agent decides when to use tools based on the prompt and context.
        This is the default and recommended choice for most use cases,
        as it allows the agent to leverage tools when beneficial while avoiding unnecessary calls.
    """

    # TODO: This enum does not support dicts, which some LLM providers allow when forcing a specific tool
    AUTO = "auto"


class ToolExecutionMode(StrEnum):
    """
    Enumeration of supported tool execution modes for durable agents.

    PARALLEL: All tool calls returned by the LLM in a single turn are executed
        concurrently via ``wf.when_all``. This is the default behaviour and
        provides the best latency when tools are independent.
    SEQUENTIAL: Tool calls are executed one after another in the order they
        were returned by the LLM. Use this when tools have side-effects that
        depend on the results of earlier calls in the same turn.
    """

    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


class OrchestrationMode(StrEnum):
    """
    Enumeration of supported orchestration strategies for durable agents.

    AGENT: Orchestration is driven by an LLM-generated plan that determines the next steps and agent interactions.
    RANDOM: Orchestration randomly selects agents or actions at each decision point, without a predetermined plan.
    ROUNDROBIN: Orchestration cycles through available agents or actions in a fixed order, ensuring equal opportunity for each participant.
    """

    AGENT = "agent"
    RANDOM = "random"
    ROUNDROBIN = "roundrobin"


class AgentStatus(StrEnum):
    """Enumeration of possible agent statuses for standardized tracking."""

    ACTIVE = "active"  # The agent is actively working on tasks
    IDLE = "idle"  # The agent is idle and waiting for tasks
    PAUSED = "paused"  # The agent is temporarily paused
    COMPLETE = "complete"  # The agent has completed all assigned tasks
    ERROR = "error"  # The agent encountered an error and needs attention


class AgentTaskStatus(StrEnum):
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
