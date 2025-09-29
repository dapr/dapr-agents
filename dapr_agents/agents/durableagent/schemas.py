from dapr_agents.types.message import BaseMessage
from pydantic import BaseModel, Field
from typing import Optional


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


class InternalTriggerAction(BaseModel):
    """
    Represents an internal message used by orchestrators to trigger agents.
    This prevents self-triggering loops.
    """

    task: Optional[str] = Field(
        None,
        description="The specific task to execute. If not provided, the agent will act based on its memory or predefined behavior.",
    )
    workflow_instance_id: Optional[str] = Field(
        default=None, description="Dapr workflow instance id from source if available"
    )
