from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dapr_agents.types import DaprWorkflowContext, EventMessageMetadata
from fastapi import Response

class OrchestratorInterface(ABC):
    """
    Interface defining the required methods for workflow orchestrators.
    Ensures consistent implementation across different orchestration strategies.
    """
    
    @abstractmethod
    def model_post_init(self, __context: Any) -> None:
        """Initialize the orchestrator service."""
        pass

    @abstractmethod
    def main_workflow(self, ctx: DaprWorkflowContext, input: Any) -> Any:
        """
        Execute the primary workflow that coordinates agent interactions.

        Args:
            ctx (DaprWorkflowContext): The workflow execution context
            input (Any): The input for this workflow iteration

        Returns:
            Any: The workflow result or continuation
        """
        pass

    @abstractmethod
    async def process_agent_response(self, message: Any, metadata: EventMessageMetadata) -> Response:
        """Process responses from agents."""
        pass

    @abstractmethod
    async def broadcast_message_to_agents(self, **kwargs) -> None:
        """Broadcast a message to all registered agents."""
        pass

    @abstractmethod
    async def trigger_agent(self, name: str, instance_id: str, **kwargs) -> None:
        """Trigger a specific agent to perform an action."""
        pass
