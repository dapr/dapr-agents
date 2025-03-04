from dapr_agents.workflow.agentic import AgenticWorkflowService
from pydantic import Field, model_validator
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class OrchestratorServiceBase(AgenticWorkflowService):

    orchestrator_topic_name: Optional[str] = Field(None, description="The topic name dedicated to this specific orchestrator, derived from the orchestrator's name if not provided.")
    
    @model_validator(mode="before")
    def set_orchestrator_topic_name(cls, values: dict):
        # Derive orchestrator_topic_name from agent name
        if not values.get("orchestrator_topic_name") and values.get("name"):
            values["orchestrator_topic_name"] = values["name"]
        
        return values
    
    def model_post_init(self, __context: Any) -> None:
        """
        Register agentic workflow.
        """

        # Complete post-initialization
        super().model_post_init(__context)

        # Prepare agent metadata
        self.agent_metadata = {
            "name": self.name,
            "topic_name": self.orchestrator_topic_name,
            "pubsub_name": self.message_bus_name
        }

        # Register agent metadata
        self.register_agentic_system()