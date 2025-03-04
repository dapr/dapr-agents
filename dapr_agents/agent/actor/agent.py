from dapr_agents.agent.actor.service import AgentActorServiceBase
from dapr_agents.messaging import message_router
from dapr_agents.types.agent import AgentActorMessage
from dapr_agents.types.message import BaseMessage, EventMessageMetadata
from fastapi import Response, status
from pydantic import BaseModel, Field
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class AgentTaskResponse(BaseMessage):
    """
    Represents a response message from an agent after completing a task.
    """

class TriggerAction(BaseModel):
    """
    Represents a message used to trigger an agent's activity within the workflow.
    """
    task: Optional[str] = Field(None, description="The specific task to execute. If not provided, the agent can act based on its memory or predefined behavior.")
    iteration: Optional[int] = Field(default=0, description="The current iteration of the workflow loop.")

class AgentActorService(AgentActorServiceBase):
    """
    A Pydantic-based class for managing services and exposing FastAPI routes with Dapr pub/sub and actor support.
    """
    
    @message_router
    async def process_trigger_action(self, message: TriggerAction, metadata: EventMessageMetadata) -> Response:
        """
        Processes TriggerAction messages sent directly to the agent's topic.
        """
        try:
            logger.info(f"{self.agent.name} received {metadata.type} from {metadata.source}.")

            # Extract workflow_instance_id if available
            workflow_instance_id = metadata.headers.get("workflow_instance_id")
            logger.debug(f"Workflow instance ID: {workflow_instance_id}")

            # Execute the task or fallback to memory
            task = message.task
            if not task:
                logger.info(f"{self.agent.name} executing default task from memory.")

            response = await self.invoke_task(task)

            # Check if the response exists
            content = response.body.decode() if response and response.body else "Task completed but no response generated."

            # Broadcast result
            response_message = BaseMessage(name=self.agent.name, role="user", content=content)
            await self.broadcast_message(message=response_message)
            
            # Prepare metadata for routing
            additional_metadata = {"event_name": "AgentTaskResponse", "workflow_instance_id": workflow_instance_id}

            # Validate and wrap response
            agent_response = AgentTaskResponse(**response_message.model_dump())
            
            # Send the message to the target agent
            await self.send_message_to_agent(name=metadata.source, message=agent_response, **additional_metadata)

            return Response(content="Task processed successfully", status_code=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error processing trigger action: {e}", exc_info=True)
            return Response(content=f"Error processing message: {str(e)}", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @message_router(broadcast=True)
    async def process_broadcast_message(self, message: BaseMessage, metadata: EventMessageMetadata) -> Response:
        """
        Processes a message from the broadcast topic.
        """
        try:
            logger.info(f"{self.agent.name} received broadcast message of type '{metadata.type}' from '{metadata.source}'.")

            # Ignore messages sent by this agent
            if metadata.source == self.agent.name:
                logger.info(f"{self.agent.name} ignored its own broadcast message of type '{metadata.type}'.")
                return Response(status_code=status.HTTP_204_NO_CONTENT)

            # Log and process the valid broadcast message
            logger.debug(f"{self.agent.name} is processing broadcast message of type '{metadata.type}' from '{metadata.source}'.")
            logger.debug(f"Message content: {message.content}")

            # Add the message to the agent's memory
            self.agent.memory.add_message(message.model_dump())

            # Add the message to the actor's state
            actor_message = AgentActorMessage(**message.model_dump())
            await self.add_message(actor_message)

            return Response(content="Broadcast message added to memory and actor state.", status_code=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error processing broadcast message: {e}", exc_info=True)
            return Response(content=f"Error processing message: {str(e)}", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)