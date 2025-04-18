import logging
from dapr_agents.agent.actor.schemas import AgentTaskResponse, TriggerAction, BroadcastMessage
from dapr_agents.agent.actor.service import AgentActorService
from dapr_agents.types.agent import AgentActorMessage
from dapr_agents.workflow.messaging.decorator import message_router

logger = logging.getLogger(__name__)

class AgentActor(AgentActorService):
    """
    A Pydantic-based class for managing services and exposing FastAPI routes with Dapr pub/sub and actor support.
    """
    
    @message_router
    async def process_trigger_action(self, message: TriggerAction):
        """
        Processes TriggerAction messages sent directly to the agent's topic.
        """
        try:
            metadata = message.pop("_message_metadata", {})
            source = metadata.get("source", "unknown_source")
            message_type = metadata.get("type", "unknown_type")

            logger.info(f"{self.agent.name} received {message_type} from {source}.")

            # Extract workflow_instance_id if available
            workflow_instance_id = message.get("workflow_instance_id") or None
            logger.debug(f"Workflow instance ID: {workflow_instance_id}")

            # Execute the task or fallback to memory
            task = message.get("task", None)
            if not task:
                logger.info(f"{self.agent.name} executing default task from memory.")

            response = await self.invoke_task(task)

            # Check if the response exists
            content = response.body.decode() if response and response.body else "Task completed but no response generated."

            # Broadcast result
            response_message = BroadcastMessage(name=self.agent.name, role="user", content=content)
            await self.broadcast_message(message=response_message)
            
            # Update response
            response_message = response_message.model_dump()
            response_message["workflow_instance_id"] = workflow_instance_id
            agent_response = AgentTaskResponse(**response_message)
            
            # Send the message to the target agent
            await self.send_message_to_agent(name=source, message=agent_response)
        except Exception as e:
            logger.error(f"Error processing trigger action: {e}", exc_info=True)

    @message_router(broadcast=True)
    async def process_broadcast_message(self, message: BroadcastMessage):
        """
        Processes a message from the broadcast topic.
        """
        try:
            metadata = message.pop("_message_metadata", {})

            if not isinstance(metadata, dict):
                logger.warning(f"{getattr(self, 'name', 'agent')} received a broadcast with invalid metadata. Ignoring.")
                return

            source = metadata.get("source", "unknown_source")
            message_type = metadata.get("type", "unknown_type")
            message_content = message.get("content", "No content")

            logger.info(f"{self.agent.name} received broadcast message of type '{message_type}' from '{source}'.")

            # Ignore messages sent by this agent
            if source == self.agent.name:
                logger.info(f"{self.agent.name} ignored its own broadcast message of type '{message_type}'.")
                return
            
            # Log and process the valid broadcast message
            logger.debug(f"{self.agent.name} is processing broadcast message of type '{message_type}' from '{source}'.")
            logger.debug(f"Message content: {message_content}")

            # Add the message to the agent's memory
            self.agent.memory.add_message(message)

            # Add the message to the actor's state
            actor_message = AgentActorMessage(**message)
            await self.add_message(actor_message)

        except Exception as e:
            logger.error(f"Error processing broadcast message: {e}", exc_info=True)