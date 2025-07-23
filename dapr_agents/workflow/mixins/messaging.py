import logging
from typing import Union

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MessagingMixin:
    """
    Mixin providing agent messaging capabilities, including broadcast, direct messaging, and agent metadata management.
    """

    async def broadcast_message(
        self,
        message: Union[BaseModel, dict],
        exclude_orchestrator: bool = False,
        **kwargs,
    ) -> None:
        """
        Send a message to all registered agents.

        Args:
            message: The message content as a Pydantic model or dictionary.
            exclude_orchestrator: If True, excludes orchestrators from receiving the message.
            **kwargs: Additional metadata fields to include in the message.
        """
        try:
            # Skip broadcasting if no broadcast topic is set
            if not self.broadcast_topic_name:
                logger.info(f"{self.name} has no broadcast topic; skipping broadcast.")
                return
            # Skip broadcasting if no agents are registered
            agents_metadata = self.get_agents_metadata(
                exclude_orchestrator=exclude_orchestrator
            )
            if not agents_metadata:
                logger.warning("No agents available for broadcast.")
                return
            # Broadcast the message to all agents
            logger.info(
                f"{self.name} broadcasting message to {self.broadcast_topic_name}."
            )
            await self.publish_event_message(
                topic_name=self.broadcast_topic_name,
                pubsub_name=self.message_bus_name,
                source=self.name,
                message=message,
                **kwargs,
            )
            logger.debug(f"{self.name} broadcasted message.")
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}", exc_info=True)

    async def send_message_to_agent(
        self, name: str, message: Union[BaseModel, dict], **kwargs
    ) -> None:
        """
        Send a message to a specific agent.

        Args:
            name: The name of the target agent.
            message: The message content as a Pydantic model or dictionary.
            **kwargs: Additional metadata fields to include in the message.
        """
        try:
            agents_metadata = self.get_agents_metadata()
            if name not in agents_metadata:
                logger.warning(
                    f"Target '{name}' is not registered as an agent. Skipping message send."
                )
                return

            agent_metadata = agents_metadata[name]
            logger.info(f"{self.name} sending message to agent '{name}'.")
            await self.publish_event_message(
                topic_name=agent_metadata["topic_name"],
                pubsub_name=agent_metadata["pubsub_name"],
                source=self.name,
                message=message,
                **kwargs,
            )
            logger.debug(f"{self.name} sent message to agent '{name}'.")
        except Exception as e:
            logger.error(
                f"Failed to send message to agent '{name}': {e}", exc_info=True
            )
