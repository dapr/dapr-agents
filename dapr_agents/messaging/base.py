from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
from pydantic import BaseModel, Field
import logging
import json

logger = logging.getLogger(__name__)

class PubSubBase(BaseModel, ABC):
    """
    Abstract base class for a generic pub/sub messaging system.
    """

    message_bus_name: str = Field(..., description="The name of the message bus component, defining the pub/sub base.")
    
    @abstractmethod
    async def publish_message(self, pubsub_name: str, topic_name: str, message: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Abstract method to publish a message to a specified topic.
        Must be implemented by subclasses.
        """
        pass

    async def serialize_message(self, message: Any) -> str:
        """
        Serializes a message to JSON format.

        Args:
            message (Any): The message content to serialize.

        Returns:
            str: JSON string of the message.

        Raises:
            ValueError: If the message is not serializable.
        """
        try:
            return json.dumps(message if message is not None else {})
        except TypeError as te:
            logger.error(f"Failed to serialize message: {message}. Error: {te}")
            raise ValueError(f"Message contains non-serializable data: {te}")