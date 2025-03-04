from typing import Optional, Any, Dict, Union
from dapr.conf import settings as dapr_settings
from dapr.aio.clients import DaprClient
from pydantic import BaseModel, Field
from dapr_agents.messaging import PubSubBase
import logging
import os

logger = logging.getLogger(__name__)

class DaprPubSub(PubSubBase):
    """
    Dapr-based implementation of pub/sub messaging.
    """

    daprGrpcHost: Optional[str] = Field(None, description="Host address for the Dapr gRPC endpoint.")
    daprGrpcPort: Optional[int] = Field(None, description="Port number for the Dapr gRPC endpoint.")
    daprGrpcAddress: Optional[str] = Field(None, description="Full address of the Dapr sidecar (host:port).")

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization to configure Dapr settings.
        """
        # Configure Dapr gRPC settings, using environment variables if provided
        env_daprGrpcHost = os.getenv('DAPR_RUNTIME_HOST')
        env_daprGrpcPort = os.getenv('DAPR_GRPC_PORT')

        # Resolve final values for Dapr settings
        self.daprGrpcHost = self.daprGrpcHost or env_daprGrpcHost or dapr_settings.DAPR_RUNTIME_HOST
        self.daprGrpcPort = int(self.daprGrpcPort or env_daprGrpcPort or dapr_settings.DAPR_GRPC_PORT)

        # Set the Dapr gRPC address based on finalized settings
        self.daprGrpcAddress = self.daprGrpcAddress or f"{self.daprGrpcHost}:{self.daprGrpcPort}"

        # Proceed with base model setup
        super().model_post_init(__context)

    async def publish_message(self, pubsub_name: str, topic_name: str, message: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Publishes a message to a specific topic with optional metadata.

        Args:
            pubsub_name (str): The pub/sub component to use.
            topic_name (str): The topic to publish the message to.
            message (Any): The message content, can be None or any JSON-serializable type.
            metadata (Optional[Dict[str, Any]]): Additional metadata to include in the publish event.

        Raises:
            ValueError: If the message contains non-serializable data.
            Exception: If publishing the message fails.
        """
        try:
            # Serialize the message, handling non-serializable data
            json_message = await self.serialize_message(message)
            
            # Publishing message
            async with DaprClient(address=self.daprGrpcAddress) as client:
                await client.publish_event(
                    pubsub_name=pubsub_name or self.message_bus_name,
                    topic_name=topic_name,
                    data=json_message,
                    data_content_type='application/json',
                    publish_metadata=metadata or {}
                )
            
            logger.debug(f"Message successfully published to topic '{topic_name}' on pub/sub '{pubsub_name}'.")
            logger.debug(f"Serialized Message: {json_message}, Metadata: {metadata}")
        except Exception as e:
            logger.error(
                f"Error publishing message to topic '{topic_name}' on pub/sub '{pubsub_name}'. "
                f"Message: {message}, Metadata: {metadata}, Error: {e}"
            )
            raise Exception(f"Failed to publish message to topic '{topic_name}' on pub/sub '{pubsub_name}': {str(e)}")
    
    async def publish_event_message(self, topic_name: str, pubsub_name: str, source: str, message: Union[BaseModel, dict], message_type: Optional[str] = None, **kwargs,) -> None:
        """
        Publishes an event message to a specified topic with dynamic metadata.

        Args:
            topic_name (str): The topic to publish the message to.
            pubsub_name (str): The pub/sub component to use.
            source (str): The source of the message (e.g., service or agent name).
            message (Union[BaseModel, dict]): The message content as a Pydantic model or dictionary.
            message_type (Optional[str]): The type of the message. Required if `message` is a dictionary.
            **kwargs: Additional metadata fields to include in the message.
        """
        if isinstance(message, BaseModel):
            # Derive `message_type` from the Pydantic model class name
            message_type = message.__class__.__name__
            message_dict = message.model_dump()
        elif isinstance(message, dict):
            # Require `message_type` for dictionary messages
            if not message_type:
                raise ValueError(
                    "message_type must be provided when message is a dictionary."
                )
            message_dict = message
        else:
            raise ValueError("Message must be a Pydantic BaseModel or a dictionary.")

        # Base metadata
        base_metadata = {
            "cloudevent.type": message_type,
            "cloudevent.source": source,
        }

        # Merge additional metadata from kwargs
        metadata = {**base_metadata, **kwargs}

        logger.debug(f"{source} preparing to publish '{message_type}' to topic '{topic_name}'.")
        logger.debug(f"Message: {message_dict}, Metadata: {metadata}")

        # Publish the message
        await self.publish_message(
            topic_name=topic_name,
            pubsub_name=pubsub_name or self.message_bus_name,
            message=message_dict,
            metadata=metadata,
        )

        logger.debug(f"{source} published '{message_type}' to topic '{topic_name}'.")