import logging
import json
from dataclasses import is_dataclass, asdict
from typing import Optional, Any, Dict, Union
from pydantic import BaseModel, Field
from dapr.aio.clients import DaprClient

logger = logging.getLogger(__name__)


class DaprPubSub(BaseModel):
    """
    Dapr-based implementation of pub/sub messaging.
    """

    message_bus_name: str = Field(
        ...,
        description="The name of the message bus component, defining the pub/sub base.",
    )

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

    async def publish_message(
        self,
        pubsub_name: str,
        topic_name: str,
        message: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
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
            json_message = await self.serialize_message(message)

            # TODO: retry publish should be configurable
            async with DaprClient() as client:
                await client.publish_event(
                    pubsub_name=pubsub_name or self.message_bus_name,
                    topic_name=topic_name,
                    data=json_message,
                    data_content_type="application/json",
                    publish_metadata=metadata or {},
                )

            logger.debug(
                f"Message successfully published to topic '{topic_name}' on pub/sub '{pubsub_name}'."
            )
            logger.debug(f"Serialized Message: {json_message}, Metadata: {metadata}")
        except Exception as e:
            logger.error(
                f"Error publishing message to topic '{topic_name}' on pub/sub '{pubsub_name}'. "
                f"Message: {message}, Metadata: {metadata}, Error: {e}"
            )
            raise Exception(
                f"Failed to publish message to topic '{topic_name}' on pub/sub '{pubsub_name}': {str(e)}"
            )

    async def publish_event_message(
        self,
        topic_name: str,
        pubsub_name: str,
        source: str,
        message: Union[BaseModel, dict, Any],
        message_type: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Publishes an event message to a specified topic with dynamic metadata.

        Args:
            topic_name (str): The topic to publish the message to.
            pubsub_name (str): The pub/sub component to use.
            source (str): The source of the message (e.g., service or agent name).
            message (Union[BaseModel, dict, dataclass, Any]): The message content, as a Pydantic model, dictionary, or dataclass instance.
            message_type (Optional[str]): The type of the message. Required if `message` is a dictionary.
            **kwargs: Additional metadata fields to include in the message.
        """
        if isinstance(message, BaseModel):
            message_type = message_type or message.__class__.__name__
            message_dict = message.model_dump()

        elif isinstance(message, dict):
            if not message_type:
                raise ValueError(
                    "message_type must be provided when message is a dictionary."
                )
            message_dict = message

        elif is_dataclass(message):
            message_type = message_type or message.__class__.__name__
            message_dict = asdict(message)

        else:
            raise ValueError(
                "Message must be a Pydantic BaseModel, a dictionary, or a dataclass instance."
            )

        metadata = {
            "cloudevent.type": message_type,
            "cloudevent.source": source,
        }
        metadata.update(kwargs)

        logger.debug(
            f"{source} preparing to publish '{message_type}' to topic '{topic_name}'."
        )
        logger.debug(f"Message: {message_dict}, Metadata: {metadata}")

        await self.publish_message(
            topic_name=topic_name,
            pubsub_name=pubsub_name or self.message_bus_name,
            message=message_dict,
            metadata=metadata,
        )

        logger.info(f"{source} published '{message_type}' to topic '{topic_name}'.")
