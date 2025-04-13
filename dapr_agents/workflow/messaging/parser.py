import logging
from typing import Any, Tuple, Type, Union, Optional
from dataclasses import is_dataclass
from dapr.common.pubsub.subscription import SubscriptionMessage
from dapr_agents.types.message import EventMessageMetadata
from dapr_agents.workflow.messaging.utils import is_supported_model, is_pydantic_model

logger = logging.getLogger(__name__)


def extract_cloudevent_data(
    message: Union[SubscriptionMessage, dict],
) -> Tuple[dict, dict]:
    """
    Extracts CloudEvent metadata and raw payload data from a SubscriptionMessage or dict.

    Args:
        message (Union[SubscriptionMessage, dict]): The raw message received from pub/sub.

    Returns:
        Tuple[dict, dict]: (event_data, metadata) where event_data is the message payload, and
                           metadata is the parsed CloudEvent metadata as a dictionary.

    Raises:
        ValueError: If message type is unsupported.
    """
    if isinstance(message, SubscriptionMessage):
        metadata = EventMessageMetadata(
            id=message.id(),
            datacontenttype=message.data_content_type(),
            pubsubname=message.pubsub_name(),
            source=message.source(),
            specversion=message.spec_version(),
            time=None,
            topic=message.topic(),
            traceid=None,
            traceparent=None,
            type=message.type(),
            tracestate=None,
            headers=message.extensions(),
        ).model_dump()
        event_data = message.data()

    elif isinstance(message, dict):
        metadata = EventMessageMetadata(
            id=message.get("id"),
            datacontenttype=message.get("datacontenttype"),
            pubsubname=message.get("pubsubname"),
            source=message.get("source"),
            specversion=message.get("specversion"),
            time=message.get("time"),
            topic=message.get("topic"),
            traceid=message.get("traceid"),
            traceparent=message.get("traceparent"),
            type=message.get("type"),
            tracestate=message.get("tracestate"),
            headers=message.get("extensions", {}),
        ).model_dump()
        event_data = message.get("data", {})

    else:
        raise ValueError(f"Unexpected message type: {type(message)}")

    return event_data, metadata


def validate_message_model(model: Type[Any], event_data: dict) -> Any:
    """
    Validates and parses event data against the provided message model.

    Args:
        model (Type[Any]): The message model class.
        event_data (dict): The raw event payload data.

    Returns:
        Any: An instance of the message model (or raw dict if `model` is `dict`).

    Raises:
        TypeError: If the model is not supported.
        ValueError: If model validation fails.
    """
    if not is_supported_model(model):
        raise TypeError(f"Unsupported model type: {model}")

    try:
        logger.info(f"Validating payload with model '{model.__name__}'...")

        if model is dict:
            return event_data
        elif is_dataclass(model):
            return model(**event_data)
        elif is_pydantic_model(model):
            return model.model_validate(event_data).model_dump()

    except Exception as e:
        logger.error(f"Message validation failed for model '{model.__name__}': {e}")
        raise ValueError(f"Message validation failed: {e}")


def parse_cloudevent(
    message: Union[SubscriptionMessage, dict], model: Optional[Type[Any]] = None
) -> Tuple[Any, dict]:
    """
    Parses and validates a CloudEvent from a SubscriptionMessage or dict.

    This combines both metadata extraction and message model validation for direct use.

    Args:
        message (Union[SubscriptionMessage, dict]): The incoming pub/sub message.
        model (Optional[Type[Any]]): The schema used to validate the message body.

    Returns:
        Tuple[Any, dict]: The validated message (or raw dict) and its metadata.

    Raises:
        ValueError: If metadata or validation fails.
    """
    try:
        event_data, metadata = extract_cloudevent_data(message)

        if model is None:
            raise ValueError("Message validation failed: No model provided.")

        validated_message = validate_message_model(model, event_data)

        logger.info("Message successfully parsed and validated")
        logger.debug(f"Data: {validated_message}")
        logger.debug(f"metadata: {metadata}")

        return validated_message, metadata

    except Exception as e:
        logger.error(f"Failed to parse CloudEvent: {e}", exc_info=True)
        raise ValueError(f"Invalid CloudEvent: {str(e)}")
