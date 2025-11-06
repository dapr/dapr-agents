from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Union

from dapr.aio.clients import DaprClient
from pydantic import BaseModel

logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]
AgentsMetadata = Mapping[str, Mapping[str, Any]]


async def serialize_message(message: Any) -> str:
    """
    Serialize an arbitrary message payload into a JSON string.

    Args:
        message: The message payload to serialize. Expected to be JSON serializable.

    Returns:
        A JSON string representation of the payload (empty object for ``None``).

    Raises:
        ValueError: If the payload cannot be serialized to JSON.
    """
    try:
        return json.dumps(message if message is not None else {})
    except TypeError as exc:  # pragma: no cover - mirrors previous behaviour
        logger.error("Failed to serialize message %r: %s", message, exc)
        raise ValueError(f"Message contains non-serializable data: {exc}") from exc


async def publish_message(
    *,
    pubsub_name: str,
    topic_name: str,
    message: Any,
    metadata: Optional[JsonDict] = None,
    default_pubsub: Optional[str] = None,
    client_factory: Callable[[], DaprClient] = DaprClient,
    logger_: logging.Logger = logger,
) -> None:
    """
    Publish a raw JSON-serializable payload to a Dapr pub/sub topic.

    Args:
        pubsub_name: Pub/Sub component to target. Falls back to ``default_pubsub`` if falsy.
        topic_name: Destination topic.
        message: Payload to publish (will be serialized via :func:`serialize_message`).
        metadata: Optional CloudEvent metadata.
        default_pubsub: Component used when ``pubsub_name`` is falsy.
        client_factory: Callable returning an async Dapr client (primarily for testing).
        logger_: Logger used for diagnostic output.
    """
    json_body = await serialize_message(message)
    target_pubsub = pubsub_name or default_pubsub
    if not target_pubsub:
        raise ValueError("pubsub_name or default_pubsub must be provided.")

    meta = metadata or {}

    try:
        async with client_factory() as client:
            await client.publish_event(
                pubsub_name=target_pubsub,
                topic_name=topic_name,
                data=json_body,
                data_content_type="application/json",
                publish_metadata=meta,
            )
        logger_.debug(
            "Published message to pubsub=%s topic=%s metadata=%s payload=%s",
            target_pubsub,
            topic_name,
            meta,
            json_body,
        )
    except Exception as exc:  # pragma: no cover - network failures
        logger_.error(
            "Error publishing message to pubsub=%s topic=%s: %s",
            target_pubsub,
            topic_name,
            exc,
            exc_info=True,
        )
        raise


async def publish_event_message(
    *,
    topic_name: str,
    pubsub_name: str,
    source: str,
    message: Union[BaseModel, JsonDict, Any],
    message_type: Optional[str] = None,
    metadata: Optional[JsonDict] = None,
    default_pubsub: Optional[str] = None,
    client_factory: Callable[[], DaprClient] = DaprClient,
    logger_: logging.Logger = logger,
) -> None:
    """
    Publish a CloudEvent-style payload to a topic with convenient schema support.

    Args:
        topic_name: Destination topic.
        pubsub_name: Pub/Sub component to use.
        source: Logical message source (used for CloudEvent metadata).
        message: Payload as Pydantic model, dataclass, dict, or JSON string.
        message_type: Optional CloudEvent type override.
        metadata: Additional metadata entries merged with CloudEvent defaults.
        default_pubsub: Component to use when ``pubsub_name`` is falsy.
        client_factory: Callable returning an async Dapr client.
        logger_: Logger used for diagnostics.

    Raises:
        ValueError: For unsupported payload types or missing ``message_type`` on dict payloads.
    """
    if isinstance(message, BaseModel):
        message_type = message_type or message.__class__.__name__
        payload = message.model_dump()
    elif isinstance(message, dict):
        if not message_type:
            raise ValueError(
                "message_type must be provided when message is a dictionary."
            )
        payload = message
    elif is_dataclass(message):
        message_type = message_type or message.__class__.__name__
        payload = asdict(message)
    else:
        raise ValueError(
            "Message must be a Pydantic BaseModel, dataclass, or dictionary.",
        )

    combined_metadata: MutableMapping[str, Any] = {
        "cloudevent.type": message_type,
        "cloudevent.source": source,
    }
    if metadata:
        combined_metadata.update(metadata)

    logger_.debug(
        "%s publishing event type=%s to topic=%s metadata=%s",
        source,
        message_type,
        topic_name,
        dict(combined_metadata),
    )

    await publish_message(
        pubsub_name=pubsub_name,
        topic_name=topic_name,
        message=payload,
        metadata=dict(combined_metadata),
        default_pubsub=default_pubsub,
        client_factory=client_factory,
        logger_=logger_,
    )

    logger_.info("%s published '%s' to topic '%s'.", source, message_type, topic_name)


async def broadcast_message(
    *,
    message: Union[BaseModel, JsonDict],
    broadcast_topic: Optional[str],
    message_bus: str,
    source: str,
    agents_metadata: AgentsMetadata,
    exclude_orchestrator: bool = False,
    metadata: Optional[JsonDict] = None,
    client_factory: Callable[[], DaprClient] = DaprClient,
    logger_: logging.Logger = logger,
) -> None:
    """
    Broadcast a message to every agent in the supplied metadata mapping.

    Args:
        message: Payload to publish (Pydantic model or dict).
        broadcast_topic: Topic used for team broadcasts; if falsy the call is ignored.
        message_bus: Default pub/sub component for broadcasts.
        source: Emitting agent/service name.
        agents_metadata: Mapping of agent name -> metadata (requires ``topic_name`` & ``pubsub_name``).
        exclude_orchestrator: Skip agents flagged with ``orchestrator=True``.
        metadata: Additional CloudEvent metadata.
        client_factory: Callable returning an async Dapr client.
        logger_: Logger used for diagnostics.
    """
    if not broadcast_topic:
        logger_.info("%s has no broadcast topic; skipping broadcast.", source)
        return

    recipients = {
        name: meta
        for name, meta in agents_metadata.items()
        if not (exclude_orchestrator and meta.get("orchestrator"))
    }
    if not recipients:
        logger_.warning("No agents available for broadcast from %s.", source)
        return

    await publish_event_message(
        topic_name=broadcast_topic,
        pubsub_name=message_bus,
        source=source,
        message=message,
        metadata=metadata,
        default_pubsub=message_bus,
        client_factory=client_factory,
        logger_=logger_,
    )
    logger_.debug("%s broadcasted message to %d agents.", source, len(recipients))


async def send_message_to_agent(
    *,
    target_agent: str,
    message: Union[BaseModel, JsonDict],
    agents_metadata: AgentsMetadata,
    source: str,
    metadata: Optional[JsonDict] = None,
    client_factory: Callable[[], DaprClient] = DaprClient,
    logger_: logging.Logger = logger,
) -> None:
    """
    Send a direct message to a single agent using its registry metadata.

    Args:
        target_agent: Logical agent name to address.
        message: Payload as Pydantic model or dict.
        agents_metadata: Mapping of agent metadata (must include ``topic_name`` & ``pubsub_name``).
        source: Name of the sender (used in CloudEvent metadata).
        metadata: Additional CloudEvent metadata.
        client_factory: Callable returning an async Dapr client.
        logger_: Logger used for diagnostics.
    """
    meta = agents_metadata.get(target_agent)
    if not meta:
        logger_.warning(
            "Target '%s' is not registered; skipping message.", target_agent
        )
        return

    topic = meta.get("topic_name")
    pubsub_name = meta.get("pubsub_name")
    if not topic or not pubsub_name:
        logger_.warning(
            "Agent '%s' metadata missing topic_name/pubsub_name; skipping message.",
            target_agent,
        )
        return

    await publish_event_message(
        topic_name=str(topic),
        pubsub_name=str(pubsub_name),
        source=source,
        message=message,
        metadata=metadata,
        default_pubsub=str(pubsub_name),
        client_factory=client_factory,
        logger_=logger_,
    )
    logger_.debug("Sent message from %s to agent %s.", source, target_agent)
