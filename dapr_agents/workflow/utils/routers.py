from __future__ import annotations

import json
import logging
import types
from dataclasses import is_dataclass
from types import NoneType
from typing import Any, Optional, Tuple, Type, Union, get_args, get_origin

from dapr.common.pubsub.subscription import SubscriptionMessage

from dapr_agents.types.message import EventMessageMetadata
from dapr_agents.workflow.utils.core import is_supported_model

logger = logging.getLogger(__name__)


def extract_message_models(type_hint: Any) -> list[type]:
    """
    Turn a single class or a Union[...] into a list of concrete classes (filters None/Any).

    Args:
        type_hint (Any):
            The type hint to extract classes from.

    Returns:
        list[type]: A list of concrete classes extracted from the type hint.
    """
    if type_hint is None:
        return []
    origin = get_origin(type_hint)
    if origin in (Union, types.UnionType):
        return [
            t for t in get_args(type_hint) if t is not NoneType and isinstance(t, type)
        ]
    return [type_hint] if isinstance(type_hint, type) else []


def _maybe_json_loads(payload: Any, content_type: Optional[str]) -> Any:
    """Best-effort: parse JSON by content-type hint or shape; otherwise return original value."""
    try:
        if isinstance(payload, (dict, list)):
            return payload

        ct = (content_type or "").lower()
        looks_json = "json" in ct

        if isinstance(payload, bytes):
            text = payload.decode("utf-8", errors="strict")
            if looks_json or (text and text[0] in "{["):
                return json.loads(text)
            return text

        if isinstance(payload, str):
            if looks_json or (payload and payload[0] in "{["):
                return json.loads(payload)
            return payload

        return payload
    except Exception:
        logger.debug("JSON parsing failed; returning raw payload", exc_info=True)
        return payload


def _maybe_json_body(body: Any) -> Any:
    """HTTP helper: parse str/bytes into JSON once; otherwise return as-is."""
    if isinstance(body, (bytes, str)):
        try:
            return json.loads(body)
        except Exception:
            return body
    return body


def validate_message_model(model: Type[Any], event_data: dict) -> Any:
    """
    Validate/coerce event_data into model (dict, dataclass, Pydantic v1/v2).

    Args:
        model (Type[Any]):
            The model class to validate against.
        event_data (dict):
            The event data to validate.

    Returns:
        Any: The validated/coerced message instance.
    """
    if not is_supported_model(model):
        raise TypeError(f"Unsupported model type: {model!r}")

    try:
        logger.info("Validating payload with model '%s'...", model.__name__)

        if model is dict:
            return event_data

        if is_dataclass(model):
            return model(**event_data)

        if hasattr(model, "model_validate"):  # Pydantic v2
            return model.model_validate(event_data)

        if hasattr(model, "parse_obj"):  # Pydantic v1
            return model.parse_obj(event_data)

        raise TypeError(f"Unsupported model type: {model!r}")

    except Exception as e:
        logger.error("Message validation failed for model '%s': %s", model.__name__, e)
        raise ValueError(f"Message validation failed: {e}") from e


def extract_cloudevent_data(
    message: Union[SubscriptionMessage, dict, bytes, str],
) -> Tuple[Any, dict]:
    """
    Extract CloudEvent .data and metadata from Dapr SubscriptionMessage or similar shapes.

    Args:
        message (Union[SubscriptionMessage, dict, bytes, str]):
            The incoming CloudEvent message from Dapr pub/sub.

    Returns:
        Tuple[Any, dict]: A tuple containing the event data and its metadata.
    """
    if isinstance(message, SubscriptionMessage):
        content_type = message.data_content_type()
        raw = message.data()
        event_data = _maybe_json_loads(raw, content_type)
        metadata = EventMessageMetadata(
            id=message.id(),
            datacontenttype=content_type,
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

    elif isinstance(message, dict):
        content_type = message.get("datacontenttype")
        raw = message.get("data", {})
        event_data = _maybe_json_loads(raw, content_type)
        metadata = EventMessageMetadata(
            id=message.get("id"),
            datacontenttype=content_type,
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

    elif isinstance(message, (bytes, str)):
        content_type = "application/json"
        event_data = _maybe_json_loads(message, content_type)
        metadata = EventMessageMetadata(
            id=None,
            datacontenttype=content_type,
            pubsubname=None,
            source=None,
            specversion=None,
            time=None,
            topic=None,
            traceid=None,
            traceparent=None,
            type=None,
            tracestate=None,
            headers={},
        ).model_dump()

    else:
        raise ValueError(f"Unexpected message type: {type(message)!r}")

    if not isinstance(event_data, dict):
        logger.debug(
            "CloudEvent data is not a dict (type=%s); value=%r",
            type(event_data),
            event_data,
        )

    return event_data, metadata


def parse_cloudevent(
    message: Union[SubscriptionMessage, dict, bytes, str],
    model: Optional[Type[Any]] = None,
) -> Tuple[Any, dict]:
    """
    Parse a pub/sub CloudEvent and validate its `.data` against model.

    Args:
        message (Union[SubscriptionMessage, dict, bytes, str]):
            The incoming CloudEvent message from Dapr pub/sub.
        model (Optional[Type[Any]], optional):
            The model class to validate the event data against. Defaults to None.

    Returns:
        Tuple[Any, dict]: A tuple containing the validated message and its metadata.
    """
    try:
        if model is None:
            raise ValueError("Message validation failed: No model provided.")

        event_data, metadata = extract_cloudevent_data(message)
        if not isinstance(event_data, dict):
            event_data = {"data": event_data}

        validated_message = validate_message_model(model, event_data)
        logger.info("CloudEvent successfully parsed and validated")
        logger.debug("Data: %r", validated_message)
        logger.debug("metadata: %r", metadata)
        return validated_message, metadata

    except Exception as e:
        logger.error("Failed to parse CloudEvent: %s", e, exc_info=True)
        raise ValueError(f"Invalid CloudEvent: {str(e)}") from e


def parse_http_json(
    body: Any,
    model: Optional[Type[Any]] = None,
    *,
    attach_metadata: bool = False,
) -> Tuple[Any, dict]:
    """
    Parse a plain JSON HTTP body and validate against model (no CloudEvent semantics).

    Args:
        body (Any):
            The incoming HTTP request body.
        model (Optional[Type[Any]], optional):
            The model class to validate the body against. Defaults to None.
        attach_metadata (bool, optional):
            Whether to attach empty metadata dict. Defaults to False.

    Returns:
        Tuple[Any, dict]: A tuple containing the validated message and its metadata.
    """
    if model is None:
        raise ValueError("Message validation failed: No model provided.")

    payload = _maybe_json_body(body)
    if isinstance(payload, dict):
        event_data = payload
    else:
        event_data = {"data": payload}

    validated = validate_message_model(model, event_data)
    metadata: dict = {} if attach_metadata else {}
    logger.info("HTTP JSON successfully parsed and validated (no CloudEvent semantics)")
    logger.debug("Data: %r", validated)
    return validated, metadata
