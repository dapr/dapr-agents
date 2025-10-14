from __future__ import annotations

import json
import logging
import types
from dataclasses import is_dataclass
from types import NoneType
from typing import Any, Optional, Tuple, Type, Union, get_args, get_origin

from dapr.common.pubsub.subscription import SubscriptionMessage

from dapr_agents.types.message import EventMessageMetadata
from dapr_agents.workflow.utils.core import is_pydantic_model, is_supported_model

logger = logging.getLogger(__name__)


def extract_message_models(type_hint: Any) -> list[type]:
    """Normalize a message type hint into a concrete list of classes.

    Supports:
      - Single class: `MyMessage` → `[MyMessage]`
      - Union: `Union[Foo, Bar]` or `Foo | Bar` → `[Foo, Bar]`
      - Optional: `Optional[Foo]` (i.e., `Union[Foo, None]`) → `[Foo]`

    Notes:
      - Forward refs should be resolved by the caller (e.g., via `typing.get_type_hints`).
      - Non-class entries (e.g., `None`, `typing.Any`) are filtered out.
      - Returns an empty list when the hint isn't a usable class or union of classes.
    """
    if type_hint is None:
        return []

    origin = get_origin(type_hint)
    if origin in (Union, types.UnionType):  # handle both `Union[...]` and `A | B`
        return [
            t for t in get_args(type_hint) if t is not NoneType and isinstance(t, type)
        ]

    return [type_hint] if isinstance(type_hint, type) else []


def _maybe_json_loads(payload: Any, content_type: Optional[str]) -> Any:
    """
    Best-effort JSON parsing based on content type and payload shape.

    - If payload is `dict`/`list` → return as-is.
    - If bytes/str and content-type hints JSON (or text looks like JSON) → parse to Python.
    - Otherwise → return the original payload.

    This helper is intentionally forgiving; callers should validate downstream.
    """
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


def extract_cloudevent_data(
    message: Union[SubscriptionMessage, dict, bytes, str],
) -> Tuple[dict, dict]:
    """
    Extract CloudEvent metadata and payload (attempting JSON parsing when appropriate).

    Accepts:
      - `SubscriptionMessage` (Dapr SDK)
      - `dict` (raw CloudEvent envelope)
      - `bytes`/`str` (data-only; metadata is synthesized)

    Returns:
        (event_data, metadata) as dictionaries. `event_data` may be non-dict JSON
        (e.g., list) if the payload is an array; callers expecting dicts should handle it.

    Raises:
        ValueError: For unsupported `message` types.
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
            time=None,  # not always populated by SDK
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
        # No CloudEvent envelope; treat payload as data-only and synthesize minimal metadata.
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
            "Event data is not a dict (type=%s); value=%r", type(event_data), event_data
        )

    return event_data, metadata


def validate_message_model(model: Type[Any], event_data: dict) -> Any:
    """
    Validate and coerce `event_data` into `model`.

    Supports:
      - dict: returns `event_data` unchanged
      - dataclass: constructs the dataclass
      - Pydantic v2 model: uses `model_validate`

    Raises:
        TypeError: If the model is not a supported kind.
        ValueError: If validation/construction fails.
    """
    if not is_supported_model(model):
        raise TypeError(f"Unsupported model type: {model!r}")

    try:
        logger.info(f"Validating payload with model '{model.__name__}'...")

        if model is dict:
            return event_data
        if is_dataclass(model):
            return model(**event_data)
        if is_pydantic_model(model):
            return model.model_validate(event_data)
        raise TypeError(f"Unsupported model type: {model!r}")
    except Exception as e:
        logger.error(f"Message validation failed for model '{model.__name__}': {e}")
        raise ValueError(f"Message validation failed: {e}")


def parse_cloudevent(
    message: Union[SubscriptionMessage, dict, bytes, str],
    model: Optional[Type[Any]] = None,
) -> Tuple[Any, dict]:
    """
    Parse a CloudEvent-like input and validate its payload against ``model``.

    Args:
        message (Union[SubscriptionMessage, dict, bytes, str]): Incoming message; can be a Dapr ``SubscriptionMessage``, a raw
                 CloudEvent ``dict``, or bare ``bytes``/``str`` payloads.
        model (Optional[Type[Any]]):   Schema for payload validation (required).

    Returns:
        Tuple[Any, dict]: A tuple containing the validated message and its metadata.

    Raises:
        ValueError: If no model is provided or validation fails.
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
