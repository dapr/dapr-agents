from __future__ import annotations

import inspect
import logging
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Optional,
    get_type_hints,
)

from dapr_agents.workflow.utils.core import is_supported_model
from dapr_agents.workflow.utils.routers import extract_message_models

logger = logging.getLogger(__name__)


def message_router(
    func: Optional[Callable[..., Any]] = None,
    *,
    pubsub: Optional[str] = None,
    topic: Optional[str] = None,
    dead_letter_topic: Optional[str] = None,
    broadcast: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorate a message handler with routing metadata.

    The handler must accept a parameter named `message`. Its type hint defines the
    expected payload model(s), e.g.:

        @message_router(pubsub="pubsub", topic="orders")
        def on_order(message: OrderCreated): ...

        @message_router(pubsub="pubsub", topic="events")
        def on_event(message: Union[Foo, Bar]): ...

    Args:
        func: (optional) bare-decorator form support.
        pubsub: Name of the Dapr pub/sub component (required when used with args).
        topic: Topic name to subscribe to (required when used with args).
        dead_letter_topic: Optional dead-letter topic (defaults to f"{topic}_DEAD").
        broadcast: Optional flag you can use downstream for fan-out semantics.

    Returns:
        The original function tagged with `_message_router_data`.
    """

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        # Validate required kwargs only when decorator is used with args
        if pubsub is None or topic is None:
            raise ValueError(
                "`pubsub` and `topic` are required when using @message_router with arguments."
            )

        sig = inspect.signature(f)
        if "message" not in sig.parameters:
            raise ValueError(f"'{f.__name__}' must have a 'message' parameter.")

        # Resolve forward refs under PEP 563 / future annotations
        try:
            hints = get_type_hints(f, globalns=f.__globals__)
        except Exception:
            logger.debug(
                "Failed to fully resolve type hints for %s", f.__name__, exc_info=True
            )
            hints = getattr(f, "__annotations__", {}) or {}

        raw_hint = hints.get("message")
        if raw_hint is None:
            raise TypeError(
                f"'{f.__name__}' must type-hint the 'message' parameter "
                "(e.g., 'message: MyModel' or 'message: Union[A, B]')"
            )

        models = extract_message_models(raw_hint)
        if not models:
            raise TypeError(
                f"Unsupported or unresolved message type for '{f.__name__}': {raw_hint!r}"
            )

        # Optional early validation of supported schema kinds
        for m in models:
            if not is_supported_model(m):
                raise TypeError(f"Unsupported model type in '{f.__name__}': {m!r}")

        data = {
            "pubsub": pubsub,
            "topic": topic,
            "dead_letter_topic": dead_letter_topic
            or (f"{topic}_DEAD" if topic else None),
            "is_broadcast": broadcast,
            "message_schemas": models,  # list[type]
            "message_types": [m.__name__ for m in models],  # list[str]
        }

        # Attach metadata; deepcopy for defensive isolation
        setattr(f, "_is_message_handler", True)
        setattr(f, "_message_router_data", deepcopy(data))

        logger.debug(
            "@message_router: '%s' => models %s (topic=%s, pubsub=%s, broadcast=%s)",
            f.__name__,
            [m.__name__ for m in models],
            topic,
            pubsub,
            broadcast,
        )
        return f

    # Support both @message_router(...) and bare @message_router usage
    return decorator if func is None else decorator(func)
