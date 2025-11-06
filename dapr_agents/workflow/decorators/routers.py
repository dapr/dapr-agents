from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Callable, List, Literal, Optional, Type, get_type_hints

from dapr_agents.workflow.utils.core import is_supported_model
from dapr_agents.workflow.utils.routers import extract_message_models

logger = logging.getLogger(__name__)

HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]


def message_router(
    func: Optional[Callable[..., Any]] = None,
    *,
    pubsub: Optional[str] = None,
    topic: Optional[str] = None,
    dead_letter_topic: Optional[str] = None,
    broadcast: bool = False,
    message_model: Optional[Any] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Tag a callable as a **Pub/Sub → Workflow** entry with routing + schema metadata.

    Args:
        func (Optional[Callable[..., Any]]):
            The function to decorate (if used without parentheses).
        pubsub (Optional[str]):
            The name of the Dapr pub/sub component. Optional when wiring via `PubSubRouteSpec`.
        topic (Optional[str]):
            The pub/sub topic to subscribe to. Optional when wiring via `PubSubRouteSpec`.
        dead_letter_topic (Optional[str]):
            The dead-letter topic to publish failed messages to.
        broadcast (bool):
            Whether to treat this as a broadcast subscription.
        message_model (Optional[Any]):
            The message model class or Union[...] to use for validation.

    Returns:
        Callable[[Callable[..., Any]], Callable[..., Any]]:
            The decorated function.
    """

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        # Resolve message model(s)
        if message_model is None:
            # Back-compat fallback: try to infer from a `message` param if present, but not required.
            try:
                hints = get_type_hints(f, globalns=f.__globals__)
            except Exception:
                logger.debug(
                    "Failed to resolve type hints for %s", f.__name__, exc_info=True
                )
                hints = getattr(f, "__annotations__", {}) or {}
            inferred = hints.get("message")
            models = extract_message_models(inferred) if inferred else []
        else:
            models = extract_message_models(message_model)

        if not models:
            raise TypeError(
                "`@message_router` requires `message_model` (class or Union[...])."
            )

        for m in models:
            if not is_supported_model(m):
                raise TypeError(f"Unsupported model type: {m!r}")

        data = {
            "pubsub": pubsub,
            "topic": topic,
            "dead_letter_topic": dead_letter_topic
            or (f"{topic}_DEAD" if topic else None),
            "is_broadcast": broadcast,
            "message_schemas": models,
            "message_types": [m.__name__ for m in models],
        }

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

    return decorator if func is None else decorator(func)


def http_router(
    func: Optional[Callable[..., Any]] = None,
    *,
    path: Optional[str] = None,
    method: HttpMethod = "POST",
    summary: Optional[str] = None,
    tags: Optional[List[str]] = None,
    response_model: Optional[Type[Any]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Tag a callable as a **plain-HTTP** endpoint with schema metadata for its JSON body.

    Args:
        func (Optional[Callable[..., Any]]):
            The function to decorate (if used without parentheses).
        path (Optional[str]):
            The HTTP path to route to.
        method (HttpMethod):
            The HTTP method to route to.
        summary (Optional[str]):
            A short summary of the endpoint.
        tags (Optional[List[str]]):
            A list of tags for grouping endpoints.
        response_model (Optional[Type[Any]]):
            The response model class to use for validation.

    Returns:
        Callable[[Callable[..., Any]], Callable[..., Any]]:
            The decorated function.
    """

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        if path is None:
            raise ValueError("`@http_router` requires `path`.")
        method_upper = method.upper()

        try:
            hints = get_type_hints(f, globalns=f.__globals__)
        except Exception:
            logger.debug(
                "Failed to fully resolve type hints for %s", f.__name__, exc_info=True
            )
            hints = getattr(f, "__annotations__", {}) or {}

        raw_hint = hints.get("request")
        models = extract_message_models(raw_hint) if raw_hint is not None else []
        if not models:
            raise TypeError(
                "`@http_router` requires a type-hinted `request` parameter."
            )

        for m in models:
            if not is_supported_model(m):
                raise TypeError(f"Unsupported request model type: {m!r}")

        data = {
            "path": path,
            "method": method_upper,
            "summary": summary,
            "tags": (tags or []),
            "response_model": response_model,
            "request_schemas": models,
            "request_type_names": [m.__name__ for m in models],
        }
        setattr(f, "_is_http_handler", True)
        setattr(f, "_http_route_data", deepcopy(data))
        logger.debug(
            "@http_router: '%s' => models %s (%s %s)",
            f.__name__,
            [m.__name__ for m in models],
            method_upper,
            path,
        )
        return f

    return decorator if func is None else decorator(func)
