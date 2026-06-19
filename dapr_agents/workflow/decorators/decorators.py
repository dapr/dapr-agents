#
# Copyright 2026 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Callable, List, Literal, Optional, Type, TypeVar, get_type_hints

from dapr_agents.workflow.utils.core import is_supported_model
from dapr_agents.workflow.utils.routers import extract_message_models
from dapr_agents.workflow.utils.subscription import MessageContext, validate_hook
from dapr_agents.utils.logger import with_logger_context

logger = logging.getLogger(__name__)

HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]

R = TypeVar("R")


def workflow_entry(func: Callable[..., R]) -> Callable[..., R]:
    """
    Mark a method/function as the workflow entrypoint for an Agent.

    This decorator wraps the function to inject context-aware logging
    state (tracking if the workflow is replaying) and annotates the
    callable with `_is_workflow_entry = True` so AgentRunner can
    discover it on the agent instance via reflection.

    Usage:
        class MyAgent:
            @workflow_entry
            def my_workflow(
                self, ctx: DaprWorkflowContext, wf_input: dict
            ) -> str:
                ...

    Returns:
        The wrapped callable, with an identifying attribute.
    """
    wrapped_func = with_logger_context(func)
    setattr(wrapped_func, "_is_workflow_entry", True)
    return wrapped_func


def message_router(
    func: Callable[..., Any] | None = None,
    *,
    pubsub: str | None = None,
    topic: str | None = None,
    dead_letter_topic: str | None = None,
    broadcast: bool = False,
    message_model: Any | None = None,
    payload_filter: Callable[[Any, MessageContext], bool] | None = None,
    model_filter: Callable[[Any, MessageContext], bool] | None = None,
    mapper: Callable[[Any, MessageContext], Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Tag a callable as a **Pub/Sub → Workflow** entry with routing + schema metadata.

    Hooks run on the per-topic consumer thread and **block message intake** for
    that topic while they execute. Keep them cheap (in-memory checks, attribute
    comparisons, header lookups). For anything I/O-bound, do the check inside the
    workflow body and short-circuit there; otherwise a single slow hook will
    block the whole topic.

    Routing is first-match per topic: if two bindings can both accept a message,
    only the first registered one runs.

    Args:
        func (Callable[..., Any] | None):
            The function to decorate (if used without parentheses).
        pubsub (str | None):
            The name of the Dapr pub/sub component. Optional when wiring via `PubSubRouteSpec`.
        topic (str | None):
            The pub/sub topic to subscribe to. Optional when wiring via `PubSubRouteSpec`.
        dead_letter_topic (str | None):
            The dead-letter topic to publish failed messages to.
        broadcast (bool):
            Whether to treat this as a broadcast subscription.
        message_model (Any | None):
            The message model class or Union[...] to use for validation.
        payload_filter (Callable[[Any, MessageContext], bool] | None):
            Sync hook called with the raw CloudEvent `data` (a dict or scalar,
            depending on what the publisher sent) and a `MessageContext`. Runs
            *before* schema validation, so it's the right place for cheap
            metadata/source checks that don't need a parsed model. Returning
            False skips this binding; the next binding on the topic is tried.
            Raising is logged and treated the same as returning False (skip
            this binding, try the next).
            Must not mutate the inputs, the same payload object is passed
            to all bindings.
        model_filter (Callable[[Any, MessageContext], bool] | None):
            Sync hook called with the validated message model and a
            `MessageContext`. Runs *after* schema validation, so it can rely
            on typed attribute access on the parsed instance. Same skip/raise
            semantics as `payload_filter`.
            Should not mutate the model, any mutations should be done by `mapper`.
        mapper (Callable[[Any, MessageContext], Any] | None):
            Sync hook called with the validated message model and a
            `MessageContext`, returning a JSON-serializable output model.
            Runs as the last step *after* schema validation and `model_filter`,
            so it can rely on typed attribute access on the parsed instance.
            If the hook returns a non-JSON-serializable output model or raises,
            the binding is skipped and the next binding is tried.
            Mutating the input message model is acceptable,
            the output model becomes the workflow input on success.

    Returns:
        Callable[[Callable[..., Any]], Callable[..., Any]]:
            The decorated function.

    Raises:
        TypeError:
            If `message_model` cannot be resolved, or if any hook is
            an `async def` callable. Hook must be synchronous because they run
            on the consumer thread; for async I/O, push the check into the
            workflow body where the runtime is async-aware.
    """
    validate_hook(payload_filter, "payload_filter")
    validate_hook(model_filter, "model_filter")
    validate_hook(mapper, "mapper")

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
            "payload_filter": payload_filter,
            "model_filter": model_filter,
            "mapper": mapper,
        }

        setattr(f, "_is_message_handler", True)
        setattr(f, "_message_router_data", data)

        logger.debug(
            "@message_router: '%s' => models %s (topic=%s, pubsub=%s, broadcast=%s, "
            "payload_filter=%s, model_filter=%s, mapper=%s)",
            f.__name__,
            [m.__name__ for m in models],
            topic,
            pubsub,
            broadcast,
            payload_filter is not None,
            model_filter is not None,
            mapper is not None,
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
