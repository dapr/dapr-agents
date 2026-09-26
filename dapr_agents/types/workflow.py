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

import math
from enum import Enum
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Type, Union

if TYPE_CHECKING:
    from dapr_agents.workflow.utils.subscription import DedupeBackend, MessageContext


class DaprWorkflowStatus(str, Enum):
    """Enumeration of possible workflow statuses for standardized tracking."""

    UNKNOWN = "unknown"  # Workflow is in an undefined state
    RUNNING = "running"  # Workflow is actively running
    COMPLETED = "completed"  # Workflow has completed
    FAILED = "failed"  # Workflow encountered an error
    TERMINATED = "terminated"  # Workflow was canceled or forcefully terminated
    SUSPENDED = "suspended"  # Workflow was temporarily paused
    PENDING = "pending"  # Workflow is waiting to start


@dataclass
class PubSubRouteSpec:
    """
    Pub/sub subscription that schedules a workflow when a message arrives.

    Attributes:
        pubsub_name: Dapr pub/sub component name.
        topic: Topic to subscribe to.
        handler_fn: Bound workflow callable to run (method or function).
        message_model: Optional schema (Pydantic/dataclass/dict). If omitted and
            `handler_fn` is decorated with `@message_router`, the decorator's
            first schema is used; otherwise `dict`.
        dead_letter_topic: Optional DLQ topic name.
        payload_filter: Optional sync callable `(payload, MessageContext) -> bool`
            run before schema validation. If omitted, falls back to the value on
            `handler_fn`'s `@message_router` decorator (if any).
        model_filter: Optional sync callable `(model, MessageContext) -> bool` run
            after schema validation. If omitted, falls back to the value on
            `handler_fn`'s `@message_router` decorator (if any).
        mapper: Optional sync callable `(model, MessageContext) -> Any` run
            after schema validation and filters. If omitted, falls back to the value on
            `handler_fn`'s `@message_router` decorator (if any).
    """

    pubsub_name: str
    topic: str
    handler_fn: Callable[..., Any]
    message_model: type[Any] | None = None
    dead_letter_topic: str | None = None
    payload_filter: Callable[[Any, MessageContext], bool] | None = None
    model_filter: Callable[[Any, MessageContext], bool] | None = None
    mapper: Callable[[Any, MessageContext], Any] | None = None


@dataclass
class HttpRouteSpec:
    """
    HTTP endpoint that schedules a workflow when a request arrives.

    Attributes:
        path: FastAPI path to mount (e.g., "/blog/start").
        handler_fn: Bound workflow callable to run (method or function).
        method: HTTP method (default: POST).
        request_model: Optional schema for request validation. If omitted and
            `handler_fn` is decorated with `@http_router`, the decorator's first
            schema is used; otherwise `dict`.
        summary: Optional OpenAPI summary.
        tags: Optional OpenAPI tags.
        response_model: Optional Pydantic response model for docs.
    """

    path: str
    handler_fn: Callable[..., Any]
    method: str = "POST"
    request_model: Optional[Type[Any]] = None
    summary: Optional[str] = None
    tags: Optional[List[str]] = None
    response_model: Optional[Type[Any]] = None


FieldResolver = Union[str, Callable[[Any, "MessageContext"], Any]]
"""Either a dotted field path into the validated message (e.g. ``"job.workflow_id"``)
or a sync callable ``(validated_message, MessageContext) -> value``."""


@dataclass(frozen=True)
class NotFoundRetryPolicy:
    """Bounded retry for messages whose target workflow instance does not exist (yet).

    The message is retried (RETRY) until either limit is reached; then it is
    treated like a terminal workflow (dead-lettered if the route has a
    ``dead_letter_topic``, otherwise dropped with a WARNING).

    Attempts are counted in memory, per process and per route. With several
    replicas the real number of deliveries can be up to replicas x
    ``max_attempts``; each replica is still bounded by ``window_seconds``. How
    quickly redeliveries arrive depends on the broker's redelivery settings.

    Attributes:
        max_attempts: Deliveries that may observe "not found" before giving up,
            counting the current one. ``1`` means never retry. Default 10.
        window_seconds: Wall-clock budget measured from the first "not found"
            delivery of this message. Default 300.0.
    """

    max_attempts: int = 10
    window_seconds: float = 300.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError(
                f"max_attempts must be an int >= 1, got {self.max_attempts!r}."
            )
        if (
            isinstance(self.window_seconds, bool)
            or not isinstance(self.window_seconds, Real)
            or not math.isfinite(float(self.window_seconds))
            or self.window_seconds <= 0
        ):
            raise ValueError(
                f"window_seconds must be a finite number > 0, got {self.window_seconds!r}."
            )


@dataclass(frozen=True)
class WorkflowEventRouteSpec:
    """Pub/sub subscription that raises an external event on an existing workflow instance.

    Each validated message becomes one
    ``DaprWorkflowClient.raise_workflow_event(instance_id, event_name, data=...)``
    call. The workflow consumes it with ``ctx.wait_for_external_event(event_name)``.

    Dapr buffers an event raised while nothing waits and hands it to the next
    wait with the same (case-insensitive) name, so event names must be unique
    per wait. Deduplication is on by default so a redelivered message cannot
    satisfy a later wait.

    Security: anyone who can publish to ``topic`` can signal any workflow
    instance whose id they can guess or learn. Restrict publishers with Dapr
    pub/sub topic scoping.

    Attributes:
        pubsub_name: Dapr pub/sub component name.
        topic: Topic to subscribe to. A topic that carries an event route
            carries nothing else.
        event_name: Event name to raise. Used when ``event_name_from`` is None.
        instance_id_from: Resolver for the target workflow instance id (required).
        event_name_from: Optional resolver that overrides ``event_name`` per message.
        data_from: Optional resolver for the event payload. Default: the validated
            message serialized the way workflow inputs are serialized (CloudEvent
            metadata included).
        message_model: Schema (Pydantic / dataclass / dict, or ``Union[...]``).
            Default ``dict``.
        dead_letter_topic: Optional dead-letter topic.
        payload_filter: Same contract as ``PubSubRouteSpec.payload_filter``.
        model_filter: Same contract as ``PubSubRouteSpec.model_filter``.
        dedupe: Deduplicate redeliveries by CloudEvent id. Default True.
        deduper: Optional backend for this route (e.g. a shared store when
            running several replicas). Must be None when ``dedupe=False``.
        not_found_retry: Policy for instances that do not exist yet.
        name: Route name for logs and ``MessageContext.handler_name``.
            Default: ``f"{event_name}@{pubsub_name}:{topic}"``.
    """

    pubsub_name: str
    topic: str
    event_name: str
    instance_id_from: FieldResolver
    event_name_from: FieldResolver | None = None
    data_from: FieldResolver | None = None
    message_model: Any | None = None
    dead_letter_topic: str | None = None
    payload_filter: Callable[[Any, MessageContext], bool] | None = None
    model_filter: Callable[[Any, MessageContext], bool] | None = None
    dedupe: bool = True
    deduper: DedupeBackend | None = None
    not_found_retry: NotFoundRetryPolicy = field(default_factory=NotFoundRetryPolicy)
    name: str | None = None
