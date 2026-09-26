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

NOT_FOUND_MAX_ATTEMPTS_LIMIT = 100
NOT_FOUND_WINDOW_SECONDS_LIMIT = 3600.0
DEFAULT_EVENT_MAX_DATA_BYTES = 1_048_576
DEFAULT_EVENT_CALL_TIMEOUT_SECONDS = 30.0
DEFAULT_EVENT_DEDUPE_MAX_ENTRIES = 65_536
DEFAULT_EVENT_HOOK_TIMEOUT_SECONDS = 5.0


def _check_int_range(value: Any, name: str, *, upper: int | None = None) -> None:
    """Raise ValueError unless ``value`` is an int in ``[1, upper]``."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or (upper is not None and value > upper)
    ):
        bound = f" and <= {upper}" if upper is not None else ""
        raise ValueError(f"{name} must be an int >= 1{bound}, got {value!r}.")


def _check_seconds(value: Any, name: str, *, upper: float | None = None) -> None:
    """Raise ValueError unless ``value`` is a finite number in ``(0, upper]``."""
    number = float(value) if isinstance(value, Real) else math.nan
    if (
        isinstance(value, bool)
        or not math.isfinite(number)
        or number <= 0
        or (upper is not None and number > upper)
    ):
        bound = f" and <= {upper:g}" if upper is not None else ""
        raise ValueError(f"{name} must be a finite number > 0{bound}, got {value!r}.")


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
    Counts are kept for at most 4096 messages per route; beyond that the least
    recently used counts are evicted and restart, so the limits hold only
    below that bound.

    Dead-letter topics: on a route with a ``dead_letter_topic``, daprd sends a
    RETRY to the dead-letter topic as soon as the pub/sub component's inbound
    resiliency retry policy is used up, and at once when there is none. Pair
    such a route with a component inbound retry policy for this policy to take
    effect.

    Frozen on purpose: value equality is used to detect an idempotent
    re-registration of the same route.

    Attributes:
        max_attempts: Deliveries that may observe "not found" before giving up,
            counting the current one. ``1`` means never retry. Default 10,
            at most 100.
        window_seconds: Wall-clock budget measured from the first "not found"
            delivery of this message. Default 300.0, at most 3600.
    """

    max_attempts: int = 10
    window_seconds: float = 300.0

    def __post_init__(self) -> None:
        _check_int_range(
            self.max_attempts, "max_attempts", upper=NOT_FOUND_MAX_ATTEMPTS_LIMIT
        )
        _check_seconds(
            self.window_seconds, "window_seconds", upper=NOT_FOUND_WINDOW_SECONDS_LIMIT
        )


@dataclass(frozen=True)
class WorkflowEventTarget:
    """The workflow instance and event a message resolved to, handed to ``authorize``.

    Attributes:
        instance_id: Resolved target workflow instance id.
        event_name: Resolved event name.
    """

    instance_id: str
    event_name: str


@dataclass(frozen=True)
class WorkflowEventRouteSpec:
    """Pub/sub subscription that raises an external event on an existing workflow instance.

    Each validated message becomes one
    ``DaprWorkflowClient.raise_workflow_event(instance_id, event_name, data=...)``
    call. The workflow consumes it with ``ctx.wait_for_external_event(event_name)``.

    Dapr buffers an event raised while nothing waits and hands it to the next
    wait with the same (case-insensitive) name, so event names must be unique
    per wait. Deduplication is on by default to stop a redelivered message
    from satisfying a later wait. It is best-effort: keyed by CloudEvent id (or
    a SHA-256 of the canonical JSON payload when there is none), in process
    memory by default and bounded by TTL and size (see ``deduper`` and
    ``dedupe_max_entries``). A crash after the raise but before the ack, or a
    redelivery to another replica without a shared ``deduper``, can still
    raise the event twice. Publishers must use unique CloudEvent ids: the
    duplicate check runs before any resolver, filter or ``authorize``, so a
    message that reuses an id already seen is treated as a duplicate,
    acknowledged and never evaluated.

    Evaluation order: schema validation and the filters, then the resolvers
    and limits, then ``authorize``, then the sidecar calls (state check, raise).

    Outcomes, in that order: an unresolvable or oversized message, an
    ``authorize`` denial, a terminal workflow (COMPLETED / FAILED /
    TERMINATED), a used-up ``not_found_retry`` budget or a permanent sidecar
    error (gRPC ``INVALID_ARGUMENT``, ``PERMISSION_DENIED``,
    ``UNAUTHENTICATED``, ``UNIMPLEMENTED``, ``OUT_OF_RANGE`` or
    ``FAILED_PRECONDITION``) is dropped; daprd dead-letters it when
    ``dead_letter_topic`` is set, otherwise it is logged at WARNING and
    discarded. Terminal states are never retried. Other sidecar errors and
    timeouts are retried. If the workflow finishes between the state check and
    the raise, the runtime discards the event and the message is still
    acknowledged.

    Security: anyone who can publish to ``topic`` can signal any workflow
    instance whose id they can guess or learn. Restrict publishers with Dapr
    pub/sub topic scoping, and use ``authorize`` to decide per message whether
    it may signal the resolved workflow. The CloudEvent ``source`` (and the
    other CloudEvent attributes) are set by the publisher, so a check on them
    is only as strong as topic scoping and pub/sub access control; prefer
    checks tied to the resolved target. Event names the SDK itself waits on
    (``approval_response_*`` and ``user_input_response:*``, compared with
    Unicode case folding as Dapr does) are rejected unless
    ``allow_reserved_event_names`` is True: a static ``event_name`` at
    registration, a name from ``event_name_from`` by dropping the message.
    Resolved instance ids and event names longer than 512 characters (counted
    as characters, not bytes), and payloads over ``max_data_bytes``, are
    dropped.

    Hooks: ``authorize``, ``payload_filter`` and ``model_filter`` run with the
    ``hook_timeout_seconds`` deadline and must return exactly ``True`` to let
    the message through; ``False``, any other value (``"False"``, ``1``), an
    exception or a timeout rejects it. Hooks fail closed by design: a timeout
    or exception drops the message even when its cause was transient. Hooks
    and sidecar calls run on two thread pools shared by every event-route topic
    of the subscriber. Resolvers (``instance_id_from``,
    ``event_name_from``, ``data_from``) and payload serialization run on the
    consumer thread without a deadline, so keep them cheap.

    Frozen on purpose: value equality is used to detect an idempotent
    re-registration of the same route.

    Example::

        class JobFinished(BaseModel):
            job: JobRef  # JobRef has a ``workflow_id`` field
            status: str

        def job_event_name(msg: JobFinished, ctx: MessageContext) -> str:
            return f"job_finished_{msg.status}"

        def authorize(
            msg: JobFinished, ctx: MessageContext, target: WorkflowEventTarget
        ) -> bool:
            # Tied to the resolved target, not to publisher-supplied CloudEvent
            # attributes such as ctx.event.source.
            return target.instance_id.startswith("job-")

        spec = WorkflowEventRouteSpec(
            pubsub_name="messagepubsub",
            topic="jobs.finished",
            event_name="job_finished",
            instance_id_from="job.workflow_id",  # dotted path into the message
            event_name_from=job_event_name,       # optional callable resolver
            message_model=JobFinished,
            dead_letter_topic="jobs.finished.dlq",
            authorize=authorize,
        )
        runner.subscribe(agent, event_routes=[spec])

        # In the workflow with instance id == job.workflow_id:
        result = yield ctx.wait_for_external_event("job_finished_done")

    Attributes:
        pubsub_name: Dapr pub/sub component name.
        topic: Topic to subscribe to. A topic that carries an event route
            carries nothing else.
        event_name: Event name to raise. Used when ``event_name_from`` is None.
        instance_id_from: Resolver for the target workflow instance id (required).
        event_name_from: Optional resolver that overrides ``event_name`` per message.
        data_from: Optional resolver for the event payload. Default: the
            validated message serialized like workflow inputs but made
            JSON-safe (Pydantic models are dumped with ``mode="json"``, so
            datetime / UUID / Decimal values survive), CloudEvent metadata
            included.
        message_model: Schema (Pydantic / dataclass / dict, or ``Union[...]``).
            Default ``dict``.
        dead_letter_topic: Optional dead-letter topic. Note that with a
            dead-letter topic and no pub/sub inbound resiliency retry policy,
            daprd dead-letters the first RETRY, so ``not_found_retry`` never
            retries (see ``NotFoundRetryPolicy``).
        payload_filter: Like ``PubSubRouteSpec.payload_filter``, but it runs
            with the ``hook_timeout_seconds`` deadline and must return exactly
            ``True`` to accept.
        model_filter: Like ``PubSubRouteSpec.model_filter``, with the same
            deadline and strict ``True`` rule as ``payload_filter``.
        authorize: Optional sync callable ``(validated_message, MessageContext,
            WorkflowEventTarget) -> bool``. The context carries the CloudEvent,
            whose attributes are publisher-supplied; the target is the resolved
            instance id and event name. It runs after schema validation, the
            filters, the resolvers and the limits, and before any sidecar call.
            Anything other than exactly ``True`` (including an exception or a
            timeout) denies the message: it is dropped with a WARNING that
            names the route, workflow instance and message id, never the
            payload.
        hook_timeout_seconds: Deadline for ``authorize``, ``payload_filter``
            and ``model_filter``. Default 5.0.
        dedupe: Deduplicate redeliveries by CloudEvent id. Default True.
        deduper: Optional backend for this route. When None, the route uses its
            own in-memory backend (TTL of at least 15 minutes and the
            not-found window, ``dedupe_max_entries`` ids). A subscriber- or
            runner-wide ``deduper`` is never used for event routes; to share
            dedupe across replicas for an event route, set ``spec.deduper``.
            Must be None when ``dedupe=False``.
        dedupe_max_entries: Capacity of the default in-memory backend. Once
            full, the oldest ids are evicted before their TTL, so a topic that
            receives more than this many messages per TTL window dedupes only
            the most recent ones. Default 65536.
        not_found_retry: Policy for instances that do not exist yet.
        max_data_bytes: Largest serialized JSON event payload accepted; larger
            payloads are dropped. Default 1 MiB.
        call_timeout_seconds: Deadline for each sidecar call (state check and
            raise). A timeout is retried; the timed-out call may still
            complete in the background. While ``dedupe`` is on, a timed-out
            raise is remembered and a redelivery of the same message to the
            same process does not raise it again: it is acknowledged once the
            first raise succeeded, retried while it is still running, and
            handled like any other failed raise if it failed. That memory is
            in process only; a restart clears it, and with ``dedupe=False``
            there is no key to track, so a timed-out raise can then be raised
            twice. Default 30.0.
        allow_reserved_event_names: Allow event names the SDK reserves for its
            own waits (``approval_response_*``, ``user_input_response:*``).
            Default False.
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
    max_data_bytes: int = DEFAULT_EVENT_MAX_DATA_BYTES
    call_timeout_seconds: float = DEFAULT_EVENT_CALL_TIMEOUT_SECONDS
    dedupe_max_entries: int = DEFAULT_EVENT_DEDUPE_MAX_ENTRIES
    allow_reserved_event_names: bool = False
    authorize: Callable[[Any, MessageContext, WorkflowEventTarget], bool] | None = None
    hook_timeout_seconds: float = DEFAULT_EVENT_HOOK_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        _check_int_range(self.max_data_bytes, "max_data_bytes")
        _check_seconds(self.call_timeout_seconds, "call_timeout_seconds")
        _check_seconds(self.hook_timeout_seconds, "hook_timeout_seconds")
        _check_int_range(self.dedupe_max_entries, "dedupe_max_entries")
