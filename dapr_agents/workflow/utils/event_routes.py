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

"""Pub/sub routes that raise external events on existing workflow instances.

A workflow event route turns a validated pub/sub message into
``DaprWorkflowClient.raise_workflow_event(instance_id, event_name, data=...)``.
The subscriber plumbing (CloudEvent parsing, schema validation, filters,
dedupe, dead-letter topic) lives in ``subscription.py``; this module holds the
per-message resolution and the dispatch decision (SUCCESS / RETRY / DROP).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

import grpc
from cachetools import TTLCache
from dapr.ext.workflow.workflow_state import WorkflowState, WorkflowStatus

from dapr_agents.types.workflow import FieldResolver, NotFoundRetryPolicy

if TYPE_CHECKING:
    from dapr_agents.workflow.utils.subscription import DedupeBackend, MessageContext

logger = logging.getLogger(__name__)

EventDispatchStatus = Literal["success", "retry", "drop"]

# Values match STATUS_SUCCESS / STATUS_RETRY / STATUS_DROP in subscription.py.
_SUCCESS: EventDispatchStatus = "success"
_RETRY: EventDispatchStatus = "retry"
_DROP: EventDispatchStatus = "drop"

TERMINAL_WORKFLOW_STATUSES: frozenset[WorkflowStatus] = frozenset(
    {WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.TERMINATED}
)

_NOT_FOUND_DETAILS = "no such instance exists"
_TRACKER_MAXSIZE = 4096
_TRACKER_MIN_TTL_SECONDS = 60.0


class EventRouteResolutionError(Exception):
    """A message could not be turned into (instance_id, event_name, data).

    Attributes:
        field: The field that failed (``instance_id``, ``event_name`` or ``data``).
        reason: Human-readable reason.
    """

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason


@dataclass(frozen=True)
class EventRouteTarget:
    """What an event binding raises, carried on ``MessageRouteBinding.event_target``."""

    event_name: str
    instance_id_from: FieldResolver
    event_name_from: FieldResolver | None
    data_from: FieldResolver | None
    dedupe: bool
    deduper: DedupeBackend | None
    not_found_retry: NotFoundRetryPolicy


@dataclass(frozen=True)
class _ResolvedEvent:
    instance_id: str
    event_name: str
    data: Any


# ---- resolvers ---------------------------------------------------------------


def parse_field_path(path: str) -> tuple[str, ...]:
    """Split and validate a dotted field path.

    Raises:
        ValueError: If the path is empty, has empty or padded segments, or a
            segment starts with ``__``.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    segments = tuple(path.split("."))
    for seg in segments:
        if not seg:
            raise ValueError("empty segment")
        if seg != seg.strip():
            raise ValueError(f"segment {seg!r} has surrounding whitespace")
        if seg.startswith("__"):
            raise ValueError(f"segment {seg!r} may not start with '__'")
    return segments


def _step(cur: Any, seg: str, path: str, field: str) -> Any:
    if isinstance(cur, Mapping):
        if seg in cur:
            return cur[seg]
    elif isinstance(cur, (list, tuple)):
        # Sequences only take numeric indexes (no `items.count` method lookups).
        if seg.isdigit() and int(seg) < len(cur):
            return cur[int(seg)]
    elif hasattr(cur, seg):
        return getattr(cur, seg)
    raise EventRouteResolutionError(field, f"path {path!r}: segment {seg!r} not found")


def resolve_field(
    resolver: FieldResolver,
    message: Any,
    msg_ctx: MessageContext,
    *,
    field: str,
) -> Any:
    """Resolve one field from the validated message.

    Raises:
        EventRouteResolutionError: If the path does not resolve or the callable raises.
    """
    if isinstance(resolver, str):
        try:
            segments = parse_field_path(resolver)
        except ValueError as exc:
            raise EventRouteResolutionError(
                field, f"invalid field path {resolver!r}: {exc}"
            ) from exc
        cur = message
        for seg in segments:
            cur = _step(cur, seg, resolver, field)
        return cur
    try:
        return resolver(message, msg_ctx)
    except Exception as exc:
        raise EventRouteResolutionError(
            field, f"resolver raised {type(exc).__name__}: {exc}"
        ) from exc


def coerce_identifier(value: Any, *, field: str) -> str:
    """Coerce a resolved instance id / event name to a non-empty string.

    Raises:
        EventRouteResolutionError: If the value is None, empty, or not str/int.
    """
    if isinstance(value, str):
        if not value.strip():
            raise EventRouteResolutionError(field, "resolved to an empty string")
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if value is None:
        raise EventRouteResolutionError(field, "resolved to None")
    raise EventRouteResolutionError(
        field, f"resolved to {type(value).__name__}, expected str"
    )


def serialize_event_data(
    resolver: FieldResolver | None,
    message: Any,
    msg_ctx: MessageContext,
    *,
    default_serializer: Callable[[Any], Any],
) -> Any:
    """Build the JSON-safe event payload.

    Raises:
        EventRouteResolutionError: If resolution fails or the result is not
            JSON-serializable (a poison message must not loop in the SDK).
    """
    if resolver is None:
        result = default_serializer(message)
    else:
        value = resolve_field(resolver, message, msg_ctx, field="data")
        if hasattr(value, "model_dump"):
            result = value.model_dump(mode="json")
        elif is_dataclass(value) and not isinstance(value, type):
            result = asdict(value)
        else:
            result = value
    try:
        json.dumps(result)
    except (TypeError, ValueError) as exc:
        raise EventRouteResolutionError(
            "data", f"not JSON-serializable: {exc}"
        ) from exc
    return result


def is_instance_not_found_error(exc: BaseException) -> bool:
    """True when ``exc`` is a gRPC error saying the workflow instance does not exist."""
    if not isinstance(exc, grpc.RpcError):
        return False
    try:
        code = exc.code()  # type: ignore[attr-defined]
        details = exc.details()  # type: ignore[attr-defined]
    except Exception:
        return False
    if code == grpc.StatusCode.NOT_FOUND:
        return True
    return isinstance(details, str) and _NOT_FOUND_DETAILS in details


# ---- not-found tracking ------------------------------------------------------


class _NotFoundTracker:
    """Per-route, per-process count of "instance not found" deliveries."""

    def __init__(self, window_seconds: float) -> None:
        ttl = max(2 * window_seconds, _TRACKER_MIN_TTL_SECONDS)
        self._cache: TTLCache = TTLCache(maxsize=_TRACKER_MAXSIZE, ttl=ttl)
        self._lock = threading.Lock()

    def record(self, key: str, now: float) -> tuple[int, float]:
        """Count one more not-found delivery; return (attempts, first_seen)."""
        with self._lock:
            previous: Optional[tuple[int, float]] = self._cache.get(key)
            updated = (1, now) if previous is None else (previous[0] + 1, previous[1])
            self._cache[key] = updated
            return updated

    def clear(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)


# ---- dispatcher --------------------------------------------------------------


@dataclass(frozen=True)
class _DispatchContext:
    target: EventRouteTarget
    route_name: str
    pubsub: str
    topic: str
    dead_letter_topic: str | None
    event_id: str | None


class WorkflowEventDispatcher:
    """Raise one external event per message and decide the broker response.

    Runs synchronously on the calling (consumer) thread.
    """

    def __init__(
        self,
        *,
        wf_client: Any,
        default_serializer: Callable[[Any], Any],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._wf_client = wf_client
        self._default_serializer = default_serializer
        self._clock = clock
        self._trackers: dict[tuple[str, str], _NotFoundTracker] = {}
        self._trackers_lock = threading.Lock()

    def dispatch(
        self,
        *,
        target: EventRouteTarget,
        route_name: str,
        pubsub: str,
        topic: str,
        dead_letter_topic: str | None,
        message: Any,
        msg_ctx: MessageContext,
    ) -> EventDispatchStatus:
        """Resolve, check the workflow state and raise the event."""
        ctx = _DispatchContext(
            target=target,
            route_name=route_name,
            pubsub=pubsub,
            topic=topic,
            dead_letter_topic=dead_letter_topic,
            event_id=msg_ctx.event.id,
        )
        try:
            resolved = self._resolve(target, message, msg_ctx)
        except EventRouteResolutionError as exc:
            logger.warning(
                "Event route %r could not resolve %s (%s) for message id=%s on topic %r; dropping.",
                route_name,
                exc.field,
                exc.reason,
                ctx.event_id,
                topic,
            )
            return _DROP

        key = ctx.event_id or f"{resolved.instance_id}\x1f{resolved.event_name}"
        try:
            state = self._check_state(resolved.instance_id)
        except Exception:
            logger.exception(
                "Event route %r: fetching state of workflow %s failed (event %r); retrying.",
                route_name,
                resolved.instance_id,
                resolved.event_name,
            )
            return _RETRY
        if state is None:
            return self._on_not_found(ctx, resolved, key)
        status = state.runtime_status
        if status in TERMINAL_WORKFLOW_STATUSES:
            self._tracker(ctx).clear(key)
            return self._give_up(ctx, resolved, reason=f"workflow is {status.name}")
        return self._raise(ctx, resolved, key)

    def _resolve(
        self, target: EventRouteTarget, message: Any, msg_ctx: MessageContext
    ) -> _ResolvedEvent:
        instance_id = coerce_identifier(
            resolve_field(
                target.instance_id_from, message, msg_ctx, field="instance_id"
            ),
            field="instance_id",
        )
        if target.event_name_from is None:
            event_name = target.event_name
        else:
            event_name = coerce_identifier(
                resolve_field(
                    target.event_name_from, message, msg_ctx, field="event_name"
                ),
                field="event_name",
            )
        data = serialize_event_data(
            target.data_from,
            message,
            msg_ctx,
            default_serializer=self._default_serializer,
        )
        return _ResolvedEvent(instance_id=instance_id, event_name=event_name, data=data)

    def _check_state(self, instance_id: str) -> Optional[WorkflowState]:
        return self._wf_client.get_workflow_state(instance_id, fetch_payloads=False)

    def _tracker(self, ctx: _DispatchContext) -> _NotFoundTracker:
        key = (ctx.pubsub, ctx.topic)
        with self._trackers_lock:
            tracker = self._trackers.get(key)
            if tracker is None:
                tracker = _NotFoundTracker(ctx.target.not_found_retry.window_seconds)
                self._trackers[key] = tracker
            return tracker

    def _on_not_found(
        self, ctx: _DispatchContext, resolved: _ResolvedEvent, key: str
    ) -> EventDispatchStatus:
        policy = ctx.target.not_found_retry
        tracker = self._tracker(ctx)
        now = self._clock()
        attempts, first_seen = tracker.record(key, now)
        elapsed = now - first_seen
        if attempts < policy.max_attempts and elapsed < policy.window_seconds:
            logger.info(
                "Event route %r: workflow %s not found yet for event %r (attempt %d/%d); retrying.",
                ctx.route_name,
                resolved.instance_id,
                resolved.event_name,
                attempts,
                policy.max_attempts,
            )
            return _RETRY
        tracker.clear(key)
        return self._give_up(
            ctx,
            resolved,
            reason=f"instance not found after {attempts} attempts / {elapsed:.0f}s",
        )

    def _give_up(
        self, ctx: _DispatchContext, resolved: _ResolvedEvent, *, reason: str
    ) -> EventDispatchStatus:
        if ctx.dead_letter_topic:
            action = f"dead-lettering to {ctx.dead_letter_topic!r}"
        else:
            action = "dropping (no dead_letter_topic configured)"
        logger.warning(
            "Event route %r on topic %r: cannot raise event %r on workflow %s "
            "(message id=%s): %s; %s.",
            ctx.route_name,
            ctx.topic,
            resolved.event_name,
            resolved.instance_id,
            ctx.event_id,
            reason,
            action,
        )
        return _DROP

    def _raise(
        self, ctx: _DispatchContext, resolved: _ResolvedEvent, key: str
    ) -> EventDispatchStatus:
        try:
            self._wf_client.raise_workflow_event(
                instance_id=resolved.instance_id,
                event_name=resolved.event_name,
                data=resolved.data,
            )
        except Exception as exc:
            if is_instance_not_found_error(exc):
                return self._on_not_found(ctx, resolved, key)
            logger.exception(
                "Event route %r: raising event %r on workflow %s failed; retrying.",
                ctx.route_name,
                resolved.event_name,
                resolved.instance_id,
            )
            return _RETRY
        self._tracker(ctx).clear(key)
        logger.debug(
            "Event route %r raised event %r on workflow %s.",
            ctx.route_name,
            resolved.event_name,
            resolved.instance_id,
        )
        return _SUCCESS


__all__ = [
    "EventDispatchStatus",
    "EventRouteResolutionError",
    "EventRouteTarget",
    "TERMINAL_WORKFLOW_STATUSES",
    "WorkflowEventDispatcher",
    "coerce_identifier",
    "is_instance_not_found_error",
    "parse_field_path",
    "resolve_field",
    "serialize_event_data",
]
