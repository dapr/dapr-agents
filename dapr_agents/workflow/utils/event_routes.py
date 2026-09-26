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

How daprd treats the response (Dapr 1.18 runtime, which this SDK targets):

* DROP: the message is not redelivered. When the subscription has a
  dead-letter topic, daprd publishes the dropped message there.
* RETRY: daprd first applies the pub/sub component's inbound resiliency retry
  policy, if any. When that is used up (at once, without a policy) the message
  goes to the dead-letter topic if one is configured, and otherwise back to
  the broker for redelivery. So on a route with a ``dead_letter_topic`` the
  "not found" bounded retry only takes effect through a component inbound
  resiliency retry policy; without one, the first RETRY dead-letters.

Known race: the workflow can finish between the state check and the raise.
The runtime then discards the event while the message is acknowledged
(SUCCESS). Nothing can wait for that event anymore, so no workflow is harmed,
but the message is not dead-lettered.

Each sidecar call runs with the route's ``call_timeout_seconds`` deadline. A
timeout answers RETRY, but the call may still complete in the background. A
timed-out raise still running is remembered by the message's dedupe key (in
process memory), so a redelivery to the same process answers SUCCESS once it
succeeded, RETRY while it runs, and follows the normal error rules if it
failed. A restart, ``dedupe=False``, a crash before the ack or a redelivery
to another replica without a shared ``spec.deduper`` can still raise twice.
Errors with a code in ``PERMANENT_SIDECAR_ERROR_CODES`` are dropped like a
terminal workflow; other errors answer RETRY.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal

import grpc
from cachetools import LRUCache
from dapr.ext.workflow.workflow_state import WorkflowState, WorkflowStatus

from dapr_agents.streaming.keys import (
    APPROVAL_RESPONSE_EVENT_PREFIX,
    USER_INPUT_EVENT_PREFIX,
)
from dapr_agents.types.workflow import (
    DEFAULT_EVENT_CALL_TIMEOUT_SECONDS,
    DEFAULT_EVENT_DEDUPE_MAX_ENTRIES,
    DEFAULT_EVENT_HOOK_TIMEOUT_SECONDS,
    DEFAULT_EVENT_MAX_DATA_BYTES,
    FieldResolver,
    NotFoundRetryPolicy,
)
from dapr_agents.workflow.utils.call_deadline import DeadlineCaller
from dapr_agents.workflow.utils.event_route_calls import (
    MIN_TIMED_OUT_RAISES_TRACKED,
    PERMANENT_SIDECAR_ERROR_CODES,
    TimedOutRaises,
    permanent_error_code,
    run_strict_hook,
)

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

# Longest resolved instance id / event name accepted from a message, in
# characters (not bytes).
MAX_EVENT_IDENTIFIER_LENGTH = 512

# Event names the SDK itself waits on, case-folded because the durabletask
# worker matches event names with ``str.casefold()``.
RESERVED_EVENT_NAME_PREFIXES: tuple[str, ...] = (
    APPROVAL_RESPONSE_EVENT_PREFIX.casefold(),
    f"{USER_INPUT_EVENT_PREFIX}:".casefold(),
)

_NOT_FOUND_DETAILS = "no such instance exists"
_TRACKER_MAXSIZE = 4096


class EventRouteResolutionError(Exception):
    """A message could not be turned into (instance_id, event_name, data).

    Attributes:
        field: The field that failed (``instance_id``, ``event_name``,
            ``data``, or ``message`` for an unexpected error).
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
    max_data_bytes: int = DEFAULT_EVENT_MAX_DATA_BYTES
    call_timeout_seconds: float = DEFAULT_EVENT_CALL_TIMEOUT_SECONDS
    dedupe_max_entries: int = DEFAULT_EVENT_DEDUPE_MAX_ENTRIES
    allow_reserved_event_names: bool = False
    authorize: Callable[[Any, MessageContext], bool] | None = None
    hook_timeout_seconds: float = DEFAULT_EVENT_HOOK_TIMEOUT_SECONDS


@dataclass(frozen=True)
class _ResolvedEvent:
    instance_id: str
    event_name: str
    data: Any


# ---- resolvers ---------------------------------------------------------------


def is_reserved_event_name(name: str) -> bool:
    """True when ``name`` starts with an event-name prefix the SDK waits on.

    Compared case-folded, the way the durabletask worker matches event names,
    so ``"approval_re\u017fponse_x"`` (long s) is reserved too.
    """
    return name.casefold().startswith(RESERVED_EVENT_NAME_PREFIXES)


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
        EventRouteResolutionError: If the value is None, empty, not str/int, or
            longer than ``MAX_EVENT_IDENTIFIER_LENGTH``.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    if isinstance(value, str):
        if not value.strip():
            raise EventRouteResolutionError(field, "resolved to an empty string")
        if len(value) > MAX_EVENT_IDENTIFIER_LENGTH:
            raise EventRouteResolutionError(
                field,
                f"resolved to {len(value)} characters "
                f"(limit {MAX_EVENT_IDENTIFIER_LENGTH})",
            )
        return value
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
    max_bytes: int | None = None,
) -> Any:
    """Build the JSON-safe event payload.

    Raises:
        EventRouteResolutionError: If resolution or serialization fails in any
            way (including recursion errors), the result is not
            JSON-serializable, or it is larger than ``max_bytes`` once encoded.
            A poison message must be dropped, never retried.
    """
    if resolver is None:
        try:
            result = default_serializer(message)
        except EventRouteResolutionError:
            raise
        except Exception as exc:
            raise EventRouteResolutionError(
                "data", f"serialization failed: {type(exc).__name__}: {exc}"
            ) from exc
    else:
        result = resolve_field(resolver, message, msg_ctx, field="data")
    try:
        result = _to_json_value(result)
        encoded = json.dumps(result)
    except Exception as exc:
        raise EventRouteResolutionError(
            "data", f"not JSON-serializable: {type(exc).__name__}: {exc}"
        ) from exc
    size = len(encoded.encode("utf-8"))
    if max_bytes is not None and size > max_bytes:
        raise EventRouteResolutionError(
            "data", f"serialized payload is {size} bytes (limit {max_bytes})"
        )
    return result


def _to_json_value(value: Any) -> Any:
    """Dump a resolved Pydantic model / dataclass; other values pass through."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


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
    """Per-route, per-process count of "instance not found" deliveries.

    Entries never expire by time, so a message redelivered slower than any
    TTL keeps its count; they are removed when the message is raised or given
    up. Bounded to ``_TRACKER_MAXSIZE`` messages: if more distinct messages
    than that are waiting for missing instances at once, the least recently
    used counts are evicted and those messages start counting again, so
    ``max_attempts`` / ``window_seconds`` are only guaranteed below that bound.
    """

    def __init__(self) -> None:
        self._cache: LRUCache = LRUCache(maxsize=_TRACKER_MAXSIZE)
        self._lock = threading.Lock()

    def record(self, key: str, now: float) -> tuple[int, float]:
        """Count one more not-found delivery; return (attempts, first_seen)."""
        with self._lock:
            previous: tuple[int, float] | None = self._cache.get(key)
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
    dedupe_key: str | None


class WorkflowEventDispatcher:
    """Raise one external event per message and decide the broker response.

    Runs on the calling (consumer) thread. Each sidecar call runs on a
    ``DeadlineCaller`` worker with the route's ``call_timeout_seconds``; call
    :meth:`close` when the subscriber shuts down.
    """

    def __init__(
        self,
        *,
        wf_client: Any,
        default_serializer: Callable[[Any], Any],
        clock: Callable[[], float] = time.monotonic,
        caller: DeadlineCaller | None = None,
    ) -> None:
        self._wf_client = wf_client
        self._default_serializer = default_serializer
        self._clock = clock
        self._caller = caller or DeadlineCaller(thread_name_prefix="event-route-call")
        self._trackers: dict[tuple[str, str], _NotFoundTracker] = {}
        self._timed_out: dict[tuple[str, str], TimedOutRaises] = {}
        self._trackers_lock = threading.Lock()

    def close(self) -> None:
        """Release the call workers without waiting for in-flight calls."""
        self._caller.close()

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
        dedupe_key: str | None = None,
    ) -> EventDispatchStatus:
        """Resolve, authorize, check the workflow state and raise the event.

        Args:
            dedupe_key: The subscriber's dedupe key for this message, or None
                when the route does not dedupe. Used to recognize a redelivery
                of a message whose raise timed out.
        """
        ctx = _DispatchContext(
            target=target,
            route_name=route_name,
            pubsub=pubsub,
            topic=topic,
            dead_letter_topic=dead_letter_topic,
            event_id=msg_ctx.event.id,
            dedupe_key=dedupe_key,
        )
        try:
            resolved = self._resolve(target, message, msg_ctx)
        except EventRouteResolutionError as exc:
            logger.warning(
                "Event route %r could not resolve %s (%s) for message id=%r on topic %r; dropping.",
                route_name,
                exc.field,
                exc.reason,
                ctx.event_id,
                topic,
            )
            return _DROP
        if not self._authorized(ctx, resolved, message, msg_ctx):
            return _DROP
        key = ctx.event_id or f"{resolved.instance_id}\x1f{resolved.event_name}"
        earlier = self._earlier_timed_out_raise(ctx, resolved, key)
        if earlier is not None:
            return earlier
        return self._deliver(ctx, resolved, key)

    def hook_accepts(
        self,
        target: EventRouteTarget,
        hook: Callable[[Any, MessageContext], Any] | None,
        value: Any,
        msg_ctx: MessageContext,
        *,
        kind: str,
        route_name: str,
    ) -> bool:
        """Run an event route filter with the route's hook deadline.

        True only when there is no filter or it returned exactly ``True``.
        """
        if hook is None:
            return True
        return run_strict_hook(
            self._caller,
            hook,
            target.hook_timeout_seconds,
            value,
            msg_ctx,
            kind=kind,
            route_name=route_name,
        )

    def _authorized(
        self,
        ctx: _DispatchContext,
        resolved: _ResolvedEvent,
        message: Any,
        msg_ctx: MessageContext,
    ) -> bool:
        if self.hook_accepts(
            ctx.target,
            ctx.target.authorize,
            message,
            msg_ctx,
            kind="authorize",
            route_name=ctx.route_name,
        ):
            return True
        logger.warning(
            "Event route %r on topic %r: authorize denied event %r on workflow %r "
            "(message id=%r); dropping.",
            ctx.route_name,
            ctx.topic,
            resolved.event_name,
            resolved.instance_id,
            ctx.event_id,
        )
        return False

    def _resolve(
        self, target: EventRouteTarget, message: Any, msg_ctx: MessageContext
    ) -> _ResolvedEvent:
        """Resolve the event; any failure becomes an ``EventRouteResolutionError``.

        Nothing unexpected may escape to the subscriber, which would RETRY a
        poison message forever.
        """
        try:
            return self._resolve_fields(target, message, msg_ctx)
        except EventRouteResolutionError:
            raise
        except Exception as exc:
            raise EventRouteResolutionError(
                "message", f"unexpected {type(exc).__name__}: {exc}"
            ) from exc

    def _resolve_fields(
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
            if not target.allow_reserved_event_names and is_reserved_event_name(
                event_name
            ):
                raise EventRouteResolutionError(
                    "event_name", f"resolved to reserved event name {event_name!r}"
                )
        data = serialize_event_data(
            target.data_from,
            message,
            msg_ctx,
            default_serializer=self._default_serializer,
            max_bytes=target.max_data_bytes,
        )
        return _ResolvedEvent(instance_id=instance_id, event_name=event_name, data=data)

    def _call(
        self,
        ctx: _DispatchContext,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self._caller.call(fn, ctx.target.call_timeout_seconds, *args, **kwargs)

    def _deliver(
        self, ctx: _DispatchContext, resolved: _ResolvedEvent, key: str
    ) -> EventDispatchStatus:
        try:
            state: WorkflowState | None = self._call(
                ctx,
                self._wf_client.get_workflow_state,
                resolved.instance_id,
                fetch_payloads=False,
            )
        except TimeoutError:
            return self._on_timeout(ctx, resolved, "fetching the workflow state")
        except Exception as exc:
            # The SDK only maps "no such instance exists" to None; a NOT_FOUND
            # with other wording must still use the bounded not-found budget.
            status = self._sidecar_error_status(
                ctx, resolved, key, exc, "fetching the workflow state"
            )
            if status is not None:
                return status
            logger.exception(
                "Event route %r: fetching state of workflow %r failed (event %r); retrying.",
                ctx.route_name,
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

    def _sidecar_error_status(
        self,
        ctx: _DispatchContext,
        resolved: _ResolvedEvent,
        key: str,
        exc: BaseException,
        what: str,
    ) -> EventDispatchStatus | None:
        """Status for a not-found or permanent sidecar error; None for a transient one."""
        if is_instance_not_found_error(exc):
            return self._on_not_found(ctx, resolved, key)
        code = permanent_error_code(exc)
        if code is None:
            return None
        self._tracker(ctx).clear(key)
        return self._give_up(
            ctx, resolved, reason=f"{what} failed with permanent gRPC code {code.name}"
        )

    def _earlier_timed_out_raise(
        self, ctx: _DispatchContext, resolved: _ResolvedEvent, key: str
    ) -> EventDispatchStatus | None:
        """Status decided by an earlier timed-out raise of this message, if any.

        None means deliver normally (nothing tracked, or the earlier raise
        failed with a transient error).
        """
        if ctx.dedupe_key is None:
            return None
        future = self._timed_out_raises(ctx).get(ctx.dedupe_key)
        if future is None:
            return None
        if not future.done():
            logger.info(
                "Event route %r: an earlier raise of event %r on workflow %r for "
                "message id=%r is still running; retrying without raising again.",
                ctx.route_name,
                resolved.event_name,
                resolved.instance_id,
                ctx.event_id,
            )
            return _RETRY
        exc = future.exception()
        if exc is None:
            self._tracker(ctx).clear(key)
            logger.info(
                "Event route %r: the earlier timed-out raise of event %r on workflow %r "
                "(message id=%r) succeeded; acknowledging without raising again.",
                ctx.route_name,
                resolved.event_name,
                resolved.instance_id,
                ctx.event_id,
            )
            return _SUCCESS
        return self._sidecar_error_status(ctx, resolved, key, exc, "raising the event")

    def _timed_out_raises(self, ctx: _DispatchContext) -> TimedOutRaises:
        key = (ctx.pubsub, ctx.topic)
        with self._trackers_lock:
            tracked = self._timed_out.get(key)
            if tracked is None:
                tracked = TimedOutRaises(
                    max(ctx.target.dedupe_max_entries, MIN_TIMED_OUT_RAISES_TRACKED)
                )
                self._timed_out[key] = tracked
            return tracked

    def _on_timeout(
        self, ctx: _DispatchContext, resolved: _ResolvedEvent, what: str
    ) -> EventDispatchStatus:
        logger.warning(
            "Event route %r: %s for event %r on workflow %r (message id=%r) timed out "
            "after %gs; retrying. The call may still complete in the background.",
            ctx.route_name,
            what,
            resolved.event_name,
            resolved.instance_id,
            ctx.event_id,
            ctx.target.call_timeout_seconds,
        )
        return _RETRY

    def _tracker(self, ctx: _DispatchContext) -> _NotFoundTracker:
        key = (ctx.pubsub, ctx.topic)
        with self._trackers_lock:
            tracker = self._trackers.get(key)
            if tracker is None:
                tracker = _NotFoundTracker()
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
                "Event route %r: workflow %r not found yet for event %r (attempt %d/%d); retrying.",
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
            # daprd publishes DROPped messages to the subscription's DLQ.
            action = f"dropping; daprd dead-letters it to {ctx.dead_letter_topic!r}"
        else:
            action = "dropping (no dead_letter_topic configured)"
        logger.warning(
            "Event route %r on topic %r: cannot raise event %r on workflow %r "
            "(message id=%r): %s; %s.",
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
            self._call_raise(ctx, resolved)
        except TimeoutError:
            return self._on_timeout(ctx, resolved, "raising the event")
        except Exception as exc:
            status = self._sidecar_error_status(
                ctx, resolved, key, exc, "raising the event"
            )
            if status is not None:
                return status
            logger.exception(
                "Event route %r: raising event %r on workflow %r failed; retrying.",
                ctx.route_name,
                resolved.event_name,
                resolved.instance_id,
            )
            return _RETRY
        self._tracker(ctx).clear(key)
        logger.debug(
            "Event route %r raised event %r on workflow %r.",
            ctx.route_name,
            resolved.event_name,
            resolved.instance_id,
        )
        return _SUCCESS

    def _call_raise(self, ctx: _DispatchContext, resolved: _ResolvedEvent) -> None:
        """Raise with the call deadline; remember a timed-out raise that is still running."""
        future = self._caller.submit(
            self._wf_client.raise_workflow_event,
            instance_id=resolved.instance_id,
            event_name=resolved.event_name,
            data=resolved.data,
        )
        try:
            future.result(timeout=ctx.target.call_timeout_seconds)
        except TimeoutError:
            # cancel() fails once a worker runs the call: it may still raise.
            if not future.cancel() and ctx.dedupe_key is not None:
                self._timed_out_raises(ctx).add(ctx.dedupe_key, future)
            raise


__all__ = [
    "EventDispatchStatus",
    "MAX_EVENT_IDENTIFIER_LENGTH",
    "PERMANENT_SIDECAR_ERROR_CODES",
    "RESERVED_EVENT_NAME_PREFIXES",
    "EventRouteResolutionError",
    "EventRouteTarget",
    "TERMINAL_WORKFLOW_STATUSES",
    "WorkflowEventDispatcher",
    "coerce_identifier",
    "is_instance_not_found_error",
    "is_reserved_event_name",
    "parse_field_path",
    "resolve_field",
    "serialize_event_data",
]
