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
dedupe, dead-letter topic) lives in ``subscription.py``, the resolvers in
``event_route_resolvers.py`` and the per-topic state in
``event_route_state.py``; this module holds the dispatch decision
(SUCCESS / RETRY / DROP).

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

Deduplication and the per-message state are keyed by a composite key
computed after resolution: a SHA-256 of the CloudEvent id (or payload digest),
the target instance, the event name and a digest of the data. A message that
reuses an id for another target or other data never collides with the
original; a redelivery of the identical message is still recognized.

Each sidecar call runs with the route's ``call_timeout_seconds`` deadline. A
timeout answers RETRY, but the call may still complete in the background. A
timed-out raise still running is remembered by the composite key (in process
memory), so a redelivery of the same message to the same process answers
SUCCESS once it succeeded, RETRY while it runs, and follows the normal error
rules if it failed. A restart, ``dedupe=False``, a crash before the ack or a
redelivery to another replica without a shared ``spec.deduper`` can still
raise twice. Errors with a code in ``PERMANENT_SIDECAR_ERROR_CODES`` are
dropped like a terminal workflow; other errors answer RETRY.

Hooks are tri-state (see ``HookVerdict``): an exact ``True`` allows, an
explicit ``False``, a non-bool or an exception drops, and a timeout or a
saturated hook pool answers RETRY within the route's ``not_found_retry``
budget, then gives up like a terminal workflow.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Literal

from dapr.ext.workflow.workflow_state import WorkflowState, WorkflowStatus

from dapr_agents.types.workflow import (
    DEFAULT_EVENT_CALL_TIMEOUT_SECONDS,
    DEFAULT_EVENT_DEDUPE_MAX_ENTRIES,
    DEFAULT_EVENT_HOOK_TIMEOUT_SECONDS,
    DEFAULT_EVENT_MAX_DATA_BYTES,
    FieldResolver,
    NotFoundRetryPolicy,
    WorkflowEventTarget,
)
from dapr_agents.workflow.utils.call_deadline import DeadlineCaller
from dapr_agents.workflow.utils.core import stable_json_sha256
from dapr_agents.workflow.utils.event_route_calls import (
    PERMANENT_SIDECAR_ERROR_CODES,
    HookVerdict,
    is_instance_not_found_error,
    permanent_error_code,
    run_strict_hook,
)
from dapr_agents.workflow.utils.event_route_resolvers import (
    MAX_EVENT_IDENTIFIER_LENGTH,
    RESERVED_EVENT_NAME_PREFIXES,
    EventRouteResolutionError,
    coerce_identifier,
    is_reserved_event_name,
    parse_field_path,
    resolve_field,
    serialize_event_data,
)
from dapr_agents.workflow.utils.event_route_state import (
    AttemptTracker,
    EventRouteTopicState,
    RaiseIdentity,
    TrackedRaise,
)
from dapr_agents.workflow.utils.log_throttle import WarningThrottle

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

# Worker threads per dispatcher (one per subscriber, shared by all its
# event-route topics) for user hooks and for sidecar calls.
DEFAULT_HOOK_POOL_SIZE = 8
DEFAULT_SIDECAR_POOL_SIZE = 16


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
    authorize: Callable[[Any, MessageContext, WorkflowEventTarget], bool] | None = None
    hook_timeout_seconds: float = DEFAULT_EVENT_HOOK_TIMEOUT_SECONDS


@dataclass(frozen=True)
class _ResolvedEvent:
    instance_id: str
    event_name: str
    data: Any

    def identity(self) -> RaiseIdentity:
        return self._identity

    @cached_property
    def _identity(self) -> RaiseIdentity:
        # data is JSON-safe (serialize_event_data checked it).
        return RaiseIdentity(
            instance_id=self.instance_id,
            event_name=self.event_name,
            data_sha256=stable_json_sha256(self.data),
        )

    def dedupe_key(self, message_key: str) -> str:
        """Composite key: the message key, target, event name and data digest."""
        identity = self.identity()
        parts = (message_key, identity.instance_id, identity.event_name)
        digest = stable_json_sha256([*parts, identity.data_sha256])
        return f"event-route:{digest}"


def _key_for_log(key: str) -> str:
    """A short, payload-free prefix of a key for WARNING logs."""
    return f"{key[:24]}..." if len(key) > 24 else key


def _backend_seen(deduper: DedupeBackend, key: str) -> bool:
    try:
        return deduper.seen(key)
    except Exception:
        logger.debug("Dedupe backend seen() error; continuing.", exc_info=True)
        return False


def _backend_mark(deduper: DedupeBackend, key: str) -> None:
    try:
        deduper.mark(key)
    except Exception:
        logger.debug("Dedupe backend mark() error; continuing.", exc_info=True)


# ---- dispatcher --------------------------------------------------------------


@dataclass(frozen=True)
class _DispatchContext:
    target: EventRouteTarget
    route_name: str
    topic: str
    dead_letter_topic: str | None
    event_id: str | None
    message_key: str
    deduper: DedupeBackend | None
    state: EventRouteTopicState


class WorkflowEventDispatcher:
    """Raise one external event per message and decide the broker response.

    Runs on the calling (consumer) thread. User hooks (``authorize`` and the
    event-route filters) run on one ``DeadlineCaller`` pool and sidecar calls
    on another, so a hook that never returns cannot starve sidecar calls or
    the other way round. Both pools belong to this dispatcher, which is one
    per subscriber: they are shared by every event-route topic of that
    subscriber. Call :meth:`close` when the subscriber shuts down.
    """

    def __init__(
        self,
        *,
        wf_client: Any,
        default_serializer: Callable[[Any], Any],
        clock: Callable[[], float] = time.monotonic,
        hook_pool_size: int = DEFAULT_HOOK_POOL_SIZE,
        sidecar_pool_size: int = DEFAULT_SIDECAR_POOL_SIZE,
        hook_caller: DeadlineCaller | None = None,
        sidecar_caller: DeadlineCaller | None = None,
    ) -> None:
        """
        Args:
            hook_pool_size: Workers for user hooks (ignored with ``hook_caller``).
            sidecar_pool_size: Workers for sidecar calls (ignored with
                ``sidecar_caller``).
            hook_caller: Pre-built pool for user hooks (tests).
            sidecar_caller: Pre-built pool for sidecar calls (tests).
        """
        self._wf_client = wf_client
        self._default_serializer = default_serializer
        self._clock = clock
        self._hook_caller = hook_caller or DeadlineCaller(
            max_workers=hook_pool_size, thread_name_prefix="event-route-hook"
        )
        self._sidecar_caller = sidecar_caller or DeadlineCaller(
            max_workers=sidecar_pool_size, thread_name_prefix="event-route-call"
        )
        self._topic_states: dict[tuple[str, str], EventRouteTopicState] = {}
        self._topic_state_creation_lock = threading.Lock()
        self._warnings = WarningThrottle(clock=clock)

    def close(self) -> None:
        """Release both pools without waiting for in-flight calls."""
        self._hook_caller.close()
        self._sidecar_caller.close()

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
        message_key: str | None = None,
        deduper: DedupeBackend | None = None,
    ) -> EventDispatchStatus:
        """Resolve, dedupe, authorize, check the workflow state and raise the event.

        Args:
            message_key: CloudEvent id, or a payload digest when there is none.
                Defaults to the CloudEvent id from ``msg_ctx``.
            deduper: The route's dedupe backend, or None when it does not
                dedupe. Checked and marked with the composite key.
        """
        ctx = _DispatchContext(
            target=target,
            route_name=route_name,
            topic=topic,
            dead_letter_topic=dead_letter_topic,
            event_id=msg_ctx.event.id,
            message_key=message_key or msg_ctx.event.id or "",
            deduper=deduper,
            state=self._topic_state(pubsub, topic, target),
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
        key = resolved.dedupe_key(ctx.message_key)
        status = self._dispatch_resolved(ctx, resolved, key, message, msg_ctx)
        if ctx.deduper is not None and status != _RETRY:
            _backend_mark(ctx.deduper, key)
        return status

    def _dispatch_resolved(
        self,
        ctx: _DispatchContext,
        resolved: _ResolvedEvent,
        key: str,
        message: Any,
        msg_ctx: MessageContext,
    ) -> EventDispatchStatus:
        """Dedupe, then an earlier timed-out raise, authorize and delivery.

        An earlier timed-out raise of this message is checked before
        ``authorize``, which the message already passed.
        """
        if ctx.deduper is not None and _backend_seen(ctx.deduper, key):
            logger.debug(
                "Event route %r: duplicate of message id=%r; acknowledging.",
                ctx.route_name,
                ctx.event_id,
            )
            return _SUCCESS
        status = self._earlier_timed_out_raise(ctx, resolved, key)
        if status is None:
            status = self._authorize(ctx, resolved, key, message, msg_ctx)
        if status is None:
            status = self._deliver(ctx, resolved, key)
        return status

    def filter_verdict(
        self,
        target: EventRouteTarget,
        hook: Callable[[Any, MessageContext], Any] | None,
        value: Any,
        msg_ctx: MessageContext,
        *,
        kind: str,
        route_name: str,
    ) -> HookVerdict:
        """Run an event route filter with the route's hook deadline.

        ALLOW when there is no filter or it returned exactly ``True``.
        """
        if hook is None:
            return HookVerdict.ALLOW
        return run_strict_hook(
            self._hook_caller,
            hook,
            target.hook_timeout_seconds,
            value,
            msg_ctx,
            kind=kind,
            route_name=route_name,
        )

    def on_filter_undecided(
        self,
        *,
        target: EventRouteTarget,
        route_name: str,
        pubsub: str,
        topic: str,
        dead_letter_topic: str | None,
        msg_ctx: MessageContext,
        message_key: str | None,
        kind: str,
    ) -> EventDispatchStatus:
        """RETRY for a filter that did not answer, within the retry budget."""
        ctx = _DispatchContext(
            target=target,
            route_name=route_name,
            topic=topic,
            dead_letter_topic=dead_letter_topic,
            event_id=msg_ctx.event.id,
            message_key=message_key or msg_ctx.event.id or "",
            deduper=None,
            state=self._topic_state(pubsub, topic, target),
        )
        return self._on_undecided(ctx, None, f"{kind}:{ctx.message_key}", kind)

    def _topic_state(
        self, pubsub: str, topic: str, target: EventRouteTarget
    ) -> EventRouteTopicState:
        """The topic's state; the lock is taken only to create it the first time."""
        key = (pubsub, topic)
        state = self._topic_states.get(key)
        if state is not None:
            return state
        with self._topic_state_creation_lock:
            state = self._topic_states.get(key)
            if state is None:
                state = EventRouteTopicState.create(target.dedupe_max_entries)
                self._topic_states[key] = state
            return state

    def _authorize(
        self,
        ctx: _DispatchContext,
        resolved: _ResolvedEvent,
        key: str,
        message: Any,
        msg_ctx: MessageContext,
    ) -> EventDispatchStatus | None:
        """None when ``authorize`` allows (or is unset); otherwise the status."""
        if ctx.target.authorize is None:
            return None
        event_target = WorkflowEventTarget(
            instance_id=resolved.instance_id, event_name=resolved.event_name
        )
        verdict = run_strict_hook(
            self._hook_caller,
            ctx.target.authorize,
            ctx.target.hook_timeout_seconds,
            message,
            msg_ctx,
            event_target,
            kind="authorize",
            route_name=ctx.route_name,
        )
        if verdict is HookVerdict.UNDECIDED:
            return self._on_undecided(ctx, resolved, key, "authorize")
        ctx.state.undecided.clear(key)
        if verdict is HookVerdict.ALLOW:
            return None
        return self._give_up(ctx, resolved, reason="authorize denied the message")

    def _on_undecided(
        self,
        ctx: _DispatchContext,
        resolved: _ResolvedEvent | None,
        key: str,
        kind: str,
    ) -> EventDispatchStatus:
        """RETRY within the ``not_found_retry`` budget, then give up."""
        attempts, elapsed = self._record_attempt(ctx, ctx.state.undecided, key)
        if attempts is None:
            logger.info(
                "Event route %r: %s did not decide for message id=%r; retrying.",
                ctx.route_name,
                kind,
                ctx.event_id,
            )
            return _RETRY
        reason = f"{kind} did not decide after {attempts} attempts / {elapsed:.0f}s"
        if ctx.dead_letter_topic:
            return self._give_up(ctx, resolved, reason=reason)
        self._warnings.warn(
            logger,
            ("undecided", ctx.route_name),
            "Event route %r on topic %r: %s (message key %s); dropping (no "
            "dead_letter_topic configured).",
            ctx.route_name,
            ctx.topic,
            reason,
            _key_for_log(key),
        )
        return _DROP

    def _record_attempt(
        self, ctx: _DispatchContext, tracker: AttemptTracker, key: str
    ) -> tuple[int | None, float]:
        """Count an attempt; (None, elapsed) while within budget, else (attempts, elapsed).

        The entry is cleared once the budget is used up.
        """
        policy = ctx.target.not_found_retry
        now = self._clock()
        attempts, first_seen = tracker.record(key, now)
        elapsed = now - first_seen
        if attempts < policy.max_attempts and elapsed < policy.window_seconds:
            return None, elapsed
        tracker.clear(key)
        return attempts, elapsed

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

    def _deliver(
        self, ctx: _DispatchContext, resolved: _ResolvedEvent, key: str
    ) -> EventDispatchStatus:
        try:
            state: WorkflowState | None = self._sidecar_caller.call(
                self._wf_client.get_workflow_state,
                ctx.target.call_timeout_seconds,
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
            ctx.state.not_found.clear(key)
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
        ctx.state.not_found.clear(key)
        return self._give_up(
            ctx, resolved, reason=f"{what} failed with permanent gRPC code {code.name}"
        )

    def _earlier_timed_out_raise(
        self, ctx: _DispatchContext, resolved: _ResolvedEvent, key: str
    ) -> EventDispatchStatus | None:
        """Status decided by an earlier timed-out raise of this exact message, if any.

        ``key`` is the composite key, so only the same message id with the same
        target and data can match. None means deliver normally (nothing
        tracked, or the earlier raise failed with a transient error).
        """
        if not ctx.target.dedupe:
            return None
        tracked = ctx.state.timed_out.peek(key)
        if tracked is None:
            return None
        # The composite key includes the identity, so a match is the same message.
        assert tracked.identity == resolved.identity(), "composite key collision"
        if not tracked.future.done():
            logger.info(
                "Event route %r: an earlier raise of event %r on workflow %r for "
                "message id=%r is still running; retrying without raising again.",
                ctx.route_name,
                resolved.event_name,
                resolved.instance_id,
                ctx.event_id,
            )
            return _RETRY
        ctx.state.timed_out.discard(key, tracked)
        exc = tracked.future.exception()
        if exc is not None:
            return self._sidecar_error_status(
                ctx, resolved, key, exc, "raising the event"
            )
        ctx.state.not_found.clear(key)
        logger.info(
            "Event route %r: the earlier timed-out raise of event %r on workflow %r "
            "(message id=%r) succeeded; acknowledging without raising again.",
            ctx.route_name,
            resolved.event_name,
            resolved.instance_id,
            ctx.event_id,
        )
        return _SUCCESS

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

    def _on_not_found(
        self, ctx: _DispatchContext, resolved: _ResolvedEvent, key: str
    ) -> EventDispatchStatus:
        attempts, elapsed = self._record_attempt(ctx, ctx.state.not_found, key)
        if attempts is None:
            logger.info(
                "Event route %r: workflow %r not found yet for event %r; retrying.",
                ctx.route_name,
                resolved.instance_id,
                resolved.event_name,
            )
            return _RETRY
        return self._give_up(
            ctx,
            resolved,
            reason=f"instance not found after {attempts} attempts / {elapsed:.0f}s",
        )

    def _give_up(
        self, ctx: _DispatchContext, resolved: _ResolvedEvent | None, *, reason: str
    ) -> EventDispatchStatus:
        if ctx.dead_letter_topic:
            # daprd publishes DROPped messages to the subscription's DLQ.
            action = f"dropping; daprd dead-letters it to {ctx.dead_letter_topic!r}"
        else:
            action = "dropping (no dead_letter_topic configured)"
        what = (
            ""
            if resolved is None
            else f" event {resolved.event_name!r} on workflow {resolved.instance_id!r}"
        )
        logger.warning(
            "Event route %r on topic %r: cannot raise%s (message id=%r): %s; %s.",
            ctx.route_name,
            ctx.topic,
            what,
            ctx.event_id,
            reason,
            action,
        )
        return _DROP

    def _raise(
        self, ctx: _DispatchContext, resolved: _ResolvedEvent, key: str
    ) -> EventDispatchStatus:
        try:
            self._call_raise(ctx, resolved, key)
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
        ctx.state.not_found.clear(key)
        logger.debug(
            "Event route %r raised event %r on workflow %r.",
            ctx.route_name,
            resolved.event_name,
            resolved.instance_id,
        )
        return _SUCCESS

    def _call_raise(
        self, ctx: _DispatchContext, resolved: _ResolvedEvent, key: str
    ) -> None:
        """Raise with the call deadline; remember a timed-out raise that is still running."""
        timeout = ctx.target.call_timeout_seconds
        future = self._sidecar_caller.start(
            self._wf_client.raise_workflow_event,
            timeout,
            instance_id=resolved.instance_id,
            event_name=resolved.event_name,
            data=resolved.data,
        )
        try:
            future.result(timeout=timeout)
        except TimeoutError:
            # cancel() fails once a worker runs the call: it may still raise.
            if not future.cancel() and ctx.target.dedupe:
                self._track_timed_out_raise(ctx, resolved, key, future)
            raise

    def _track_timed_out_raise(
        self,
        ctx: _DispatchContext,
        resolved: _ResolvedEvent,
        key: str,
        future: Future[Any],
    ) -> None:
        tracked = TrackedRaise(identity=resolved.identity(), future=future)
        if ctx.state.timed_out.add(key, tracked):
            return
        logger.warning(
            "Event route %r: cannot track the timed-out raise of event %r on "
            "workflow %r (message id=%r): too many raises are still running. A "
            "redelivery may raise the event again.",
            ctx.route_name,
            resolved.event_name,
            resolved.instance_id,
            ctx.event_id,
        )


__all__ = [
    "DEFAULT_HOOK_POOL_SIZE",
    "DEFAULT_SIDECAR_POOL_SIZE",
    "EventDispatchStatus",
    "MAX_EVENT_IDENTIFIER_LENGTH",
    "PERMANENT_SIDECAR_ERROR_CODES",
    "RESERVED_EVENT_NAME_PREFIXES",
    "EventRouteResolutionError",
    "EventRouteTarget",
    "HookVerdict",
    "TERMINAL_WORKFLOW_STATUSES",
    "WorkflowEventDispatcher",
    "coerce_identifier",
    "is_instance_not_found_error",
    "is_reserved_event_name",
    "parse_field_path",
    "resolve_field",
    "serialize_event_data",
]
