"""Streaming pub/sub subscription utilities for workflow message routing."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Type,
)

import dapr.ext.workflow as wf
from dapr.clients import DaprClient
from dapr.clients.grpc._response import TopicEventResponse
from dapr.common.pubsub.subscription import SubscriptionMessage
from dapr.ext.workflow.workflow_state import WorkflowState, WorkflowStatus
from cachetools import TTLCache

from dapr_agents.types.message import EventMessageMetadata
from dapr_agents.workflow.utils.routers import (
    extract_cloudevent_data,
    validate_message_model,
)

logger = logging.getLogger(__name__)

# Delivery mode constants
DELIVERY_MODE_SYNC: Literal["sync"] = "sync"
DELIVERY_MODE_ASYNC: Literal["async"] = "async"

# Topic event response status constants
STATUS_SUCCESS = "success"
STATUS_RETRY = "retry"
STATUS_DROP = "drop"

# Metadata key for attaching message metadata to payloads
METADATA_KEY = "_message_metadata"

# Thread shutdown timeout in seconds
THREAD_SHUTDOWN_TIMEOUT_SECONDS = 10.0


class DedupeBackend(Protocol):
    """Idempotency backend contract (best-effort duplicate detection)."""

    def seen(self, key: str) -> bool: ...

    def mark(self, key: str) -> None: ...


class TTLDedupeBackend:
    """Thread-safe in-memory deduplication backend using a TTL cache.

    Entries expire after `ttl` seconds, so memory is bounded and old IDs
    are not retained indefinitely.  Suitable for Dapr at-least-once delivery
    where duplicate messages typically arrive within a short retry window.
    """

    def __init__(self, maxsize: int = 4096, ttl: float = 60.0) -> None:
        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._lock = threading.Lock()

    def seen(self, key: str) -> bool:
        with self._lock:
            return key in self._cache

    def mark(self, key: str) -> None:
        with self._lock:
            self._cache[key] = True


TopicKey = tuple[str, str]
BindingSchemaPair = tuple["MessageRouteBinding", type[Any]]

PayloadFilter = Callable[[Any, "MessageContext"], bool]
ModelFilter = Callable[[Any, "MessageContext"], bool]


def _validate_filter(
    filter_fn: Callable[[Any, "MessageContext"], bool] | None,
    filter_name: str,
) -> None:
    """Reject async or non-callable filters at registration time"""
    if filter_fn is None:
        return
    if not callable(filter_fn):
        raise TypeError(
            f"`{filter_name}` must be callable, got {type(filter_fn).__name__}."
        )
    call_attr = getattr(filter_fn, "__call__", None)
    is_async_callable = asyncio.iscoroutinefunction(
        filter_fn
    ) or asyncio.iscoroutinefunction(call_attr)
    if is_async_callable:
        raise TypeError(
            f"`{filter_name}` must be a synchronous callable; "
            "filters run on the consumer thread and cannot be `async def` "
            "(including callable objects whose `__call__` is async). "
            "For I/O-bound checks, do them in the workflow body."
        )


@dataclass(frozen=True)
class MessageContext:
    """Per-message context handed to ``@message_router`` filters.

    Built once per incoming CloudEvent and passed as the second positional
    argument to ``payload_filter`` and ``model_filter`` callables.

    Attributes:
        event: Parsed CloudEvent envelope (id, source, type, topic, headers, ...).
    """

    event: EventMessageMetadata


@dataclass
class MessageRouteBinding:
    """Internal binding definition for a message route.

    Attributes:
        handler: The workflow callable to invoke when a matching message arrives.
        schemas: List of Pydantic/dataclass models to validate incoming payloads.
        pubsub: The Dapr pub/sub component name.
        topic: The topic to subscribe to.
        dead_letter_topic: Optional topic for failed messages.
        name: Human-readable name for logging (typically the handler function name).
        payload_filter: Optional predicate on the raw event data + MessageContext.
            Returning False or raising skips this binding (next is tried).
            Runs before schema validation. Must not mutate its inputs.
        model_filter: Optional predicate on the validated message model + MessageContext.
            Returning False or raising skips this binding (next is tried).
            Runs after schema validation. Must not mutate the model.
    """

    handler: Callable[..., Any]
    schemas: list[type[Any]]
    pubsub: str
    topic: str
    dead_letter_topic: str | None
    name: str
    payload_filter: PayloadFilter | None = None
    model_filter: ModelFilter | None = None


def _resolve_event_loop(
    loop: Optional[asyncio.AbstractEventLoop],
) -> asyncio.AbstractEventLoop:
    """Resolve the event loop to use for async operations.

    Args:
        loop: Optional explicitly provided event loop.

    Returns:
        The resolved event loop.

    Raises:
        RuntimeError: If no event loop is available and none was provided.
    """
    if loop is not None:
        return loop
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError(
                    "Event loop is closed. Provide an active event loop."
                )
            return loop
        except RuntimeError as exc:
            raise RuntimeError(
                "No running event loop available. "
                "Provide an explicit loop or run from within an async context."
            ) from exc


def _validate_delivery_mode(delivery_mode: str) -> None:
    """Validate that delivery_mode is one of the allowed values."""
    if delivery_mode not in (DELIVERY_MODE_SYNC, DELIVERY_MODE_ASYNC):
        raise ValueError(
            f"delivery_mode must be '{DELIVERY_MODE_SYNC}' or '{DELIVERY_MODE_ASYNC}', "
            f"got '{delivery_mode}'"
        )


def _validate_dead_letter_topics(bindings: List[MessageRouteBinding]) -> None:
    """Validate that bindings don't have conflicting dead letter topics.

    Raises:
        ValueError: If multiple different dead_letter_topics are configured
            for bindings that share the same (pubsub, topic).
    """
    by_topic: Dict[TopicKey, set] = defaultdict(set)
    for binding in bindings:
        key = (binding.pubsub, binding.topic)
        if binding.dead_letter_topic:
            by_topic[key].add(binding.dead_letter_topic)

    for topic_key, dead_letter_topics in by_topic.items():
        if len(dead_letter_topics) > 1:
            raise ValueError(
                f"Multiple dead_letter_topics configured for {topic_key[0]}:{topic_key[1]}: "
                f"{dead_letter_topics}. Only one dead_letter_topic is supported per topic."
            )


def _warn_unreachable_bindings(bindings: List[MessageRouteBinding]) -> None:
    """Each message is routed to a single binding. An unconditional binding takes
    every message for its schema, so warn that later bindings sharing its
    (pubsub, topic) and schema never run."""
    grouped = _group_bindings_by_topic(bindings)
    for (pubsub, topic), topic_bindings in grouped.items():
        first_match: dict[type[Any], str] = {}
        for binding in topic_bindings:
            schemas = binding.schemas or [dict]
            for schema in schemas:
                winner = first_match.get(schema)
                if winner is None:
                    continue
                schema_name = getattr(schema, "__name__", schema)
                logger.warning(
                    f"{binding.name!r} never runs for {schema_name!r} on {pubsub}:{topic}: "
                    f"each message is routed to a single binding and {winner!r} is registered "
                    f"first. If both need to handle it, fan out from one workflow."
                )
            has_filter = (
                binding.payload_filter is not None or binding.model_filter is not None
            )
            if has_filter:
                continue
            for schema in schemas:
                first_match.setdefault(schema, binding.name)


def _group_bindings_by_topic(
    bindings: List[MessageRouteBinding],
) -> Dict[TopicKey, List[MessageRouteBinding]]:
    """Group bindings by (pubsub, topic) key."""
    bindings_by_topic_key: Dict[TopicKey, List[MessageRouteBinding]] = defaultdict(list)
    for binding in bindings:
        key = (binding.pubsub, binding.topic)
        bindings_by_topic_key[key].append(binding)
    return dict(bindings_by_topic_key)


def _build_binding_schema_pairs(
    bindings: List[MessageRouteBinding],
) -> List[BindingSchemaPair]:
    """Build a list of (binding, schema) pairs for message routing.

    Each binding can have multiple schemas; this flattens them into pairs
    to try in order when matching incoming messages.
    """
    pairs: List[BindingSchemaPair] = []
    for binding in bindings:
        schemas = binding.schemas or [dict]
        for schema in schemas:
            pairs.append((binding, schema))
    return pairs


def _order_pairs_by_cloudevent_type(
    pairs: List[BindingSchemaPair],
    cloudevent_type: Optional[str],
) -> List[BindingSchemaPair]:
    """Reorder binding-schema pairs to prioritize those matching the CloudEvent type.

    If the CloudEvent 'type' header matches a schema's class name, those pairs
    are tried first, followed by the remaining pairs in original order.
    """
    if not cloudevent_type:
        return pairs

    matching_ce_type_pairs = [
        pair for pair in pairs if getattr(pair[1], "__name__", "") == cloudevent_type
    ]
    if not matching_ce_type_pairs:
        return pairs

    remaining_pairs = [pair for pair in pairs if pair not in matching_ce_type_pairs]
    return matching_ce_type_pairs + remaining_pairs


def _normalize_status(status: Any) -> str | None:
    """Coerce a `TopicEventResponse` into a status constant, or `None`
    when the value does not match any of them.

    Accepts enums, strings, and the various string-like shapes the gRPC SDK has produced
    over time.
    """
    if hasattr(status, "name"):
        raw = status.name
    elif hasattr(status, "value"):
        raw = str(status.value)
    else:
        raw = str(status)
    lowered = raw.lower()
    for known in (STATUS_SUCCESS, STATUS_RETRY, STATUS_DROP):
        if known in lowered:
            return known
    return None


def _filter_accepts(
    filter_fn: Callable[[Any, "MessageContext"], bool] | None,
    value: Any,
    msg_ctx: "MessageContext",
    *,
    kind: str,
    binding_name: str,
) -> bool:
    """Run an optional message filter.

    True means proceed, False means skip the binding.
    An exception from a user-supplied filter is logged and treated as a filter rejection
    to avoid an infinite retry loop.
    """
    if filter_fn is None:
        return True
    try:
        return bool(filter_fn(value, msg_ctx))
    except Exception:
        logger.exception(
            f"{kind} for binding '{binding_name}' raised; skipping binding."
        )
        return False


def _attach_metadata_to_payload(parsed: Any, metadata: Optional[dict]) -> None:
    """Attach message metadata to the parsed payload (best effort)."""
    if metadata is None:
        return
    try:
        if isinstance(parsed, dict):
            parsed[METADATA_KEY] = metadata
        else:
            setattr(parsed, METADATA_KEY, metadata)
    except Exception:
        logger.debug(f"Could not attach {METADATA_KEY} to payload; continuing.")


def _serialize_workflow_input(parsed: Any) -> Tuple[dict, Optional[dict]]:
    """Convert parsed message to workflow input dict and extract metadata."""
    metadata: Optional[dict] = None

    if isinstance(parsed, dict):
        wf_input = dict(parsed)
        metadata = wf_input.get(METADATA_KEY)
    elif hasattr(parsed, "model_dump"):
        metadata = getattr(parsed, METADATA_KEY, None)
        wf_input = parsed.model_dump()
    elif is_dataclass(parsed):
        metadata = getattr(parsed, METADATA_KEY, None)
        wf_input = asdict(parsed)
    else:
        metadata = getattr(parsed, METADATA_KEY, None)
        wf_input = {"data": parsed}

    if metadata:
        wf_input[METADATA_KEY] = dict(metadata)

    return wf_input, metadata


def _log_workflow_outcome(
    instance_id: str,
    state: Optional[WorkflowState],
    log_outcome: bool,
) -> None:
    """Log workflow completion status."""
    if not state:
        logger.warning(f"[wf] {instance_id}: no state (timeout/missing).")
        return

    status = state.runtime_status
    if status == WorkflowStatus.COMPLETED:
        if log_outcome:
            output = getattr(state, "serialized_output", None)
            logger.debug(f"[wf] {instance_id} COMPLETED output={output}")
        return

    failure = getattr(state, "failure_details", None)
    if failure:
        error_type = getattr(failure, "error_type", None)
        error_message = getattr(failure, "message", None)
        stack_trace = getattr(failure, "stack_trace", "") or ""
        logger.error(
            f"[wf] {instance_id} FAILED type={error_type} message={error_message}\n{stack_trace}"
        )
    else:
        status_name = status.name if hasattr(status, "name") else str(status)
        custom_status = getattr(state, "serialized_custom_status", None)
        logger.error(
            f"[wf] {instance_id} finished with status={status_name} custom_status={custom_status}"
        )


def _shutdown_thread(
    thread: threading.Thread,
    subscription: Any,
    pubsub_name: str,
    topic_name: str,
) -> None:
    """Shutdown a consumer thread, raising if it becomes a zombie.

    Raises:
        RuntimeError: If the thread does not stop within the timeout.
    """
    try:
        subscription.close()
    except Exception:
        logger.exception(f"Error closing subscription for {pubsub_name}:{topic_name}")

    thread.join(timeout=THREAD_SHUTDOWN_TIMEOUT_SECONDS)
    if thread.is_alive():
        if thread.daemon:
            logger.warning(
                f"Consumer thread for {pubsub_name}:{topic_name} did not stop within "
                f"{THREAD_SHUTDOWN_TIMEOUT_SECONDS}s; it is a daemon thread and will "
                f"be reaped on process exit."
            )
        else:
            raise RuntimeError(
                f"Consumer thread for {pubsub_name}:{topic_name} did not stop within "
                f"{THREAD_SHUTDOWN_TIMEOUT_SECONDS}s timeout. Thread may be a zombie."
            )


class _StreamSubscriber:
    """Drives one round of streaming subscriptions.

    Holds the runtime config so methods don't have to thread a dozen parameters
    through their call sites.
    One instance per `subscribe_message_bindings` invocation, not reusable.
    """

    def __init__(
        self,
        *,
        dapr_client: DaprClient,
        loop: asyncio.AbstractEventLoop | None,
        delivery_mode: Literal["sync", "async"],
        deduper: DedupeBackend | None,
        wf_client: wf.DaprWorkflowClient,
        queue_maxsize: int,
        await_result: bool,
        await_timeout: int | None,
        fetch_payloads: bool,
        log_outcome: bool,
    ) -> None:
        self.dapr_client = dapr_client
        self.loop = loop
        self.delivery_mode = delivery_mode
        self.deduper = deduper
        self.wf_client = wf_client
        self.await_result = await_result
        self.await_timeout = await_timeout
        self.fetch_payloads = fetch_payloads
        self.log_outcome = log_outcome
        self.queue: asyncio.Queue | None = None
        self._worker_tasks: list[asyncio.Task] = []

        if delivery_mode == DELIVERY_MODE_ASYNC:
            if loop is None or not loop.is_running():
                raise RuntimeError(
                    f"delivery_mode='{DELIVERY_MODE_ASYNC}' requires an active running event loop."
                )
            self.queue = asyncio.Queue(maxsize=max(1, queue_maxsize))

    # ---- workflow scheduling --------------------------------------------------

    def _wait_for_completion(self, instance_id: str) -> WorkflowState | None:
        try:
            return self.wf_client.wait_for_workflow_completion(
                instance_id,
                fetch_payloads=self.fetch_payloads,
                timeout_in_seconds=self.await_timeout,
            )
        except Exception:
            logger.exception(f"[wf] {instance_id}: error while waiting for completion")
            return None

    async def _await_and_log(self, instance_id: str) -> None:
        state = await asyncio.to_thread(self._wait_for_completion, instance_id)
        _log_workflow_outcome(instance_id, state, self.log_outcome)

    def _spawn_outcome_logger(self, instance_id: str) -> None:
        """Fire-and-forget logging of workflow completion"""
        try:
            asyncio.get_running_loop()
            # We have a running loop, create a detached task
            asyncio.create_task(self._await_and_log(instance_id))
        except RuntimeError:
            # No running loop - use a background thread for outcome logging
            def _log_in_thread() -> None:
                state = self._wait_for_completion(instance_id)
                _log_workflow_outcome(instance_id, state, self.log_outcome)

            thread = threading.Thread(target=_log_in_thread, daemon=True)
            thread.start()

    async def _await_sync_result(self, instance_id: str) -> TopicEventResponse:
        state = await asyncio.to_thread(self._wait_for_completion, instance_id)
        _log_workflow_outcome(instance_id, state, self.log_outcome)
        if state and state.runtime_status == WorkflowStatus.COMPLETED:
            return TopicEventResponse(STATUS_SUCCESS)
        if state and state.runtime_status == WorkflowStatus.FAILED:
            logger.warning(
                f"Workflow {instance_id} failed; dropping message to avoid infinite retries."
            )
            return TopicEventResponse(STATUS_DROP)
        return TopicEventResponse(STATUS_RETRY)

    async def _schedule_workflow(
        self, handler: Callable[..., Any], parsed: Any
    ) -> TopicEventResponse:
        # All pub/sub-triggered handlers go through schedule_new_workflow for
        # durable execution. Downstream the workflow may be a full durable agent
        # run, or lightweight context storage from team peers -- kept as a
        # workflow for durability and future hook extensibility.
        try:
            wf_input, _ = _serialize_workflow_input(parsed)
            workflow_name = getattr(handler, "__name__", str(handler))
            input_json = json.dumps(wf_input, ensure_ascii=False, indent=2)
            logger.debug(f"Scheduling workflow: {workflow_name} | input={input_json}")

            instance_id = await asyncio.to_thread(
                self.wf_client.schedule_new_workflow,
                workflow=handler,
                input=wf_input,
            )
            logger.debug(f"Scheduled workflow={workflow_name} instance={instance_id}")

            if self.await_result and self.delivery_mode == DELIVERY_MODE_SYNC:
                return await self._await_sync_result(instance_id)

            self._spawn_outcome_logger(instance_id)
            return TopicEventResponse(STATUS_SUCCESS)
        except Exception:
            logger.exception("Workflow scheduling failed; requesting retry.")
            return TopicEventResponse(STATUS_RETRY)

    async def _async_worker(self) -> None:
        assert self.queue is not None
        while True:
            handler, payload = await self.queue.get()
            try:
                await self._schedule_workflow(handler, payload)
            except Exception:
                logger.exception("Async worker error while scheduling workflow.")
                raise
            finally:
                self.queue.task_done()

    # ---- per-message dispatch -------------------------------------------------

    def _has_running_loop(self) -> bool:
        return self.loop is not None and self.loop.is_running()

    def _enqueue_async(
        self, binding: MessageRouteBinding, parsed: Any
    ) -> TopicEventResponse:
        assert self.queue is not None and self.loop is not None
        # Backpressure-aware enqueue: block until the item is queued
        fut = asyncio.run_coroutine_threadsafe(
            self.queue.put((binding.handler, parsed)), self.loop
        )
        try:
            fut.result()
            return TopicEventResponse(STATUS_SUCCESS)
        except Exception:
            logger.exception(
                f"Failed to enqueue workflow task for handler {binding.name}; requesting retry."
            )
            return TopicEventResponse(STATUS_RETRY)

    def _schedule_on_loop(
        self, binding: MessageRouteBinding, parsed: Any
    ) -> TopicEventResponse:
        assert self.loop is not None
        fut = asyncio.run_coroutine_threadsafe(
            self._schedule_workflow(binding.handler, parsed), self.loop
        )
        try:
            return fut.result()
        except Exception:
            logger.exception(
                f"Failed to schedule workflow for handler {binding.name}; requesting retry."
            )
            return TopicEventResponse(STATUS_RETRY)

    def _schedule_standalone(
        self, binding: MessageRouteBinding, parsed: Any
    ) -> TopicEventResponse:
        try:
            return asyncio.run(self._schedule_workflow(binding.handler, parsed))
        except Exception:
            logger.exception(
                f"Failed to schedule workflow for handler {binding.name}; requesting retry."
            )
            return TopicEventResponse(STATUS_RETRY)

    def _dispatch(
        self, binding: MessageRouteBinding, parsed: Any
    ) -> TopicEventResponse:
        """Send a validated payload to workflow scheduling via the appropriate path."""
        if self.delivery_mode == DELIVERY_MODE_ASYNC and self._has_running_loop():
            return self._enqueue_async(binding, parsed)
        if self._has_running_loop():
            return self._schedule_on_loop(binding, parsed)
        return self._schedule_standalone(binding, parsed)

    def _dedup_id(
        self, metadata: dict | None, event_data: Any, topic_name: str
    ) -> str | None:
        """Compute the dedup key for this message, or None when dedup is disabled."""
        if self.deduper is None:
            return None
        return (metadata or {}).get("id") or f"{topic_name}:{hash(str(event_data))}"

    def _is_seen(self, candidate_id: str) -> bool:
        """Best-effort check; backend errors are swallowed and reported as 'not seen'."""
        assert self.deduper is not None
        try:
            return self.deduper.seen(candidate_id)
        except Exception:
            logger.debug("Dedupe backend seen() error; continuing.", exc_info=True)
            return False

    def _mark_seen(self, candidate_id: str) -> None:
        """Best-effort mark; backend errors are logged but never raise."""
        assert self.deduper is not None
        try:
            self.deduper.mark(candidate_id)
        except Exception:
            logger.debug("Dedupe backend mark() error; continuing.", exc_info=True)

    def _route_to_binding(
        self,
        pairs: list[BindingSchemaPair],
        topic_name: str,
        event_data: Any,
        metadata: dict | None,
    ) -> TopicEventResponse:
        """Pick the first matching binding and dispatch; DROP when nothing matches."""
        msg_ctx = MessageContext(
            event=EventMessageMetadata.model_validate(metadata or {}),
        )
        ordered_pairs = _order_pairs_by_cloudevent_type(
            pairs, (metadata or {}).get("type")
        )

        # Filters are per-binding, not per-schema. We iterate flattened
        # (binding, schema) pairs, so cache the payload_filter result and
        # remember model_filter rejections; once a binding rejects, every
        # remaining pair for that binding is skipped.
        payload_filter_cache: dict[int, bool] = {}
        model_filter_rejected: set[int] = set()

        for binding, schema in ordered_pairs:
            binding_key = id(binding)
            if binding_key in model_filter_rejected:
                continue
            if binding_key not in payload_filter_cache:
                payload_filter_cache[binding_key] = _filter_accepts(
                    binding.payload_filter,
                    event_data,
                    msg_ctx,
                    kind="payload_filter",
                    binding_name=binding.name,
                )
            if not payload_filter_cache[binding_key]:
                continue

            try:
                payload = (
                    event_data if isinstance(event_data, dict) else {"data": event_data}
                )
                parsed = validate_message_model(schema, payload)
                _attach_metadata_to_payload(parsed, metadata)
            except (ValueError, TypeError):
                # Validation/coercion errors, try next schema
                continue

            if not _filter_accepts(
                binding.model_filter,
                parsed,
                msg_ctx,
                kind="model_filter",
                binding_name=binding.name,
            ):
                model_filter_rejected.add(binding_key)
                continue

            return self._dispatch(binding, parsed)

        logger.warning(
            f"No binding accepted message on topic={topic_name!r} "
            f"(no schema matched or all filters rejected); dropping. raw={event_data!r}"
        )
        return TopicEventResponse(STATUS_DROP)

    def _handle_message(
        self,
        pairs: list[BindingSchemaPair],
        topic_name: str,
        message: SubscriptionMessage,
    ) -> TopicEventResponse:
        """Route one message to the matching binding.

        Dedup marks happen after a terminal outcome (SUCCESS or DROP). Marking
        on arrival would silently neutralize RETRY: the broker redelivers, the
        retry hits the dedup cache, and the message is ack'd without ever being
        re-processed.
        """
        try:
            event_data, metadata = extract_cloudevent_data(message)

            dedup_id = self._dedup_id(metadata, event_data, topic_name)
            if dedup_id is not None and self._is_seen(dedup_id):
                logger.debug(
                    f"Duplicate detected id={dedup_id} topic={topic_name}; dropping."
                )
                return TopicEventResponse(STATUS_SUCCESS)

            response = self._route_to_binding(pairs, topic_name, event_data, metadata)

            if dedup_id is not None and _normalize_status(response.status) in (
                STATUS_SUCCESS,
                STATUS_DROP,
            ):
                self._mark_seen(dedup_id)

            return response
        except Exception:
            logger.exception("Message handler error; requesting retry.")
            return TopicEventResponse(STATUS_RETRY)

    # ---- consumer thread ------------------------------------------------------

    def _run_consumer_loop(
        self,
        sub: Any,
        handler: Callable[[SubscriptionMessage], TopicEventResponse],
        pubsub_name: str,
        topic_name: str,
    ) -> None:
        logger.debug(f"Starting stream consumer for {pubsub_name}:{topic_name}")
        try:
            for msg in sub:
                if msg is None:
                    continue
                self._consume_one(sub, msg, handler, pubsub_name, topic_name)
        except Exception:
            logger.exception(
                f"Stream consumer {pubsub_name}:{topic_name} exited with error"
            )
            raise
        finally:
            try:
                sub.close()
            except Exception:
                pass

    def _consume_one(
        self,
        sub: Any,
        msg: SubscriptionMessage,
        handler: Callable[[SubscriptionMessage], TopicEventResponse],
        pubsub_name: str,
        topic_name: str,
    ) -> None:
        """Process one message and send the matching broker ack.

        Handler exceptions and unknown status values both fall through to RETRY.
        Ack failures propagate so the outer loop kills the consumer thread.
        """
        try:
            response = handler(msg)
            status = _normalize_status(response.status)
            if status is None:
                logger.warning(f"Unknown status {response.status}; retrying.")
                status = STATUS_RETRY
        except Exception:
            logger.exception(f"Handler exception in stream {pubsub_name}:{topic_name}")
            status = STATUS_RETRY

        responders = {
            STATUS_SUCCESS: sub.respond_success,
            STATUS_RETRY: sub.respond_retry,
            STATUS_DROP: sub.respond_drop,
        }
        try:
            responders[status](msg)
        except Exception:
            logger.exception(
                f"Failed to send {status} response for {pubsub_name}:{topic_name}"
            )
            raise

    # ---- public entry ---------------------------------------------------------

    def subscribe_all(
        self, bindings: list[MessageRouteBinding]
    ) -> list[Callable[[], None]]:
        closers: list[Callable[[], None]] = []

        if self.queue is not None:
            assert self.loop is not None
            for _ in range(max(1, len(bindings))):
                self._worker_tasks.append(self.loop.create_task(self._async_worker()))

        for (pubsub_name, topic_name), topic_bindings in _group_bindings_by_topic(
            bindings
        ).items():
            pairs = _build_binding_schema_pairs(topic_bindings)
            dead_letter_topic = topic_bindings[0].dead_letter_topic
            handler_fn = partial(self._handle_message, pairs, topic_name)

            subscription = self.dapr_client.subscribe(
                pubsub_name=pubsub_name,
                topic=topic_name,
                dead_letter_topic=dead_letter_topic,
            )
            consumer_thread = threading.Thread(
                target=self._run_consumer_loop,
                args=(subscription, handler_fn, pubsub_name, topic_name),
                daemon=True,
            )
            consumer_thread.start()

            closers.append(
                partial(
                    _shutdown_thread,
                    consumer_thread,
                    subscription,
                    pubsub_name,
                    topic_name,
                )
            )
            logger.debug(
                f"Subscribed streaming to pubsub={pubsub_name} topic={topic_name} "
                f"(delivery={self.delivery_mode} await={self.await_result})"
            )

        if self._worker_tasks:
            tasks_snapshot = list(self._worker_tasks)
            closers.append(partial(_cancel_tasks, tasks_snapshot))

        return closers


def _cancel_tasks(tasks: list[asyncio.Task]) -> None:
    for task in tasks:
        try:
            task.cancel()
        except Exception:
            logger.debug("Error cancelling worker task.", exc_info=True)


def subscribe_message_bindings(
    bindings: list[MessageRouteBinding],
    *,
    dapr_client: DaprClient,
    loop: asyncio.AbstractEventLoop | None,
    delivery_mode: Literal["sync", "async"],
    queue_maxsize: int,
    deduper: DedupeBackend | None,
    wf_client: wf.DaprWorkflowClient | None,
    await_result: bool,
    await_timeout: int | None,
    fetch_payloads: bool,
    log_outcome: bool,
) -> list[Callable[[], None]]:
    """Set up streaming subscriptions for message route bindings.

    Args:
        bindings: List of message route bindings to subscribe.
        dapr_client: Active Dapr client for creating subscriptions.
        loop: Event loop for async operations (required for async delivery mode).
        delivery_mode: 'sync' blocks the Dapr thread; 'async' enqueues onto workers.
        queue_maxsize: Max in-flight messages for async mode.
        deduper: Optional idempotency backend.
        wf_client: Workflow client for scheduling workflows.
        await_result: If True (sync only), wait for workflow completion.
        await_timeout: Timeout in seconds when awaiting workflow completion.
        fetch_payloads: Include payloads when waiting for completion.
        log_outcome: Log workflow completion status.

    Returns:
        List of closer functions to unsubscribe and cleanup resources.

    Raises:
        ValueError: If delivery_mode is invalid or dead_letter_topics conflict.
        RuntimeError: If async mode is used without a running event loop.
    """
    if not bindings:
        return []

    _validate_delivery_mode(delivery_mode)
    _validate_dead_letter_topics(bindings)
    _warn_unreachable_bindings(bindings)

    if delivery_mode == DELIVERY_MODE_ASYNC:
        resolved_loop = _resolve_event_loop(loop)
    else:
        # In sync mode we rely on asyncio.run(...) and don't require an existing
        # running event loop; avoid resolving it unconditionally.
        resolved_loop = loop
    resolved_wf_client = wf_client or wf.DaprWorkflowClient()

    subscriber = _StreamSubscriber(
        dapr_client=dapr_client,
        loop=resolved_loop,
        delivery_mode=delivery_mode,
        deduper=deduper,
        wf_client=resolved_wf_client,
        queue_maxsize=queue_maxsize,
        await_result=await_result,
        await_timeout=await_timeout,
        fetch_payloads=fetch_payloads,
        log_outcome=log_outcome,
    )
    return subscriber.subscribe_all(bindings)
