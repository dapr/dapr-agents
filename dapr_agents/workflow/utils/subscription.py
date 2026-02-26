"""Streaming pub/sub subscription utilities for workflow message routing."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass
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


SchedulerFn = Callable[[Callable[..., Any], dict], Optional[str]]
TopicKey = Tuple[str, str]
BindingSchemaPair = Tuple["MessageRouteBinding", Type[Any]]


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
    """

    handler: Callable[..., Any]
    schemas: List[Type[Any]]
    pubsub: str
    topic: str
    dead_letter_topic: Optional[str]
    name: str


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
        raise RuntimeError(
            f"Consumer thread for {pubsub_name}:{topic_name} did not stop within "
            f"{THREAD_SHUTDOWN_TIMEOUT_SECONDS}s timeout. Thread may be a zombie."
        )


def _subscribe_message_bindings(
    bindings: List[MessageRouteBinding],
    *,
    dapr_client: DaprClient,
    loop: asyncio.AbstractEventLoop,
    delivery_mode: Literal["sync", "async"],
    queue_maxsize: int,
    deduper: Optional[DedupeBackend],
    wf_client: wf.DaprWorkflowClient,
    await_result: bool,
    await_timeout: Optional[int],
    fetch_payloads: bool,
    log_outcome: bool,
) -> List[Callable[[], None]]:
    """Internal implementation of streaming subscriptions.

    This function sets up streaming subscriptions for all bindings,
    grouping by (pubsub, topic) to create one subscription per unique topic.
    """
    queue: Optional[asyncio.Queue] = None
    worker_tasks: List[asyncio.Task] = []

    if delivery_mode == DELIVERY_MODE_ASYNC:
        if not loop.is_running():
            raise RuntimeError(
                f"delivery_mode='{DELIVERY_MODE_ASYNC}' requires an active running event loop."
            )
        queue = asyncio.Queue(maxsize=max(1, queue_maxsize))

    def _wait_for_completion(instance_id: str) -> Optional[WorkflowState]:
        try:
            return wf_client.wait_for_workflow_completion(
                instance_id,
                fetch_payloads=fetch_payloads,
                timeout_in_seconds=await_timeout,
            )
        except Exception:
            logger.exception(f"[wf] {instance_id}: error while waiting for completion")
            return None

    async def _await_and_log(instance_id: str) -> None:
        state = await asyncio.to_thread(_wait_for_completion, instance_id)
        _log_workflow_outcome(instance_id, state, log_outcome)

    async def _schedule_workflow(
        bound_workflow: Callable[..., Any], parsed: Any
    ) -> TopicEventResponse:
        try:
            wf_input, _ = _serialize_workflow_input(parsed)

            workflow_name = getattr(bound_workflow, "__name__", str(bound_workflow))
            input_json = json.dumps(wf_input, ensure_ascii=False, indent=2)
            logger.debug(f"Scheduling workflow: {workflow_name} | input={input_json}")

            instance_id = await asyncio.to_thread(
                wf_client.schedule_new_workflow,
                workflow=bound_workflow,
                input=wf_input,
            )
            logger.debug(f"Scheduled workflow={workflow_name} instance={instance_id}")

            if await_result and delivery_mode == DELIVERY_MODE_SYNC:
                state = await asyncio.to_thread(_wait_for_completion, instance_id)
                _log_workflow_outcome(instance_id, state, log_outcome)
                if state and state.runtime_status == WorkflowStatus.COMPLETED:
                    return TopicEventResponse(STATUS_SUCCESS)
                return TopicEventResponse(STATUS_RETRY)

            asyncio.create_task(_await_and_log(instance_id))
            return TopicEventResponse(STATUS_SUCCESS)
        except Exception:
            logger.exception("Workflow scheduling failed; requesting retry.")
            return TopicEventResponse(STATUS_RETRY)

    if queue is not None:

        async def _async_worker() -> None:
            assert queue is not None
            while True:
                workflow_callable, payload = await queue.get()
                try:
                    await _schedule_workflow(workflow_callable, payload)
                except Exception:
                    logger.exception("Async worker error while scheduling workflow.")
                    raise
                finally:
                    queue.task_done()

        for _ in range(max(1, len(bindings))):
            worker_tasks.append(loop.create_task(_async_worker()))

    bindings_by_topic_key = _group_bindings_by_topic(bindings)
    closers: List[Callable[[], None]] = []

    for (pubsub_name, topic_name), topic_bindings in bindings_by_topic_key.items():
        binding_schema_pairs = _build_binding_schema_pairs(topic_bindings)
        dead_letter_topic = topic_bindings[0].dead_letter_topic

        def _create_message_handler(
            pairs: List[BindingSchemaPair],
        ) -> Callable[[SubscriptionMessage], TopicEventResponse]:
            """Create a composite handler for a topic that routes to the correct binding."""

            def handler(message: SubscriptionMessage) -> TopicEventResponse:
                try:
                    event_data, metadata = extract_cloudevent_data(message)

                    if deduper is not None:
                        candidate_id = (metadata or {}).get("id") or (
                            f"{topic_name}:{hash(str(event_data))}"
                        )
                        try:
                            if deduper.seen(candidate_id):
                                logger.debug(
                                    f"Duplicate detected id={candidate_id} topic={topic_name}; dropping."
                                )
                                return TopicEventResponse(STATUS_SUCCESS)
                            deduper.mark(candidate_id)
                        except Exception:
                            logger.debug(
                                "Dedupe backend error; continuing.", exc_info=True
                            )

                    cloudevent_type = (metadata or {}).get("type")
                    ordered_pairs = _order_pairs_by_cloudevent_type(
                        pairs, cloudevent_type
                    )

                    for binding, schema in ordered_pairs:
                        try:
                            payload = (
                                event_data
                                if isinstance(event_data, dict)
                                else {"data": event_data}
                            )
                            parsed = validate_message_model(schema, payload)
                            _attach_metadata_to_payload(parsed, metadata)

                            if delivery_mode == DELIVERY_MODE_ASYNC:
                                assert queue is not None
                                loop.call_soon_threadsafe(
                                    queue.put_nowait,
                                    (binding.handler, parsed),
                                )
                                return TopicEventResponse(STATUS_SUCCESS)

                            if loop.is_running():
                                fut = asyncio.run_coroutine_threadsafe(
                                    _schedule_workflow(binding.handler, parsed), loop
                                )
                                return fut.result()

                            return asyncio.run(
                                _schedule_workflow(binding.handler, parsed)
                            )

                        except Exception:
                            continue

                    logger.warning(
                        f"No matching schema for topic={topic_name!r}; dropping. raw={event_data!r}"
                    )
                    return TopicEventResponse(STATUS_DROP)

                except Exception:
                    logger.exception("Message handler error; requesting retry.")
                    return TopicEventResponse(STATUS_RETRY)

            return handler

        handler_fn = _create_message_handler(binding_schema_pairs)

        subscription = dapr_client.subscribe(
            pubsub_name=pubsub_name,
            topic=topic_name,
            dead_letter_topic=dead_letter_topic,
        )

        def _run_consumer_loop(
            sub: Any,
            handler: Callable[[SubscriptionMessage], TopicEventResponse],
            ps_name: str,
            t_name: str,
        ) -> None:
            logger.debug(f"Starting stream consumer for {ps_name}:{t_name}")
            try:
                for msg in sub:
                    if msg is None:
                        continue
                    try:
                        response = handler(msg)
                        if response.status == STATUS_SUCCESS:
                            sub.respond_success(msg)
                        elif response.status == STATUS_RETRY:
                            sub.respond_retry(msg)
                        elif response.status == STATUS_DROP:
                            sub.respond_drop(msg)
                        else:
                            logger.warning(
                                f"Unknown status {response.status}, retrying"
                            )
                            sub.respond_retry(msg)
                    except Exception:
                        logger.exception(
                            f"Handler exception in stream {ps_name}:{t_name}"
                        )
                        try:
                            sub.respond_retry(msg)
                        except Exception:
                            logger.exception(
                                f"Failed to send retry response for {ps_name}:{t_name}"
                            )
                            raise
            except Exception as exc:
                logger.exception(
                    f"Stream consumer {ps_name}:{t_name} exited with error"
                )
                raise exc
            finally:
                try:
                    sub.close()
                except Exception:
                    pass

        consumer_thread = threading.Thread(
            target=_run_consumer_loop,
            args=(subscription, handler_fn, pubsub_name, topic_name),
            daemon=True,
        )
        consumer_thread.start()

        def _make_closer(
            sub: Any,
            thread: threading.Thread,
            ps_name: str,
            t_name: str,
        ) -> Callable[[], None]:
            def _close() -> None:
                _shutdown_thread(thread, sub, ps_name, t_name)

            return _close

        closers.append(
            _make_closer(subscription, consumer_thread, pubsub_name, topic_name)
        )
        logger.debug(
            f"Subscribed streaming to pubsub={pubsub_name} topic={topic_name} "
            f"(delivery={delivery_mode} await={await_result})"
        )

    if worker_tasks:

        def _make_cancel_all(tasks: List[asyncio.Task]) -> Callable[[], None]:
            def _cancel() -> None:
                for task in tasks:
                    try:
                        task.cancel()
                    except Exception:
                        logger.debug("Error cancelling worker task.", exc_info=True)

            return _cancel

        closers.append(_make_cancel_all(worker_tasks))

    return closers


def subscribe_message_bindings(
    bindings: List[MessageRouteBinding],
    *,
    dapr_client: DaprClient,
    loop: Optional[asyncio.AbstractEventLoop],
    delivery_mode: Literal["sync", "async"],
    queue_maxsize: int,
    deduper: Optional[DedupeBackend],
    scheduler: Optional[SchedulerFn],
    wf_client: Optional[wf.DaprWorkflowClient],
    await_result: bool,
    await_timeout: Optional[int],
    fetch_payloads: bool,
    log_outcome: bool,
) -> List[Callable[[], None]]:
    """Set up streaming subscriptions for message route bindings.

    Args:
        bindings: List of message route bindings to subscribe.
        dapr_client: Active Dapr client for creating subscriptions.
        loop: Event loop for async operations (required for async delivery mode).
        delivery_mode: 'sync' blocks the Dapr thread; 'async' enqueues onto workers.
        queue_maxsize: Max in-flight messages for async mode.
        deduper: Optional idempotency backend.
        scheduler: Unused (retained for API compatibility).
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

    resolved_loop = _resolve_event_loop(loop)
    resolved_wf_client = wf_client or wf.DaprWorkflowClient()

    return _subscribe_message_bindings(
        bindings,
        dapr_client=dapr_client,
        loop=resolved_loop,
        delivery_mode=delivery_mode,
        queue_maxsize=queue_maxsize,
        deduper=deduper,
        wf_client=resolved_wf_client,
        await_result=await_result,
        await_timeout=await_timeout,
        fetch_payloads=fetch_payloads,
        log_outcome=log_outcome,
    )
