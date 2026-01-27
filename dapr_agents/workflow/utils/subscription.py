from __future__ import annotations

import asyncio
import logging
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from typing import (
    Any,
    Callable,
    List,
    Literal,
    Optional,
    Protocol,
    Type,
)

import dapr.ext.workflow as wf
from dapr.clients import DaprClient
from dapr.clients.grpc._response import TopicEventResponse
from dapr.common.pubsub.subscription import SubscriptionMessage
from dapr.ext.workflow.workflow_state import WorkflowState

from dapr_agents.workflow.utils.routers import (
    extract_cloudevent_data,
    validate_message_model,
)

logger = logging.getLogger(__name__)


class DedupeBackend(Protocol):
    """Idempotency backend contract (best-effort duplicate detection)."""

    def seen(self, key: str) -> bool: ...

    def mark(self, key: str) -> None: ...


SchedulerFn = Callable[[Callable[..., Any], dict], Optional[str]]


@dataclass
class MessageRouteBinding:
    """Internal binding definition for a message route."""

    handler: Callable[..., Any]
    schemas: List[Type[Any]]
    pubsub: str
    topic: str
    dead_letter_topic: Optional[str]
    name: str


def resolve_loop(
    loop: Optional[asyncio.AbstractEventLoop],
) -> asyncio.AbstractEventLoop:
    if loop is not None:
        return loop
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.get_event_loop()


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
    if not bindings:
        return []

    loop = resolve_loop(loop)
    if delivery_mode not in ("sync", "async"):
        raise ValueError("delivery_mode must be 'sync' or 'async'")

    queue: Optional[asyncio.Queue] = None
    worker_tasks: List[asyncio.Task] = []

    if delivery_mode == "async":
        if not loop or not loop.is_running():
            raise RuntimeError(
                "delivery_mode='async' requires an active running event loop."
            )
        queue = asyncio.Queue(maxsize=max(1, queue_maxsize))

    _wf_client = wf_client or wf.DaprWorkflowClient()

    def _default_scheduler(
        workflow_callable: Callable[..., Any], wf_input: dict
    ) -> Optional[str]:
        try:
            import json

            logger.debug(
                "-> Scheduling workflow: %s | input=%s",
                getattr(workflow_callable, "__name__", str(workflow_callable)),
                json.dumps(wf_input, ensure_ascii=False, indent=2),
            )
        except Exception:
            logger.warning("Could not serialize wf_input for logging", exc_info=True)
        return _wf_client.schedule_new_workflow(
            workflow=workflow_callable, input=wf_input
        )

    _scheduler: SchedulerFn = scheduler or _default_scheduler

    def _log_state(instance_id: str, state: Optional[WorkflowState]) -> None:
        if not state:
            logger.warning("[wf] %s: no state (timeout/missing).", instance_id)
            return
        status = getattr(state.runtime_status, "name", str(state.runtime_status))
        if status == "COMPLETED":
            if log_outcome:
                logger.info(
                    "[wf] %s COMPLETED output=%s",
                    instance_id,
                    getattr(state, "serialized_output", None),
                )
            return
        failure = getattr(state, "failure_details", None)
        if failure:
            logger.error(
                "[wf] %s FAILED type=%s message=%s\n%s",
                instance_id,
                getattr(failure, "error_type", None),
                getattr(failure, "message", None),
                getattr(failure, "stack_trace", "") or "",
            )
        else:
            logger.error(
                "[wf] %s finished with status=%s custom_status=%s",
                instance_id,
                status,
                getattr(state, "serialized_custom_status", None),
            )

    def _wait_for_completion(instance_id: str) -> Optional[WorkflowState]:
        try:
            return _wf_client.wait_for_workflow_completion(
                instance_id,
                fetch_payloads=fetch_payloads,
                timeout_in_seconds=await_timeout,
            )
        except Exception:
            logger.exception("[wf] %s: error while waiting for completion", instance_id)
            return None

    async def _await_and_log(instance_id: str) -> None:
        state = await asyncio.to_thread(_wait_for_completion, instance_id)
        _log_state(instance_id, state)

    async def _schedule(
        bound_workflow: Callable[..., Any], parsed: Any
    ) -> TopicEventResponse:
        try:
            metadata: Optional[dict] = None
            if isinstance(parsed, dict):
                wf_input = dict(parsed)
                metadata = wf_input.get("_message_metadata")
            elif hasattr(parsed, "model_dump"):
                metadata = getattr(parsed, "_message_metadata", None)
                wf_input = parsed.model_dump()
            elif is_dataclass(parsed):
                metadata = getattr(parsed, "_message_metadata", None)
                wf_input = asdict(parsed)
            else:
                metadata = getattr(parsed, "_message_metadata", None)
                wf_input = {"data": parsed}

            if metadata:
                wf_input["_message_metadata"] = dict(metadata)

            instance_id = await asyncio.to_thread(_scheduler, bound_workflow, wf_input)
            logger.info(
                "Scheduled workflow=%s instance=%s",
                getattr(bound_workflow, "__name__", str(bound_workflow)),
                instance_id,
            )

            if await_result and delivery_mode == "sync":
                state = await asyncio.to_thread(_wait_for_completion, instance_id)
                _log_state(instance_id, state)
                if state and getattr(state.runtime_status, "name", "") == "COMPLETED":
                    return TopicEventResponse("success")
                return TopicEventResponse("retry")

            asyncio.create_task(_await_and_log(instance_id))
            return TopicEventResponse("success")
        except Exception:
            logger.exception("Workflow scheduling failed; requesting retry.")
            return TopicEventResponse("retry")

    if queue is not None:

        async def _worker() -> None:
            while True:
                workflow_callable, payload = await queue.get()
                try:
                    await _schedule(workflow_callable, payload)
                except Exception:
                    logger.exception("Async worker crashed while scheduling workflow.")
                finally:
                    queue.task_done()

        for _ in range(max(1, len(bindings))):
            worker_tasks.append(loop.create_task(_worker()))  # type: ignore[union-attr]

    # ---------------- NEW: group by (pubsub, topic) and build ONE composite handler per topic -------------
    grouped: dict[tuple[str, str], list[MessageRouteBinding]] = defaultdict(list)
    for b in bindings:
        grouped[(b.pubsub, b.topic)].append(b)

    def _composite_handler_fn(
        group: list[MessageRouteBinding],
    ) -> Callable[[SubscriptionMessage], TopicEventResponse]:
        # Flatten a plan: [(binding, model), ...] preserving declaration order
        plan: list[tuple[MessageRouteBinding, Type[Any]]] = []
        for b in group:
            for m in b.schemas or [dict]:
                plan.append((b, m))

        def handler(message: SubscriptionMessage) -> TopicEventResponse:
            try:
                event_data, metadata = extract_cloudevent_data(message)

                # Optional: simple idempotency hook
                if deduper is not None:
                    candidate_id = (metadata or {}).get(
                        "id"
                    ) or f"{group[0].topic}:{hash(str(event_data))}"
                    try:
                        if deduper.seen(candidate_id):
                            logger.info(
                                "Duplicate detected id=%s topic=%s; dropping.",
                                candidate_id,
                                group[0].topic,
                            )
                            return TopicEventResponse("success")
                        deduper.mark(candidate_id)
                    except Exception:
                        logger.debug("Dedupe backend error; continuing.", exc_info=True)

                # (Optional) fast-path by CloudEvent type == model name (if publisher sets ce-type)
                ce_type = (metadata or {}).get("type")
                ordered_iter = plan
                if ce_type:
                    preferred = [
                        pair
                        for pair in plan
                        if getattr(pair[1], "__name__", "") == ce_type
                    ]
                    if preferred:
                        # Try preferred models first, then the rest
                        tail = [pair for pair in plan if pair not in preferred]
                        ordered_iter = preferred + tail

                # Try to validate against each model and dispatch to its handler
                for binding, model in ordered_iter:
                    try:
                        payload = (
                            event_data
                            if isinstance(event_data, dict)
                            else {"data": event_data}
                        )
                        parsed = validate_message_model(model, payload)
                        # attach metadata
                        try:
                            if isinstance(parsed, dict):
                                parsed["_message_metadata"] = metadata
                            else:
                                setattr(parsed, "_message_metadata", metadata)
                        except Exception:
                            logger.debug(
                                "Could not attach _message_metadata; continuing.",
                                exc_info=True,
                            )

                        # enqueue/schedule to the right handler
                        if delivery_mode == "async":
                            assert queue is not None
                            loop.call_soon_threadsafe(
                                queue.put_nowait,
                                (binding.handler, parsed),  # type: ignore[union-attr]
                            )
                            return TopicEventResponse("success")

                        if loop and loop.is_running():
                            fut = asyncio.run_coroutine_threadsafe(
                                _schedule(binding.handler, parsed), loop
                            )
                            return fut.result()

                        return asyncio.run(_schedule(binding.handler, parsed))

                    except Exception:
                        # Not a match for this model → keep trying
                        continue

                # No model matched for this topic → drop (or switch to "retry" if you prefer)
                logger.warning(
                    "No matching schema for topic=%r; dropping. raw=%r",
                    group[0].topic,
                    event_data,
                )
                return TopicEventResponse("drop")

            except Exception:
                logger.exception("Message handler error; requesting retry.")
                return TopicEventResponse("retry")

        return handler

    closers: List[Callable[[], None]] = []

    # subscribe one composite handler per (pubsub, topic)
    for (pubsub_name, topic_name), group in grouped.items():
        # Validate dead_letter_topic consistency
        dl_topics = {b.dead_letter_topic for b in group}
        if len(dl_topics) > 1:
            logger.warning(
                "Multiple dead_letter_topics found for %s:%s %s. Using %s.",
                pubsub_name,
                topic_name,
                dl_topics,
                group[0].dead_letter_topic,
            )

        handler_fn = _composite_handler_fn(group)

        # Streaming subscription (Default)
        subscription = dapr_client.subscribe(
            pubsub_name=pubsub_name,
            topic=topic_name,
            dead_letter_topic=group[0].dead_letter_topic,
        )

        def _consumer_loop(
            subscription: Any,
            handler_fn: Callable[[SubscriptionMessage], TopicEventResponse],
            pubsub_name: str,
            topic_name: str,
        ) -> None:
            logger.info("Starting stream consumer for %s:%s", pubsub_name, topic_name)
            try:
                for message in subscription:
                    if message is None:
                        continue
                    try:
                        response = handler_fn(message)
                        if response.status == "success":
                            subscription.respond_success(message)
                        elif response.status == "retry":
                            subscription.respond_retry(message)
                        elif response.status == "drop":
                            subscription.respond_drop(message)
                        else:
                            logger.warning(
                                "Unknown status %s, retrying", response.status
                            )
                            subscription.respond_retry(message)
                    except Exception:
                        logger.exception(
                            "Handler exception in stream %s:%s",
                            pubsub_name,
                            topic_name,
                        )
                        # Default to retry on handler crash
                        try:
                            subscription.respond_retry(message)
                        except Exception:
                            logger.exception(
                                "Failed to send retry response for %s:%s",
                                pubsub_name,
                                topic_name,
                            )
            except Exception as e:
                # If closed explicitly, we might get an error or simple exit
                logger.debug(
                    "Stream consumer %s:%s exited: %s", pubsub_name, topic_name, e
                )
            finally:
                subscription.close()

        # Start consumer in a separate thread to avoid blocking the event loop
        t = threading.Thread(
            target=_consumer_loop,
            args=(subscription, handler_fn, pubsub_name, topic_name),
            daemon=True,  # Ensure process can exit if thread hangs
        )
        t.start()

        def _make_streaming_closer(
            sub: Any,
            thread: threading.Thread,
        ) -> Callable[[], None]:
            def _close() -> None:
                try:
                    sub.close()  # Signal the subscription to stop
                except Exception:
                    logger.debug("Error closing subscription", exc_info=True)

                # Wait for thread to finish with a timeout
                thread.join(timeout=10.0)
                if thread.is_alive():
                    logger.warning(
                        "Consumer thread for %s:%s did not stop within timeout; abandoning.",
                        pubsub_name,
                        topic_name,
                    )

            return _close

        closers.append(_make_streaming_closer(subscription, t))
        logger.info(
            "Subscribed STREAMING to pubsub=%s topic=%s (delivery=%s await=%s)",
            pubsub_name,
            topic_name,
            delivery_mode,
            await_result,
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
