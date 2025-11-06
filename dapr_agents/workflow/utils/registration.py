from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import asdict, dataclass, is_dataclass
from typing import (
    Any,
    Callable,
    Iterable,
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
from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse, Response

from dapr_agents.types.workflow import HttpRouteSpec, PubSubRouteSpec
from dapr_agents.workflow.utils.routers import (
    extract_cloudevent_data,
    parse_http_json,
    validate_message_model,
)

logger = logging.getLogger(__name__)


class DedupeBackend(Protocol):
    """Idempotency backend contract (best-effort duplicate detection)."""

    def seen(self, key: str) -> bool:
        ...

    def mark(self, key: str) -> None:
        ...


SubscribeFn = Callable[..., Callable[[], None]]
SchedulerFn = Callable[[Callable[..., Any], dict], Optional[str]]


@dataclass
class _MessageRouteBinding:
    handler: Callable[..., Any]
    schemas: List[Type[Any]]
    pubsub: str
    topic: str
    dead_letter_topic: Optional[str]
    name: str


@dataclass
class _HttpRouteBinding:
    handler: Callable[..., Any]
    schemas: List[Type[Any]]
    method: str
    path: str
    summary: Optional[str]
    tags: List[str]
    response_model: Optional[Type[Any]]
    name: str


def _resolve_loop(
    loop: Optional[asyncio.AbstractEventLoop],
) -> asyncio.AbstractEventLoop:
    if loop is not None:
        return loop
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.get_event_loop()


def _iter_decorated(target: Any, attr: str):
    meta = getattr(target, attr, None)
    if callable(target) and meta:
        yield None, target, meta
        return
    for name in dir(target):
        fn = getattr(target, name)
        meta = getattr(fn, attr, None)
        if callable(fn) and meta:
            yield target, fn, meta


def _collect_message_bindings(
    *,
    targets: Optional[Iterable[Any]],
    routes: Optional[Iterable[PubSubRouteSpec]],
) -> List[_MessageRouteBinding]:
    bindings: List[_MessageRouteBinding] = []

    if targets:
        for target in targets:
            for owner, handler, meta in _iter_decorated(target, "_message_router_data"):
                bound = (
                    handler
                    if owner is None
                    else handler.__get__(owner, owner.__class__)
                )
                schemas = list(meta.get("message_schemas") or [dict])
                pubsub = meta.get("pubsub")
                topic = meta.get("topic")
                if not pubsub or not topic:
                    raise ValueError(
                        f"@message_router '{getattr(bound, '__name__', bound)}' is missing pubsub/topic. "
                        "Provide them in the decorator or register via PubSubRouteSpec with explicit values."
                    )
                bindings.append(
                    _MessageRouteBinding(
                        handler=bound,
                        schemas=schemas,
                        pubsub=pubsub,
                        topic=topic,
                        dead_letter_topic=meta.get("dead_letter_topic"),
                        name=getattr(bound, "__name__", str(bound)),
                    )
                )

    if routes:
        for spec in routes:
            bound = spec.handler_fn
            meta = getattr(bound, "_message_router_data", None)
            if spec.message_model is not None:
                schemas = [spec.message_model]
            elif meta and meta.get("message_schemas"):
                schemas = list(meta.get("message_schemas"))
            else:
                schemas = [dict]
            bindings.append(
                _MessageRouteBinding(
                    handler=bound,
                    schemas=schemas,
                    pubsub=spec.pubsub_name,
                    topic=spec.topic,
                    dead_letter_topic=spec.dead_letter_topic,
                    name=getattr(bound, "__name__", str(bound)),
                )
            )

    return bindings


def _collect_http_bindings(
    *,
    targets: Optional[Iterable[Any]],
    routes: Optional[Iterable[HttpRouteSpec]],
) -> List[_HttpRouteBinding]:
    bindings: List[_HttpRouteBinding] = []

    if targets:
        for target in targets:
            for owner, handler, meta in _iter_decorated(target, "_http_route_data"):
                bound = (
                    handler
                    if owner is None
                    else handler.__get__(owner, owner.__class__)
                )
                schemas = list(meta.get("request_schemas") or [dict])
                bindings.append(
                    _HttpRouteBinding(
                        handler=bound,
                        schemas=schemas,
                        method=meta["method"],
                        path=meta["path"],
                        summary=meta.get("summary"),
                        tags=list(meta.get("tags") or []),
                        response_model=meta.get("response_model"),
                        name=getattr(bound, "__name__", str(bound)),
                    )
                )

    if routes:
        for spec in routes:
            bound = spec.handler_fn
            meta = getattr(bound, "_http_route_data", None)
            if spec.request_model is not None:
                schemas = [spec.request_model]
            elif meta and meta.get("request_schemas"):
                schemas = list(meta.get("request_schemas"))
            else:
                schemas = [dict]

            summary = (
                spec.summary
                if spec.summary is not None
                else (meta.get("summary") if meta else None)
            )
            tags = list(spec.tags or (meta.get("tags") if meta else []) or [])
            response_model = (
                spec.response_model
                if spec.response_model is not None
                else (meta.get("response_model") if meta else None)
            )
            method = spec.method or (meta.get("method") if meta else "POST")

            bindings.append(
                _HttpRouteBinding(
                    handler=bound,
                    schemas=schemas,
                    method=method,
                    path=spec.path,
                    summary=summary,
                    tags=tags,
                    response_model=response_model,
                    name=getattr(bound, "__name__", str(bound)),
                )
            )

    return bindings


def _subscribe_message_bindings(
    bindings: List[_MessageRouteBinding],
    *,
    dapr_client: DaprClient,
    loop: Optional[asyncio.AbstractEventLoop],
    delivery_mode: Literal["sync", "async"],
    queue_maxsize: int,
    deduper: Optional[DedupeBackend],
    subscribe: Optional[SubscribeFn],
    scheduler: Optional[SchedulerFn],
    wf_client: Optional[wf.DaprWorkflowClient],
    await_result: bool,
    await_timeout: Optional[int],
    fetch_payloads: bool,
    log_outcome: bool,
) -> List[Callable[[], None]]:
    if not bindings:
        return []

    loop = _resolve_loop(loop)
    if subscribe is None:
        subscribe = dapr_client.subscribe_with_handler  # type: ignore[assignment]
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
                "➡️ Scheduling workflow: %s | input=%s",
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
    from collections import defaultdict

    grouped: dict[tuple[str, str], list[_MessageRouteBinding]] = defaultdict(list)
    for b in bindings:
        grouped[(b.pubsub, b.topic)].append(b)

    def _composite_handler_fn(
        group: list[_MessageRouteBinding],
    ) -> Callable[[SubscriptionMessage], TopicEventResponse]:
        # Flatten a plan: [(binding, model), ...] preserving declaration order
        plan: list[tuple[_MessageRouteBinding, Type[Any]]] = []
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
                                queue.put_nowait, (binding.handler, parsed)
                            )  # type: ignore[union-attr]
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
        handler_fn = _composite_handler_fn(group)
        close_fn = subscribe(  # type: ignore[misc]
            pubsub_name=pubsub_name,
            topic=topic_name,
            handler_fn=handler_fn,
            dead_letter_topic=group[0].dead_letter_topic,
        )
        logger.info(
            "Subscribed COMPOSITE(%d handlers) to pubsub=%s topic=%s (delivery=%s await=%s)",
            len(group),
            pubsub_name,
            topic_name,
            delivery_mode,
            await_result,
        )
        closers.append(close_fn)

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


def _mount_http_bindings(
    bindings: List[_HttpRouteBinding],
    *,
    app: FastAPI,
    loop: Optional[asyncio.AbstractEventLoop],
) -> List[Callable[[], None]]:
    if not bindings:
        return []

    _ = _resolve_loop(
        loop
    )  # Parity with message registrar; FastAPI does not require it yet.
    closers: List[Callable[[], None]] = []

    async def _invoke(bound_handler: Callable[..., Any], parsed: Any) -> Any:
        result = bound_handler(parsed)
        if inspect.iscoroutine(result):
            result = await result
        return result

    for binding in bindings:
        _schemas = binding.schemas or [dict]
        _method = binding.method
        _path = binding.path
        _summary = binding.summary
        _tags = list(binding.tags)
        _response_model = binding.response_model
        _name = binding.name
        _handler = binding.handler

        def _make_endpoint(
            *,
            bound_handler: Callable[..., Any],
            schemas_b: List[Type[Any]],
            method_b: str,
            path_b: str,
            name_b: str,
        ) -> Callable[..., Any]:
            async def endpoint(body: Any = Body(...)) -> Any:
                try:
                    parsed = None
                    matched_model: Optional[Type[Any]] = None
                    for model in schemas_b:
                        try:
                            candidate, _ = parse_http_json(
                                body, model=model, attach_metadata=False
                            )
                            parsed = candidate
                            matched_model = model
                            break
                        except Exception:
                            logger.debug(
                                "HTTP schema %r did not match; trying next.",
                                model,
                                exc_info=True,
                            )

                    if parsed is None:
                        return JSONResponse(
                            status_code=422,
                            content={
                                "detail": "Request body did not match any expected schema"
                            },
                        )

                    if matched_model is not None:
                        logger.debug(
                            "Validated HTTP request for %s %s with model=%s",
                            method_b,
                            path_b,
                            getattr(matched_model, "__name__", str(matched_model)),
                        )

                    result = await _invoke(bound_handler, parsed)

                    if isinstance(result, Response):
                        return result
                    if isinstance(
                        result, (dict, list, str, int, float, bool, type(None))
                    ):
                        return result
                    return JSONResponse(content=result)

                except Exception:
                    logger.exception("HTTP handler error for %s %s.", method_b, path_b)
                    return JSONResponse(
                        status_code=500, content={"detail": "Internal Server Error"}
                    )

            endpoint.__name__ = f"{name_b}_endpoint"
            return endpoint

        endpoint = _make_endpoint(
            bound_handler=_handler,
            schemas_b=_schemas,
            method_b=_method,
            path_b=_path,
            name_b=_name,
        )

        app.add_api_route(
            _path,
            endpoint,
            methods=[_method],
            summary=_summary,
            tags=_tags,
            response_model=_response_model,
        )

        closers.append(lambda: None)
        logger.info("Mounted HTTP route %s %s -> %s", _method, _path, _name)

    return closers


def register_message_routes(
    *,
    dapr_client: DaprClient,
    targets: Optional[Iterable[Any]] = None,
    routes: Optional[Iterable[PubSubRouteSpec]] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
    delivery_mode: Literal["sync", "async"] = "sync",
    queue_maxsize: int = 1024,
    deduper: Optional[DedupeBackend] = None,
    subscribe: Optional[SubscribeFn] = None,
    scheduler: Optional[SchedulerFn] = None,
    wf_client: Optional[wf.DaprWorkflowClient] = None,
    await_result: bool = False,
    await_timeout: Optional[int] = None,
    fetch_payloads: bool = True,
    log_outcome: bool = True,
) -> List[Callable[[], None]]:
    """
    Register workflow-backed pub/sub routes via decorator discovery and/or explicit specs.

    Args:
        dapr_client: Active Dapr client used to create subscriptions.
        targets: Objects/functions containing `@message_router` callables to auto-discover.
        routes: Explicit `PubSubRouteSpec` entries to register.
        loop: Event loop used to await async work (required for `delivery_mode="async"`).
        delivery_mode: `"sync"` blocks the Dapr thread; `"async"` enqueues onto a worker queue.
        queue_maxsize: Max in-flight messages when `delivery_mode="async"`.
        deduper: Optional idempotency backend keyed by CloudEvent id/hash.
        subscribe: Optional override for `dapr_client.subscribe_with_handler`.
        scheduler: Optional `(callable, input_dict) -> instance_id` function.
        wf_client: Reused `DaprWorkflowClient` for scheduling/waiting.
        await_result: If `True` (sync only), wait for workflow completion and request retry on failure.
        await_timeout: Optional wait timeout in seconds.
        fetch_payloads: Include workflow payloads when waiting for completion.
        log_outcome: Log COMPLETED/FAILED status (either inline or via detached task).

    Returns:
        List of closers that unsubscribe handlers and cancel async workers.
    """
    if targets is None and routes is None:
        raise ValueError(
            "Provide `targets` and/or `routes` when registering message routes."
        )

    bindings = _collect_message_bindings(targets=targets, routes=routes)
    if not bindings:
        logger.info("No message routes discovered.")
        return []

    return _subscribe_message_bindings(
        bindings,
        dapr_client=dapr_client,
        loop=loop,
        delivery_mode=delivery_mode,
        queue_maxsize=queue_maxsize,
        deduper=deduper,
        subscribe=subscribe,
        scheduler=scheduler,
        wf_client=wf_client,
        await_result=await_result,
        await_timeout=await_timeout,
        fetch_payloads=fetch_payloads,
        log_outcome=log_outcome,
    )


def register_http_routes(
    *,
    app: FastAPI,
    targets: Optional[Iterable[Any]] = None,
    routes: Optional[Iterable[HttpRouteSpec]] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> List[Callable[[], None]]:
    """
    Mount FastAPI endpoints from `@http_router` targets and/or explicit `HttpRouteSpec` entries.

    Args:
        app: FastAPI application to register routes on.
        targets: Objects/functions containing decorated HTTP handlers to auto-discover.
        routes: Explicit HTTP specs to mount.
        loop: Optional loop reference (retained for symmetry/future async needs).

    Returns:
        List of no-op closers (API symmetry with message registrar).
    """
    if targets is None and routes is None:
        raise ValueError(
            "Provide `targets` and/or `routes` when registering HTTP routes."
        )

    bindings = _collect_http_bindings(targets=targets, routes=routes)
    if not bindings:
        logger.info("No HTTP routes discovered.")
        return []

    return _mount_http_bindings(bindings, app=app, loop=loop)
