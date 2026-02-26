from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Literal,
    Optional,
    Type,
)

import dapr.ext.workflow as wf
from dapr.clients import DaprClient
from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse, Response

from dapr_agents.types.workflow import HttpRouteSpec, PubSubRouteSpec
from dapr_agents.workflow.utils.routers import parse_http_json
from dapr_agents.workflow.utils.subscription import (
    DedupeBackend,
    MessageRouteBinding,
    SchedulerFn,
    _resolve_event_loop,
    subscribe_message_bindings,
)

logger = logging.getLogger(__name__)


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
) -> List[MessageRouteBinding]:
    bindings: List[MessageRouteBinding] = []

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
                    MessageRouteBinding(
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
                MessageRouteBinding(
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


def _mount_http_bindings(
    bindings: List[_HttpRouteBinding],
    *,
    app: FastAPI,
    loop: Optional[asyncio.AbstractEventLoop],
) -> List[Callable[[], None]]:
    if not bindings:
        return []

    _ = _resolve_event_loop(
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

    return subscribe_message_bindings(
        bindings,
        dapr_client=dapr_client,
        loop=loop,
        delivery_mode=delivery_mode,
        queue_maxsize=queue_maxsize,
        deduper=deduper,
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
