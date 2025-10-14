from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Callable, Iterable, List, Optional, Type

from dapr.clients import DaprClient
from dapr.clients.grpc._response import TopicEventResponse
from dapr.common.pubsub.subscription import SubscriptionMessage

from dapr_agents.workflow.utils.messaging import (
    extract_cloudevent_data,
    validate_message_model,
)

logger = logging.getLogger(__name__)


def register_message_handlers(
    targets: Iterable[Any],
    dapr_client: DaprClient,
    *,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> List[Callable[[], None]]:
    """Discover and subscribe handlers decorated with `@message_router`.

    Scans each target:
      - If the target itself is a decorated function (has `_message_router_data`), it is registered.
      - If the target is an object, all its attributes are scanned for decorated callables.

    Subscriptions use Dapr's streaming API (`subscribe_with_handler`) which invokes your handler
    on a background thread. This function returns a list of "closer" callables. Invoking a closer
    will unsubscribe the corresponding handler.

    Args:
        targets: Functions and/or instances to inspect for `_message_router_data`.
        dapr_client: Active Dapr client used to create subscriptions.
        loop: Event loop to await async handlers. If omitted, uses the running loop
              or falls back to `asyncio.get_event_loop()`.

    Returns:
        A list of callables. Each callable, when invoked, closes the associated subscription.
    """
    # Resolve loop strategy once up front.
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

    closers: List[Callable[[], None]] = []

    def _iter_handlers(obj: Any):
        """Yield (owner, fn) pairs for decorated handlers on `obj`.

        If `obj` is itself a decorated function, yield (None, obj).
        If `obj` is an instance, scan its attributes for decorated callables.
        """
        meta = getattr(obj, "_message_router_data", None)
        if callable(obj) and meta:
            yield None, obj
            return

        for name in dir(obj):
            fn = getattr(obj, name)
            if callable(fn) and getattr(fn, "_message_router_data", None):
                yield obj, fn

    for target in targets:
        for owner, handler in _iter_handlers(target):
            meta = getattr(handler, "_message_router_data")
            schemas: List[Type[Any]] = meta.get("message_schemas") or []

            # Bind method to instance if needed (descriptor protocol).
            bound = (
                handler if owner is None else handler.__get__(owner, owner.__class__)
            )

            async def _invoke(
                bound_handler: Callable[..., Any],
                parsed: Any,
            ) -> TopicEventResponse:
                """Invoke the user handler (sync or async) and normalize the result."""
                result = bound_handler(parsed)
                if inspect.iscoroutine(result):
                    result = await result
                if isinstance(result, TopicEventResponse):
                    return result
                # Treat any truthy/None return as success unless user explicitly returns a response.
                return TopicEventResponse("success")

            def _make_handler(
                bound_handler: Callable[..., Any],
            ) -> Callable[[SubscriptionMessage], TopicEventResponse]:
                """Create a Dapr-compatible handler for a single decorated function."""

                def handler_fn(message: SubscriptionMessage) -> TopicEventResponse:
                    try:
                        # 1) Extract payload + CloudEvent metadata (bytes/str/dict are also supported by the extractor)
                        event_data, metadata = extract_cloudevent_data(message)

                        # 2) Validate against the first matching schema (or dict as fallback)
                        parsed = None
                        for model in schemas or [dict]:
                            try:
                                parsed = validate_message_model(model, event_data)
                                break
                            except Exception:
                                # Try the next schema; log at debug for signal without noise.
                                logger.debug(
                                    "Schema %r did not match payload; trying next.",
                                    model,
                                    exc_info=True,
                                )
                                continue

                        if parsed is None:
                            # Permanent schema mismatch → drop (DLQ if configured by Dapr)
                            logger.warning(
                                "No matching schema for message on topic %r; dropping. Raw payload: %r",
                                meta["topic"],
                                event_data,
                            )
                            return TopicEventResponse("drop")

                        # 3) Attach CE metadata for downstream consumers
                        if isinstance(parsed, dict):
                            parsed["_message_metadata"] = metadata
                        else:
                            setattr(parsed, "_message_metadata", metadata)

                        # 4) Bridge worker thread → event loop
                        if loop and loop.is_running():
                            fut = asyncio.run_coroutine_threadsafe(
                                _invoke(bound_handler, parsed), loop
                            )
                            return fut.result()
                        return asyncio.run(_invoke(bound_handler, parsed))

                    except Exception:
                        # Transient failure (I/O, handler crash, etc.) → retry
                        logger.exception("Message handler error; requesting retry.")
                        return TopicEventResponse("retry")

                return handler_fn

            close_fn = dapr_client.subscribe_with_handler(
                pubsub_name=meta["pubsub"],
                topic=meta["topic"],
                handler_fn=_make_handler(bound),
                dead_letter_topic=meta.get("dead_letter_topic"),
            )
            closers.append(close_fn)

    return closers
