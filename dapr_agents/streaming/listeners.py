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

"""Pluggable transports for ``AgentStreamChunk`` delivery.

``StreamListener`` is a small synchronous protocol used by ``StreamEmitter``
from inside workflow activities. Built-in implementations cover the common
transports; users register custom types via ``register_stream_listener``.

Listeners never raise from ``emit`` — failures are logged and swallowed. The
authoritative final message is always persisted via workflow state, so a
dropped chunk never corrupts durable state.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import queue
import threading
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol

from dapr.aio.clients import DaprClient as AsyncDaprClient

from dapr_agents.types.streaming import AgentStreamChunk
from dapr_agents.workflow.utils.pubsub import publish_message

logger = logging.getLogger(__name__)

ListenerFactory = Callable[..., "StreamListener"]

_BUILTIN_LISTENERS: Dict[str, ListenerFactory] = {}
_USER_LISTENERS: Dict[str, ListenerFactory] = {}
_REGISTRY_LOCK = threading.Lock()


class StreamListener(Protocol):
    """Receives ``AgentStreamChunk`` events from a ``StreamEmitter``.

    Contract for implementers (all four built-in listeners follow it):

    - **Thread safety.** ``emit`` is called from the workflow-activity
      thread, which is a different OS thread from any asyncio consumer
      loop. Implementations MUST NOT call asyncio primitives directly
      from ``emit``. If your transport feeds an asyncio queue, use
      ``loop.call_soon_threadsafe(queue.put_nowait, chunk)`` to bridge —
      see ``InProcessQueueListener`` for the reference pattern.
    - **Synchronous + non-blocking.** ``emit`` runs on the LLM read loop's
      critical path. Offload any blocking I/O (HTTP POST, gRPC publish) to
      a background worker thread with a bounded queue.
    - **Fire-and-forget.** ``emit`` MUST NOT raise. Log and swallow
      transport errors; the authoritative final message is persisted via
      workflow state, so a dropped chunk never corrupts durable state.
    - **Idempotent close.** ``close`` may be called multiple times (and
      from multiple sites — the session listener cache, the
      per-activity emitter, and the runner). Each call after the first
      must be a no-op.
    - **Order within the topic.** Emit order is the ``(sequence)`` order
      the emitter produced; implementations may reorder across transports
      (pub/sub brokers, webhook retries) but clients dedupe/order via
      ``chunk_id`` and ``(workflow_instance_id, sequence)``.
    """

    def emit(self, chunk: AgentStreamChunk) -> None: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# PubSubListener — Dapr pub/sub transport (topology-safe default)
# ---------------------------------------------------------------------------


class PubSubListener:
    """Publishes chunks to a Dapr pub/sub topic.

    Safe across processes and replicas; the right default when consumers may
    live elsewhere. Emits are enqueued on a bounded background queue so
    network latency does not dominate the LLM read loop. Dropped chunks are
    logged but not re-raised; ``TURN_COMPLETE`` / ``SESSION_COMPLETE`` acts as
    the backstop when deltas are lost.
    """

    def __init__(
        self,
        *,
        pubsub_name: str,
        topic: str,
        queue_max_size: int = 1024,
        drain_timeout_seconds: float = 5.0,
        batch_size: int = 8,
    ) -> None:
        if not pubsub_name:
            raise ValueError("PubSubListener requires a non-empty pubsub_name")
        if not topic:
            raise ValueError("PubSubListener requires a non-empty topic")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._pubsub_name = pubsub_name
        self._topic = topic
        self._queue: "queue.Queue[Optional[AgentStreamChunk]]" = queue.Queue(
            maxsize=queue_max_size
        )
        self._drain_timeout = drain_timeout_seconds
        self._batch_size = batch_size
        self._closed = False
        self._worker = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"pubsub-stream-{topic}",
        )
        self._worker.start()

    def emit(self, chunk: AgentStreamChunk) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            logger.warning(
                "PubSub stream listener queue full (topic=%s); dropping chunk %s",
                self._topic,
                chunk.chunk_id,
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Poison pill. Use put (blocking) so the worker eventually drains
        # everything still in the queue before exiting.
        try:
            self._queue.put(None, timeout=self._drain_timeout)
        except queue.Full:
            logger.error(
                "PubSub stream listener queue still full on close (topic=%s)",
                self._topic,
            )
        self._worker.join(timeout=self._drain_timeout)

    # -- internal ----------------------------------------------------------

    def _run(self) -> None:
        """Persistent worker loop.

        Runs a single ``asyncio`` event loop for the listener's lifetime,
        batches up to ``batch_size`` chunks per round-trip via
        ``asyncio.gather``, and reuses a single ``DaprClient`` across all
        publishes (the previous per-chunk ``loop.run_until_complete`` tore
        down the client every time, costing ~2-5 ms of gRPC warmup per
        chunk). Within-batch ordering is preserved by broker-side sequence
        + the envelope's ``sequence`` field; ``asyncio.gather`` fans out
        to amortise round-trip latency across the batch.
        """
        try:
            asyncio.run(self._worker_loop())
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "PubSub listener worker exited with error (topic=%s): %s",
                self._topic,
                exc,
            )

    async def _worker_loop(self) -> None:
        loop = asyncio.get_running_loop()

        pending_exit = False

        # Drain a bounded batch from the thread-safe queue in the executor
        # so we don't block the loop. The first ``get`` blocks with a
        # short timeout; if no work arrives, we poll again so close() can
        # still deliver the sentinel promptly. Subsequent non-blocking
        # gets fill the batch up to ``batch_size`` when the producer is
        # faster than the broker.
        def _drain_batch() -> List[Optional[AgentStreamChunk]]:
            try:
                first = self._queue.get(timeout=1.0)
            except queue.Empty:
                return []
            items: List[Optional[AgentStreamChunk]] = [first]
            while len(items) < self._batch_size:
                try:
                    items.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            return items

        # Open a single Dapr gRPC channel for the listener's lifetime.
        # Previously every publish opened+closed its own ``DaprClient``
        # context (via ``publish_message``'s ``client_factory``), so
        # ``asyncio.gather`` across a batch actually created N concurrent
        # channels — strictly worse than sequential on a shared channel.
        # Reusing one client cuts ~2-5ms of gRPC warmup per chunk.
        async with AsyncDaprClient() as client:
            while not pending_exit:
                items = await loop.run_in_executor(None, _drain_batch)
                if not items:
                    continue

                # Separate sentinel from real chunks: sentinel signals shutdown.
                publishable: List[AgentStreamChunk] = []
                for item in items:
                    if item is None:
                        pending_exit = True
                    else:
                        publishable.append(item)

                if not publishable:
                    continue

                for chunk in publishable:
                    try:
                        await publish_message(
                            pubsub_name=self._pubsub_name,
                            topic_name=self._topic,
                            message=chunk.model_dump(mode="json"),
                            metadata={
                                "cloudevent.type": f"agent.stream.{chunk.type.value}",
                                "cloudevent.subject": chunk.root_instance_id,
                            },
                            client=client,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "PubSub stream publish failed (topic=%s chunk=%s): %s",
                            self._topic,
                            chunk.chunk_id,
                            exc,
                        )


# ---------------------------------------------------------------------------
# InProcessQueueListener — asyncio.Queue via process-local registry
# ---------------------------------------------------------------------------


class _InProcessQueueEntry:
    """Asyncio queue bound to a specific consumer event loop.

    The listener side is synchronous (it runs inside a workflow activity
    thread); the consumer side awaits ``asyncio.Queue.get`` on its own
    event loop. Bridging requires ``loop.call_soon_threadsafe`` — the
    listener thread schedules ``put_nowait`` on the consumer's loop so
    the waiter wakes up correctly.
    """

    __slots__ = ("queue", "loop")

    def __init__(
        self,
        queue_: "asyncio.Queue[Optional[AgentStreamChunk]]",
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.queue = queue_
        self.loop = loop


_INPROCESS_REGISTRY: Dict[str, _InProcessQueueEntry] = {}


def register_in_process_queue(
    registry_key: str,
    *,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> "asyncio.Queue[Optional[AgentStreamChunk]]":
    """Create and register an asyncio queue for the given key.

    Callers (typically ``AgentRunner.run_stream``) must register a queue
    before scheduling the workflow and call
    :func:`unregister_in_process_queue` when done. Chunks are drained by
    the consumer side (see ``dapr_agents.streaming.consumers``).

    Args:
        registry_key: Session-scoped key (typically the root instance id).
        loop: Explicit event loop to bind the queue to. When ``None``, the
            caller's currently running loop is captured — ``run_stream``
            calls this while its own coroutine is active, which is the
            loop the consumer will use.
    """

    if loop is None:
        loop = asyncio.get_running_loop()
    q: "asyncio.Queue[Optional[AgentStreamChunk]]" = asyncio.Queue()
    entry = _InProcessQueueEntry(q, loop)
    with _REGISTRY_LOCK:
        if registry_key in _INPROCESS_REGISTRY:
            raise ValueError(
                f"In-process queue already registered for key={registry_key}"
            )
        _INPROCESS_REGISTRY[registry_key] = entry
    return q


def unregister_in_process_queue(registry_key: str) -> None:
    """Remove the in-process queue entry and wake any blocked consumer."""

    with _REGISTRY_LOCK:
        entry = _INPROCESS_REGISTRY.pop(registry_key, None)
    if entry is None:
        return
    # Signal drain from whichever thread called us. If we're on the
    # consumer's loop, put_nowait is safe; otherwise, bounce through
    # call_soon_threadsafe so the async waiter actually wakes up.
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is entry.loop:
        entry.queue.put_nowait(None)
    else:
        try:
            entry.loop.call_soon_threadsafe(entry.queue.put_nowait, None)
        except RuntimeError:
            # Loop is already closed — nothing to signal.
            pass


def _get_in_process_entry(registry_key: str) -> _InProcessQueueEntry:
    with _REGISTRY_LOCK:
        entry = _INPROCESS_REGISTRY.get(registry_key)
    if entry is None:
        raise LookupError(
            f"No in-process queue registered for key={registry_key}. "
            "Did you forget to call register_in_process_queue() before "
            "scheduling the workflow?"
        )
    return entry


# Backwards-compat alias for external code that imported the helper
# under its old name; returns the underlying asyncio.Queue.
def _get_in_process_queue(
    registry_key: str,
) -> "asyncio.Queue[Optional[AgentStreamChunk]]":
    return _get_in_process_entry(registry_key).queue


class InProcessQueueListener:
    """Writes chunks to a process-local queue keyed by ``registry_key``.

    Suitable only when the producing activity runs in the same process as the
    consumer. Multi-replica deployments must use ``PubSubListener`` instead —
    see the streaming design doc.

    Bridges the listener's synchronous ``emit`` call (running on the
    workflow-activity thread) onto the consumer's ``asyncio`` loop via
    ``loop.call_soon_threadsafe`` so the waiter wakes promptly without the
    thread-pool hop that ``queue.SimpleQueue`` + ``run_in_executor``
    imposed previously.
    """

    def __init__(self, *, registry_key: str) -> None:
        if not registry_key:
            raise ValueError("InProcessQueueListener requires a registry_key")
        self._registry_key = registry_key
        # Resolve eagerly so misconfiguration fails fast at the root.
        entry = _get_in_process_entry(registry_key)
        self._queue = entry.queue
        self._loop = entry.loop
        self._closed = False

    def emit(self, chunk: AgentStreamChunk) -> None:
        if self._closed:
            return
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, chunk)
        except RuntimeError as exc:
            # Consumer's loop closed before we could enqueue — session
            # likely finished. Drop silently; the final message is in
            # durable state.
            logger.debug(
                "In-process stream loop closed (key=%s chunk=%s): %s",
                self._registry_key,
                chunk.chunk_id,
                exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "In-process stream enqueue failed (key=%s chunk=%s): %s",
                self._registry_key,
                chunk.chunk_id,
                exc,
            )

    def close(self) -> None:
        # Closing the listener does not drain/destroy the registry entry —
        # the consumer owns that lifecycle (it must call
        # unregister_in_process_queue when it finishes). Avoid double-posting
        # the sentinel: the consumer will detect end-of-stream via
        # SESSION_COMPLETE or workflow terminal.
        self._closed = True


# ---------------------------------------------------------------------------
# WebhookListener — HTTP POST per chunk
# ---------------------------------------------------------------------------


class WebhookListener:
    """POSTs each chunk as JSON to a user-supplied URL.

    Uses a bounded background thread + synchronous HTTP client. Failures are
    retried briefly then dropped with a warning. Stream chunks are best-effort.
    """

    def __init__(
        self,
        *,
        url: str,
        headers: Optional[Mapping[str, str]] = None,
        timeout_seconds: float = 5.0,
        retry_attempts: int = 2,
        queue_max_size: int = 1024,
        drain_timeout_seconds: float = 5.0,
    ) -> None:
        if not url:
            raise ValueError("WebhookListener requires a non-empty url")
        self._url = url
        self._headers = dict(headers or {})
        self._headers.setdefault("Content-Type", "application/json")
        self._timeout = timeout_seconds
        self._retry = max(0, int(retry_attempts))
        self._queue: "queue.Queue[Optional[AgentStreamChunk]]" = queue.Queue(
            maxsize=queue_max_size
        )
        self._drain_timeout = drain_timeout_seconds
        self._closed = False
        self._worker = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"webhook-stream-{url}",
        )
        self._worker.start()

    def emit(self, chunk: AgentStreamChunk) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            logger.warning(
                "Webhook listener queue full (url=%s); dropping chunk %s",
                self._url,
                chunk.chunk_id,
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put(None, timeout=self._drain_timeout)
        except queue.Full:
            logger.error(
                "Webhook listener queue still full on close (url=%s)", self._url
            )
        self._worker.join(timeout=self._drain_timeout)

    # -- internal ----------------------------------------------------------

    def _run(self) -> None:
        # Import lazily so we don't impose httpx on users who never use the
        # webhook listener.
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - optional dep
            logger.error(
                "WebhookListener requires httpx but it is not installed: %s", exc
            )
            return
        client = httpx.Client(timeout=self._timeout, headers=self._headers)
        try:
            while True:
                chunk = self._queue.get()
                if chunk is None:
                    return
                self._post_with_retries(client, chunk)
        finally:
            client.close()

    def _post_with_retries(self, client: Any, chunk: AgentStreamChunk) -> None:
        body = json.dumps(chunk.model_dump(mode="json"))
        last_exc: Optional[Exception] = None
        for attempt in range(self._retry + 1):
            try:
                response = client.post(self._url, content=body)
                if 200 <= response.status_code < 300:
                    return
                last_exc = RuntimeError(
                    f"webhook returned status {response.status_code}"
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        logger.warning(
            "Webhook emit failed after %d attempts (url=%s chunk=%s): %s",
            self._retry + 1,
            self._url,
            chunk.chunk_id,
            last_exc,
        )


# ---------------------------------------------------------------------------
# CompositeListener — fan-out to N listeners
# ---------------------------------------------------------------------------


class CompositeListener:
    """Fans each chunk out to every wrapped listener, isolating errors."""

    def __init__(self, listeners: Iterable[StreamListener]) -> None:
        self._listeners: List[StreamListener] = list(listeners)
        if not self._listeners:
            raise ValueError("CompositeListener requires at least one child listener")

    def emit(self, chunk: AgentStreamChunk) -> None:
        for listener in self._listeners:
            try:
                listener.emit(chunk)
            except Exception as exc:  # noqa: BLE001 - never raise from emit
                logger.error(
                    "Child listener %s raised during emit (chunk=%s): %s",
                    type(listener).__name__,
                    chunk.chunk_id,
                    exc,
                )

    def close(self) -> None:
        for listener in self._listeners:
            try:
                listener.close()
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Child listener %s raised during close: %s",
                    type(listener).__name__,
                    exc,
                )


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------


def _build_pubsub(config: Mapping[str, Any], **_: Any) -> StreamListener:
    return PubSubListener(
        pubsub_name=config["pubsub_name"],
        topic=config["topic"],
        queue_max_size=int(config.get("queue_max_size", 1024)),
        drain_timeout_seconds=float(config.get("drain_timeout_seconds", 5.0)),
    )


def _build_in_process(config: Mapping[str, Any], **_: Any) -> StreamListener:
    return InProcessQueueListener(registry_key=config["registry_key"])


def _build_webhook(config: Mapping[str, Any], **_: Any) -> StreamListener:
    return WebhookListener(
        url=config["url"],
        headers=config.get("headers"),
        timeout_seconds=float(config.get("timeout_seconds", 5.0)),
        retry_attempts=int(config.get("retry_attempts", 2)),
        queue_max_size=int(config.get("queue_max_size", 1024)),
        drain_timeout_seconds=float(config.get("drain_timeout_seconds", 5.0)),
    )


def _build_composite(
    config: Mapping[str, Any], *, allow_custom: bool = False, **kwargs: Any
) -> StreamListener:
    children = [
        build_listener(c, allow_custom=allow_custom, **kwargs)
        for c in config.get("listeners", [])
    ]
    return CompositeListener(children)


def _build_custom(
    config: Mapping[str, Any], *, allow_custom: bool = False, **kwargs: Any
) -> StreamListener:
    """Build a listener via a dotted-import factory path.

    **Never reachable from HTTP-supplied configs.** ``build_listener`` gates
    this factory behind ``allow_custom=True`` which the HTTP surface never
    passes. Keep it as a power-user escape hatch for Python callers that
    explicitly construct a ``listener`` arg to ``run_stream``.
    """
    if not allow_custom:
        raise ValueError(
            "custom listener factories require allow_custom=True; register "
            "the listener at startup via register_stream_listener() for "
            "HTTP-accessible configs."
        )
    factory_path = config.get("factory")
    if not factory_path or not isinstance(factory_path, str):
        raise ValueError("custom listener requires 'factory' as a dotted import string")
    module_name, _, attr_name = factory_path.rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid factory path '{factory_path}'")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name)
    extra_kwargs = dict(config.get("kwargs", {}))
    return factory(**extra_kwargs, **kwargs)


_BUILTIN_LISTENERS.update(
    {
        "pubsub": _build_pubsub,
        "in_process": _build_in_process,
        "webhook": _build_webhook,
        "composite": _build_composite,
        "custom": _build_custom,
    }
)


def register_stream_listener(name: str, factory: ListenerFactory) -> None:
    """Register a user-supplied listener factory.

    ``name`` becomes the value of the ``type`` discriminator in
    ``listener_config`` dictionaries. Registrations are process-local and
    should happen at agent construction time. Registered factories can be
    selected via HTTP-supplied configs by name — prefer this over the
    ``custom`` dotted-import factory for anything reachable from
    ``POST /stream`` since it enforces an explicit allowlist.
    """

    if not name or not isinstance(name, str):
        raise ValueError("listener name must be a non-empty string")
    if name in _BUILTIN_LISTENERS:
        raise ValueError(f"Cannot override built-in listener type '{name}'")
    with _REGISTRY_LOCK:
        _USER_LISTENERS[name] = factory


def build_listener(
    config: Mapping[str, Any],
    *,
    allow_custom: bool = False,
    **kwargs: Any,
) -> StreamListener:
    """Instantiate a listener from a JSON-serializable config dict.

    Args:
        config: Listener config with a ``type`` discriminator and type-specific
            fields (see each built-in's ``__init__`` for shapes).
        allow_custom: If ``True``, the ``"custom"`` listener type is dispatched
            via ``importlib.import_module``. HTTP surfaces must pass ``False``
            (the default) to prevent arbitrary-factory RCE via request bodies.
            Direct Python callers that trust their config can pass ``True``.
        **kwargs: Forwarded to the factory (e.g., ``infra=<DaprInfra>`` for
            factories that need framework services).

    Raises:
        ValueError: Unknown listener type, missing ``type`` field, or
            ``custom`` type without ``allow_custom=True``.
    """

    if not isinstance(config, Mapping):
        raise TypeError(
            f"listener config must be a mapping, got {type(config).__name__}"
        )
    ltype = config.get("type")
    if not ltype or not isinstance(ltype, str):
        raise ValueError("listener config must include a 'type' field")
    with _REGISTRY_LOCK:
        factory = _BUILTIN_LISTENERS.get(ltype) or _USER_LISTENERS.get(ltype)
    if factory is None:
        raise ValueError(f"Unknown listener type '{ltype}'")
    # Built-in composite/custom factories accept ``allow_custom`` to gate the
    # dotted-import path. User-registered factories are trusted (they had to
    # be registered at startup) and don't need the flag.
    if factory in (_build_composite, _build_custom):
        return factory(config, allow_custom=allow_custom, **kwargs)
    return factory(config, **kwargs)


__all__ = [
    "CompositeListener",
    "InProcessQueueListener",
    "PubSubListener",
    "StreamListener",
    "WebhookListener",
    "build_listener",
    "register_in_process_queue",
    "register_stream_listener",
    "unregister_in_process_queue",
]
