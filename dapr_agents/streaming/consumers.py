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

"""Consumer helpers that turn a ``StreamListener`` transport into an async
iterator of ``AgentStreamChunk``.

Used by ``AgentRunner.run_stream`` and ``.serve()`` SSE/NDJSON routes. Each
consumer yields chunks in arrival order and terminates on ``SESSION_COMPLETE``
(root) or an explicit stop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from typing import AsyncIterator, Optional, Protocol

from dapr_agents.streaming.listeners import (
    _get_in_process_queue,
    unregister_in_process_queue,
)
from dapr_agents.types.streaming import AgentStreamChunk, StreamChunkType

logger = logging.getLogger(__name__)


class StreamConsumer(Protocol):
    """Async iterator of ``AgentStreamChunk`` drained from some transport."""

    def __aiter__(self) -> "StreamConsumer": ...

    async def __anext__(self) -> AgentStreamChunk: ...

    async def astart(self) -> None:
        """Open any underlying subscription or registry handle eagerly.

        ``AgentRunner.run_stream`` calls this *before* scheduling the
        workflow so a fast activity cannot emit ``START`` before the
        consumer is ready. Must be idempotent — subsequent ``astart`` calls
        are no-ops. In-process consumers whose registry entry is populated
        at construction time may implement this as a no-op.
        """

    async def aclose(self) -> None: ...


def _should_terminate(chunk: AgentStreamChunk, root_instance_id: str) -> bool:
    """Return True when this chunk signals the user-visible end of the session."""

    return (
        chunk.type is StreamChunkType.SESSION_COMPLETE
        and chunk.root_instance_id == root_instance_id
        and chunk.depth == 0
    )


# ---------------------------------------------------------------------------
# In-process queue consumer
# ---------------------------------------------------------------------------


class InProcessQueueConsumer:
    """Drains chunks from the process-local registry queue.

    Paired with :class:`dapr_agents.streaming.listeners.InProcessQueueListener`.
    The consumer owns the lifecycle of the registry entry: it calls
    :func:`unregister_in_process_queue` on close. Awaits directly on the
    underlying ``asyncio.Queue`` — no thread-pool hop per chunk.
    """

    def __init__(self, *, registry_key: str, root_instance_id: str) -> None:
        self._registry_key = registry_key
        self._root_instance_id = root_instance_id
        self._queue = _get_in_process_queue(registry_key)
        self._closed = False
        self._terminated = False

    def __aiter__(self) -> "InProcessQueueConsumer":
        return self

    async def astart(self) -> None:
        """No-op — the registry entry is populated at construction time."""

    async def __anext__(self) -> AgentStreamChunk:
        if self._closed or self._terminated:
            raise StopAsyncIteration

        while True:
            item = await self._queue.get()
            if item is None:
                # Sentinel — producer closed or consumer was cancelled.
                self._terminated = True
                raise StopAsyncIteration
            if item.root_instance_id != self._root_instance_id:
                continue
            if _should_terminate(item, self._root_instance_id):
                self._terminated = True
                return item
            return item

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Remove the registry entry and wake a blocked reader (if any) with
        # the sentinel. Safe to call even if the producer already finished.
        unregister_in_process_queue(self._registry_key)


# ---------------------------------------------------------------------------
# Pub/Sub consumer
# ---------------------------------------------------------------------------


class PubSubStreamConsumer:
    """Subscribes to a Dapr pub/sub topic and yields chunks for a session.

    Filters by ``root_instance_id`` so multiple concurrent sessions on the same
    topic stay isolated. Runs the blocking Dapr subscription in a background
    thread and bridges to an asyncio queue the consumer awaits on.
    """

    def __init__(
        self,
        *,
        pubsub_name: str,
        topic: str,
        root_instance_id: str,
        dead_letter_topic: Optional[str] = None,
    ) -> None:
        self._pubsub_name = pubsub_name
        self._topic = topic
        self._root_instance_id = root_instance_id
        self._dead_letter_topic = dead_letter_topic

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_queue: Optional[asyncio.Queue[Optional[AgentStreamChunk]]] = None
        self._stop_event = threading.Event()
        self._subscription_ready = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._subscription: Optional[object] = None
        self._started = False
        self._terminated = False

    def __aiter__(self) -> "PubSubStreamConsumer":
        if not self._started:
            self._start()
        return self

    async def astart(self) -> None:
        """Open the Dapr subscription and block until it's live.

        Called by ``AgentRunner.run_stream`` *before* scheduling the workflow
        so fast activities don't emit ``START`` before the consumer can
        receive it. The background thread signals ``_subscription_ready`` as
        soon as ``client.subscribe(...)`` returns; we await that signal here.
        Idempotent — subsequent calls after the subscription is live are
        no-ops.
        """
        if self._started:
            return
        self._start()
        # Park off-event-loop so we don't block the caller's loop while the
        # background thread opens the gRPC subscription.
        await asyncio.get_running_loop().run_in_executor(
            None, self._subscription_ready.wait, 30.0
        )
        if not self._subscription_ready.is_set():
            raise RuntimeError(
                f"PubSubStreamConsumer subscription to {self._topic!r} "
                "did not become ready within 30s"
            )

    async def __anext__(self) -> AgentStreamChunk:
        if not self._started:
            self._start()
        if self._terminated:
            raise StopAsyncIteration
        assert self._async_queue is not None
        item = await self._async_queue.get()
        if item is None:
            self._terminated = True
            raise StopAsyncIteration
        if _should_terminate(item, self._root_instance_id):
            self._terminated = True
        return item

    async def aclose(self) -> None:
        self._stop_event.set()
        sub = self._subscription
        if sub is not None and hasattr(sub, "close"):
            try:
                sub.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("PubSub subscription close raised: %s", exc)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._async_queue is not None:
            # Unblock any pending consumer.
            self._async_queue.put_nowait(None)

    # -- internal ----------------------------------------------------------

    def _start(self) -> None:
        self._started = True
        self._loop = asyncio.get_running_loop()
        self._async_queue = asyncio.Queue()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"pubsub-stream-consumer-{self._topic}",
        )
        self._thread.start()

    def _run(self) -> None:
        # Import locally to avoid pulling Dapr at module import time.
        from dapr.clients import DaprClient

        try:
            with DaprClient() as client:
                subscription = client.subscribe(
                    pubsub_name=self._pubsub_name,
                    topic=self._topic,
                    dead_letter_topic=self._dead_letter_topic,
                )
                self._subscription = subscription
                # Signal ``astart()`` that the subscription is live. Setting
                # this AFTER ``subscribe`` returns means callers can rely on
                # the Dapr gRPC channel having accepted the subscription
                # before the workflow is scheduled.
                self._subscription_ready.set()
                for msg in subscription:
                    if self._stop_event.is_set():
                        break
                    if msg is None:
                        continue
                    try:
                        chunk = self._parse(msg)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Dropping malformed stream chunk on %s: %s",
                            self._topic,
                            exc,
                        )
                        self._ack(subscription, msg, "drop")
                        continue

                    if (
                        chunk is None
                        or chunk.root_instance_id != self._root_instance_id
                    ):
                        # Not ours or unparseable. Ack so it doesn't redeliver.
                        self._ack(subscription, msg, "success")
                        continue

                    self._post_async(chunk)
                    self._ack(subscription, msg, "success")

                    if _should_terminate(chunk, self._root_instance_id):
                        break
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "PubSub stream consumer terminated with error (topic=%s): %s",
                self._topic,
                exc,
            )
        finally:
            # If the subscription never opened, astart() is still blocked on
            # the readiness event — unblock it so the caller sees the error
            # promptly rather than hitting the 30s timeout.
            self._subscription_ready.set()
            # Signal end-of-stream to the async consumer.
            self._post_async(None)

    def _parse(self, msg: object) -> Optional[AgentStreamChunk]:
        # Dapr SubscriptionMessage exposes `.data()` for the CloudEvent data.
        data = getattr(msg, "data", None)
        raw = data() if callable(data) else data
        if raw is None:
            return None
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            return None
        return AgentStreamChunk.model_validate(raw)

    def _ack(self, subscription: object, msg: object, action: str) -> None:
        method_name = {
            "success": "respond_success",
            "retry": "respond_retry",
            "drop": "respond_drop",
        }.get(action, "respond_success")
        method = getattr(subscription, method_name, None)
        if method is None:
            return
        try:
            method(msg)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Subscription ack (%s) raised: %s", action, exc)

    def _post_async(self, item: Optional[AgentStreamChunk]) -> None:
        loop = self._loop
        q = self._async_queue
        if loop is None or q is None:
            return
        loop.call_soon_threadsafe(q.put_nowait, item)


__all__ = [
    "InProcessQueueConsumer",
    "PubSubStreamConsumer",
    "StreamConsumer",
]
