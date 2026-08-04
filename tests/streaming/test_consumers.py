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

"""Tests for stream consumer helpers."""

from __future__ import annotations

import asyncio

import pytest

from dapr_agents.streaming.consumers import InProcessQueueConsumer
from dapr_agents.streaming.listeners import (
    InProcessQueueListener,
    register_in_process_queue,
    unregister_in_process_queue,
)
from dapr_agents.types.streaming import AgentStreamChunk, StreamChunkType


def _chunk(
    *,
    seq: int,
    typ: StreamChunkType = StreamChunkType.CONTENT_DELTA,
    root: str = "root-1",
    depth: int = 0,
) -> AgentStreamChunk:
    return AgentStreamChunk(
        sequence=seq,
        type=typ,
        agent="alice",
        workflow_instance_id="wf-1",
        turn=0,
        root_instance_id=root,
        depth=depth,
    )


@pytest.mark.asyncio
async def test_in_process_consumer_yields_chunks() -> None:
    register_in_process_queue("consumer-1")
    try:
        listener = InProcessQueueListener(registry_key="consumer-1")
        listener.emit(_chunk(seq=1))
        listener.emit(_chunk(seq=2))
        listener.emit(_chunk(seq=3, typ=StreamChunkType.SESSION_COMPLETE))

        consumer = InProcessQueueConsumer(
            registry_key="consumer-1", root_instance_id="root-1"
        )
        seen: list[AgentStreamChunk] = []
        async for item in consumer:
            seen.append(item)
        await consumer.aclose()

        assert [c.sequence for c in seen] == [1, 2, 3]
        assert seen[-1].type is StreamChunkType.SESSION_COMPLETE
    finally:
        unregister_in_process_queue("consumer-1")


@pytest.mark.asyncio
async def test_in_process_consumer_filters_by_root() -> None:
    register_in_process_queue("consumer-2")
    try:
        listener = InProcessQueueListener(registry_key="consumer-2")
        listener.emit(_chunk(seq=1, root="root-a"))
        listener.emit(_chunk(seq=2, root="root-b"))
        listener.emit(
            _chunk(seq=3, root="root-a", typ=StreamChunkType.SESSION_COMPLETE)
        )

        consumer = InProcessQueueConsumer(
            registry_key="consumer-2", root_instance_id="root-a"
        )
        seen: list[AgentStreamChunk] = []
        async for item in consumer:
            seen.append(item)
        await consumer.aclose()

        assert [c.sequence for c in seen] == [1, 3]
    finally:
        unregister_in_process_queue("consumer-2")


@pytest.mark.asyncio
async def test_in_process_consumer_sentinel_terminates() -> None:
    """Producer unregisters mid-stream → consumer sees the sentinel and stops.

    Drives the consumer through an actual ``async for`` loop rather than
    just constructing and closing it. The previous version exercised
    register/unregister plumbing without ever iterating, so the sentinel
    termination path was untested.
    """

    register_in_process_queue("consumer-sentinel")
    try:
        listener = InProcessQueueListener(registry_key="consumer-sentinel")
        consumer = InProcessQueueConsumer(
            registry_key="consumer-sentinel", root_instance_id="root-1"
        )
        listener.emit(_chunk(seq=1))
        listener.emit(_chunk(seq=2))

        async def producer_closes_after_yield() -> None:
            # Yield control once so the consumer starts draining, then
            # unregister the queue. ``unregister_in_process_queue`` posts
            # the ``None`` sentinel, which should terminate the consumer.
            await asyncio.sleep(0)
            unregister_in_process_queue("consumer-sentinel")

        producer_task = asyncio.create_task(producer_closes_after_yield())
        seen: list[AgentStreamChunk] = []
        async for item in consumer:
            seen.append(item)
        await producer_task
        await consumer.aclose()

        assert [c.sequence for c in seen] == [1, 2]
    finally:
        # aclose() already unregistered; this is defensive for failure paths.
        unregister_in_process_queue("consumer-sentinel")


@pytest.mark.asyncio
async def test_in_process_consumer_nonroot_turn_complete_does_not_terminate() -> None:
    register_in_process_queue("consumer-4")
    try:
        listener = InProcessQueueListener(registry_key="consumer-4")
        listener.emit(_chunk(seq=1))
        # Sub-agent TURN_COMPLETE (depth > 0) should not terminate the iterator.
        listener.emit(_chunk(seq=2, typ=StreamChunkType.TURN_COMPLETE, depth=1))
        listener.emit(_chunk(seq=3, typ=StreamChunkType.SESSION_COMPLETE))

        consumer = InProcessQueueConsumer(
            registry_key="consumer-4", root_instance_id="root-1"
        )
        seen: list[AgentStreamChunk] = []
        async for item in consumer:
            seen.append(item)
        await consumer.aclose()

        assert [c.sequence for c in seen] == [1, 2, 3]
    finally:
        unregister_in_process_queue("consumer-4")
