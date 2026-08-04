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

"""Tests for the pluggable StreamListener transports."""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dapr_agents.streaming.listeners import (
    CompositeListener,
    InProcessQueueListener,
    PubSubListener,
    StreamListener,
    WebhookListener,
    build_listener,
    register_in_process_queue,
    register_stream_listener,
    unregister_in_process_queue,
    _USER_LISTENERS,
)
from dapr_agents.types.streaming import AgentStreamChunk, StreamChunkType


def _chunk(root: str = "root-1", seq: int = 1) -> AgentStreamChunk:
    return AgentStreamChunk(
        sequence=seq,
        type=StreamChunkType.CONTENT_DELTA,
        agent="alice",
        workflow_instance_id="wf-1",
        turn=0,
        root_instance_id=root,
    )


# ---------------------------------------------------------------------------
# InProcessQueueListener
# ---------------------------------------------------------------------------


class TestInProcessQueueListener:
    def test_requires_registered_queue(self) -> None:
        with pytest.raises(LookupError):
            InProcessQueueListener(registry_key="never-registered")

    @pytest.mark.asyncio
    async def test_emit_roundtrip(self) -> None:
        register_in_process_queue("inproc-1")
        try:
            listener = InProcessQueueListener(registry_key="inproc-1")
            listener.emit(_chunk(seq=1))
            listener.emit(_chunk(seq=2))
            listener.close()

            from dapr_agents.streaming.listeners import _get_in_process_queue

            q = _get_in_process_queue("inproc-1")
            # call_soon_threadsafe has been scheduled; let the loop run so
            # put_nowait actually lands in the queue before we inspect it.
            await asyncio.sleep(0)
            first = await asyncio.wait_for(q.get(), timeout=1.0)
            second = await asyncio.wait_for(q.get(), timeout=1.0)
            assert first.sequence == 1
            assert second.sequence == 2
        finally:
            unregister_in_process_queue("inproc-1")

    @pytest.mark.asyncio
    async def test_double_register_rejects(self) -> None:
        register_in_process_queue("inproc-dup")
        try:
            with pytest.raises(ValueError):
                register_in_process_queue("inproc-dup")
        finally:
            unregister_in_process_queue("inproc-dup")

    @pytest.mark.asyncio
    async def test_unregister_idempotent(self) -> None:
        register_in_process_queue("inproc-once")
        unregister_in_process_queue("inproc-once")
        unregister_in_process_queue("inproc-once")  # no error

    @pytest.mark.asyncio
    async def test_close_idempotent(self) -> None:
        register_in_process_queue("inproc-close")
        try:
            listener = InProcessQueueListener(registry_key="inproc-close")
            listener.close()
            listener.close()  # must not raise
        finally:
            unregister_in_process_queue("inproc-close")

    @pytest.mark.asyncio
    async def test_emit_after_close_drops_silently(self) -> None:
        register_in_process_queue("inproc-drop")
        try:
            listener = InProcessQueueListener(registry_key="inproc-drop")
            listener.close()
            listener.emit(_chunk())  # no-op, no raise
        finally:
            unregister_in_process_queue("inproc-drop")


# ---------------------------------------------------------------------------
# PubSubListener
# ---------------------------------------------------------------------------


class TestPubSubListener:
    def test_rejects_empty_pubsub(self) -> None:
        with pytest.raises(ValueError):
            PubSubListener(pubsub_name="", topic="t")

    def test_rejects_empty_topic(self) -> None:
        with pytest.raises(ValueError):
            PubSubListener(pubsub_name="p", topic="")

    def test_emit_publishes_via_dapr(self) -> None:
        captured: List[dict] = []

        async def fake_publish(**kwargs):
            captured.append(kwargs)

        with patch(
            "dapr_agents.streaming.listeners.publish_message",
            new=fake_publish,
        ):
            listener = PubSubListener(pubsub_name="bus", topic="t1")
            listener.emit(_chunk(seq=1))
            listener.emit(_chunk(seq=2))
            listener.close()

        assert len(captured) == 2
        assert captured[0]["pubsub_name"] == "bus"
        assert captured[0]["topic_name"] == "t1"
        first_payload = captured[0]["message"]
        assert first_payload["sequence"] == 1
        assert first_payload["type"] == "content_delta"
        assert first_payload["root_instance_id"] == "root-1"

    def test_emit_after_close_drops(self) -> None:
        fake = AsyncMock()
        with patch(
            "dapr_agents.streaming.listeners.publish_message",
            new=fake,
        ):
            listener = PubSubListener(pubsub_name="bus", topic="t")
            listener.close()
            listener.emit(_chunk())
        fake.assert_not_awaited()

    def test_publish_failure_does_not_raise(self) -> None:
        async def boom(**kwargs):
            raise RuntimeError("network down")

        with patch(
            "dapr_agents.streaming.listeners.publish_message",
            new=boom,
        ):
            listener = PubSubListener(pubsub_name="bus", topic="t")
            # Must not raise even though the publish fails.
            listener.emit(_chunk())
            listener.close()

    def test_metadata_carries_event_type(self) -> None:
        captured: List[dict] = []

        async def fake_publish(**kwargs):
            captured.append(kwargs)

        with patch(
            "dapr_agents.streaming.listeners.publish_message",
            new=fake_publish,
        ):
            listener = PubSubListener(pubsub_name="bus", topic="t")
            listener.emit(
                AgentStreamChunk(
                    sequence=1,
                    type=StreamChunkType.SESSION_COMPLETE,
                    agent="a",
                    workflow_instance_id="wf",
                    turn=0,
                    root_instance_id="root-x",
                )
            )
            listener.close()

        assert (
            captured[0]["metadata"]["cloudevent.type"]
            == "agent.stream.session_complete"
        )
        assert captured[0]["metadata"]["cloudevent.subject"] == "root-x"


# ---------------------------------------------------------------------------
# CompositeListener
# ---------------------------------------------------------------------------


class _RecordingListener:
    def __init__(self) -> None:
        self.received: List[AgentStreamChunk] = []
        self.closed = False

    def emit(self, chunk: AgentStreamChunk) -> None:
        self.received.append(chunk)

    def close(self) -> None:
        self.closed = True


class _ExplodingListener:
    def emit(self, chunk: AgentStreamChunk) -> None:
        raise RuntimeError("boom")

    def close(self) -> None:
        raise RuntimeError("close-boom")


class TestCompositeListener:
    def test_rejects_empty_children(self) -> None:
        with pytest.raises(ValueError):
            CompositeListener([])

    def test_fans_out(self) -> None:
        a, b = _RecordingListener(), _RecordingListener()
        composite = CompositeListener([a, b])
        c = _chunk()
        composite.emit(c)
        composite.close()
        assert a.received == [c]
        assert b.received == [c]
        assert a.closed and b.closed

    def test_isolates_child_errors_on_emit(self) -> None:
        good = _RecordingListener()
        composite = CompositeListener([_ExplodingListener(), good])
        composite.emit(_chunk())
        assert len(good.received) == 1

    def test_isolates_child_errors_on_close(self) -> None:
        good = _RecordingListener()
        composite = CompositeListener([_ExplodingListener(), good])
        composite.close()  # must not raise
        assert good.closed


# ---------------------------------------------------------------------------
# WebhookListener
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code


class _FakeClient:
    def __init__(self, *_, **__) -> None:
        self.posts: List[tuple] = []
        self.closed = False

    def post(self, url: str, content: str) -> _FakeResponse:
        self.posts.append((url, content))
        return _FakeResponse(200)

    def close(self) -> None:
        self.closed = True


class _BrokenClient(_FakeClient):
    def post(self, url: str, content: str) -> _FakeResponse:
        self.posts.append((url, content))
        return _FakeResponse(500)


class TestWebhookListener:
    def test_rejects_empty_url(self) -> None:
        with pytest.raises(ValueError):
            WebhookListener(url="")

    def test_posts_chunks(self) -> None:
        fake_httpx = MagicMock()
        client = _FakeClient()
        fake_httpx.Client.return_value = client

        with patch.dict("sys.modules", {"httpx": fake_httpx}):
            listener = WebhookListener(url="http://x/hook", retry_attempts=0)
            listener.emit(_chunk(seq=1))
            listener.emit(_chunk(seq=2))
            listener.close()

        assert len(client.posts) == 2
        assert client.posts[0][0] == "http://x/hook"
        assert client.closed

    def test_retries_then_logs_on_failure(self) -> None:
        fake_httpx = MagicMock()
        client = _BrokenClient()
        fake_httpx.Client.return_value = client

        with patch.dict("sys.modules", {"httpx": fake_httpx}):
            listener = WebhookListener(url="http://x/hook", retry_attempts=2)
            listener.emit(_chunk(seq=1))
            listener.close()

        # 1 original + 2 retries = 3 attempts
        assert len(client.posts) == 3

    def test_missing_httpx_does_not_raise(self) -> None:
        import sys

        original = sys.modules.pop("httpx", None)
        try:
            # Prime the import system to fail for 'httpx'.
            with patch.dict("sys.modules", {"httpx": None}):
                listener = WebhookListener(url="http://x/hook", retry_attempts=0)
                listener.emit(_chunk())
                listener.close()
        finally:
            if original is not None:
                sys.modules["httpx"] = original


# ---------------------------------------------------------------------------
# build_listener factory + registration
# ---------------------------------------------------------------------------


class TestBuildListener:
    def test_rejects_non_mapping(self) -> None:
        with pytest.raises(TypeError):
            build_listener("pubsub")  # type: ignore[arg-type]

    def test_rejects_missing_type(self) -> None:
        with pytest.raises(ValueError):
            build_listener({})

    def test_rejects_unknown_type(self) -> None:
        with pytest.raises(ValueError):
            build_listener({"type": "nope"})

    def test_custom_factory_rejected_without_allow_custom(self) -> None:
        """The ``custom`` dotted-import factory must not be usable by default.

        Any HTTP caller that can POST a listener config would otherwise
        trigger arbitrary ``importlib.import_module`` calls.
        """
        with pytest.raises(ValueError, match="allow_custom"):
            build_listener(
                {
                    "type": "custom",
                    "factory": "dapr_agents.streaming.listeners.PubSubListener",
                }
            )

    def test_custom_factory_allowed_when_opt_in(self) -> None:
        """Trusted Python callers can opt in explicitly. This test uses
        ``PubSubListener`` as the factory since it's a real class on the
        module path and doesn't require extra registration."""
        listener = build_listener(
            {
                "type": "custom",
                "factory": "dapr_agents.streaming.listeners.PubSubListener",
                "kwargs": {"pubsub_name": "bus", "topic": "t"},
            },
            allow_custom=True,
        )
        try:
            assert isinstance(listener, PubSubListener)
        finally:
            listener.close()

    def test_builds_pubsub(self) -> None:
        listener = build_listener(
            {"type": "pubsub", "pubsub_name": "bus", "topic": "t"}
        )
        try:
            assert isinstance(listener, PubSubListener)
        finally:
            listener.close()

    @pytest.mark.asyncio
    async def test_builds_in_process(self) -> None:
        register_in_process_queue("factory-inproc")
        try:
            listener = build_listener(
                {"type": "in_process", "registry_key": "factory-inproc"}
            )
            assert isinstance(listener, InProcessQueueListener)
            listener.close()
        finally:
            unregister_in_process_queue("factory-inproc")

    @pytest.mark.asyncio
    async def test_builds_composite(self) -> None:
        register_in_process_queue("factory-comp")
        try:
            listener = build_listener(
                {
                    "type": "composite",
                    "listeners": [
                        {"type": "in_process", "registry_key": "factory-comp"},
                    ],
                }
            )
            assert isinstance(listener, CompositeListener)
            listener.close()
        finally:
            unregister_in_process_queue("factory-comp")


class TestRegisterStreamListener:
    def setup_method(self) -> None:
        _USER_LISTENERS.pop("custom-test", None)

    def teardown_method(self) -> None:
        _USER_LISTENERS.pop("custom-test", None)

    def test_cannot_override_builtin(self) -> None:
        with pytest.raises(ValueError):
            register_stream_listener("pubsub", lambda *_: None)  # type: ignore[arg-type]

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError):
            register_stream_listener("", lambda *_: None)  # type: ignore[arg-type]

    def test_registers_and_builds(self) -> None:
        recorded: List[dict] = []

        def factory(config, **kwargs):
            recorded.append(dict(config))
            return _RecordingListener()

        register_stream_listener("custom-test", factory)
        listener = build_listener({"type": "custom-test", "foo": "bar"})
        assert isinstance(listener, _RecordingListener)
        assert recorded == [{"type": "custom-test", "foo": "bar"}]
