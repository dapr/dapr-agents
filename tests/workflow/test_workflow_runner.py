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

"""Tests for WorkflowRunner concurrent workflow scheduling (issue #563)."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

import grpc
import pytest

from dapr_agents.workflow.runners.base import WorkflowRunner


def _make_runner(mock_client: MagicMock | None = None) -> WorkflowRunner:
    """Create a WorkflowRunner with a mock DaprWorkflowClient."""
    client = mock_client or MagicMock()
    client.schedule_new_workflow.return_value = "instance-123"
    with patch(
        "dapr_agents.workflow.runners.base.DaprWorkflowClient", return_value=client
    ):
        return WorkflowRunner(name="test-runner", wf_client=client)


def _dummy_workflow() -> None:
    pass


# ---------------------------------------------------------------------------
# Regression guard: lock must not exist
# ---------------------------------------------------------------------------


def test_no_client_lock_attribute():
    runner = _make_runner()
    assert not hasattr(runner, "_client_lock"), (
        "_client_lock must not exist — it serializes all workflow operations (issue #563)"
    )


# ---------------------------------------------------------------------------
# Concurrent scheduling must not serialize
# ---------------------------------------------------------------------------


def test_concurrent_run_workflow():
    """Three concurrent schedule calls must not be serialized by a lock."""
    client = MagicMock()
    call_count = 0

    def _slow_schedule(**kwargs):
        nonlocal call_count
        call_count += 1
        time.sleep(0.1)
        return f"id-{call_count}"

    client.schedule_new_workflow.side_effect = _slow_schedule
    runner = _make_runner(client)

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(runner.run_workflow, _dummy_workflow) for _ in range(3)]
        results = [f.result() for f in as_completed(futures)]
    elapsed = time.monotonic() - start

    assert len(results) == 3
    assert client.schedule_new_workflow.call_count == 3
    # If serialized, would take ~0.3s; concurrent should be ~0.1s
    assert elapsed < 0.25, f"Scheduling was serialized (took {elapsed:.2f}s)"


# ---------------------------------------------------------------------------
# Scheduling must not block while a completion-wait is in progress
# ---------------------------------------------------------------------------


def test_concurrent_schedule_and_wait():
    """Scheduling a new workflow must not block while another is being awaited."""
    client = MagicMock()
    client.schedule_new_workflow.return_value = "id-new"

    def _slow_wait(*args, **kwargs):
        time.sleep(0.5)
        return MagicMock(runtime_status="COMPLETED")

    client.wait_for_workflow_completion.side_effect = _slow_wait
    runner = _make_runner(client)

    # Start a long-running wait in a background thread
    with ThreadPoolExecutor(max_workers=2) as pool:
        wait_future = pool.submit(runner.wait_for_workflow_completion, "id-existing")
        # Brief pause so the wait thread is actively blocking
        time.sleep(0.05)

        # Schedule a new workflow — must not be blocked by the wait
        start = time.monotonic()
        schedule_future = pool.submit(runner.run_workflow, _dummy_workflow)
        instance_id = schedule_future.result(timeout=1.0)
        schedule_elapsed = time.monotonic() - start

        # Clean up the wait
        wait_future.result(timeout=2.0)

    assert instance_id is not None
    assert schedule_elapsed < 0.2, (
        f"Scheduling blocked by wait ({schedule_elapsed:.2f}s) — lock regression"
    )


# ---------------------------------------------------------------------------
# Retry logic unaffected
# ---------------------------------------------------------------------------


def test_run_workflow_retry_on_transient_grpc_error():
    """Transient gRPC errors should trigger retries without a lock."""
    import grpc

    client = MagicMock()

    class FakeRpcError(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.UNAVAILABLE

        def details(self):
            return "connection reset"

    client.schedule_new_workflow.side_effect = [FakeRpcError(), "id-retry-success"]
    runner = _make_runner(client)

    with patch("dapr_agents.workflow.runners.base.time.sleep"):
        result = runner.run_workflow(_dummy_workflow)

    assert result == "id-retry-success"
    assert client.schedule_new_workflow.call_count == 2


# ---------------------------------------------------------------------------
# Issue #520: 'actor is closed' during completion wait must be retried
# ---------------------------------------------------------------------------


class _FakeRpcError(grpc.RpcError):
    def __init__(self, code, details):
        self._code = code
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details


class _FakeActorClosedError(_FakeRpcError):
    def __init__(self):
        super().__init__(
            grpc.StatusCode.UNKNOWN, "actor is closed, cannot handle deactivated"
        )


class _FakeClock:
    """Deterministic monotonic clock so retry loops advance instantly."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _patch_clock(clock: _FakeClock):
    return patch.multiple(
        "dapr_agents.workflow.runners.base.time",
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )


def _completed_state() -> MagicMock:
    state = MagicMock()
    state.runtime_status = MagicMock()
    state.runtime_status.name = "COMPLETED"
    return state


def test_wait_for_workflow_completion_retries_actor_closed(caplog):
    """'actor is closed' is transient: keep waiting instead of raising."""
    client = MagicMock()
    state = _completed_state()
    client.wait_for_workflow_completion.side_effect = [_FakeActorClosedError(), state]
    runner = _make_runner(client)

    with _patch_clock(_FakeClock()):
        result = runner.wait_for_workflow_completion("instance-123")

    assert result is state
    assert client.wait_for_workflow_completion.call_count == 2
    assert "actor was deactivated" in caplog.text


def test_wait_for_workflow_completion_gives_up_after_budget(caplog):
    """Persistent deactivation exhausts the budget and returns None."""
    client = MagicMock()
    client.wait_for_workflow_completion.side_effect = _FakeActorClosedError()
    runner = _make_runner(client)

    with _patch_clock(_FakeClock()):
        result = runner.wait_for_workflow_completion(
            "instance-123", timeout_in_seconds=1
        )

    assert result is None
    assert "stayed deactivated" in caplog.text


def test_wait_for_workflow_completion_non_transient_error_unchanged(caplog):
    """Other gRPC errors keep the existing error-and-None contract."""
    client = MagicMock()
    client.wait_for_workflow_completion.side_effect = _FakeRpcError(
        grpc.StatusCode.INTERNAL, "boom"
    )
    runner = _make_runner(client)

    with patch("dapr_agents.workflow.runners.base.time.sleep"):
        result = runner.wait_for_workflow_completion("instance-123")

    assert result is None
    assert client.wait_for_workflow_completion.call_count == 1
    assert "Error while waiting" in caplog.text


def test_await_and_log_state_survives_actor_closed(caplog):
    """The fire-and-forget monitor retries and logs the final state."""
    client = MagicMock()
    state = _completed_state()
    state.serialized_output = "done"
    client.wait_for_workflow_completion.side_effect = [_FakeActorClosedError(), state]
    runner = _make_runner(client)

    with (
        caplog.at_level(logging.INFO, logger="dapr_agents.workflow.runners.base"),
        _patch_clock(_FakeClock()),
    ):
        asyncio.run(runner._await_and_log_state("instance-123", 600, True))

    assert client.wait_for_workflow_completion.call_count == 2
    assert "completed" in caplog.text
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)
