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

"""Unit tests for DeadlineCaller (deadline-bounded calls on daemon threads)."""

from __future__ import annotations

import threading

import pytest

from dapr_agents.workflow.utils.call_deadline import (
    DeadlineCaller,
    DeadlineCallerClosedError,
)

_WAIT = 5.0  # upper bound for events that should fire at once


def test_returns_result_and_passes_arguments():
    caller = DeadlineCaller()
    assert caller.call(lambda a, *, b: a + b, _WAIT, 1, b=2) == 3
    caller.close()


def test_propagates_exceptions():
    caller = DeadlineCaller()

    def boom() -> None:
        raise KeyError("x")

    with pytest.raises(KeyError):
        caller.call(boom, _WAIT)
    caller.close()


def test_reuses_an_idle_worker():
    caller = DeadlineCaller(max_workers=4)
    for _ in range(5):
        caller.call(lambda: None, _WAIT)
    assert len(caller._threads) == 1
    assert all(t.daemon for t in caller._threads.values())
    caller.close()


def test_timeout_raises_and_a_queued_call_never_runs():
    caller = DeadlineCaller(max_workers=1)
    release = threading.Event()
    started = threading.Event()
    ran: list[str] = []

    def slow() -> str:
        started.set()
        release.wait(_WAIT)
        return "late"

    with pytest.raises(TimeoutError):
        caller.call(slow, 0.01)
    assert started.wait(_WAIT)
    # The only worker is busy, so this call is still queued at its deadline.
    with pytest.raises(TimeoutError):
        caller.call(lambda: ran.append("queued"), 0.01)
    release.set()
    # The worker skips the cancelled job and serves the next one.
    assert caller.call(lambda: "next", _WAIT) == "next"
    assert ran == []
    assert caller._queued == 0 and caller._running == {}
    assert len(caller._threads) == 1
    caller.close()


def test_grows_when_all_workers_are_busy():
    caller = DeadlineCaller(max_workers=2)
    release = threading.Event()
    with pytest.raises(TimeoutError):
        caller.call(release.wait, 0.01, _WAIT)
    assert caller.call(lambda: "ok", _WAIT) == "ok"
    assert len(caller._threads) == 2
    release.set()
    caller.close()


def test_close_stops_workers_and_rejects_new_calls():
    caller = DeadlineCaller()
    caller.call(lambda: None, _WAIT)
    threads = list(caller._threads.values())
    caller.close()
    caller.close()  # idempotent
    for thread in threads:
        thread.join(_WAIT)
        assert not thread.is_alive()
    with pytest.raises(DeadlineCallerClosedError):
        caller.call(lambda: None, _WAIT)


def test_close_without_workers_is_a_noop():
    caller = DeadlineCaller()
    caller.close()
    assert caller._threads == {}


def test_rejects_non_positive_max_workers():
    with pytest.raises(ValueError, match="max_workers"):
        DeadlineCaller(max_workers=0)
