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

"""Calls that never return: stuck-worker replacement, thread ceiling, backlog."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from typing import Any, Callable

import pytest

from dapr_agents.workflow.utils.call_deadline import (
    DeadlineCaller,
    DeadlineCallerBusyError,
)
from dapr_agents.workflow.utils.event_routes import (
    DEFAULT_HOOK_POOL_SIZE,
    DEFAULT_SIDECAR_POOL_SIZE,
)
from tests.workflow._event_route_helpers import (
    make_dispatcher,
    make_target,
    make_wf,
    run_dispatch,
)

_WAIT = 5.0
_TINY = 0.02


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def hang() -> Iterator[threading.Event]:
    """An Event that hung calls wait on; always released at teardown."""
    release = threading.Event()
    yield release
    release.set()


def _wait_until(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + _WAIT
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.005)


def _caller(clock: _Clock, **kwargs: Any) -> DeadlineCaller:
    values: dict[str, Any] = dict(max_workers=1, stuck_grace_seconds=0.0, clock=clock)
    values.update(kwargs)
    return DeadlineCaller(**values)


def _time_out(caller: DeadlineCaller, release: threading.Event) -> None:
    started = threading.Event()

    def stuck() -> None:
        started.set()
        release.wait(_WAIT)

    with pytest.raises(TimeoutError):
        caller.call(stuck, _TINY)
    assert started.wait(_WAIT)


# ---- DeadlineCaller ------------------------------------------------------------


def test_stuck_worker_is_replaced_and_pool_keeps_serving(hang, caplog):
    clock = _Clock()
    caller = _caller(clock, max_threads=3)
    _time_out(caller, hang)
    clock.now = 10.0  # well past deadline + grace
    with caplog.at_level(logging.WARNING):
        assert caller.call(lambda: "ok", _WAIT) == "ok"
    assert len(caller._threads) == 2
    assert "presuming its worker stuck" in caplog.text
    caller.close()


def test_worker_within_grace_is_not_replaced(hang):
    clock = _Clock()
    caller = _caller(clock, max_threads=3, stuck_grace_seconds=60.0)
    _time_out(caller, hang)
    clock.now = 10.0
    with pytest.raises(TimeoutError):
        caller.call(lambda: "queued", _TINY)
    assert len(caller._threads) == 1
    caller.close()


def test_call_without_deadline_is_never_presumed_stuck(hang):
    clock = _Clock()
    caller = _caller(clock, max_threads=3)
    started = threading.Event()
    caller.submit(lambda: (started.set(), hang.wait(_WAIT)))
    assert started.wait(_WAIT)
    clock.now = 1_000.0
    with pytest.raises(TimeoutError):
        caller.call(lambda: "queued", _TINY)
    assert len(caller._threads) == 1
    caller.close()


def test_thread_ceiling_is_respected(hang, caplog):
    clock = _Clock()
    caller = _caller(clock, max_threads=2)
    _time_out(caller, hang)
    clock.now = 10.0
    _time_out(caller, hang)  # served by the replacement, which also hangs
    clock.now = 20.0
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            with pytest.raises(TimeoutError):
                caller.call(lambda: "never", _TINY)
    assert len(caller._threads) == 2
    assert caplog.text.count("the ceiling") == 1  # logged once
    caller.close()


def test_returning_stuck_worker_retires_when_surplus(hang):
    clock = _Clock()
    caller = _caller(clock, max_threads=3)
    _time_out(caller, hang)
    stuck_thread = next(iter(caller._threads.values()))
    clock.now = 10.0
    assert caller.call(lambda: "ok", _WAIT) == "ok"
    hang.set()
    stuck_thread.join(_WAIT)
    assert not stuck_thread.is_alive()
    assert len(caller._threads) == 1
    assert caller.call(lambda: "still ok", _WAIT) == "still ok"
    caller.close()


def test_full_backlog_fails_fast_without_running_the_call(hang, caplog):
    clock = _Clock()
    caller = _caller(clock, max_threads=1, max_backlog=1)
    ran: list[str] = []
    _time_out(caller, hang)
    with pytest.raises(TimeoutError):
        caller.call(lambda: ran.append("queued"), _TINY)  # stays queued, cancelled
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            with pytest.raises(DeadlineCallerBusyError):
                caller.call(lambda: ran.append("rejected"), _WAIT)
    assert caplog.text.count("wait for a worker") == 1  # logged once
    hang.set()
    _wait_until(lambda: caller._queued == 0)  # the worker skips the cancelled job
    assert caller.call(lambda: "drained", _WAIT) == "drained"
    assert ran == []
    assert caller._backlog_reported is False
    caller.close()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(max_workers=2, max_threads=1), "max_threads"),
        (dict(max_backlog=0), "max_backlog"),
        (dict(stuck_grace_seconds=-1.0), "stuck_grace_seconds"),
    ],
)
def test_rejects_invalid_limits(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DeadlineCaller(**kwargs)


def test_default_limits_scale_with_max_workers():
    caller = DeadlineCaller(max_workers=3)
    assert caller._max_threads == 12 and caller._max_backlog == 12
    caller.close()


# ---- WorkflowEventDispatcher pools -------------------------------------------


def test_dispatcher_uses_separate_pools_with_default_sizes():
    dispatcher = make_dispatcher(make_wf())
    assert dispatcher._hook_caller is not dispatcher._sidecar_caller
    assert dispatcher._hook_caller._max_workers == DEFAULT_HOOK_POOL_SIZE
    assert dispatcher._sidecar_caller._max_workers == DEFAULT_SIDECAR_POOL_SIZE
    dispatcher.close()
    sized = make_dispatcher(make_wf(), hook_pool_size=2, sidecar_pool_size=3)
    assert sized._hook_caller._max_workers == 2
    assert sized._sidecar_caller._max_workers == 3
    sized.close()


def test_hung_hook_does_not_block_sidecar_calls(hang):
    hooks = DeadlineCaller(max_workers=1, max_threads=1, max_backlog=1)
    wf = make_wf()
    dispatcher = make_dispatcher(wf, hook_caller=hooks)
    hung = make_target(
        authorize=lambda *_: hang.wait(_WAIT), hook_timeout_seconds=_TINY
    )
    assert run_dispatch(dispatcher, target=hung) == "drop"
    assert run_dispatch(dispatcher, target=hung) == "drop"  # queued behind it
    assert run_dispatch(dispatcher) == "success"  # no authorize: sidecar only
    wf.raise_workflow_event.assert_called_once()
    dispatcher.close()


def test_saturated_hook_pool_denies(hang, caplog):
    hooks = DeadlineCaller(max_workers=1, max_threads=1, max_backlog=1)
    dispatcher = make_dispatcher(make_wf(), hook_caller=hooks)
    hung = make_target(
        authorize=lambda *_: hang.wait(_WAIT), hook_timeout_seconds=_TINY
    )
    for _ in range(2):
        assert run_dispatch(dispatcher, target=hung) == "drop"
    with caplog.at_level(logging.WARNING):
        assert run_dispatch(dispatcher, target=hung) == "drop"
    assert "hook pool is saturated" in caplog.text
    dispatcher.close()


def test_saturated_sidecar_pool_retries_without_calling(hang):
    calls = DeadlineCaller(max_workers=1, max_threads=1, max_backlog=1)
    wf = make_wf()
    wf.get_workflow_state.side_effect = lambda *a, **k: hang.wait(_WAIT)
    dispatcher = make_dispatcher(wf, sidecar_caller=calls)
    target = make_target(call_timeout_seconds=_TINY)
    assert [run_dispatch(dispatcher, target=target) for _ in range(3)] == ["retry"] * 3
    assert wf.get_workflow_state.call_count == 1  # the rest never ran
    dispatcher.close()
