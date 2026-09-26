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

"""Timed-out raises: exact-identity reuse, in-flight tracking, authorize order."""

from __future__ import annotations

import functools
import logging
import threading
from concurrent.futures import Future
from typing import Any, Optional

from dapr_agents.workflow.utils.event_route_state import (
    EventRouteTopicState,
    NotFoundTracker,
    RaiseIdentity,
    TimedOutRaises,
    TrackedRaise,
)
from dapr_agents.workflow.utils.event_routes import WorkflowEventDispatcher
from tests.workflow._event_route_helpers import (
    make_dispatcher,
    make_target,
    make_wf,
    run_dispatch,
)

_KEY = "evt-1"
_WAIT = 5.0
_SECRET = "top-secret-payload"
_TOPIC = ("messagepubsub", "t")

_target = functools.partial(make_target, call_timeout_seconds=0.05)


class _FirstRaiseHangs:
    """raise_workflow_event stand-in: the first call blocks until released."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            self.release.wait(_WAIT)


def _setup(**dispatcher_kwargs: Any):
    gate = _FirstRaiseHangs()
    wf = make_wf()
    wf.raise_workflow_event.side_effect = gate
    return gate, make_dispatcher(wf, **dispatcher_kwargs)


def _send(
    dispatcher: WorkflowEventDispatcher,
    message: dict[str, Any],
    target: Any = None,
) -> str:
    return run_dispatch(
        dispatcher, target=target or _target(), message=message, dedupe_key=_KEY
    )


def _settle(gate: _FirstRaiseHangs, dispatcher: WorkflowEventDispatcher) -> None:
    gate.release.set()
    tracked = dispatcher._topic_states[_TOPIC].timed_out.peek(_KEY)
    assert tracked is not None
    tracked.future.result(timeout=_WAIT)


def _identity(instance_id: str = "wf-1", digest: str = "d") -> RaiseIdentity:
    return RaiseIdentity(instance_id=instance_id, event_name="evt", data_sha256=digest)


# ---- C1: exact identity ------------------------------------------------------


def test_reused_id_for_another_instance_is_raised(caplog):
    # Security repro: A's raise timed out, then succeeded in the background.
    gate, dispatcher = _setup()
    assert _send(dispatcher, {"wf_id": "wf-A", "note": _SECRET}) == "retry"
    _settle(gate, dispatcher)
    with caplog.at_level(logging.WARNING):
        assert _send(dispatcher, {"wf_id": "wf-B", "note": _SECRET}) == "success"
    assert [c["instance_id"] for c in gate.calls] == ["wf-A", "wf-B"]
    assert "matches an earlier timed-out raise" in caplog.text
    assert "('wf-A', 'evt')" in caplog.text and "('wf-B', 'evt')" in caplog.text
    assert _SECRET not in caplog.text
    # A's outcome is still tracked for A's own redelivery.
    assert _send(dispatcher, {"wf_id": "wf-A", "note": _SECRET}) == "success"
    assert len(gate.calls) == 2
    dispatcher.close()


def test_reused_id_with_same_target_and_other_data_is_raised(caplog):
    gate, dispatcher = _setup()
    assert _send(dispatcher, {"wf_id": "wf-1", "v": 1}) == "retry"
    _settle(gate, dispatcher)
    with caplog.at_level(logging.WARNING):
        assert _send(dispatcher, {"wf_id": "wf-1", "v": 2}) == "success"
    assert [c["data"]["v"] for c in gate.calls] == [1, 2]
    assert "with different data" in caplog.text
    dispatcher.close()


def test_reused_id_while_first_raise_runs_is_raised_and_first_stays_tracked():
    gate, dispatcher = _setup()
    try:
        assert _send(dispatcher, {"wf_id": "wf-A"}) == "retry"
        assert _send(dispatcher, {"wf_id": "wf-B"}) == "success"
        # A's raise is still running and still tracked: no second raise of A.
        assert _send(dispatcher, {"wf_id": "wf-A"}) == "retry"
        assert [c["instance_id"] for c in gate.calls] == ["wf-A", "wf-B"]
    finally:
        gate.release.set()
        dispatcher.close()


def test_exact_redelivery_with_reordered_keys_matches():
    gate, dispatcher = _setup()
    assert _send(dispatcher, {"wf_id": "wf-1", "a": 1, "b": 2}) == "retry"
    _settle(gate, dispatcher)
    assert _send(dispatcher, {"b": 2, "a": 1, "wf_id": "wf-1"}) == "success"
    assert len(gate.calls) == 1
    dispatcher.close()


# ---- M3: authorize is not re-run for the same timed-out message -------------


def test_redelivery_of_timed_out_raise_skips_authorize():
    calls: list[Any] = []

    def authorize(msg: Any, ctx: Any, target: Any) -> bool:
        calls.append(target)
        return True

    gate, dispatcher = _setup()
    target = _target(authorize=authorize)
    try:
        assert _send(dispatcher, {"wf_id": "wf-1"}, target) == "retry"
        assert _send(dispatcher, {"wf_id": "wf-1"}, target) == "retry"
        assert len(calls) == 1
        _settle(gate, dispatcher)
        assert _send(dispatcher, {"wf_id": "wf-1"}, target) == "success"
        assert len(calls) == 1
        # A different message under the same key is authorized as usual.
        assert _send(dispatcher, {"wf_id": "wf-2"}, target) == "success"
        assert len(calls) == 2
    finally:
        gate.release.set()
        dispatcher.close()


# ---- H1: running raises are never evicted -----------------------------------


def test_running_raises_survive_capacity_pressure():
    tracked = TimedOutRaises(maxsize=1, max_running=2)
    running = {key: TrackedRaise(_identity(key), Future()) for key in "abc"}
    assert tracked.add("a", running["a"]) and tracked.add("b", running["b"])
    assert not tracked.add("c", running["c"])  # running cap reached
    assert tracked.peek("a") is running["a"] and tracked.peek("b") is running["b"]
    assert tracked.running_count() == 2

    running["a"].future.set_result(None)  # a settles into the LRU
    assert tracked.running_count() == 1
    assert tracked.add("c", running["c"])
    running["b"].future.set_result(None)  # the LRU holds one: a is evicted
    assert tracked.peek("a") is None
    assert tracked.peek("b") is running["b"] and tracked.peek("c") is running["c"]
    assert len(tracked) == 2


def test_add_refuses_a_key_held_by_another_running_raise():
    tracked = TimedOutRaises(maxsize=4)
    first = TrackedRaise(_identity("wf-A"), Future())
    assert tracked.add("k", first)
    assert not tracked.add("k", TrackedRaise(_identity("wf-B"), Future()))
    same = TrackedRaise(_identity("wf-A"), Future())
    assert tracked.add("k", same)  # same identity replaces
    assert tracked.peek("k") is same


def test_discard_removes_running_and_settled_entries():
    tracked = TimedOutRaises(maxsize=4)
    done = TrackedRaise(_identity(), Future())
    running = TrackedRaise(_identity("wf-2"), Future())
    tracked.add("done", done)
    tracked.add("running", running)
    done.future.set_result(None)
    tracked.discard("done", running)  # a different entry: kept
    assert tracked.peek("done") is done
    tracked.discard("done", done)
    tracked.discard("running", running)
    assert len(tracked) == 0
    running.future.set_result(None)  # a late settle of a discarded raise is a no-op
    assert len(tracked) == 0


def test_untracked_timed_out_raise_logs_warning(caplog):
    gate, dispatcher = _setup()
    dispatcher._topic_states[_TOPIC] = EventRouteTopicState(
        not_found=NotFoundTracker(), timed_out=TimedOutRaises(16, max_running=0)
    )
    try:
        with caplog.at_level(logging.WARNING):
            assert _send(dispatcher, {"wf_id": "wf-1"}) == "retry"
        assert "cannot track the timed-out raise" in caplog.text
        assert "may raise the event again" in caplog.text
    finally:
        gate.release.set()
        dispatcher.close()


def test_topic_state_is_created_once_per_topic():
    dispatcher = make_dispatcher(make_wf())
    first = dispatcher._topic_state("p", "t", _target())
    assert dispatcher._topic_state("p", "t", _target()) is first
    assert dispatcher._topic_state("p", "other", _target()) is not first
    dispatcher.close()


def test_topic_state_created_by_a_racing_thread_is_reused():
    dispatcher = make_dispatcher(make_wf())
    racing = EventRouteTopicState.create(1)

    class _RacingLock:
        """Another thread creates the state while this one waits for the lock."""

        def __enter__(self) -> None:
            dispatcher._topic_states[("p", "t")] = racing

        def __exit__(self, *exc: Any) -> None:
            return None

    dispatcher._topic_state_creation_lock = _RacingLock()  # type: ignore[assignment]
    assert dispatcher._topic_state("p", "t", _target()) is racing
    dispatcher.close()


def test_tracked_raise_records_resolved_identity():
    gate, dispatcher = _setup()
    try:
        assert _send(dispatcher, {"wf_id": "wf-1"}) == "retry"
        tracked: Optional[TrackedRaise] = dispatcher._topic_states[
            _TOPIC
        ].timed_out.peek(_KEY)
        assert tracked is not None
        assert tracked.identity.instance_id == "wf-1"
        assert len(tracked.identity.data_sha256) == 64
    finally:
        gate.release.set()
        dispatcher.close()
