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

"""Workflow event routes: timed-out raises, permanent sidecar errors, dedupe keys."""

from __future__ import annotations

import functools
import logging
import os
import subprocess
import sys
import threading
from concurrent.futures import Future
from typing import Any, Optional

import grpc
import pytest

from dapr_agents.workflow.utils.call_deadline import DeadlineCaller
from dapr_agents.workflow.utils.core import stable_json_sha256
from dapr_agents.workflow.utils.event_route_calls import (
    PERMANENT_SIDECAR_ERROR_CODES,
    is_instance_not_found_error,
    permanent_error_code,
)
from dapr_agents.workflow.utils.event_route_state import (
    MIN_TIMED_OUT_RAISES_TRACKED,
    EventRouteTopicState,
)
from dapr_agents.workflow.utils.event_routes import (
    EventRouteTarget,
    WorkflowEventDispatcher,
)
from dapr_agents.workflow.utils.subscription import (
    TTLDedupeBackend,
    _StreamSubscriber,
)
from tests.workflow._event_route_helpers import (
    BrokenRpcError,
    FakeRpcError,
    make_dispatcher,
    make_target,
    make_wf,
    only_tracked_raise,
    run_dispatch,
)

_KEY = "evt-1"
_WAIT = 5.0
_TOPIC = ("messagepubsub", "t")

_target = functools.partial(make_target, call_timeout_seconds=0.05)
_wf = make_wf
_dispatcher = make_dispatcher


def _dispatch(
    dispatcher: WorkflowEventDispatcher,
    *,
    dedupe_key: Optional[str] = _KEY,
    dlq: Optional[str] = None,
    target: Optional[EventRouteTarget] = None,
    message: Any = None,
) -> str:
    return run_dispatch(
        dispatcher,
        target=target or _target(),
        message=message,
        dlq=dlq,
        dedupe_key=dedupe_key,
    )


def _state(dispatcher: WorkflowEventDispatcher) -> EventRouteTopicState:
    return dispatcher._topic_states[_TOPIC]


class _GatedRaise:
    """raise_workflow_event stand-in that blocks until released, then does ``outcome``."""

    def __init__(self, outcome: Optional[BaseException] = None) -> None:
        self.release = threading.Event()
        self.finished = threading.Event()
        self.calls: list[dict[str, Any]] = []
        self.outcome = outcome

    def __call__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        if len(self.calls) > 1:
            return  # redelivered raises complete at once
        try:
            self.release.wait(_WAIT)
            if self.outcome is not None:
                raise self.outcome
        finally:
            self.finished.set()


def _time_out_first_raise(outcome: Optional[BaseException] = None):
    gate = _GatedRaise(outcome)
    wf = _wf()
    wf.raise_workflow_event.side_effect = gate
    dispatcher = _dispatcher(wf)
    assert _dispatch(dispatcher) == "retry"
    return gate, wf, dispatcher


def _finish(gate: _GatedRaise, dispatcher: WorkflowEventDispatcher) -> None:
    gate.release.set()
    assert gate.finished.wait(_WAIT)
    tracked = only_tracked_raise(dispatcher)
    # The future settles just after the call returns.
    tracked.future.exception(timeout=_WAIT)


# ---- timed-out raise tracking ----------------------------------------------


def test_redelivery_while_timed_out_raise_runs_retries_without_raising():
    gate, wf, dispatcher = _time_out_first_raise()
    try:
        assert _dispatch(dispatcher) == "retry"
        assert len(gate.calls) == 1
    finally:
        gate.release.set()
        dispatcher.close()


def test_redelivery_after_timed_out_raise_succeeded_acks_without_raising():
    gate, wf, dispatcher = _time_out_first_raise()
    _finish(gate, dispatcher)
    assert _dispatch(dispatcher) == "success"
    assert len(gate.calls) == 1
    assert len(_state(dispatcher).timed_out) == 0  # resolved entries are removed
    dispatcher.close()


def test_redelivery_after_transient_failure_raises_again():
    gate, wf, dispatcher = _time_out_first_raise(
        FakeRpcError(grpc.StatusCode.UNAVAILABLE, "down")
    )
    _finish(gate, dispatcher)
    assert _dispatch(dispatcher) == "success"
    assert len(gate.calls) == 2
    dispatcher.close()


def test_redelivery_after_permanent_failure_drops_without_raising(caplog):
    gate, wf, dispatcher = _time_out_first_raise(
        FakeRpcError(grpc.StatusCode.PERMISSION_DENIED, "no")
    )
    _finish(gate, dispatcher)
    with caplog.at_level(logging.WARNING):
        assert _dispatch(dispatcher, dlq="dlq") == "drop"
    assert len(gate.calls) == 1
    assert "PERMISSION_DENIED" in caplog.text
    assert "'dlq'" in caplog.text
    dispatcher.close()


def test_timed_out_raise_without_dedupe_key_is_not_tracked():
    # dedupe=False: not tracked, so a timed-out raise can be raised again (documented).
    gate = _GatedRaise()
    wf = _wf()
    wf.raise_workflow_event.side_effect = gate
    dispatcher = _dispatcher(wf)
    try:
        target = _target(dedupe=False)
        assert _dispatch(dispatcher, target=target) == "retry"
        assert _dispatch(dispatcher, target=target) == "success"
        assert len(gate.calls) == 2
        assert len(_state(dispatcher).timed_out) == 0
    finally:
        gate.release.set()
        dispatcher.close()


class _NeverRunsCaller:
    """Runs state checks inline; queues raises that no worker ever picks up."""

    def call(self, fn: Any, timeout: float, *args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    def start(self, fn: Any, timeout: float, *args: Any, **kwargs: Any) -> Future:
        return Future()

    def close(self) -> None:
        return None


def test_timed_out_raise_cancelled_while_queued_is_not_tracked():
    wf = _wf()
    dispatcher = _dispatcher(wf, sidecar_caller=_NeverRunsCaller())
    assert _dispatch(dispatcher) == "retry"
    assert len(_state(dispatcher).timed_out) == 0
    wf.raise_workflow_event.assert_not_called()


def test_timed_out_raises_sized_from_dedupe_entries():
    small = EventRouteTopicState.create(8).timed_out
    large = EventRouteTopicState.create(10_000).timed_out
    assert small._settled.maxsize == MIN_TIMED_OUT_RAISES_TRACKED
    assert large._settled.maxsize == 10_000


def test_deadline_caller_submit_returns_future():
    caller = DeadlineCaller(max_workers=1)
    assert caller.submit(lambda a, *, b: a + b, 1, b=2).result(_WAIT) == 3
    caller.close()


# ---- permanent sidecar errors ------------------------------------------------


def test_permanent_error_code_set():
    assert PERMANENT_SIDECAR_ERROR_CODES == {
        grpc.StatusCode.INVALID_ARGUMENT,
        grpc.StatusCode.PERMISSION_DENIED,
        grpc.StatusCode.UNAUTHENTICATED,
        grpc.StatusCode.UNIMPLEMENTED,
        grpc.StatusCode.OUT_OF_RANGE,
        grpc.StatusCode.FAILED_PRECONDITION,
    }


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("x"),
        BrokenRpcError(),
        FakeRpcError(grpc.StatusCode.UNAVAILABLE),
        FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED),
    ],
)
def test_non_permanent_errors(exc):
    assert permanent_error_code(exc) is None


class _DetailsRaise(grpc.RpcError):
    def code(self) -> Any:
        return grpc.StatusCode.UNKNOWN

    def details(self) -> str:
        raise RuntimeError("boom")


class _DetailsNotStr(_DetailsRaise):
    def details(self) -> Any:
        return None


@pytest.mark.parametrize("exc", [_DetailsRaise(), _DetailsNotStr()])
def test_unreadable_details_are_not_instance_not_found(exc):
    assert not is_instance_not_found_error(exc)
    assert permanent_error_code(exc) is None


class _NotFoundNoDetails(_DetailsRaise):
    def code(self) -> Any:
        return grpc.StatusCode.NOT_FOUND


def test_not_found_code_needs_no_details():
    assert is_instance_not_found_error(_NotFoundNoDetails())


def test_transient_raise_error_still_retries():
    wf = _wf()
    wf.raise_workflow_event.side_effect = FakeRpcError(
        grpc.StatusCode.UNAVAILABLE, "down"
    )
    dispatcher = _dispatcher(wf)
    assert _dispatch(dispatcher) == "retry"
    dispatcher.close()


@pytest.mark.parametrize("code", sorted(PERMANENT_SIDECAR_ERROR_CODES, key=str))
def test_permanent_state_check_error_drops(code, caplog):
    wf = _wf()
    wf.get_workflow_state.side_effect = FakeRpcError(code, "rejected")
    dispatcher = _dispatcher(wf)
    with caplog.at_level(logging.WARNING):
        assert _dispatch(dispatcher) == "drop"
    wf.raise_workflow_event.assert_not_called()
    assert code.name in caplog.text and "'wf-1'" in caplog.text
    assert "'evt'" in caplog.text and "'route'" in caplog.text
    dispatcher.close()


def test_transient_state_check_error_still_retries():
    wf = _wf()
    wf.get_workflow_state.side_effect = FakeRpcError(
        grpc.StatusCode.UNAVAILABLE, "down"
    )
    dispatcher = _dispatcher(wf)
    assert _dispatch(dispatcher) == "retry"
    dispatcher.close()


# ---- stable dedupe key without a CloudEvent id -------------------------------


def test_dedupe_key_without_id_is_sha256_of_canonical_json():
    key = _StreamSubscriber._dedup_id(TTLDedupeBackend(), {}, {"a": 1}, "t")
    # sha256('{"a":1}'), precomputed.
    assert key == "t:015abd7f5cc57a2dd94b7590f04ad8084273905ee33ec5cebeae62276a97f862"


def test_dedupe_key_ignores_key_order_and_keeps_unicode():
    backend = TTLDedupeBackend()
    first = _StreamSubscriber._dedup_id(backend, None, {"b": "ø", "a": 1}, "t")
    second = _StreamSubscriber._dedup_id(backend, None, {"a": 1, "b": "ø"}, "t")
    assert first == second == f"t:{stable_json_sha256({'a': 1, 'b': 'ø'})}"


@pytest.mark.parametrize(
    "value",
    [{1: "x", "a": "y"}, object(), {"s": "\ud800"}],
    ids=["mixed-keys", "object", "lone-surrogate"],
)
def test_stable_json_sha256_falls_back_or_survives(value):
    # Mixed key types and objects are not JSON-serializable with sort_keys:
    # str(value) is hashed instead. A lone surrogate still encodes.
    assert len(stable_json_sha256(value)) == 64


def test_dedupe_key_with_id_uses_id():
    assert _StreamSubscriber._dedup_id(TTLDedupeBackend(), {"id": "x"}, {}, "t") == "x"


def test_dedupe_key_is_stable_across_processes():
    code = (
        "from dapr_agents.workflow.utils.subscription import _StreamSubscriber, "
        "TTLDedupeBackend;"
        "print(_StreamSubscriber._dedup_id(TTLDedupeBackend(), None, {'a': [1, 'b']}, 't'))"
    )
    keys = set()
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        keys.add(out.stdout.strip())
    assert len(keys) == 1
