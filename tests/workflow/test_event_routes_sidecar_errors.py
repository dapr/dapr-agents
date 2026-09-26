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

import logging
import os
import subprocess
import sys
import threading
from concurrent.futures import Future
from typing import Any, Optional
from unittest.mock import MagicMock

import grpc
import pytest
from dapr.ext.workflow.workflow_state import WorkflowStatus

from dapr_agents.types.message import EventMessageMetadata
from dapr_agents.types.workflow import NotFoundRetryPolicy
from dapr_agents.workflow.utils.call_deadline import DeadlineCaller
from dapr_agents.workflow.utils.event_route_calls import (
    MIN_TIMED_OUT_RAISES_TRACKED,
    PERMANENT_SIDECAR_ERROR_CODES,
    TimedOutRaises,
    permanent_error_code,
)
from dapr_agents.workflow.utils.event_routes import (
    EventRouteTarget,
    WorkflowEventDispatcher,
)
from dapr_agents.workflow.utils.subscription import (
    MessageContext,
    TTLDedupeBackend,
    _StreamSubscriber,
)
from tests.workflow._event_route_helpers import FakeRpcError, workflow_state

_KEY = "evt-1"
_WAIT = 5.0


def _ctx(event_id: Optional[str] = _KEY) -> MessageContext:
    fields = dict.fromkeys(EventMessageMetadata.model_fields)
    fields.update(id=event_id, topic="t")
    return MessageContext(
        event=EventMessageMetadata.model_validate(fields), handler_name="route"
    )


def _target(**overrides: Any) -> EventRouteTarget:
    values: dict[str, Any] = dict(
        event_name="evt",
        instance_id_from="wf_id",
        event_name_from=None,
        data_from=None,
        dedupe=True,
        deduper=None,
        not_found_retry=NotFoundRetryPolicy(),
        call_timeout_seconds=0.05,
    )
    values.update(overrides)
    return EventRouteTarget(**values)


def _wf() -> MagicMock:
    wf = MagicMock()
    wf.get_workflow_state.return_value = workflow_state(WorkflowStatus.RUNNING)
    return wf


def _dispatcher(wf: MagicMock, caller: Any = None) -> WorkflowEventDispatcher:
    return WorkflowEventDispatcher(
        wf_client=wf, default_serializer=lambda m: m, caller=caller
    )


def _dispatch(
    dispatcher: WorkflowEventDispatcher,
    *,
    dedupe_key: Optional[str] = _KEY,
    dlq: Optional[str] = None,
    target: Optional[EventRouteTarget] = None,
) -> str:
    return dispatcher.dispatch(
        target=target or _target(),
        route_name="route",
        pubsub="p",
        topic="t",
        dead_letter_topic=dlq,
        message={"wf_id": "wf-1"},
        msg_ctx=_ctx(),
        dedupe_key=dedupe_key,
    )


class _GatedRaise:
    """raise_workflow_event stand-in that blocks until released, then does ``outcome``."""

    def __init__(self, outcome: Optional[BaseException] = None) -> None:
        self.release = threading.Event()
        self.finished = threading.Event()
        self.calls = 0
        self.outcome = outcome

    def __call__(self, **kwargs: Any) -> None:
        self.calls += 1
        if self.calls > 1:
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
    tracked = dispatcher._timed_out_raises(
        MagicMock(pubsub="p", topic="t", target=_target())
    )
    # The future settles just after the call returns.
    tracked._futures[_KEY].exception(timeout=_WAIT)


# ---- timed-out raise tracking ----------------------------------------------


def test_redelivery_while_timed_out_raise_runs_retries_without_raising():
    gate, wf, dispatcher = _time_out_first_raise()
    try:
        assert _dispatch(dispatcher) == "retry"
        assert gate.calls == 1
    finally:
        gate.release.set()
        dispatcher.close()


def test_redelivery_after_timed_out_raise_succeeded_acks_without_raising():
    gate, wf, dispatcher = _time_out_first_raise()
    _finish(gate, dispatcher)
    assert _dispatch(dispatcher) == "success"
    assert gate.calls == 1
    tracked = dispatcher._timed_out[("p", "t")]
    assert len(tracked) == 0  # resolved entries are removed
    dispatcher.close()


def test_redelivery_after_transient_failure_raises_again():
    gate, wf, dispatcher = _time_out_first_raise(
        FakeRpcError(grpc.StatusCode.UNAVAILABLE, "down")
    )
    _finish(gate, dispatcher)
    assert _dispatch(dispatcher) == "success"
    assert gate.calls == 2
    dispatcher.close()


def test_redelivery_after_permanent_failure_drops_without_raising(caplog):
    gate, wf, dispatcher = _time_out_first_raise(
        FakeRpcError(grpc.StatusCode.PERMISSION_DENIED, "no")
    )
    _finish(gate, dispatcher)
    with caplog.at_level(logging.WARNING):
        assert _dispatch(dispatcher, dlq="dlq") == "drop"
    assert gate.calls == 1
    assert "PERMISSION_DENIED" in caplog.text
    assert "'dlq'" in caplog.text
    dispatcher.close()


def test_timed_out_raise_without_dedupe_key_is_not_tracked():
    # dedupe=False: no key, so a timed-out raise can be raised again (documented).
    gate = _GatedRaise()
    wf = _wf()
    wf.raise_workflow_event.side_effect = gate
    dispatcher = _dispatcher(wf)
    try:
        assert _dispatch(dispatcher, dedupe_key=None) == "retry"
        assert _dispatch(dispatcher, dedupe_key=None) == "success"
        assert gate.calls == 2
        assert dispatcher._timed_out == {}
    finally:
        gate.release.set()
        dispatcher.close()


class _NeverRunsCaller:
    """Runs state checks inline; queues raises that no worker ever picks up."""

    def call(self, fn: Any, timeout: float, *args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Future:
        return Future()

    def close(self) -> None:
        return None


def test_timed_out_raise_cancelled_while_queued_is_not_tracked():
    wf = _wf()
    dispatcher = _dispatcher(wf, caller=_NeverRunsCaller())
    assert _dispatch(dispatcher) == "retry"
    assert len(dispatcher._timed_out[("p", "t")]) == 0
    wf.raise_workflow_event.assert_not_called()


def test_timed_out_raises_bounded_and_sized_from_dedupe_entries():
    tracked = TimedOutRaises(maxsize=2)
    for key in ("a", "b", "c"):
        tracked.add(key, Future())
    assert len(tracked) == 2
    assert tracked.get("a") is None
    pending = tracked.get("c")
    assert pending is not None and len(tracked) == 2  # unfinished stays tracked

    dispatcher = _dispatcher(_wf())
    small = dispatcher._timed_out_raises(
        MagicMock(pubsub="p", topic="t1", target=_target(dedupe_max_entries=8))
    )
    large = dispatcher._timed_out_raises(
        MagicMock(pubsub="p", topic="t2", target=_target(dedupe_max_entries=10_000))
    )
    assert small._futures.maxsize == MIN_TIMED_OUT_RAISES_TRACKED
    assert large._futures.maxsize == 10_000
    dispatcher.close()


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


class _BrokenRpcError(grpc.RpcError):
    def code(self) -> Any:
        raise RuntimeError("boom")


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("x"),
        _BrokenRpcError(),
        FakeRpcError(grpc.StatusCode.UNAVAILABLE),
        FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED),
    ],
)
def test_non_permanent_errors(exc):
    assert permanent_error_code(exc) is None


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


def test_dedupe_key_without_id_is_sha256_of_payload():
    key = _StreamSubscriber._dedup_id(TTLDedupeBackend(), {}, {"a": 1}, "t")
    # sha256("{'a': 1}"), precomputed.
    assert key == "t:240f5ff9499fabe7952369a2a095ad1d8dedba650ea844f3fa0027e5ddc12f49"


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
