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

"""Workflow event routes: poison payloads, limits, reserved names, timeouts."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from typing import Any

import pytest
from dapr.ext.workflow.workflow_state import WorkflowStatus
from pydantic import BaseModel

from dapr_agents.types.workflow import NotFoundRetryPolicy
from dapr_agents.workflow.utils.event_routes import (
    MAX_EVENT_IDENTIFIER_LENGTH,
    EventRouteResolutionError,
    WorkflowEventDispatcher,
    coerce_identifier,
    is_reserved_event_name,
    serialize_event_data,
)
from dapr_agents.workflow.utils.event_route_state import NotFoundTracker
from dapr_agents.workflow.utils.registration import _collect_message_bindings
from dapr_agents.workflow.utils.subscription import (
    _serialize_event_default_data,
    _serialize_workflow_input,
    topic_has_event_route_conflict,
)
from tests.workflow._event_route_helpers import (
    make_ctx,
    make_spec,
    make_target,
    make_wf,
    workflow_state,
)

_WAIT = 5.0


_ctx = make_ctx
_target = make_target
_wf = make_wf


def _spec(**overrides: Any):
    return make_spec(dict(event_name="evt", instance_id_from="wf_id"), **overrides)


def _dispatch(wf_client: Any, message: Any, **target_overrides: Any) -> str:
    dispatcher = WorkflowEventDispatcher(
        wf_client=wf_client, default_serializer=_serialize_event_default_data
    )
    try:
        return dispatcher.dispatch(
            target=_target(**target_overrides),
            route_name="route",
            pubsub="messagepubsub",
            topic="t",
            dead_letter_topic=None,
            message=message,
            msg_ctx=_ctx(),
        )
    finally:
        dispatcher.close()


def _deep_dict(depth: int) -> dict:
    root: dict = {"wf_id": "wf-1"}
    node = root
    for _ in range(depth):
        node["n"] = {}
        node = node["n"]
    return root


class _Opaque:
    pass


class _AnyHolder(BaseModel):
    wf_id: str
    blob: Any


# ---- S1: poison payloads drop -------------------------------------------------


def test_deeply_nested_payload_drops():
    wf_client = _wf()
    assert _dispatch(wf_client, _deep_dict(50_000)) == "drop"
    wf_client.get_workflow_state.assert_not_called()


def test_deeply_nested_resolved_data_drops():
    wf_client = _wf()
    deep = _deep_dict(50_000)
    assert _dispatch(wf_client, {"wf_id": "wf-1"}, data_from=lambda m, c: deep) == (
        "drop"
    )


def test_model_with_unserializable_any_field_drops_with_default_serializer(caplog):
    wf_client = _wf()
    message = _AnyHolder(wf_id="wf-1", blob=_Opaque())
    with caplog.at_level(logging.WARNING):
        assert _dispatch(wf_client, message) == "drop"
    assert "data" in caplog.text
    wf_client.raise_workflow_event.assert_not_called()


def test_model_with_unserializable_any_field_drops_from_data_from():
    wf_client = _wf()
    bad = _AnyHolder(wf_id="wf-1", blob=_Opaque())
    assert _dispatch(wf_client, {"wf_id": "wf-1"}, data_from=lambda m, c: bad) == (
        "drop"
    )
    wf_client.raise_workflow_event.assert_not_called()


def test_default_serializer_failure_becomes_resolution_error():
    with pytest.raises(EventRouteResolutionError) as info:
        _serialize_event_default_data(_AnyHolder(wf_id="w", blob=_Opaque()))
    assert info.value.field == "data"


def test_custom_default_serializer_failure_drops():
    def boom(_: Any) -> Any:
        raise RecursionError("deep")

    with pytest.raises(EventRouteResolutionError, match="serialization failed"):
        serialize_event_data(None, {}, _ctx(), default_serializer=boom)


class _Exploding:
    @property
    def wf_id(self) -> str:
        raise RuntimeError("property blew up")


def test_unexpected_resolver_error_is_rewrapped_and_drops(caplog):
    # hasattr() lets non-AttributeError exceptions escape the path walk.
    wf_client = _wf()
    with caplog.at_level(logging.WARNING):
        assert _dispatch(wf_client, _Exploding()) == "drop"
    assert "unexpected RuntimeError" in caplog.text


# ---- S2: identifier and payload limits ---------------------------------------


def test_identifier_length_limit():
    at_limit = "x" * MAX_EVENT_IDENTIFIER_LENGTH
    assert coerce_identifier(at_limit, field="instance_id") == at_limit
    with pytest.raises(EventRouteResolutionError, match="limit 512"):
        coerce_identifier(at_limit + "x", field="instance_id")
    with pytest.raises(EventRouteResolutionError, match="limit 512"):
        coerce_identifier(int("9" * 513), field="event_name")


def test_oversized_instance_id_and_event_name_drop():
    wf_client = _wf()
    long_id = "w" * (MAX_EVENT_IDENTIFIER_LENGTH + 1)
    assert _dispatch(wf_client, {"wf_id": long_id}) == "drop"
    assert _dispatch(
        wf_client, {"wf_id": "wf-1", "k": long_id}, event_name_from="k"
    ) == ("drop")
    wf_client.get_workflow_state.assert_not_called()


def test_max_data_bytes_limit():
    payload = {"wf_id": "wf-1", "s": "x" * 100}
    size = len(json.dumps(_serialize_event_default_data(payload)).encode())
    assert _dispatch(_wf(), payload, max_data_bytes=size) == "success"
    wf_client = _wf()
    assert _dispatch(wf_client, payload, max_data_bytes=size - 1) == "drop"
    wf_client.get_workflow_state.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_data_bytes", 0),
        ("max_data_bytes", True),
        ("call_timeout_seconds", 0),
        ("call_timeout_seconds", float("inf")),
        ("call_timeout_seconds", "1"),
        ("dedupe_max_entries", 0),
        ("dedupe_max_entries", 1.5),
    ],
)
def test_spec_rejects_bad_limits(field, value):
    with pytest.raises(ValueError, match=field):
        _spec(**{field: value})


def test_spec_limits_reach_the_target():
    spec = _spec(
        max_data_bytes=10,
        call_timeout_seconds=1.5,
        dedupe_max_entries=3,
        allow_reserved_event_names=True,
    )
    target = _collect_message_bindings(targets=None, routes=[spec])[0].event_target
    assert target is not None
    assert (
        target.max_data_bytes,
        target.call_timeout_seconds,
        target.dedupe_max_entries,
        target.allow_reserved_event_names,
    ) == (10, 1.5, 3, True)


# ---- P3: not-found policy upper bounds ----------------------------------------


def test_not_found_policy_upper_bounds():
    assert (
        NotFoundRetryPolicy(max_attempts=100, window_seconds=3600).max_attempts == 100
    )
    with pytest.raises(ValueError, match="<= 100"):
        NotFoundRetryPolicy(max_attempts=101)
    with pytest.raises(ValueError, match="<= 3600"):
        NotFoundRetryPolicy(window_seconds=3600.5)


# ---- S3: reserved event names ------------------------------------------------


@pytest.mark.parametrize(
    "name,reserved",
    [
        ("approval_response_abc", True),
        ("Approval_Response_abc", True),
        ("user_input_response:req-1", True),
        ("USER_INPUT_RESPONSE:req-1", True),
        ("user_input_response", False),
        ("approval_responses", False),
        ("job_finished", False),
    ],
)
def test_is_reserved_event_name(name, reserved):
    assert is_reserved_event_name(name) is reserved


@pytest.mark.parametrize("name", ["approval_response_x", "User_Input_Response:1"])
def test_static_reserved_event_name_rejected_at_registration(name):
    with pytest.raises(ValueError, match="allow_reserved_event_names"):
        _collect_message_bindings(targets=None, routes=[_spec(event_name=name)])


def test_static_reserved_event_name_allowed_with_flag():
    spec = _spec(event_name="approval_response_x", allow_reserved_event_names=True)
    binding = _collect_message_bindings(targets=None, routes=[spec])[0]
    assert binding.event_target is not None
    assert binding.event_target.event_name == "approval_response_x"


def test_dynamic_reserved_event_name_drops_with_warning(caplog):
    wf_client = _wf()
    message = {"wf_id": "wf-1", "k": "approval_response_123"}
    with caplog.at_level(logging.WARNING):
        assert _dispatch(wf_client, message, event_name_from="k") == "drop"
    assert "reserved event name 'approval_response_123'" in caplog.text
    wf_client.get_workflow_state.assert_not_called()


def test_dynamic_reserved_event_name_allowed_with_flag():
    wf_client = _wf()
    message = {"wf_id": "wf-1", "k": "user_input_response:r1"}
    assert (
        _dispatch(
            wf_client, message, event_name_from="k", allow_reserved_event_names=True
        )
        == "success"
    )
    assert (
        wf_client.raise_workflow_event.call_args.kwargs["event_name"]
        == "user_input_response:r1"
    )


# ---- P2: per-call timeout ------------------------------------------------------


class _SlowClient:
    """Workflow client whose chosen method blocks until released."""

    def __init__(self, slow: str) -> None:
        self.slow = slow
        self.release = threading.Event()
        self.raised: list[dict] = []

    def _maybe_block(self, name: str) -> None:
        if name == self.slow:
            self.release.wait(_WAIT)

    def get_workflow_state(self, instance_id: str, *, fetch_payloads: bool) -> Any:
        self._maybe_block("get_workflow_state")
        return workflow_state(WorkflowStatus.RUNNING)

    def raise_workflow_event(self, **kwargs: Any) -> None:
        self._maybe_block("raise_workflow_event")
        self.raised.append(kwargs)


@pytest.mark.parametrize(
    "slow,what", [("get_workflow_state", "state"), ("raise_workflow_event", "raising")]
)
def test_timed_out_call_retries_with_warning(slow, what, caplog):
    client = _SlowClient(slow)
    try:
        with caplog.at_level(logging.WARNING):
            status = _dispatch(
                client,
                {"wf_id": "wf-1"},
                call_timeout_seconds=0.25,  # the fast call finishes well within this
            )
    finally:
        client.release.set()
    assert status == "retry"
    assert "timed out" in caplog.text and what in caplog.text
    assert "may still complete in the background" in caplog.text


# ---- A3 / Q4 / P7 ------------------------------------------------------------


@pytest.mark.parametrize(
    "flags,conflict",
    [
        ([True], False),
        ([False], False),
        ([False, False], False),
        ([True, False], True),
        ([False, True], True),
        ([True, True], True),
    ],
)
def test_topic_has_event_route_conflict(flags, conflict):
    assert topic_has_event_route_conflict(flags) is conflict


class _Stamped(BaseModel):
    at: datetime


def test_schedule_serialization_is_unchanged_by_json_mode_default():
    model = _Stamped(at=datetime(2026, 1, 2))
    assert _serialize_workflow_input(model) == (model.model_dump(), None)
    assert _serialize_workflow_input(model, json_mode=True)[0] == {
        "at": "2026-01-02T00:00:00"
    }
    for value in ({"a": 1}, 5):
        assert _serialize_workflow_input(value) == _serialize_workflow_input(
            value, json_mode=True
        )


def test_model_default_data_skips_second_jsonable_pass(monkeypatch):
    def _fail(_: Any) -> Any:
        raise AssertionError("to_jsonable_python must not run for models")

    monkeypatch.setattr(
        "dapr_agents.workflow.utils.subscription.to_jsonable_python", _fail
    )
    assert _serialize_event_default_data(_Stamped(at=datetime(2026, 1, 2))) == {
        "at": "2026-01-02T00:00:00"
    }


# ---- T2: not-found tracker lock ----------------------------------------------


def test_not_found_tracker_is_thread_safe():
    tracker = NotFoundTracker()
    threads_n, per_thread = 8, 200
    barrier = threading.Barrier(threads_n + 1)
    seen: list[list[int]] = [[] for _ in range(threads_n)]

    def record(slot: int) -> None:
        barrier.wait()
        for _ in range(per_thread):
            seen[slot].append(tracker.record("shared", 0.0)[0])

    def churn() -> None:
        barrier.wait()
        for i in range(per_thread):
            tracker.record(f"other-{i}", 0.0)
            tracker.clear(f"other-{i}")

    workers = [threading.Thread(target=record, args=(i,)) for i in range(threads_n)]
    workers.append(threading.Thread(target=churn))
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(_WAIT)
        assert not worker.is_alive()

    counts = sorted(n for slot in seen for n in slot)
    # Every increment was observed exactly once: no lost or duplicated update.
    assert counts == list(range(1, threads_n * per_thread + 1))
    assert tracker.record("shared", 0.0) == (threads_n * per_thread + 1, 0.0)
    assert tracker.record("other-0", 0.0) == (1, 0.0)
