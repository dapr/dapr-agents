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

"""Unit tests for workflow event routes: specs, validation, resolution, dispatch."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional, Union
from unittest.mock import MagicMock, patch
from uuid import UUID

import grpc
import pytest
from dapr.ext.workflow.workflow_state import WorkflowStatus
from pydantic import BaseModel

from dapr_agents.types.exceptions import PubSubNotAvailableError
from dapr_agents.types.message import EventMessageMetadata
from dapr_agents.types.workflow import (
    NotFoundRetryPolicy,
    PubSubRouteSpec,
    WorkflowEventRouteSpec,
)
from dapr_agents.workflow.utils.event_routes import (
    EventRouteResolutionError,
    EventRouteTarget,
    WorkflowEventDispatcher,
    coerce_identifier,
    is_instance_not_found_error,
    parse_field_path,
    resolve_field,
    serialize_event_data,
)
from dapr_agents.workflow.utils.registration import (
    _collect_message_bindings,
    register_message_routes,
)
from dapr_agents.workflow.utils.subscription import (
    METADATA_KEY,
    MessageContext,
    MessageRouteBinding,
    TTLDedupeBackend,
    _serialize_event_default_data,
    _serialize_workflow_input,
    _validate_event_bindings,
)

_PATCH_TARGET = "dapr_agents.workflow.utils.registration.default_dapr_client_factory"


# ---- helpers ----------------------------------------------------------------


class _FakeRpcError(grpc.RpcError):
    def __init__(self, code: Any, details: str = "") -> None:
        super().__init__(details)
        self._code = code
        self._details = details

    def code(self) -> Any:
        return self._code

    def details(self) -> str:
        return self._details


class _BrokenRpcError(grpc.RpcError):
    def code(self) -> Any:
        raise RuntimeError("boom")

    def details(self) -> str:
        raise RuntimeError("boom")


class _JobRef(BaseModel):
    workflow_id: str


class _JobFinished(BaseModel):
    job: _JobRef
    status: str


class _Other(BaseModel):
    x: int


@dataclass
class _DC:
    ref: str


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _ctx(event_id: Optional[str] = "evt-1", name: str = "route") -> MessageContext:
    fields = dict.fromkeys(EventMessageMetadata.model_fields)
    fields.update(id=event_id, topic="t")
    return MessageContext(
        event=EventMessageMetadata.model_validate(fields), handler_name=name
    )


def _state(status: WorkflowStatus) -> MagicMock:
    state = MagicMock()
    state.runtime_status = status
    return state


def _spec(**overrides: Any) -> WorkflowEventRouteSpec:
    values: dict[str, Any] = dict(
        pubsub_name="messagepubsub",
        topic="t",
        event_name="evt",
        instance_id_from="wf_id",
    )
    values.update(overrides)
    return WorkflowEventRouteSpec(**values)


def _target(**overrides: Any) -> EventRouteTarget:
    values: dict[str, Any] = dict(
        event_name="evt",
        instance_id_from="wf_id",
        event_name_from=None,
        data_from=None,
        dedupe=True,
        deduper=None,
        not_found_retry=NotFoundRetryPolicy(),
    )
    values.update(overrides)
    return EventRouteTarget(**values)


def _dispatcher(wf_client: MagicMock, clock: Optional[_Clock] = None):
    return WorkflowEventDispatcher(
        wf_client=wf_client,
        default_serializer=lambda p: _serialize_workflow_input(p)[0],
        clock=clock or _Clock(),
    )


def _dispatch(dispatcher, target=None, message=None, ctx=None, dlq=None) -> str:
    return dispatcher.dispatch(
        target=target or _target(),
        route_name="route",
        pubsub="messagepubsub",
        topic="t",
        dead_letter_topic=dlq,
        message=message if message is not None else {"wf_id": "wf-1"},
        msg_ctx=ctx or _ctx(),
    )


def _wf(status: Optional[WorkflowStatus] = WorkflowStatus.RUNNING) -> MagicMock:
    wf_client = MagicMock()
    wf_client.get_workflow_state.return_value = (
        _state(status) if status is not None else None
    )
    return wf_client


# ---- 9.1 registration and validation ----------------------------------------


@pytest.mark.parametrize("path", ["id", "job.workflow_id", "items.0.id"])
def test_valid_field_paths_accepted(path):
    _collect_message_bindings(targets=None, routes=[_spec(instance_id_from=path)])
    assert parse_field_path(path) == tuple(path.split("."))


@pytest.mark.parametrize(
    "path", ["", ".", ".a", "a.", "a..b", " a", "a. b", "a.__class__"]
)
def test_invalid_field_paths_rejected(path):
    with pytest.raises(ValueError, match="invalid field path"):
        _collect_message_bindings(targets=None, routes=[_spec(instance_id_from=path)])


class _AsyncCallable:
    async def __call__(self, msg, ctx):
        return "x"


async def _async_resolver(msg, ctx):
    return "x"


@pytest.mark.parametrize("field", ["instance_id_from", "event_name_from", "data_from"])
@pytest.mark.parametrize("resolver", [_async_resolver, _AsyncCallable()])
def test_async_resolvers_rejected(field, resolver):
    with pytest.raises(TypeError, match="synchronous"):
        _collect_message_bindings(targets=None, routes=[_spec(**{field: resolver})])


@pytest.mark.parametrize("field", ["instance_id_from", "event_name_from", "data_from"])
def test_non_str_non_callable_resolver_rejected(field):
    with pytest.raises(TypeError, match="dotted field path"):
        _collect_message_bindings(targets=None, routes=[_spec(**{field: 5})])


def test_missing_instance_id_resolver_rejected():
    with pytest.raises(TypeError, match="required"):
        _collect_message_bindings(targets=None, routes=[_spec(instance_id_from=None)])


@pytest.mark.parametrize("field", ["event_name", "topic", "pubsub_name"])
@pytest.mark.parametrize("value", ["", "   "])
def test_empty_identity_fields_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        _collect_message_bindings(targets=None, routes=[_spec(**{field: value})])


def test_not_found_policy_defaults():
    policy = NotFoundRetryPolicy()
    assert policy.max_attempts == 10
    assert policy.window_seconds == 300.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"max_attempts": True},
        {"max_attempts": 1.5},
        {"window_seconds": 0},
        {"window_seconds": -1},
        {"window_seconds": float("inf")},
        {"window_seconds": "10"},
    ],
)
def test_not_found_policy_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        NotFoundRetryPolicy(**kwargs)


def test_dedupe_false_with_deduper_rejected():
    with pytest.raises(ValueError, match="dedupe=False"):
        _collect_message_bindings(
            targets=None, routes=[_spec(dedupe=False, deduper=TTLDedupeBackend())]
        )


def test_unsupported_message_model_rejected():
    with pytest.raises(TypeError, match="Unsupported"):
        _collect_message_bindings(targets=None, routes=[_spec(message_model=int)])
    with pytest.raises(TypeError, match="Unsupported"):
        _collect_message_bindings(targets=None, routes=[_spec(message_model=5)])


def test_union_message_model_gives_both_schemas():
    bindings = _collect_message_bindings(
        targets=None, routes=[_spec(message_model=Union[_JobFinished, _Other])]
    )
    assert bindings[0].schemas == [_JobFinished, _Other]


def test_collect_mixed_routes():
    def wf_handler(ctx, msg):
        return None

    bindings = _collect_message_bindings(
        targets=None,
        routes=[
            PubSubRouteSpec(
                pubsub_name="messagepubsub", topic="a", handler_fn=wf_handler
            ),
            _spec(),
        ],
    )
    normal, event = bindings
    assert normal.event_target is None
    assert normal.handler is wf_handler
    assert event.event_target is not None
    assert event.mapper is None
    assert event.name == "evt@messagepubsub:t"
    assert event.handler.__name__ == "evt@messagepubsub:t"
    assert event.schemas == [dict]
    assert event.event_target.dedupe is True


def test_collect_uses_custom_name():
    (binding,) = _collect_message_bindings(targets=None, routes=[_spec(name="mine")])
    assert binding.name == "mine"
    assert binding.handler.__name__ == "mine"


def _binding(name: str, topic: str, *, event: bool, mapper=None) -> MessageRouteBinding:
    return MessageRouteBinding(
        handler=lambda *_: None,
        schemas=[dict],
        pubsub="messagepubsub",
        topic=topic,
        dead_letter_topic=None,
        name=name,
        mapper=mapper,
        event_target=_target() if event else None,
    )


def test_validate_event_bindings_rejects_two_event_routes_on_one_topic():
    with pytest.raises(ValueError, match="one topic per event route"):
        _validate_event_bindings(
            [_binding("a", "t", event=True), _binding("b", "t", event=True)]
        )


def test_validate_event_bindings_rejects_event_plus_schedule_route():
    with pytest.raises(ValueError, match="cannot have other routes"):
        _validate_event_bindings(
            [_binding("sched", "t", event=False), _binding("evt", "t", event=True)]
        )


def test_validate_event_bindings_accepts_distinct_topics():
    _validate_event_bindings(
        [
            _binding("a", "t1", event=True),
            _binding("b", "t2", event=True),
            _binding("c", "t3", event=False),
            _binding("d", "t3", event=False),
        ]
    )


def test_validate_event_bindings_rejects_mapper():
    with pytest.raises(ValueError, match="mapper"):
        _validate_event_bindings(
            [_binding("a", "t", event=True, mapper=lambda m, c: m)]
        )


def test_register_event_route_on_unknown_pubsub_raises():
    from tests.workflow.test_message_router import create_mock_dapr_client

    mock_dapr = create_mock_dapr_client(["otherpubsub"])
    with patch(_PATCH_TARGET, return_value=mock_dapr):
        with pytest.raises(PubSubNotAvailableError):
            register_message_routes(
                dapr_client=mock_dapr, routes=[_spec()], wf_client=MagicMock()
            )


# ---- 9.2 resolution -------------------------------------------------------------


@pytest.mark.parametrize(
    "message,path,expected",
    [
        ({"a": {"b": "v"}}, "a.b", "v"),
        (
            _JobFinished(job=_JobRef(workflow_id="w"), status="ok"),
            "job.workflow_id",
            "w",
        ),
        (_DC(ref="r"), "ref", "r"),
        ({"items": [{"id": "x"}, {"id": "y"}]}, "items.1.id", "y"),
        ({"items": ({"obj": _DC(ref="deep")},)}, "items.0.obj.ref", "deep"),
    ],
)
def test_resolve_path(message, path, expected):
    assert resolve_field(path, message, _ctx(), field="instance_id") == expected


@pytest.mark.parametrize(
    "message,path,segment",
    [
        ({"a": {}}, "a.b", "b"),
        (_DC(ref="r"), "missing", "missing"),
        ({"items": [1]}, "items.3", "3"),
        ({"items": [1]}, "items.count", "count"),
    ],
)
def test_resolve_path_missing(message, path, segment):
    with pytest.raises(EventRouteResolutionError) as info:
        resolve_field(path, message, _ctx(), field="instance_id")
    assert repr(segment) in info.value.reason
    assert info.value.field == "instance_id"


def test_resolve_callable_gets_message_and_context():
    seen: dict[str, Any] = {}

    def resolver(msg, ctx):
        seen["msg"], seen["ctx"] = msg, ctx
        return "wf-9"

    message = {"k": 1}
    ctx = _ctx(name="my-route")
    assert resolve_field(resolver, message, ctx, field="instance_id") == "wf-9"
    assert seen["msg"] is message
    assert seen["ctx"].handler_name == "my-route"


def test_resolve_callable_error_is_wrapped():
    boom = KeyError("nope")

    def resolver(msg, ctx):
        raise boom

    with pytest.raises(EventRouteResolutionError) as info:
        resolve_field(resolver, {}, _ctx(), field="data")
    assert info.value.__cause__ is boom
    assert "KeyError" in info.value.reason


def test_coerce_identifier():
    assert coerce_identifier("abc", field="instance_id") == "abc"
    assert coerce_identifier(42, field="instance_id") == "42"
    for bad in (True, None, "", "  ", ["x"]):
        with pytest.raises(EventRouteResolutionError):
            coerce_identifier(bad, field="instance_id")


def test_serialize_default_matches_workflow_input():
    message = {"wf_id": "w", METADATA_KEY: {"id": "evt-1"}}
    result = serialize_event_data(
        None,
        message,
        _ctx(),
        default_serializer=lambda p: _serialize_workflow_input(p)[0],
    )
    assert result == _serialize_workflow_input(message)[0]
    assert METADATA_KEY in result


def test_serialize_resolved_values():
    ser = lambda p: pytest.fail("default serializer must not run")  # noqa: E731
    model = _JobRef(workflow_id="w")
    assert serialize_event_data(
        lambda m, c: model, {}, _ctx(), default_serializer=ser
    ) == {"workflow_id": "w"}
    assert serialize_event_data(
        lambda m, c: _DC(ref="r"), {}, _ctx(), default_serializer=ser
    ) == {"ref": "r"}
    assert (
        serialize_event_data(lambda m, c: None, {}, _ctx(), default_serializer=ser)
        is None
    )
    assert serialize_event_data("a", {"a": [1]}, _ctx(), default_serializer=ser) == [1]


def test_serialize_rejects_non_json():
    with pytest.raises(EventRouteResolutionError, match="JSON"):
        serialize_event_data(
            lambda m, c: object(), {}, _ctx(), default_serializer=lambda p: p
        )


def test_is_instance_not_found_error():
    assert is_instance_not_found_error(_FakeRpcError(grpc.StatusCode.NOT_FOUND))
    assert is_instance_not_found_error(
        _FakeRpcError(grpc.StatusCode.UNKNOWN, "error: no such instance exists")
    )
    assert not is_instance_not_found_error(_FakeRpcError(grpc.StatusCode.UNAVAILABLE))
    assert not is_instance_not_found_error(RuntimeError("no such instance exists"))
    assert not is_instance_not_found_error(_BrokenRpcError())


# ---- 9.3 dispatcher branches ---------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        WorkflowStatus.RUNNING,
        WorkflowStatus.PENDING,
        WorkflowStatus.SUSPENDED,
        WorkflowStatus.STALLED,
        WorkflowStatus.UNKNOWN,
    ],
)
def test_dispatch_raises_event_for_live_workflow(status):
    wf_client = _wf(status)
    assert _dispatch(_dispatcher(wf_client)) == "success"
    wf_client.get_workflow_state.assert_called_once_with("wf-1", fetch_payloads=False)
    wf_client.raise_workflow_event.assert_called_once_with(
        instance_id="wf-1", event_name="evt", data={"wf_id": "wf-1"}
    )


@pytest.mark.parametrize(
    "status",
    [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.TERMINATED],
)
@pytest.mark.parametrize("dlq,wording", [("t_DEAD", "dead-letter"), (None, "dropping")])
def test_dispatch_terminal_state_drops(status, dlq, wording, caplog):
    wf_client = _wf(status)
    with caplog.at_level(logging.WARNING):
        assert _dispatch(_dispatcher(wf_client), dlq=dlq) == "drop"
    wf_client.raise_workflow_event.assert_not_called()
    text = caplog.text
    assert "wf-1" in text and "'evt'" in text and "workflow is" in text
    # conftest mocks `dapr`, so WorkflowStatus members may be mocks here.
    if isinstance(status.name, str):
        assert status.name in text
    assert wording in text


def test_dispatch_not_found_retries_until_max_attempts():
    wf_client = _wf(None)
    target = _target(not_found_retry=NotFoundRetryPolicy(max_attempts=3))
    dispatcher = _dispatcher(wf_client)
    results = [_dispatch(dispatcher, target=target) for _ in range(3)]
    assert results == ["retry", "retry", "drop"]
    wf_client.raise_workflow_event.assert_not_called()


def test_dispatch_not_found_window_expires():
    wf_client = _wf(None)
    clock = _Clock()
    target = _target(
        not_found_retry=NotFoundRetryPolicy(max_attempts=100, window_seconds=10)
    )
    dispatcher = _dispatcher(wf_client, clock)
    assert _dispatch(dispatcher, target=target) == "retry"
    clock.now = 11.0
    assert _dispatch(dispatcher, target=target) == "drop"


def test_dispatch_not_found_then_found_resets_tracker():
    wf_client = _wf(None)
    target = _target(not_found_retry=NotFoundRetryPolicy(max_attempts=2))
    dispatcher = _dispatcher(wf_client)
    assert _dispatch(dispatcher, target=target) == "retry"
    wf_client.get_workflow_state.return_value = _state(WorkflowStatus.RUNNING)
    assert _dispatch(dispatcher, target=target) == "success"
    wf_client.get_workflow_state.return_value = None
    # Counting restarted: attempt 1 of 2 again.
    assert _dispatch(dispatcher, target=target) == "retry"


def test_dispatch_single_attempt_drops_immediately():
    target = _target(not_found_retry=NotFoundRetryPolicy(max_attempts=1))
    assert _dispatch(_dispatcher(_wf(None)), target=target) == "drop"


def test_dispatch_state_error_retries():
    wf_client = _wf()
    wf_client.get_workflow_state.side_effect = RuntimeError("sidecar down")
    assert _dispatch(_dispatcher(wf_client)) == "retry"
    wf_client.raise_workflow_event.assert_not_called()


def test_dispatch_raise_not_found_goes_through_not_found_branch():
    wf_client = _wf()
    wf_client.raise_workflow_event.side_effect = _FakeRpcError(
        grpc.StatusCode.NOT_FOUND
    )
    target = _target(not_found_retry=NotFoundRetryPolicy(max_attempts=2))
    dispatcher = _dispatcher(wf_client)
    assert _dispatch(dispatcher, target=target) == "retry"
    assert _dispatch(dispatcher, target=target) == "drop"


def test_dispatch_raise_transient_error_always_retries():
    wf_client = _wf()
    wf_client.raise_workflow_event.side_effect = _FakeRpcError(
        grpc.StatusCode.UNAVAILABLE
    )
    target = _target(not_found_retry=NotFoundRetryPolicy(max_attempts=1))
    dispatcher = _dispatcher(wf_client)
    assert [_dispatch(dispatcher, target=target) for _ in range(5)] == ["retry"] * 5


def test_dispatch_resolution_error_drops_without_state_call(caplog):
    wf_client = _wf()
    with caplog.at_level(logging.WARNING):
        assert _dispatch(_dispatcher(wf_client), message={"other": 1}) == "drop"
    wf_client.get_workflow_state.assert_not_called()
    assert "instance_id" in caplog.text and "evt-1" in caplog.text


def test_dispatch_key_falls_back_without_cloudevent_id():
    wf_client = _wf(None)
    target = _target(not_found_retry=NotFoundRetryPolicy(max_attempts=2))
    dispatcher = _dispatcher(wf_client)
    ctx = _ctx(event_id=None)
    assert _dispatch(dispatcher, target=target, ctx=ctx) == "retry"
    assert _dispatch(dispatcher, target=target, ctx=ctx) == "drop"


def test_dispatch_event_name_resolver_overrides_literal():
    wf_client = _wf()
    target = _target(event_name_from="kind", data_from=lambda m, c: {"v": 1})
    _dispatch(_dispatcher(wf_client), target=target, message={"wf_id": 7, "kind": "k"})
    wf_client.raise_workflow_event.assert_called_once_with(
        instance_id="7", event_name="k", data={"v": 1}
    )


def test_dispatch_state_check_not_found_error_is_bounded():
    # The SDK only returns None for "no such instance exists"; a NOT_FOUND with
    # other wording is re-raised and must still use the not-found budget.
    wf_client = _wf()
    wf_client.get_workflow_state.side_effect = _FakeRpcError(
        grpc.StatusCode.NOT_FOUND, "workflow instance 'x' not found"
    )
    target = _target(not_found_retry=NotFoundRetryPolicy(max_attempts=2))
    dispatcher = _dispatcher(wf_client)
    results = [_dispatch(dispatcher, target=target) for _ in range(2)]
    assert results == ["retry", "drop"]
    wf_client.raise_workflow_event.assert_not_called()


def test_not_found_tracker_evicts_beyond_bound(monkeypatch):
    monkeypatch.setattr("dapr_agents.workflow.utils.event_routes._TRACKER_MAXSIZE", 2)
    target = _target(not_found_retry=NotFoundRetryPolicy(max_attempts=2))
    dispatcher = _dispatcher(_wf(None))
    assert _dispatch(dispatcher, target=target, ctx=_ctx("a")) == "retry"
    assert _dispatch(dispatcher, target=target, ctx=_ctx("b")) == "retry"
    assert _dispatch(dispatcher, target=target, ctx=_ctx("c")) == "retry"
    # "a" was evicted (documented bound), so its count restarted.
    assert _dispatch(dispatcher, target=target, ctx=_ctx("a")) == "retry"
    assert _dispatch(dispatcher, target=target, ctx=_ctx("a")) == "drop"


class _Timed(BaseModel):
    wf_id: str
    at: datetime
    ref: UUID


def test_default_event_data_is_json_safe_for_pydantic_models():
    message = _Timed(
        wf_id="w",
        at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        ref=UUID(int=1),
    )
    data = _serialize_event_default_data(message)
    assert data == {
        "wf_id": "w",
        "at": "2026-01-02T00:00:00Z",
        "ref": "00000000-0000-0000-0000-000000000001",
    }
    # The schedule path is unchanged: it still dumps Python objects.
    assert isinstance(_serialize_workflow_input(message)[0]["at"], datetime)


def test_default_event_data_keeps_metadata_and_non_model_shapes():
    assert _serialize_event_default_data({"a": 1, METADATA_KEY: {"id": "e"}}) == {
        "a": 1,
        METADATA_KEY: {"id": "e"},
    }
    assert _serialize_event_default_data(_DC(ref="r")) == {"ref": "r"}
    assert _serialize_event_default_data({"d": Decimal("1.5")}) == {"d": "1.5"}


def test_dispatch_default_path_accepts_datetime_model():
    wf_client = _wf()
    dispatcher = WorkflowEventDispatcher(
        wf_client=wf_client, default_serializer=_serialize_event_default_data
    )
    message = _Timed(wf_id="wf-1", at=datetime(2026, 1, 2), ref=UUID(int=2))
    assert _dispatch(dispatcher, message=message) == "success"
    data = wf_client.raise_workflow_event.call_args.kwargs["data"]
    assert data["at"] == "2026-01-02T00:00:00"
