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

"""In-process tests: CloudEvents through register_message_routes to raise_workflow_event.

No Dapr sidecar: the Dapr client and workflow client are mocks, but the real
subscriber thread, parsing, validation, filters, dedupe and dispatch run.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import grpc
import pytest
from cachetools import TTLCache
from dapr.ext.workflow.workflow_state import WorkflowStatus
from pydantic import BaseModel

from dapr_agents.types.workflow import (
    NotFoundRetryPolicy,
    PubSubRouteSpec,
    WorkflowEventRouteSpec,
)
from dapr_agents.workflow.decorators.decorators import message_router
from dapr_agents.workflow.utils.registration import register_message_routes
from dapr_agents.workflow.utils.subscription import (
    EVENT_ROUTE_DEDUPE_MIN_TTL_SECONDS,
    METADATA_KEY,
    TTLDedupeBackend,
)
from tests.workflow.test_message_router import (
    _FakeTopicEventResponse,
    create_mock_dapr_client,
)

_PATCH_TARGET = "dapr_agents.workflow.utils.registration.default_dapr_client_factory"


class _JobRef(BaseModel):
    workflow_id: str


class _JobFinished(BaseModel):
    job: _JobRef
    status: str


class _FakeRpcError(grpc.RpcError):
    def __init__(self, code: Any) -> None:
        super().__init__("rpc")
        self._code = code

    def code(self) -> Any:
        return self._code

    def details(self) -> str:
        return ""


def _state(status: Any) -> MagicMock:
    state = MagicMock()
    state.runtime_status = status
    return state


def _event(data: dict, *, event_id: Optional[str] = "evt-1", topic: str = "t") -> dict:
    return {
        "id": event_id,
        "data": data,
        "datacontenttype": "application/json",
        "pubsubname": "messagepubsub",
        "source": "/test",
        "specversion": "1.0",
        "topic": topic,
        "type": "JobFinished",
        "extensions": {},
    }


def _job(workflow_id: str = "wf-1", status: str = "done") -> dict:
    return {"job": {"workflow_id": workflow_id}, "status": status}


def _spec(**overrides: Any) -> WorkflowEventRouteSpec:
    values: dict[str, Any] = dict(
        pubsub_name="messagepubsub",
        topic="t",
        event_name="job_finished",
        instance_id_from="job.workflow_id",
        message_model=_JobFinished,
    )
    values.update(overrides)
    return WorkflowEventRouteSpec(**values)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(
        "dapr_agents.workflow.utils.subscription.TopicEventResponse",
        _FakeTopicEventResponse,
    )
    mock_dapr = create_mock_dapr_client(["messagepubsub"])
    mock_wf = MagicMock()
    mock_wf.get_workflow_state.return_value = _state(WorkflowStatus.RUNNING)
    mock_wf.schedule_new_workflow.return_value = "new-instance"
    return mock_dapr, mock_wf


def _run(
    mock_dapr, mock_wf, messages: list[dict], *, routes=None, targets=None, **kwargs
):
    sub = mock_dapr.subscribe.return_value
    sub.__iter__.return_value = iter(messages)
    with patch(_PATCH_TARGET, return_value=mock_dapr):
        closers = register_message_routes(
            dapr_client=mock_dapr,
            routes=routes,
            targets=targets,
            wf_client=mock_wf,
            **kwargs,
        )
    for closer in closers:
        closer()
    return sub


def _per_topic_subs(mock_dapr, messages_by_topic: dict[str, list[dict]]) -> dict:
    subs: dict[str, MagicMock] = {}
    for topic, messages in messages_by_topic.items():
        sub = MagicMock()
        sub.__iter__.return_value = iter(messages)
        subs[topic] = sub
    mock_dapr.subscribe.side_effect = lambda **kw: subs[kw["topic"]]
    return subs


def _run_multi(mock_dapr, mock_wf, **kwargs) -> None:
    with patch(_PATCH_TARGET, return_value=mock_dapr):
        closers = register_message_routes(
            dapr_client=mock_dapr, wf_client=mock_wf, **kwargs
        )
    for closer in closers:
        closer()


def test_happy_path_dotted_path_default_data(env):
    mock_dapr, mock_wf = env
    sub = _run(mock_dapr, mock_wf, [_event(_job())], routes=[_spec()])

    sub.respond_success.assert_called_once()
    mock_wf.schedule_new_workflow.assert_not_called()
    kwargs = mock_wf.raise_workflow_event.call_args.kwargs
    assert kwargs["instance_id"] == "wf-1"
    assert kwargs["event_name"] == "job_finished"
    assert kwargs["data"]["job"] == {"workflow_id": "wf-1"}
    assert kwargs["data"]["status"] == "done"
    assert kwargs["data"][METADATA_KEY]["id"] == "evt-1"


def test_callable_resolvers_and_path_event_name(env):
    mock_dapr, mock_wf = env
    spec = _spec(
        message_model=None,
        instance_id_from=lambda m, c: m["ref"],
        data_from=lambda m, c: {"ok": m["status"] == "done", "by": c.handler_name},
        event_name_from="kind",
        name="signal",
    )
    _run(
        mock_dapr,
        mock_wf,
        [_event({"ref": "wf-7", "status": "done", "kind": "Approved"})],
        routes=[spec],
    )
    mock_wf.raise_workflow_event.assert_called_once_with(
        instance_id="wf-7", event_name="Approved", data={"ok": True, "by": "signal"}
    )


def test_all_callable_resolvers(env):
    mock_dapr, mock_wf = env
    spec = _spec(
        instance_id_from=lambda m, c: m.job.workflow_id,
        event_name_from=lambda m, c: f"job_finished:{m.status}",
        data_from=lambda m, c: m.job,
    )
    _run(mock_dapr, mock_wf, [_event(_job())], routes=[spec])
    mock_wf.raise_workflow_event.assert_called_once_with(
        instance_id="wf-1",
        event_name="job_finished:done",
        data={"workflow_id": "wf-1"},
    )


def test_terminal_state_dead_letters(env):
    mock_dapr, mock_wf = env
    mock_wf.get_workflow_state.return_value = _state(WorkflowStatus.COMPLETED)
    sub = _run(
        mock_dapr,
        mock_wf,
        [_event(_job())],
        routes=[_spec(dead_letter_topic="t_DEAD")],
    )
    assert mock_dapr.subscribe.call_args.kwargs["dead_letter_topic"] == "t_DEAD"
    sub.respond_drop.assert_called_once()
    sub.respond_retry.assert_not_called()
    mock_wf.raise_workflow_event.assert_not_called()


def test_not_found_retries_then_drops(env):
    mock_dapr, mock_wf = env
    mock_wf.get_workflow_state.return_value = None
    spec = _spec(not_found_retry=NotFoundRetryPolicy(max_attempts=2))
    sub = _run(mock_dapr, mock_wf, [_event(_job()), _event(_job())], routes=[spec])
    assert sub.respond_retry.call_count == 1
    assert sub.respond_drop.call_count == 1
    mock_wf.raise_workflow_event.assert_not_called()


def test_schema_mismatch_drops_without_state_call(env):
    mock_dapr, mock_wf = env
    sub = _run(mock_dapr, mock_wf, [_event({"nope": 1})], routes=[_spec()])
    sub.respond_drop.assert_called_once()
    mock_wf.get_workflow_state.assert_not_called()


@pytest.mark.parametrize(
    "overrides",
    [
        {"payload_filter": lambda p, c: False},
        {"model_filter": lambda m, c: False},
        {"payload_filter": lambda p, c: 1 / 0},
    ],
)
def test_filters_reject_drop(env, overrides):
    mock_dapr, mock_wf = env
    sub = _run(mock_dapr, mock_wf, [_event(_job())], routes=[_spec(**overrides)])
    sub.respond_drop.assert_called_once()
    mock_wf.get_workflow_state.assert_not_called()


def test_filters_accept(env):
    mock_dapr, mock_wf = env
    spec = _spec(
        payload_filter=lambda p, c: p["status"] == "done",
        model_filter=lambda m, c: m.job.workflow_id.startswith("wf-"),
    )
    sub = _run(mock_dapr, mock_wf, [_event(_job())], routes=[spec])
    sub.respond_success.assert_called_once()


def test_resolution_failure_drops(env):
    mock_dapr, mock_wf = env
    sub = _run(
        mock_dapr,
        mock_wf,
        [_event(_job())],
        routes=[_spec(instance_id_from="job.missing")],
    )
    sub.respond_drop.assert_called_once()
    mock_wf.get_workflow_state.assert_not_called()


def test_dedupe_on_by_default(env):
    mock_dapr, mock_wf = env
    sub = _run(mock_dapr, mock_wf, [_event(_job()), _event(_job())], routes=[_spec()])
    assert mock_wf.raise_workflow_event.call_count == 1
    assert sub.respond_success.call_count == 2


def test_dedupe_disabled(env):
    mock_dapr, mock_wf = env
    _run(
        mock_dapr,
        mock_wf,
        [_event(_job()), _event(_job())],
        routes=[_spec(dedupe=False)],
        deduper=TTLDedupeBackend(),
    )
    assert mock_wf.raise_workflow_event.call_count == 2


def test_spec_deduper_wins_over_subscriber_deduper(env):
    mock_dapr, mock_wf = env
    spec_backend, sub_backend = TTLDedupeBackend(), TTLDedupeBackend()
    _run(
        mock_dapr,
        mock_wf,
        [_event(_job())],
        routes=[_spec(deduper=spec_backend)],
        deduper=sub_backend,
    )
    assert spec_backend.seen("evt-1")
    assert not sub_backend.seen("evt-1")


def test_subscriber_deduper_used_without_spec_deduper(env):
    mock_dapr, mock_wf = env
    sub_backend = TTLDedupeBackend()
    _run(mock_dapr, mock_wf, [_event(_job())], routes=[_spec()], deduper=sub_backend)
    assert sub_backend.seen("evt-1")


def test_schedule_topic_dedupe_unchanged_next_to_event_topic(env):
    mock_dapr, mock_wf = env

    def wf_handler(ctx, msg):
        return None

    subs = _per_topic_subs(
        mock_dapr,
        {
            "a": [_event({"x": 1}, topic="a"), _event({"x": 1}, topic="a")],
            "b": [_event(_job(), topic="b"), _event(_job(), topic="b")],
        },
    )
    _run_multi(
        mock_dapr,
        mock_wf,
        routes=[
            PubSubRouteSpec(
                pubsub_name="messagepubsub", topic="a", handler_fn=wf_handler
            ),
            _spec(topic="b"),
        ],
        deduper=None,
    )
    assert mock_wf.schedule_new_workflow.call_count == 2
    assert mock_wf.raise_workflow_event.call_count == 1
    assert subs["a"].respond_success.call_count == 2
    assert subs["b"].respond_success.call_count == 2


def test_retry_does_not_mark_but_terminal_drop_does(env):
    mock_dapr, mock_wf = env
    backend = TTLDedupeBackend()
    mock_wf.raise_workflow_event.side_effect = [
        _FakeRpcError(grpc.StatusCode.UNAVAILABLE),
        None,
    ]
    sub = _run(
        mock_dapr,
        mock_wf,
        [_event(_job()), _event(_job())],
        routes=[_spec(deduper=backend)],
    )
    assert sub.respond_retry.call_count == 1
    assert sub.respond_success.call_count == 1
    assert mock_wf.raise_workflow_event.call_count == 2

    mock_wf.get_workflow_state.return_value = _state(WorkflowStatus.TERMINATED)
    mock_dapr.subscribe.return_value = MagicMock()
    sub = _run(
        mock_dapr,
        mock_wf,
        [_event(_job(), event_id="evt-2")],
        routes=[_spec(deduper=backend)],
    )
    sub.respond_drop.assert_called_once()
    assert backend.seen("evt-2")


async def test_async_delivery_still_handles_event_routes_synchronously(env):
    mock_dapr, mock_wf = env
    mock_wf.get_workflow_state.return_value = None
    sub = mock_dapr.subscribe.return_value
    sub.__iter__.return_value = iter([_event(_job())])
    with patch(_PATCH_TARGET, return_value=mock_dapr):
        closers = register_message_routes(
            dapr_client=mock_dapr,
            routes=[_spec()],
            wf_client=mock_wf,
            loop=asyncio.get_running_loop(),
            delivery_mode="async",
        )
    for closer in closers:
        closer()
    sub.respond_retry.assert_called_once()
    sub.respond_success.assert_not_called()
    mock_wf.schedule_new_workflow.assert_not_called()


def test_mixed_decorated_target_and_event_route(env):
    mock_dapr, mock_wf = env

    @message_router(pubsub="messagepubsub", topic="orders", message_model=dict)
    def handler(message: dict):
        return None

    subs = _per_topic_subs(
        mock_dapr,
        {
            "orders": [_event({"x": 1}, topic="orders", event_id="o-1")],
            "t": [_event(_job())],
        },
    )
    _run_multi(mock_dapr, mock_wf, targets=[handler], routes=[_spec()])
    assert mock_dapr.subscribe.call_count == 2
    mock_wf.schedule_new_workflow.assert_called_once()
    mock_wf.raise_workflow_event.assert_called_once()
    subs["orders"].respond_success.assert_called_once()
    subs["t"].respond_success.assert_called_once()


def test_event_route_sharing_topic_with_schedule_route_rejected(env):
    mock_dapr, mock_wf = env

    def wf_handler(ctx, msg):
        return None

    with pytest.raises(ValueError, match="one topic per event route"):
        _run(
            mock_dapr,
            mock_wf,
            [],
            routes=[
                PubSubRouteSpec(
                    pubsub_name="messagepubsub", topic="t", handler_fn=wf_handler
                ),
                _spec(),
            ],
        )


class _TimedJob(BaseModel):
    workflow_id: str
    at: datetime


def test_default_data_serializes_pydantic_datetime(env):
    mock_dapr, mock_wf = env
    spec = _spec(instance_id_from="workflow_id", message_model=_TimedJob)
    payload = {"workflow_id": "wf-1", "at": "2026-01-02T03:04:05Z"}
    sub = _run(mock_dapr, mock_wf, [_event(payload)], routes=[spec])

    sub.respond_success.assert_called_once()
    data = mock_wf.raise_workflow_event.call_args.kwargs["data"]
    assert data["at"].startswith("2026-01-02T03:04:05")
    assert data[METADATA_KEY]["id"] == "evt-1"


def test_default_event_deduper_outlives_not_found_window(env, monkeypatch):
    mock_dapr, mock_wf = env
    ttls: list[float] = []
    real_backend = TTLDedupeBackend

    def _capture(*args: Any, **kwargs: Any) -> TTLDedupeBackend:
        ttls.append(kwargs.get("ttl", 60.0))
        return real_backend(*args, **kwargs)

    monkeypatch.setattr(
        "dapr_agents.workflow.utils.subscription.TTLDedupeBackend", _capture
    )
    spec = _spec(not_found_retry=NotFoundRetryPolicy(window_seconds=3600))
    _run(mock_dapr, mock_wf, [_event(_job())], routes=[spec])
    assert ttls == [max(EVENT_ROUTE_DEDUPE_MIN_TTL_SECONDS, 3600)]


def test_redelivery_after_dedupe_ttl_is_raised_again(env):
    mock_dapr, mock_wf = env
    now = [0.0]
    backend = TTLDedupeBackend(ttl=60)
    backend._cache = TTLCache(maxsize=16, ttl=60, timer=lambda: now[0])
    spec = _spec(deduper=backend)

    _run(mock_dapr, mock_wf, [_event(_job())], routes=[spec])
    _run(mock_dapr, mock_wf, [_event(_job())], routes=[spec])
    assert mock_wf.raise_workflow_event.call_count == 1

    now[0] = 61.0  # past the TTL: the id is forgotten, the redelivery goes through
    _run(mock_dapr, mock_wf, [_event(_job())], routes=[spec])
    assert mock_wf.raise_workflow_event.call_count == 2
