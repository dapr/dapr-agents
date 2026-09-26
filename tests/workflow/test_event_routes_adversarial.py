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

"""Regression tests for adversarial workflow event route cases."""

from __future__ import annotations

import functools
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import grpc
import pytest
from cachetools import TTLCache
from dapr.ext.workflow.workflow_state import WorkflowStatus

from dapr_agents.workflow.utils import event_routes as er
from dapr_agents.workflow.utils.event_routes import (
    EventRouteTarget,
    WorkflowEventDispatcher,
    is_reserved_event_name,
)
from dapr_agents.types.workflow import NotFoundRetryPolicy
from dapr_agents.workflow.utils.registration import register_message_routes
from tests.workflow._event_route_helpers import (
    PATCH_TARGET,
    FakeRpcError,
    workflow_state,
)
from tests.workflow.test_workflow_event_routes_e2e_inprocess import (  # noqa: F401
    _event,
    _job,
    _run,
    _spec,
)
from tests.workflow.test_workflow_event_routes_e2e_inprocess import env as _env


@pytest.fixture
def env(monkeypatch):
    return _env.__wrapped__(monkeypatch)


# ---- 1. reserved-name bypass via Unicode case folding -------------------------


@pytest.mark.parametrize(
    "name",
    [
        "approval_reſponse_abc",  # LATIN SMALL LETTER LONG S
        "user_input_reſponse:req-1",
    ],
)
def test_reserved_check_matches_durabletask_casefold(name):
    # durabletask's worker matches event names with str.casefold(), so this
    # name reaches the SDK's own approval / ask_user wait.
    assert name.casefold().startswith(("approval_response_", "user_input_response:"))
    assert is_reserved_event_name(name), (
        "reserved-name check uses lower(), not casefold()"
    )


def test_reserved_bypass_end_to_end(env):
    mock_dapr, mock_wf = env
    spec = _spec(message_model=None, event_name_from="kind")
    sub = _run(
        mock_dapr,
        mock_wf,
        [_event({**_job(), "kind": "approval_reſponse_abc"})],
        routes=[spec],
    )
    mock_wf.raise_workflow_event.assert_not_called()
    sub.respond_drop.assert_called_once()


# ---- 2. timeout -> RETRY -> redelivery raises the event a second time ---------


def test_timed_out_raise_is_not_raised_again_on_redelivery(env):
    mock_dapr, mock_wf = env
    raised: list[str] = []
    done = threading.Event()

    def slow_raise(**kwargs: Any) -> None:
        time.sleep(0.3)
        raised.append(kwargs["event_name"])
        done.set()

    mock_wf.raise_workflow_event.side_effect = slow_raise
    msg = _event(_job(), event_id="evt-timeout")

    redelivered = threading.Event()

    def delayed_redelivery():
        # Broker redelivers the same CloudEvent id after the RETRY.
        yield msg
        done.wait(2)
        yield msg
        redelivered.set()

    sub = mock_dapr.subscribe.return_value
    sub.__iter__.side_effect = lambda: delayed_redelivery()
    with patch(PATCH_TARGET, return_value=mock_dapr):
        closers = register_message_routes(
            dapr_client=mock_dapr,
            routes=[_spec(call_timeout_seconds=0.05)],
            wf_client=mock_wf,
        )
    assert redelivered.wait(3)
    time.sleep(0.5)  # let the redelivered raise finish
    for closer in closers:
        closer()
    assert sub.respond_retry.call_count >= 1
    # The first (timed-out) call did raise the event in the background, so the
    # redelivery must not raise it again: a second buffered copy would satisfy
    # the NEXT wait with the same name.
    assert len(raised) == 1, f"event raised {len(raised)} times: {raised}"


# ---- 3. permanent gRPC errors are retried forever ------------------------------


@pytest.mark.parametrize(
    "code",
    [
        grpc.StatusCode.INVALID_ARGUMENT,
        grpc.StatusCode.PERMISSION_DENIED,
        grpc.StatusCode.UNIMPLEMENTED,
    ],
)
def test_permanent_sidecar_error_is_bounded(env, code):
    mock_dapr, mock_wf = env
    mock_wf.raise_workflow_event.side_effect = FakeRpcError(code, "rejected")
    msgs = [_event(_job(), event_id="evt-perm")] * 50
    sub = _run(mock_dapr, mock_wf, msgs, routes=[_spec()])
    # 50 deliveries of the same message; a permanent error must end in DROP.
    assert sub.respond_drop.call_count >= 1, (
        f"{code.name}: {sub.respond_retry.call_count} RETRYs, never dropped"
    )


# ---- 4. not-found budget resets when redeliveries are slower than tracker TTL --


def test_not_found_budget_survives_slow_redelivery(monkeypatch):
    now = [0.0]
    clock = lambda: now[0]  # noqa: E731
    # The tracker no longer uses a TTL cache; the patch (a no-op now) keeps
    # this test reproducing the old TTL-expiry bug if one comes back.
    monkeypatch.setattr(
        er, "TTLCache", functools.partial(TTLCache, timer=clock), raising=False
    )
    wf = MagicMock()
    wf.get_workflow_state.return_value = None  # instance never appears
    dispatcher = WorkflowEventDispatcher(
        wf_client=wf, default_serializer=lambda m: m, clock=clock
    )
    target = EventRouteTarget(
        event_name="e",
        instance_id_from="id",
        event_name_from=None,
        data_from=None,
        dedupe=True,
        deduper=None,
        not_found_retry=NotFoundRetryPolicy(max_attempts=3, window_seconds=10),
    )
    ctx = MagicMock()
    ctx.event.id = "evt-slow"
    statuses = []
    for _ in range(20):
        statuses.append(
            dispatcher.dispatch(
                target=target,
                route_name="r",
                pubsub="p",
                topic="t",
                dead_letter_topic=None,
                message={"id": "never"},
                msg_ctx=ctx,
            )
        )
        now[0] += 90.0  # broker backoff longer than the tracker TTL (60s)
    dispatcher.close()
    assert "drop" in statuses, f"20 deliveries over 30 min, all {set(statuses)}"
