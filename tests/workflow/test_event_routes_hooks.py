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

"""Workflow event routes: authorize hook, strict filters, reserved-name folding."""

from __future__ import annotations

import logging
import math
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from dapr_agents.workflow import WorkflowEventTarget
from dapr_agents.workflow.utils.event_routes import is_reserved_event_name
from dapr_agents.workflow.utils.registration import _collect_message_bindings
from tests.workflow.test_workflow_event_routes_e2e_inprocess import (  # noqa: F401
    _event,
    _job,
    _run,
    _spec,
)
from tests.workflow.test_workflow_event_routes_e2e_inprocess import env as _env

_SECRET = "top-secret-payload"


@pytest.fixture
def env(monkeypatch):
    return _env.__wrapped__(monkeypatch)


def _slow(result: Any = True):
    release = threading.Event()

    def hook(*_: Any) -> Any:
        release.wait(5)
        return result

    return hook, release


# ---- reserved names ------------------------------------------------------------


def test_static_reserved_event_name_with_long_s_rejected_at_registration():
    with pytest.raises(ValueError, match="reserves"):
        _collect_message_bindings(
            targets=None, routes=[_spec(event_name="approval_reſponse_x")]
        )


def test_reserved_check_is_case_insensitive():
    assert is_reserved_event_name("APPROVAL_RESPONSE_x")
    assert not is_reserved_event_name("job_finished")


# ---- authorize -----------------------------------------------------------------


def test_authorize_true_raises_event_with_message_cloudevent_and_target(env):
    mock_dapr, mock_wf = env
    seen: list[Any] = []

    def authorize(msg: Any, ctx: Any, target: WorkflowEventTarget) -> bool:
        seen.append((msg.job.workflow_id, ctx.event.id, ctx.event.source, target))
        return True

    sub = _run(
        mock_dapr, mock_wf, [_event(_job())], routes=[_spec(authorize=authorize)]
    )
    sub.respond_success.assert_called_once()
    mock_wf.raise_workflow_event.assert_called_once()
    assert seen == [
        (
            "wf-1",
            "evt-1",
            "/test",
            WorkflowEventTarget(instance_id="wf-1", event_name="job_finished"),
        )
    ]


def test_authorize_sees_the_resolved_event_name(env):
    mock_dapr, mock_wf = env
    targets: list[WorkflowEventTarget] = []

    def authorize(msg: Any, ctx: Any, target: WorkflowEventTarget) -> bool:
        targets.append(target)
        return target.instance_id.startswith("wf-")

    spec = _spec(
        event_name_from=lambda m, c: f"job_{m.status}",
        authorize=authorize,
    )
    msgs = [_event(_job()), _event(_job(workflow_id="other"), event_id="evt-2")]
    sub = _run(mock_dapr, mock_wf, msgs, routes=[spec])
    assert targets == [
        WorkflowEventTarget(instance_id="wf-1", event_name="job_done"),
        WorkflowEventTarget(instance_id="other", event_name="job_done"),
    ]
    sub.respond_success.assert_called_once()
    sub.respond_drop.assert_called_once()
    mock_wf.raise_workflow_event.assert_called_once()


@pytest.mark.parametrize(
    "result", [False, "False", "True", 1, None, MagicMock()], ids=repr
)
def test_authorize_denies_anything_but_true(env, caplog, result):
    mock_dapr, mock_wf = env
    with caplog.at_level(logging.WARNING):
        sub = _run(
            mock_dapr,
            mock_wf,
            [_event(_job(status=_SECRET))],
            routes=[_spec(authorize=lambda m, c, t: result, name="r1")],
        )
    sub.respond_drop.assert_called_once()
    mock_wf.get_workflow_state.assert_not_called()
    mock_wf.raise_workflow_event.assert_not_called()
    assert "authorize denied" in caplog.text
    assert "no dead_letter_topic configured" in caplog.text
    assert "'r1'" in caplog.text and "'wf-1'" in caplog.text
    assert "'evt-1'" in caplog.text
    assert _SECRET not in caplog.text


def test_authorize_exception_denies(env, caplog):
    mock_dapr, mock_wf = env

    def authorize(msg: Any, ctx: Any, target: Any) -> bool:
        raise RuntimeError(_SECRET)

    with caplog.at_level(logging.WARNING):
        sub = _run(
            mock_dapr, mock_wf, [_event(_job())], routes=[_spec(authorize=authorize)]
        )
    sub.respond_drop.assert_called_once()
    mock_wf.raise_workflow_event.assert_not_called()
    assert "RuntimeError" in caplog.text and _SECRET not in caplog.text


def test_authorize_timeout_retries(env, caplog):
    mock_dapr, mock_wf = env
    hook, release = _slow(True)
    try:
        with caplog.at_level(logging.WARNING):
            sub = _run(
                mock_dapr,
                mock_wf,
                [_event(_job())],
                routes=[_spec(authorize=hook, hook_timeout_seconds=0.05)],
            )
    finally:
        release.set()
    # UNDECIDED: retried within the budget, never raised, never dropped yet.
    sub.respond_retry.assert_called_once()
    sub.respond_drop.assert_not_called()
    mock_wf.raise_workflow_event.assert_not_called()
    assert "timed out" in caplog.text and "retrying" in caplog.text


def test_authorize_denial_names_the_dead_letter_topic(env, caplog):
    mock_dapr, mock_wf = env
    with caplog.at_level(logging.WARNING):
        sub = _run(
            mock_dapr,
            mock_wf,
            [_event(_job())],
            routes=[_spec(authorize=lambda m, c, t: False, dead_letter_topic="t.dlq")],
        )
    sub.respond_drop.assert_called_once()
    assert "authorize denied" in caplog.text
    assert "daprd dead-letters it to 't.dlq'" in caplog.text


def test_authorize_must_be_sync():
    async def authorize(msg: Any, ctx: Any, target: Any) -> bool:
        return True

    with pytest.raises(TypeError, match="authorize"):
        _collect_message_bindings(targets=None, routes=[_spec(authorize=authorize)])


def test_authorize_with_two_parameters_rejected_at_registration():
    with pytest.raises(TypeError, match="three positional arguments"):
        _collect_message_bindings(
            targets=None, routes=[_spec(authorize=lambda msg, ctx: True)]
        )


@pytest.mark.parametrize(
    "authorize",
    [lambda *args: True, lambda msg, ctx, target, extra=None: True, max],
    ids=["varargs", "extra-default", "builtin-without-signature"],
)
def test_authorize_signatures_accepted(authorize):
    bindings = _collect_message_bindings(
        targets=None, routes=[_spec(authorize=authorize)]
    )
    assert bindings[0].event_target.authorize is authorize


@pytest.mark.parametrize("value", [0, -1.0, math.inf, math.nan, True, "5"])
def test_hook_timeout_seconds_validated(value):
    with pytest.raises(ValueError, match="hook_timeout_seconds"):
        _spec(hook_timeout_seconds=value)


def test_hook_timeout_seconds_default():
    assert _spec().hook_timeout_seconds == 5.0
    assert _spec().authorize is None


# ---- strict filters with a deadline --------------------------------------------


@pytest.mark.parametrize("kind", ["payload_filter", "model_filter"])
@pytest.mark.parametrize("result", [1, "yes", MagicMock()], ids=repr)
def test_event_route_filter_needs_real_true(env, kind, result):
    mock_dapr, mock_wf = env
    sub = _run(
        mock_dapr,
        mock_wf,
        [_event(_job())],
        routes=[_spec(**{kind: lambda m, c: result})],
    )
    sub.respond_drop.assert_called_once()
    mock_wf.raise_workflow_event.assert_not_called()


@pytest.mark.parametrize("kind", ["payload_filter", "model_filter"])
def test_event_route_filter_true_accepts(env, kind):
    mock_dapr, mock_wf = env
    sub = _run(
        mock_dapr,
        mock_wf,
        [_event(_job())],
        routes=[_spec(**{kind: lambda m, c: True})],
    )
    sub.respond_success.assert_called_once()


@pytest.mark.parametrize("kind", ["payload_filter", "model_filter"])
def test_event_route_filter_timeout_retries(env, caplog, kind):
    mock_dapr, mock_wf = env
    hook, release = _slow(True)
    try:
        with caplog.at_level(logging.WARNING):
            sub = _run(
                mock_dapr,
                mock_wf,
                [_event(_job())],
                routes=[_spec(**{kind: hook}, hook_timeout_seconds=0.05)],
            )
    finally:
        release.set()
    sub.respond_retry.assert_called_once()
    sub.respond_drop.assert_not_called()
    mock_wf.raise_workflow_event.assert_not_called()
    assert f"{kind} timed out" in caplog.text


def test_event_route_filter_exception_rejects(env):
    mock_dapr, mock_wf = env

    def boom(msg: Any, ctx: Any) -> bool:
        raise ValueError("x")

    sub = _run(
        mock_dapr, mock_wf, [_event(_job())], routes=[_spec(payload_filter=boom)]
    )
    sub.respond_drop.assert_called_once()
    mock_wf.raise_workflow_event.assert_not_called()
