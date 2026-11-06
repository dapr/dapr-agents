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

"""Hooks that do not decide (timeout, saturated pool): bounded RETRY, then give up."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from dapr_agents.types.workflow import NotFoundRetryPolicy
from dapr_agents.workflow.utils.call_deadline import DeadlineCallerBusyError
from dapr_agents.workflow.utils.event_route_calls import HookVerdict, run_strict_hook
from dapr_agents.workflow.utils.log_throttle import WarningThrottle
from tests.workflow._event_route_helpers import (
    make_ctx,
    make_dispatcher,
    make_target,
    make_wf,
    run_dispatch,
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


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _BusyCaller:
    """A hook pool that is always saturated."""

    def call(self, fn: Any, timeout: float, *args: Any) -> Any:
        raise DeadlineCallerBusyError("full")

    def close(self) -> None:
        return None


def _undecided_dispatcher(clock: _Clock, wf: Any = None):
    return make_dispatcher(wf or make_wf(), hook_caller=_BusyCaller(), clock=clock)


def _target(max_attempts: int = 2):
    return make_target(
        authorize=lambda *_: True,
        not_found_retry=NotFoundRetryPolicy(max_attempts=max_attempts),
    )


# ---- run_strict_hook verdicts ------------------------------------------------


def test_verdicts():
    busy = _BusyCaller()
    assert run_strict_hook(busy, lambda: True, 1.0, kind="k", route_name="r") is (
        HookVerdict.UNDECIDED
    )
    dispatcher = make_dispatcher(make_wf())
    caller = dispatcher._hook_caller
    assert run_strict_hook(caller, lambda: True, 5.0, kind="k", route_name="r") is (
        HookVerdict.ALLOW
    )
    for value in (False, 1, "True"):
        assert run_strict_hook(
            caller, lambda v=value: v, 5.0, kind="k", route_name="r"
        ) is (HookVerdict.DENY)
    dispatcher.close()


# ---- authorize UNDECIDED -----------------------------------------------------


def test_undecided_authorize_retries_then_dead_letters(caplog):
    clock = _Clock()
    wf = make_wf()
    dispatcher = _undecided_dispatcher(clock, wf)
    target = _target()
    with caplog.at_level(logging.WARNING):
        assert run_dispatch(dispatcher, target=target, dlq="t.dlq") == "retry"
        assert run_dispatch(dispatcher, target=target, dlq="t.dlq") == "drop"
    wf.get_workflow_state.assert_not_called()
    wf.raise_workflow_event.assert_not_called()
    assert "authorize did not decide after 2 attempts" in caplog.text
    assert "daprd dead-letters it to 't.dlq'" in caplog.text
    # The budget was cleared: a later delivery starts counting again.
    assert run_dispatch(dispatcher, target=target, dlq="t.dlq") == "retry"
    dispatcher.close()


def test_undecided_drop_without_dlq_is_rate_limited(caplog):
    clock = _Clock()
    dispatcher = _undecided_dispatcher(clock)
    target = _target(max_attempts=1)
    with caplog.at_level(logging.WARNING):
        for event_id in ("a", "b", "c"):
            ctx = make_ctx(event_id)
            assert run_dispatch(dispatcher, target=target, ctx=ctx) == "drop"
        clock.now = 61.0
        assert run_dispatch(dispatcher, target=target, ctx=make_ctx("d")) == "drop"
    drops = [r for r in caplog.records if "did not decide" in r.getMessage()]
    assert len(drops) == 2  # the first, then one after the interval
    assert "2 similar warnings suppressed" in drops[1].getMessage()
    assert "no dead_letter_topic configured" in drops[0].getMessage()
    dispatcher.close()


def test_undecided_drop_logs_only_a_key_prefix(caplog):
    clock = _Clock()
    dispatcher = _undecided_dispatcher(clock)
    ctx = make_ctx(None)  # id-less: the key would otherwise be a full digest
    with caplog.at_level(logging.WARNING):
        run_dispatch(dispatcher, target=_target(max_attempts=1), ctx=ctx)
    message = next(
        r.getMessage() for r in caplog.records if "did not decide" in r.getMessage()
    )
    key_text = message.split("message key ")[1].split(")")[0]
    assert key_text.endswith("...") and len(key_text) == 27
    dispatcher.close()


def test_decision_after_undecided_resets_the_budget():
    clock = _Clock()
    wf = make_wf()
    dispatcher = _undecided_dispatcher(clock, wf)
    target = _target()
    real = make_dispatcher(wf)
    assert run_dispatch(dispatcher, target=target) == "retry"
    dispatcher._hook_caller = real._hook_caller  # the hook decides again
    assert run_dispatch(dispatcher, target=target) == "success"
    dispatcher._hook_caller = _BusyCaller()
    assert run_dispatch(dispatcher, target=target) == "retry"  # counting restarted
    real.close()


# ---- filter UNDECIDED through the subscriber -----------------------------------


@pytest.mark.parametrize("kind", ["payload_filter", "model_filter"])
def test_undecided_filter_retries_then_dead_letters(env, monkeypatch, kind):
    mock_dapr, mock_wf = env
    monkeypatch.setattr(
        "dapr_agents.workflow.utils.event_routes.DeadlineCaller.call",
        lambda self, fn, timeout, *a: (_ for _ in ()).throw(
            DeadlineCallerBusyError("full")
        ),
    )
    spec = _spec(
        **{kind: lambda m, c: True},
        dead_letter_topic="t.dlq",
        not_found_retry=NotFoundRetryPolicy(max_attempts=2),
    )
    msg = _event(_job())
    sub = _run(mock_dapr, mock_wf, [msg, msg], routes=[spec])
    sub.respond_retry.assert_called_once()
    sub.respond_drop.assert_called_once()
    mock_wf.raise_workflow_event.assert_not_called()


# ---- WarningThrottle -----------------------------------------------------------


def test_warning_throttle_per_key(caplog):
    clock = _Clock()
    throttle = WarningThrottle(interval_seconds=10.0, clock=clock)
    log = logging.getLogger("throttle-test")
    with caplog.at_level(logging.WARNING, logger="throttle-test"):
        assert throttle.warn(log, "a", "first %s", 1)
        assert not throttle.warn(log, "a", "second %s", 2)
        assert throttle.warn(log, "b", "other key")
        clock.now = 10.0
        assert throttle.warn(log, "a", "third %s", 3)
    assert [r.getMessage() for r in caplog.records] == [
        "first 1",
        "other key",
        "third 3 (1 similar warnings suppressed in the last 10s)",
    ]
