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

"""
Approval ids are scoped to the turn and the call's position, and fail closed.

These tests run the workflow through the pinned SDK's real orchestration
executor (see workflow_harness.py), so event delivery follows the SDK: an
event goes to the oldest waiter for its name, including a waiter left behind
by a request that already timed out, and is buffered only when nobody waits.
Every step also replays the full history, so a replay mismatch fails the test.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest

# The test suite replaces the `dapr` package with a mock (tests/conftest.py),
# but the real SDK submodules loaded before that stay importable by path.
from dapr.ext.workflow.dapr_workflow_context import (
    DaprWorkflowContext,
    when_any as real_when_any,
)

from dapr_agents.agents import durable
from dapr_agents.agents.durable import _legacy_approval_request_id
from dapr_agents.hooks import Hooks, RequireApproval
from dapr_agents.llm import OpenAIChatClient
from tests.agents.durableagent.test_hitl_workflow import (  # noqa: F401 (autouse)
    _make_agent,
    _tool_call,
    setup_env,
)
from tests.agents.durableagent.workflow_harness import ReplayHarness

INSTANCE = "wf-replay"
FINAL = {"role": "assistant", "content": "done"}


@pytest.fixture(autouse=True)
def use_real_when_any(monkeypatch):
    # durable.py composes tasks with wf.when_any; use the SDK's real one.
    monkeypatch.setattr(durable.wf, "when_any", real_when_any)


@pytest.fixture
def llm():
    m = Mock(spec=OpenAIChatClient)
    m.prompt_template = None
    m.provider = "MockProvider"
    m.api = "MockAPI"
    m.model = "gpt-4o-mock"
    return m


def _approval_agent(llm, timeout_seconds: Optional[int] = 30):
    def hook(ctx):
        return RequireApproval(timeout_seconds=timeout_seconds)

    agent = _make_agent(llm, hooks=Hooks(before_tool_call=[hook]))
    agent._retry_policy = None  # the suite's RetryPolicy is a mock
    return agent


def _reply(request_id: str, approved: bool = True, body_id: str = "") -> Dict:
    return {"approval_request_id": body_id or request_id, "approved": approved}


def _name(request_id: str) -> str:
    return f"approval_response_{request_id}"


class _Activities:
    """Answers activities: scripted LLM turns, recorded publishes and tool runs."""

    def __init__(self, llm_turns: List[Dict[str, Any]], legacy: bool = False):
        self.llm_turns = list(llm_turns)
        self.legacy = legacy  # publish ran on older code: it returns None
        self.published: List[Dict[str, Any]] = []
        self.tools_run: List[str] = []

    def __call__(self, name: str, payload: Any) -> Any:
        if name.endswith("call_llm"):
            return self.llm_turns.pop(0)
        if name.endswith("publish_approval_request"):
            event = payload["event"]
            self.published.append(event)
            return None if self.legacy else event["approval_request_id"]
        if name.endswith("run_tool"):
            tc = payload["tool_call"]
            self.tools_run.append(tc["function"]["name"])
            return {"role": "tool", "content": "ran", "tool_call_id": tc["id"]}
        return None

    @property
    def ids(self) -> List[str]:
        return [e["approval_request_id"] for e in self.published]


def _agent_harness(agent, acts: _Activities, **kwargs) -> ReplayHarness:
    def orchestrator(ctx, message):
        return (yield from agent.agent_workflow(DaprWorkflowContext(ctx), message))

    harness = ReplayHarness(orchestrator, acts, instance_id=INSTANCE, **kwargs)
    harness.start({"task": "go"})
    return harness


def _approval_harness(agent, acts: _Activities, timeout: Optional[int] = 30):
    tc = _tool_call(call_id="call_0")

    def orchestrator(ctx, _):
        return (
            yield from agent._request_approval(
                DaprWorkflowContext(ctx),
                ctx.instance_id,
                tc,
                RequireApproval(timeout_seconds=timeout),
                turn=1,
                call_index=0,
            )
        )

    harness = ReplayHarness(orchestrator, acts, instance_id=INSTANCE)
    harness.start()
    return harness


LEGACY_ID = _legacy_approval_request_id(INSTANCE, "call_0")


# Collisions that the old tool_call_id-only id allowed                         #


def test_late_turn1_reply_cannot_decide_turn2_call(llm):
    """
    Turn 1 times out; its waiter stays registered for the event name. Turn 2
    reuses call_0. The human denies turn 2, then a late turn-1 approval lands.
    With a shared id the deny is swallowed by the dead turn-1 waiter and the
    late approval approves turn 2.
    """
    agent = _approval_agent(llm)
    turn = {"tool_calls": [_tool_call(name="DeleteRepo", call_id="call_0")]}
    acts = _Activities([turn, turn, FINAL])
    h = _agent_harness(agent, acts)
    h.fire_timers()  # turn 1 times out; turn 2 publishes and waits
    turn1_id, turn2_id = acts.ids

    h.raise_event(_name(turn2_id), _reply(turn2_id, approved=False))
    h.raise_event(_name(turn1_id), _reply(turn1_id, approved=True))

    assert acts.tools_run == [], "a late turn-1 approval approved the turn-2 call"
    assert h.done
    assert turn1_id != turn2_id


def test_reply_after_earlier_timeout_reaches_its_own_call(llm):
    """
    Two calls in one turn share an empty id. The first times out. The reply to
    the second must reach the second, not the first call's dead waiter.
    """
    agent = _approval_agent(llm)
    calls = [_tool_call(name="A", call_id=""), _tool_call(name="B", call_id="")]
    acts = _Activities([{"tool_calls": calls}, FINAL])
    h = _agent_harness(agent, acts)
    h.fire_timers()  # A times out; B publishes and waits

    h.raise_event(_name(acts.ids[1]), _reply(acts.ids[1]))

    assert h.done
    assert acts.tools_run == ["B"]
    assert acts.ids[0] != acts.ids[1]


def test_calls_sharing_an_id_in_one_turn_get_their_own_decisions(llm):
    agent = _approval_agent(llm)
    calls = [_tool_call(name="A", call_id=""), _tool_call(name="B", call_id="")]
    acts = _Activities([{"tool_calls": calls}, FINAL])
    h = _agent_harness(agent, acts)
    h.raise_event(_name(acts.ids[0]), _reply(acts.ids[0], approved=False))
    h.raise_event(_name(acts.ids[1]), _reply(acts.ids[1], approved=True))

    assert h.done
    assert acts.tools_run == ["B"]


def test_ids_stable_across_replay_and_scoped_to_orchestration_time(llm):
    def ids(start: datetime) -> List[str]:
        agent = _approval_agent(llm)
        turn = {"tool_calls": [_tool_call(call_id="call_0")] * 2}
        acts = _Activities([turn, turn, FINAL])
        h = _agent_harness(agent, acts, start_time=start)
        while not h.done:  # approve each request as it is published
            h.raise_event(_name(acts.ids[-1]), _reply(acts.ids[-1]))
        assert len(acts.ids) == 4
        h.resume()  # full replay from history reproduces the same actions
        return acts.ids

    first = ids(datetime(2024, 1, 1))
    assert len(set(first)) == 4
    assert ids(datetime(2024, 1, 1)) == first
    assert set(ids(datetime(2024, 1, 2))).isdisjoint(first)


# Fail closed                                                                  #


def test_mismatched_reply_is_ignored_and_logged(llm, caplog):
    agent = _approval_agent(llm)
    acts = _Activities([])
    h = _approval_harness(agent, acts)
    rid = acts.ids[0]
    with caplog.at_level(logging.WARNING, logger="dapr_agents.agents.durable"):
        h.raise_event(_name(rid), _reply(rid, body_id="someone-else"))

    assert not h.done
    assert any(
        r.levelno == logging.WARNING and "someone-else" in r.getMessage()
        for r in caplog.records
    )
    h.fire_timers()
    assert h.done and h.output is False
    assert h.timers_created == 1


def test_malformed_replies_are_ignored_and_keep_waiting(llm, caplog):
    agent = _approval_agent(llm)
    acts = _Activities([])
    h = _approval_harness(agent, acts)
    rid = acts.ids[0]
    with caplog.at_level(logging.WARNING, logger="dapr_agents.agents.durable"):
        h.raise_event(_name(rid), None)
        h.raise_event(_name(rid), {"not": "a response"})
    assert not h.done
    assert any("unparseable" in r.getMessage() for r in caplog.records)

    h.raise_event(_name(rid), _reply(rid))
    assert h.done and h.output is True
    assert h.timers_created == 1


def test_ignored_replies_share_one_timer_until_timeout(llm):
    agent = _approval_agent(llm)
    acts = _Activities([])
    h = _approval_harness(agent, acts)
    rid = acts.ids[0]
    h.raise_event(_name(rid), _reply(rid, body_id="stale-1"))
    h.raise_event(_name(rid), _reply(rid, body_id="stale-2"))

    raced: List[List[Any]] = []

    def record(tasks):
        raced.append(list(tasks))
        return real_when_any(tasks)

    with patch.object(durable.wf, "when_any", side_effect=record):
        h.fire_timers()  # this step replays every wait, then fires the timer

    assert h.done and h.output is False
    assert h.timers_created == 1
    assert len(raced) == 3
    assert all(tasks[-1] is raced[0][-1] for tasks in raced)


# Runs whose publish was recorded by older code                                #


def _old_code_orchestrator(agent, timeout: Optional[int]):
    """The request/wait sequence of the previous release, for recording history."""
    publish = agent._activity_name(agent.publish_approval_request)

    def orchestrator(ctx, _):
        dctx = DaprWorkflowContext(ctx)
        yield dctx.call_activity(
            publish, input={"event": {"approval_request_id": LEGACY_ID}}
        )
        event_task = dctx.wait_for_external_event(_name(LEGACY_ID))
        if timeout is None:
            yield event_task
        else:
            timer = dctx.create_timer(timedelta(seconds=timeout))
            if (yield real_when_any([event_task, timer])) is timer:
                return False
        return event_task.get_result()["approved"]

    return orchestrator


@pytest.mark.parametrize("timeout", [30, None])
def test_run_waiting_at_upgrade_is_released_by_its_old_id(llm, timeout):
    agent = _approval_agent(llm, timeout_seconds=timeout)
    old = ReplayHarness(
        _old_code_orchestrator(agent, timeout),
        _Activities([], legacy=True),
        instance_id=INSTANCE,
    )
    old.start()
    assert not old.done

    # Upgrade: the new code replays the old history, then the approver replies
    # with the id it was shown, the legacy one.
    acts = _Activities([], legacy=True)
    h = _approval_harness_on(agent, acts, old.history, timeout)
    h.raise_event(_name(LEGACY_ID), _reply(LEGACY_ID))

    assert h.done and h.output is True


def _approval_harness_on(agent, acts, history, timeout):
    tc = _tool_call(call_id="call_0")

    def orchestrator(ctx, _):
        return (
            yield from agent._request_approval(
                DaprWorkflowContext(ctx),
                ctx.instance_id,
                tc,
                RequireApproval(timeout_seconds=timeout),
                turn=1,
                call_index=0,
            )
        )

    h = ReplayHarness(orchestrator, acts, instance_id=INSTANCE, history=history)
    h.resume()
    return h


@pytest.mark.parametrize("on", ["new", "legacy"])
def test_publish_on_old_replica_is_released_on_either_name(llm, on):
    """Rolling upgrade: the publish ran on an old replica and returned None."""
    agent = _approval_agent(llm)
    acts = _Activities([], legacy=True)
    h = _approval_harness(agent, acts)
    rid = acts.ids[0] if on == "new" else LEGACY_ID

    h.raise_event(_name(rid), _reply(rid))

    assert h.done and h.output is True


@pytest.mark.parametrize("on", ["new", "legacy"])
def test_publish_on_old_replica_ignores_wrong_body_id(llm, on):
    agent = _approval_agent(llm)
    acts = _Activities([], legacy=True)
    h = _approval_harness(agent, acts)
    new_id = acts.ids[0]
    rid, other = (new_id, LEGACY_ID) if on == "new" else (LEGACY_ID, new_id)

    h.raise_event(_name(rid), _reply(rid, body_id=other))
    assert not h.done
    h.raise_event(_name(rid), _reply(rid, body_id="someone-else"))
    assert not h.done

    h.raise_event(_name(other), _reply(other, approved=False))
    assert h.done and h.output is False
    assert h.timers_created == 1


def test_new_request_does_not_listen_on_the_legacy_name(llm):
    agent = _approval_agent(llm)
    acts = _Activities([])
    h = _approval_harness(agent, acts)

    h.raise_event(_name(LEGACY_ID), _reply(LEGACY_ID))

    assert not h.done
    h.fire_timers()
    assert h.done and h.output is False


def test_publish_activity_rejects_an_empty_id(llm):
    agent = _approval_agent(llm)
    with pytest.raises(ValueError):
        agent.publish_approval_request(Mock(), {"event": {"approval_request_id": ""}})
