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

LLMs reuse tool_call ids (``call_0``) across turns and may leave them empty or
repeat them in one turn. These tests drive the workflow against a small fake
event bus that buffers raised events by name, the way Dapr does, so a response
that arrives after its request timed out stays buffered for the next waiter.
"""

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest
from dapr.ext.workflow import DaprWorkflowContext

from dapr_agents.agents.durable import _legacy_approval_request_id
from dapr_agents.hooks import Hooks, RequireApproval
from dapr_agents.llm import OpenAIChatClient
from tests.agents.durableagent.test_hitl_workflow import (  # noqa: F401 (autouse)
    _make_agent,
    _tool_call,
    patch_dapr_check,
    setup_env,
)

DISPATCH_TIME = "2024-01-01T00:00:00.000000"


@pytest.fixture
def llm():
    m = Mock(spec=OpenAIChatClient)
    m.generate = AsyncMock()
    m.prompt_template = None
    m.provider = "MockProvider"
    m.api = "MockAPI"
    m.model = "gpt-4o-mock"
    return m


class _EventTask:
    def __init__(self, name: str) -> None:
        self.name = name
        self.result: Optional[Dict[str, Any]] = None

    def get_result(self) -> Optional[Dict[str, Any]]:
        return self.result


class _TimerTask:
    pass


class _Activity:
    def __init__(self, name: str, input: Any) -> None:
        self.name = name
        self.input = input


class _FakeRuntime:
    """Workflow context plus a buffered external-event bus.

    ``on_publish(event)`` runs when an approval request is published and
    ``on_timeout(request_id)`` when its timer wins; both may buffer responses.
    """

    def __init__(
        self,
        llm_responses: List[Dict[str, Any]],
        on_publish: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_timeout: Optional[Callable[[str], None]] = None,
        dispatch_time: str = DISPATCH_TIME,
        legacy_publishes: int = 0,
    ) -> None:
        self.llm_responses = list(llm_responses)
        self.on_publish = on_publish or (lambda event: None)
        self.on_timeout = on_timeout or (lambda request_id: None)
        self.bus: Dict[str, List[Any]] = defaultdict(list)
        self.timers_created = 0
        self.when_any_timers: List[_TimerTask] = []
        # the first N publishes replay history recorded by the older code,
        # whose publish activity returned None and used the legacy id
        self.legacy_publishes = legacy_publishes
        self.published: List[Dict[str, Any]] = []
        self.tools_run: List[Dict[str, Any]] = []
        self.ctx = DaprWorkflowContext()
        self.ctx.instance_id = "wf-approval-scope"
        self.ctx.is_replaying = False
        self.ctx.set_custom_status = Mock()
        self.ctx.current_utc_datetime = Mock()
        self.ctx.current_utc_datetime.isoformat = Mock(return_value=dispatch_time)
        self.ctx.call_activity = lambda fn, input=None, retry_policy=None: _Activity(
            fn, input
        )
        self.ctx.wait_for_external_event = _EventTask
        self.ctx.create_timer = self._create_timer

    def _create_timer(self, td: Any) -> _TimerTask:
        self.timers_created += 1
        return _TimerTask()

    def raise_raw(self, request_id: str, body: Any) -> None:
        self.bus[f"approval_response_{request_id}"].append(body)

    def raise_event(self, request_id: str, approved: bool, body_id: str = "") -> None:
        self.bus[f"approval_response_{request_id}"].append(
            {"approval_request_id": body_id or request_id, "approved": approved}
        )

    def _deliver(self, event: _EventTask) -> bool:
        if not self.bus[event.name]:
            return False
        event.result = self.bus[event.name].pop(0)
        return True

    def _respond(self, yielded: Any) -> Any:
        if isinstance(yielded, _Activity):
            if yielded.name.endswith("call_llm"):
                return self.llm_responses.pop(0)
            if yielded.name.endswith("publish_approval_request"):
                event = dict(yielded.input["event"])
                if len(self.published) < self.legacy_publishes:
                    event["approval_request_id"] = _legacy_approval_request_id(
                        self.ctx.instance_id, event["tool_call_id"]
                    )
                    self.published.append(event)
                    self.on_publish(event)
                    return None
                self.published.append(event)
                self.on_publish(event)
                return event["approval_request_id"]
            if yielded.name.endswith("run_tool"):
                tc = yielded.input["tool_call"]
                self.tools_run.append(tc)
                return {"role": "tool", "content": "ran", "tool_call_id": tc["id"]}
            return None
        if isinstance(yielded, tuple) and yielded[0] == "when_any":
            event, timer = yielded[1]
            self.when_any_timers.append(timer)
            if self._deliver(event):
                return event
            self.on_timeout(self.published[-1]["approval_request_id"])
            return timer
        if isinstance(yielded, _EventTask):
            assert self._deliver(yielded), "workflow would wait forever"
            return None
        raise AssertionError(f"unexpected yield: {yielded!r}")

    def drive(self, gen: Any, value: Any = None) -> Any:
        with patch("dapr.ext.workflow.when_any", side_effect=lambda t: ("when_any", t)):
            while True:
                try:
                    yielded = gen.send(value)
                except StopIteration as stop:
                    return stop.value
                value = self._respond(yielded)


def _approval_agent(llm, timeout_seconds: Optional[int] = 30):
    def hook(ctx):
        return RequireApproval(timeout_seconds=timeout_seconds)

    return _make_agent(llm, hooks=Hooks(before_tool_call=[hook]))


FINAL = {"role": "assistant", "content": "done"}


def test_late_turn1_approval_does_not_approve_turn2_call_with_reused_id(llm):
    """A turn-1 approval that lands after its timeout must not approve turn 2."""
    agent = _approval_agent(llm)
    turn_call = {"tool_calls": [_tool_call(name="DeleteRepo", call_id="call_0")]}
    late: List[str] = []

    def approve_late(request_id: str) -> None:
        # the human approves turn 1 only after it timed out; Dapr buffers it
        if not late:
            late.append(request_id)
            rt.raise_event(request_id, approved=True)

    rt = _FakeRuntime([turn_call, turn_call, FINAL], on_timeout=approve_late)
    rt.drive(agent.agent_workflow(rt.ctx, {"task": "delete it"}))

    assert rt.tools_run == [], "stale turn-1 approval approved the turn-2 call"
    assert len(rt.published) == 2
    turn1_id, turn2_id = (e["approval_request_id"] for e in rt.published)
    assert turn1_id != turn2_id
    # the stale approval is still buffered under turn 1's name, never consumed
    assert rt.bus[f"approval_response_{turn1_id}"]


def test_ids_distinct_for_repeated_tool_call_id_in_one_turn(llm):
    """Two calls in one turn sharing an id (here empty) get distinct approval ids."""
    agent = _approval_agent(llm)
    calls = [_tool_call(name="A", call_id=""), _tool_call(name="B", call_id="")]
    rt = _FakeRuntime(
        [{"tool_calls": calls}, FINAL],
        on_publish=lambda e: rt.raise_event(e["approval_request_id"], approved=True),
    )
    rt.drive(agent.agent_workflow(rt.ctx, {"task": "both"}))

    ids = [e["approval_request_id"] for e in rt.published]
    assert len(ids) == 2 and ids[0] != ids[1]
    assert not any(rt.bus.values()), "each wait consumes exactly its own response"


def test_two_waits_in_one_turn_resolve_to_their_own_responses(llm):
    """Responses buffered out of order still reach the call they were sent for."""
    agent = _approval_agent(llm, timeout_seconds=None)
    rt = _FakeRuntime([])
    tc = _tool_call(call_id="call_0")
    decision = RequireApproval(timeout_seconds=None)
    requests = [
        agent._request_approval(
            rt.ctx, rt.ctx.instance_id, tc, decision, turn=1, call_index=i
        )
        for i in range(2)
    ]
    # publish both requests first to learn their ids, then answer in reverse
    ids = [rt._respond(next(gen)) for gen in requests]
    assert ids[0] != ids[1]
    rt.raise_event(ids[1], approved=False)
    rt.raise_event(ids[0], approved=True)

    results = [rt.drive(gen, rid) for gen, rid in zip(requests, ids)]
    assert results == [True, False]


def test_mismatched_response_id_is_ignored_and_logged(llm, caplog):
    """A response whose body id does not match never approves; it is logged."""
    agent = _approval_agent(llm)
    rt = _FakeRuntime([])
    rt.on_publish = lambda e: rt.raise_event(
        e["approval_request_id"], approved=True, body_id="someone-else"
    )
    gen = agent._request_approval(
        rt.ctx,
        rt.ctx.instance_id,
        _tool_call(),
        RequireApproval(timeout_seconds=30),
        turn=1,
        call_index=0,
    )
    with caplog.at_level(logging.WARNING, logger="dapr_agents.agents.durable"):
        assert rt.drive(gen) is False

    assert any(
        r.levelno == logging.WARNING and "someone-else" in r.getMessage()
        for r in caplog.records
    )
    assert rt.timers_created == 1


def test_unparseable_response_is_ignored_and_keeps_waiting(llm, caplog):
    """A None or malformed body neither approves nor denies; the wait goes on."""
    agent = _approval_agent(llm)
    rt = _FakeRuntime([])

    def respond(event: Dict[str, Any]) -> None:
        rid = event["approval_request_id"]
        rt.raise_raw(rid, None)
        rt.raise_raw(rid, {"not": "a response"})
        rt.raise_event(rid, approved=True)

    rt.on_publish = respond
    gen = agent._request_approval(
        rt.ctx,
        rt.ctx.instance_id,
        _tool_call(),
        RequireApproval(timeout_seconds=30),
        turn=1,
        call_index=0,
    )
    with caplog.at_level(logging.WARNING, logger="dapr_agents.agents.durable"):
        assert rt.drive(gen) is True

    unparseable = [r for r in caplog.records if "unparseable" in r.getMessage()]
    assert len(unparseable) == 2
    assert all(r.levelno == logging.WARNING for r in unparseable)
    assert rt.timers_created == 1
    assert not any(rt.bus.values())


def test_mismatched_response_keeps_waiting_for_the_real_one(llm):
    """After ignoring a mismatched response the request still accepts its own."""
    agent = _approval_agent(llm)
    rt = _FakeRuntime([])

    def respond(event: Dict[str, Any]) -> None:
        rid = event["approval_request_id"]
        rt.raise_event(rid, approved=True, body_id="stale")
        rt.raise_event(rid, approved=True)

    rt.on_publish = respond
    gen = agent._request_approval(
        rt.ctx,
        rt.ctx.instance_id,
        _tool_call(),
        RequireApproval(timeout_seconds=30),
        turn=1,
        call_index=0,
    )
    assert rt.drive(gen) is True
    # one timer spans the whole wait; the mismatch did not restart it
    assert rt.timers_created == 1
    assert not any(rt.bus.values())


def test_approval_ids_stable_across_replay(llm):
    """Same history gives the same ids; a different orchestration time does not."""

    def run(dispatch_time: str) -> List[str]:
        agent = _approval_agent(llm)
        calls = {"tool_calls": [_tool_call(call_id="call_0")] * 2}
        rt = _FakeRuntime(
            [calls, calls, FINAL],
            dispatch_time=dispatch_time,
            on_publish=lambda e: rt.raise_event(
                e["approval_request_id"], approved=True
            ),
        )
        rt.drive(agent.agent_workflow(rt.ctx, {"task": "replay"}))
        return [e["approval_request_id"] for e in rt.published]

    first, replayed = run(DISPATCH_TIME), run(DISPATCH_TIME)
    assert len(first) == 4 and len(set(first)) == 4
    assert first == replayed
    other = run("2024-01-01T00:05:00.000000")
    assert set(other).isdisjoint(first)


def test_two_mismatches_then_timeout_share_one_timer(llm):
    """Ignored replies never restart the timeout: one timer spans every wait."""
    agent = _approval_agent(llm)
    rt = _FakeRuntime([])

    def respond(event: Dict[str, Any]) -> None:
        rid = event["approval_request_id"]
        rt.raise_event(rid, approved=True, body_id="stale-1")
        rt.raise_event(rid, approved=True, body_id="stale-2")

    rt.on_publish = respond
    gen = agent._request_approval(
        rt.ctx,
        rt.ctx.instance_id,
        _tool_call(),
        RequireApproval(timeout_seconds=30),
        turn=1,
        call_index=0,
    )
    assert rt.drive(gen) is False
    assert rt.timers_created == 1
    assert len(rt.when_any_timers) == 3
    assert all(t is rt.when_any_timers[0] for t in rt.when_any_timers)


def test_run_waiting_before_upgrade_is_released_by_its_old_id(llm):
    """A publish recorded by the older code keeps waiting on the legacy id."""
    agent = _approval_agent(llm)
    turn_call = {"tool_calls": [_tool_call(name="DeleteRepo", call_id="call_0")]}
    rt = _FakeRuntime(
        [turn_call, FINAL],
        legacy_publishes=1,
        on_publish=lambda e: rt.raise_event(e["approval_request_id"], approved=True),
    )
    rt.drive(agent.agent_workflow(rt.ctx, {"task": "delete it"}))

    legacy_id = _legacy_approval_request_id(rt.ctx.instance_id, "call_0")
    assert rt.published[0]["approval_request_id"] == legacy_id
    assert [tc["id"] for tc in rt.tools_run] == ["call_0"]
    assert not any(rt.bus.values())


def test_late_legacy_approval_does_not_approve_new_request(llm):
    """An old-format approval that arrives late never approves a new-code request."""
    agent = _approval_agent(llm)
    turn_call = {"tool_calls": [_tool_call(name="DeleteRepo", call_id="call_0")]}
    late: List[str] = []

    def approve_late(request_id: str) -> None:
        if not late:
            late.append(request_id)
            rt.raise_event(request_id, approved=True)

    rt = _FakeRuntime(
        [turn_call, turn_call, FINAL], legacy_publishes=1, on_timeout=approve_late
    )
    rt.drive(agent.agent_workflow(rt.ctx, {"task": "delete it"}))

    assert rt.tools_run == []
    legacy_id = _legacy_approval_request_id(rt.ctx.instance_id, "call_0")
    assert late == [legacy_id]
    assert rt.published[1]["approval_request_id"] != legacy_id
    assert rt.bus[f"approval_response_{legacy_id}"]
