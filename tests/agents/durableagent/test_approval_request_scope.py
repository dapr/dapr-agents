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
    ) -> None:
        self.llm_responses = list(llm_responses)
        self.on_publish = on_publish or (lambda event: None)
        self.on_timeout = on_timeout or (lambda request_id: None)
        self.bus: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.published: List[Dict[str, Any]] = []
        self.tools_run: List[Dict[str, Any]] = []
        self.ctx = DaprWorkflowContext()
        self.ctx.instance_id = "wf-approval-scope"
        self.ctx.is_replaying = False
        self.ctx.set_custom_status = Mock()
        self.ctx.current_utc_datetime = Mock()
        self.ctx.current_utc_datetime.isoformat = Mock(return_value=DISPATCH_TIME)
        self.ctx.call_activity = lambda fn, input=None, retry_policy=None: _Activity(
            fn, input
        )
        self.ctx.wait_for_external_event = _EventTask
        self.ctx.create_timer = lambda td: _TimerTask()

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
                event = yielded.input["event"]
                self.published.append(event)
                self.on_publish(event)
            if yielded.name.endswith("run_tool"):
                tc = yielded.input["tool_call"]
                self.tools_run.append(tc)
                return {"role": "tool", "content": "ran", "tool_call_id": tc["id"]}
            return None
        if isinstance(yielded, tuple) and yielded[0] == "when_any":
            event, timer = yielded[1]
            if self._deliver(event):
                return event
            self.on_timeout(self.published[-1]["approval_request_id"])
            return timer
        if isinstance(yielded, _EventTask):
            assert self._deliver(yielded), "workflow would wait forever"
            return None
        raise AssertionError(f"unexpected yield: {yielded!r}")

    def drive(self, gen: Any) -> Any:
        value = None
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
    ids = []
    for gen in requests:
        rt._respond(next(gen))
        ids.append(rt.published[-1]["approval_request_id"])
    assert ids[0] != ids[1]
    rt.raise_event(ids[1], approved=False)
    rt.raise_event(ids[0], approved=True)

    results = [rt.drive(gen) for gen in requests]
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


def test_mismatched_response_keeps_waiting_for_the_real_one(llm):
    """After ignoring a mismatched response the request still accepts its own."""
    agent = _approval_agent(llm, timeout_seconds=None)
    rt = _FakeRuntime([])

    def respond(event: Dict[str, Any]) -> None:
        rid = event["approval_request_id"]
        rt.raise_event(rid, approved=True, body_id="stale")
        rt.raise_event(rid, approved=False)

    rt.on_publish = respond
    gen = agent._request_approval(
        rt.ctx,
        rt.ctx.instance_id,
        _tool_call(),
        RequireApproval(timeout_seconds=None),
        turn=1,
        call_index=0,
    )
    assert rt.drive(gen) is False
    assert not any(rt.bus.values())


def test_approval_ids_stable_across_replay(llm):
    """Re-running the same history yields the same approval ids."""

    def run() -> List[str]:
        agent = _approval_agent(llm)
        calls = {"tool_calls": [_tool_call(call_id="call_0")] * 2}
        rt = _FakeRuntime(
            [calls, calls, FINAL],
            on_publish=lambda e: rt.raise_event(
                e["approval_request_id"], approved=True
            ),
        )
        rt.drive(agent.agent_workflow(rt.ctx, {"task": "replay"}))
        return [e["approval_request_id"] for e in rt.published]

    first, second = run(), run()
    assert len(first) == 4 and len(set(first)) == 4
    assert first == second
