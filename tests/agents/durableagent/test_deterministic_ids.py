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

"""Replay-determinism tests for child workflow instance IDs.

Dapr replays workflow code from history, so every ID the workflow body
creates must be identical on each replay. Each test drives the same workflow
generator twice with the same inputs, as a replay does, and compares the child
workflow instance IDs it produced.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
from dapr.ext.workflow import DaprWorkflowContext

from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    BuiltinTool,
    OrchestrationMode,
    ToolExecutionMode,
)
from dapr_agents.agents.durable import DurableAgent, child_workflow_instance_id
from dapr_agents.agents.schemas import AgentWorkflowEntry, AgentWorkflowMessage
from dapr_agents.hooks import Deny, Hooks, Mutate, Skip
from dapr_agents.tool import tool

PARENT_INSTANCE_ID = "parent-wf-1"
ORCHESTRATION_TIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def patch_dapr(monkeypatch):
    import dapr.ext.workflow as wf

    monkeypatch.setattr(wf, "WorkflowRuntime", lambda: MagicMock())

    class _RetryPolicy:
        def __init__(
            self,
            *,
            max_number_of_attempts=1,
            first_retry_interval=timedelta(seconds=1),
            max_retry_interval=timedelta(seconds=60),
            backoff_coefficient=2.0,
            retry_timeout: Optional[timedelta] = None,
        ):
            pass

    monkeypatch.setattr(wf, "RetryPolicy", _RetryPolicy)


@pytest.fixture(autouse=True)
def patch_dapr_client(monkeypatch):
    mock_client = MagicMock()
    mock_client.get_state.return_value = MagicMock(data=None)
    mock_client.get_metadata.return_value = MagicMock(
        registered_components=[], application_id="test-app"
    )
    monkeypatch.setattr(
        "dapr_agents.storage.daprstores.base.default_dapr_client_factory",
        lambda: mock_client,
    )
    monkeypatch.setattr("dapr.clients.DaprClient", lambda: mock_client)


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.prompt_template = None
    llm.__class__.__name__ = "MockLLM"
    llm.provider = "mock"
    llm.api = "mock"
    llm.model = "mock-model"
    llm.component_name = None
    llm.base_url = None
    llm.azure_endpoint = None
    llm.azure_deployment = None
    return llm


def _make_ctx(
    instance_id: str = PARENT_INSTANCE_ID, now: datetime = ORCHESTRATION_TIME
) -> DaprWorkflowContext:
    """A fresh workflow context, as the runtime builds for each replay."""
    ctx = DaprWorkflowContext()
    ctx.instance_id = instance_id
    ctx.is_replaying = False
    ctx.current_utc_datetime = now
    ctx.call_activity = Mock()
    ctx.call_child_workflow = Mock()
    ctx.set_custom_status = Mock()
    return ctx


def _tool_call(call_id: str, task: str) -> Dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "Sam", "arguments": json.dumps({"task": task})},
    }


def _child_ids(ctx: DaprWorkflowContext) -> List[str]:
    return [c.kwargs["instance_id"] for c in ctx.call_child_workflow.call_args_list]


def _run_agent_turn(
    agent: DurableAgent,
    tool_calls: List[Dict[str, Any]],
    now: datetime = ORCHESTRATION_TIME,
):
    """Drive agent_workflow up to the agent-as-tool dispatch of turn 1."""
    ctx = _make_ctx(now=now)
    gen = agent.agent_workflow(ctx, {"task": "ask sam"})
    next(gen)  # record_initial_entry
    gen.send(None)  # call_llm
    gen.send({"role": "assistant", "content": None, "tool_calls": tool_calls})
    gen.close()
    return ctx


def _make_frodo(mock_llm) -> DurableAgent:
    sam = DurableAgent(name="Sam", role="r", goal="g", llm=mock_llm)
    return DurableAgent(
        name="Frodo",
        role="r",
        goal="g",
        llm=mock_llm,
        tools=[sam],
        execution=AgentExecutionConfig(
            tool_execution_mode=ToolExecutionMode.SEQUENTIAL
        ),
    )


class TestChildWorkflowInstanceId:
    def test_same_inputs_same_id(self):
        a = child_workflow_instance_id("wf", "t0", "tool", 1, 0, "call-1")
        b = child_workflow_instance_id("wf", "t0", "tool", 1, 0, "call-1")
        assert a == b

    @pytest.mark.parametrize(
        "other",
        [
            ("wf-2", "t0", "tool", 1, 0, "call-1"),
            ("wf", "t1", "tool", 1, 0, "call-1"),
            ("wf", "t0", "tool", 2, 0, "call-1"),
            ("wf", "t0", "tool", 1, 1, "call-1"),
            ("wf", "t0", "tool", 1, 0, "call-2"),
            ("wf", "t0", "orchestration", 1),
        ],
    )
    def test_any_input_change_changes_id(self, other):
        base = child_workflow_instance_id("wf", "t0", "tool", 1, 0, "call-1")
        assert child_workflow_instance_id(*other) != base


class TestAgentAsToolReplay:
    def test_child_instance_id_stable_across_replay(self, mock_llm):
        frodo = _make_frodo(mock_llm)
        calls = [_tool_call("call-1", "first"), _tool_call("call-2", "second")]

        first = _child_ids(_run_agent_turn(frodo, calls))
        replay = _child_ids(_run_agent_turn(frodo, calls))

        assert len(first) == 2
        assert first == replay

    def test_empty_tool_call_ids_get_distinct_stable_ids(self, mock_llm):
        """The Dapr conversation client defaults tool_call_id to ""."""
        frodo = _make_frodo(mock_llm)
        calls = [_tool_call("", "first"), _tool_call("", "second")]

        first = _child_ids(_run_agent_turn(frodo, calls))
        replay = _child_ids(_run_agent_turn(frodo, calls))

        assert len(first) == 2
        assert first[0] != first[1]
        assert first == replay

    def test_duplicate_tool_call_ids_get_distinct_ids(self, mock_llm):
        frodo = _make_frodo(mock_llm)
        calls = [_tool_call("call-1", "first"), _tool_call("call-1", "second")]

        ids = _child_ids(_run_agent_turn(frodo, calls))

        assert len(set(ids)) == 2

    def test_id_is_scoped_to_parent_instance(self, mock_llm):
        frodo = _make_frodo(mock_llm)
        calls = [_tool_call("call-1", "first")]
        ctx_a = _run_agent_turn(frodo, calls)

        ctx_b = _make_ctx("another-parent")
        gen = frodo.agent_workflow(ctx_b, {"task": "ask sam"})
        next(gen)
        gen.send(None)
        gen.send({"role": "assistant", "content": None, "tool_calls": calls})
        gen.close()

        assert _child_ids(ctx_a) != _child_ids(ctx_b)

    def test_reused_instance_id_gets_new_ids_at_a_new_time(self, mock_llm):
        """A caller that reuses an instance id for a new run must not reuse child ids."""
        frodo = _make_frodo(mock_llm)
        calls = [_tool_call("call-1", "first"), _tool_call("call-2", "second")]

        first_run = _child_ids(_run_agent_turn(frodo, calls))
        second_run = _child_ids(
            _run_agent_turn(frodo, calls, now=ORCHESTRATION_TIME + timedelta(hours=1))
        )

        assert len(first_run) == len(second_run) == 2
        assert set(first_run).isdisjoint(second_run)


class TestOrchestrationReplay:
    def _run_orchestration_turn(self, boss: DurableAgent) -> DaprWorkflowContext:
        """Drive orchestration_workflow up to the child dispatch of turn 1."""
        ctx = _make_ctx("parent-wf-1:0001")
        gen = boss.orchestration_workflow(
            ctx, {"task": "do it", "instance_id": PARENT_INSTANCE_ID}
        )
        next(gen)  # get_team_members
        gen.send({"metadata": {"Sam": {"agent": {"appid": "sam-app"}}}})
        gen.send({})  # initialize_orchestration -> orch_state
        gen.send({"agent": "Sam", "instruction": "work"})  # select_next_task
        gen.close()
        return ctx

    def test_orchestrator_child_instance_id_stable_across_replay(self, mock_llm):
        boss = DurableAgent(
            name="Boss",
            role="r",
            goal="g",
            llm=mock_llm,
            execution=AgentExecutionConfig(
                orchestration_mode=OrchestrationMode.ROUNDROBIN, max_iterations=2
            ),
        )

        first = _child_ids(self._run_orchestration_turn(boss))
        replay = _child_ids(self._run_orchestration_turn(boss))

        assert len(first) == 1
        assert first == replay


def _save_tool_results_payload(
    agent: DurableAgent, tool_calls: List[Dict[str, Any]]
) -> tuple[DaprWorkflowContext, Dict[str, Any]]:
    """Drive a turn through its tool calls and return the save_tool_results input."""
    ctx = _make_ctx()
    gen = agent.agent_workflow(ctx, {"task": "ask sam"})
    next(gen)  # record_initial_entry
    gen.send(None)  # call_llm
    gen.send({"role": "assistant", "content": None, "tool_calls": tool_calls})
    for i in range(len(tool_calls) - 1):
        gen.send(f"result {i}")  # sequential child workflows
    gen.send(f"result {len(tool_calls) - 1}")  # yields save_tool_results
    gen.close()
    save_calls = [
        c
        for c in ctx.call_activity.call_args_list
        if "tool_call_meta" in c.kwargs.get("input", {})
    ]
    assert len(save_calls) == 1
    return ctx, save_calls[0].kwargs["input"]


class TestToolHistoryWithSharedToolCallIds:
    """tool_history must name the child that actually ran each call."""

    @pytest.mark.parametrize("call_id", ["", "call-1"])
    def test_payload_keeps_each_child_id(self, mock_llm, call_id):
        frodo = _make_frodo(mock_llm)
        calls = [_tool_call(call_id, "first"), _tool_call(call_id, "second")]

        ctx, payload = _save_tool_results_payload(frodo, calls)

        child_ids = _child_ids(ctx)
        assert len(set(child_ids)) == 2
        assert [m["child_instance_id"] for m in payload["tool_call_meta"]] == (
            child_ids
        )

    @pytest.mark.parametrize("call_id", ["", "call-1"])
    def test_tool_history_records_each_child_id(self, mock_llm, call_id):
        frodo = _make_frodo(mock_llm)
        calls = [_tool_call(call_id, "first"), _tool_call(call_id, "second")]
        ctx, payload = _save_tool_results_payload(frodo, calls)
        assistant = AgentWorkflowMessage(
            role="assistant", content=None, tool_calls=calls
        )
        entry = AgentWorkflowEntry(
            source="test",
            triggering_workflow_instance_id=None,
            messages=[assistant],
            tool_history=[],
            last_message=assistant,
        )

        with (
            patch.object(frodo, "save_state"),
            patch.object(frodo._infra, "get_state", return_value=entry),
        ):
            frodo.save_tool_results(Mock(), payload)

        assert [r.agent_workflow_instance_id for r in entry.tool_history] == (
            _child_ids(ctx)
        )
        assert [r.tool_args["task"] for r in entry.tool_history] == [
            "first",
            "second",
        ]

    def test_falls_back_to_tool_calls_by_id_without_meta(self, mock_llm):
        """Payloads scheduled before tool_call_meta existed still record."""
        frodo = _make_frodo(mock_llm)
        calls = [_tool_call("call-1", "first")]
        ctx, payload = _save_tool_results_payload(frodo, calls)
        payload = {k: v for k, v in payload.items() if k != "tool_call_meta"}
        assistant = AgentWorkflowMessage(
            role="assistant", content=None, tool_calls=calls
        )
        entry = AgentWorkflowEntry(
            source="test",
            triggering_workflow_instance_id=None,
            messages=[assistant],
            tool_history=[],
            last_message=assistant,
        )

        with (
            patch.object(frodo, "save_state"),
            patch.object(frodo._infra, "get_state", return_value=entry),
        ):
            frodo.save_tool_results(Mock(), payload)

        assert [r.agent_workflow_instance_id for r in entry.tool_history] == (
            _child_ids(ctx)
        )

    @pytest.mark.parametrize("bad_meta", [[], [{"bogus": 1}] * 3])
    def test_mismatched_tool_call_meta_falls_back_without_raising(
        self, mock_llm, bad_meta
    ):
        """A tool_call_meta whose length differs from tool_results is ignored."""
        frodo = _make_frodo(mock_llm)
        calls = [_tool_call("call-1", "first")]
        ctx, payload = _save_tool_results_payload(frodo, calls)
        payload = {**payload, "tool_call_meta": bad_meta}
        assistant = AgentWorkflowMessage(
            role="assistant", content=None, tool_calls=calls
        )
        entry = AgentWorkflowEntry(
            source="test",
            triggering_workflow_instance_id=None,
            messages=[assistant],
            tool_history=[],
            last_message=assistant,
        )

        with (
            patch.object(frodo, "save_state"),
            patch.object(frodo._infra, "get_state", return_value=entry),
        ):
            frodo.save_tool_results(Mock(), payload)

        assert [r.agent_workflow_instance_id for r in entry.tool_history] == (
            _child_ids(ctx)
        )

    def test_tool_calls_by_id_keeps_last_entry_for_shared_id(self, mock_llm):
        """Old code reads tool_calls_by_id on rollback; the last dispatch wins."""
        frodo = _make_frodo(mock_llm)
        calls = [_tool_call("call-1", "first"), _tool_call("call-1", "second")]

        ctx, payload = _save_tool_results_payload(frodo, calls)

        by_id = payload["tool_calls_by_id"]
        assert list(by_id) == ["call-1"]
        assert by_id["call-1"]["child_instance_id"] == _child_ids(ctx)[1]
        assert by_id["call-1"] == payload["tool_call_meta"][1]


def _call(call_id: str, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def test_tool_call_meta_lines_up_with_results_across_branches(mock_llm):
    """Every dispatch branch in one turn, with shared ids, stays aligned."""

    @tool
    def lookup(q: str) -> str:
        """Look something up."""
        return q

    def hook(ctx):
        args = ctx.payload
        if args.get("q") == "deny":
            return Deny(reason="no")
        if args.get("q") == "skip":
            return Skip(result="skipped")
        if args.get("task") == "mutate":
            return Mutate(payload={"task": "mutated"})
        return None

    sam = DurableAgent(name="Sam", role="r", goal="g", llm=mock_llm)
    frodo = DurableAgent(
        name="Frodo",
        role="r",
        goal="g",
        llm=mock_llm,
        tools=[sam, lookup],
        hooks=Hooks(before_tool_call=[hook]),
        execution=AgentExecutionConfig(
            tool_execution_mode=ToolExecutionMode.SEQUENTIAL,
            builtin_tools=[BuiltinTool.ASK_USER],
        ),
    )
    calls = [
        _call("d", "Lookup", {"q": "deny"}),
        _call("", "Lookup", {"q": "run"}),
        _call("", "Sam", {"task": "first"}),
        _call("", "ask_user", {"question": "?"}),
        _call("m", "Sam", {"task": "mutate"}),
        _call("s", "Lookup", {"q": "skip"}),
    ]
    ctx = _make_ctx()
    gen = frodo.agent_workflow(ctx, {"task": "go"})
    next(gen)  # record_initial_entry
    gen.send(None)  # call_llm
    gen.send({"role": "assistant", "content": None, "tool_calls": calls})
    gen.send("sam first")  # child workflow for calls[2]
    gen.send("sam mutated")  # child workflow for calls[4]
    gen.send(  # run_tool activity for calls[1]
        {"role": "tool", "name": "Lookup", "tool_call_id": "", "content": "run"}
    )
    gen.close()
    (save,) = [
        c.kwargs["input"]
        for c in ctx.call_activity.call_args_list
        if "tool_call_meta" in c.kwargs.get("input", {})
    ]
    results, meta = save["tool_results"], save["tool_call_meta"]

    assert len(results) == len(meta) == len(calls)
    assert [m["tool_call"]["function"]["name"] for m in meta] == [
        c["function"]["name"] for c in calls
    ]
    assert [r["name"] for r in results] == [c["function"]["name"] for c in calls]
    assert [m.get("hook_decision") for m in meta] == [
        "denied",
        None,
        None,
        None,
        None,
        "skipped",
    ]
    assert [m["child_instance_id"] for m in meta if m["is_agent_call"]] == (
        _child_ids(ctx)
    )
    assert json.loads(meta[4]["tool_call"]["function"]["arguments"]) == {
        "task": "mutated"
    }
    assert all(m.get("dispatch_time") for m in meta)
