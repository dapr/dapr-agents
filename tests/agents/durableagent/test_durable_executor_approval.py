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

"""DurableAgent executor integration: binding, approval pauses, streaming.

These tests drive ``agent_workflow`` and ``_consume_executor`` with scripted
executors; the Claude-specific ones use a fake SDK client (no network).
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from dapr_agents.agents.executor_run import EXECUTOR_PAUSED_KEY, PausedExecutorRun
from dapr_agents.agents.executors import (
    AgentEvent,
    DaprSessionStore,
    EchoAgentExecutor,
    ExecutorBinding,
)
from dapr_agents.agents.executors.observer import ExecutorRunObserver
from dapr_agents.agents.schemas import AgentWorkflowEntry
from dapr_agents.hooks import Hooks, RequireApproval
from dapr_agents.tool import tool
from dapr_agents.tool.mcp.dapr_workflow_client import mcp_tool_def_to_workflow_tool
from dapr_agents.tool.workflow.tool_context import WorkflowContextInjectedTool
from dapr_agents.types import AgentError
from dapr_agents.types.tools import ToolExecutionStatus
from dapr.ext.workflow import MCPToolDef
from tests.agents.durableagent.test_durable_executor import (  # noqa: F401
    _make_agent,
    _ScriptedExecutor,
    patch_dapr_check,
    setup_env,
)

SESSION = "a79daed0-7ae8-4ca5-9385-0d62f44d05d3"


@tool
def get_weather(city: str) -> str:
    """Get weather."""
    return f"sunny in {city}"


class _ApprovalExecutor(_ScriptedExecutor):
    """Scripted executor that can pause and records its bindings."""

    supports_tool_approval = True

    def __init__(self, script: List[AgentEvent]) -> None:
        super().__init__(script)
        self.bindings: List[ExecutorBinding] = []

    def bind(self, binding: ExecutorBinding) -> "_ApprovalExecutor":
        self.bindings.append(binding)
        return self


class _RecordingObserver(ExecutorRunObserver):
    def __init__(self) -> None:
        self.events: List[AgentEvent] = []
        self.finished: List[Optional[BaseException]] = []

    def on_event(self, event: AgentEvent) -> None:
        self.events.append(event)

    def finish(self, error: Optional[BaseException] = None) -> None:
        self.finished.append(error)


def _entry(**kwargs) -> AgentWorkflowEntry:
    return AgentWorkflowEntry(
        source="test",
        triggering_workflow_instance_id=None,
        messages=[],
        tool_history=[],
        **kwargs,
    )


def _paused(tool_call_id="t1", session_id=SESSION, **approval) -> Dict[str, Any]:
    return PausedExecutorRun(
        tool_call_id=tool_call_id,
        name="transfer",
        arguments={"to": "alice"},
        session_id=session_id,
        approval=approval,
    ).to_activity_result()


# ---------------------------------------------------------------------------
# Workflow body: pause -> _request_approval -> resume
# ---------------------------------------------------------------------------


class _WorkflowDriver:
    """Runs ``agent_workflow`` against a mock context.

    ``run_executor`` returns the queued results in order; every other
    activity returns ``None``. The approval race is resolved by ``outcome``:
    ``"approve"`` / ``"deny"`` deliver an approval event, ``"timeout"``
    fires the timer.
    """

    def __init__(self, agent, run_results: List[Any], outcome: str = "approve"):
        self.agent = agent
        self.results = list(run_results)
        self.outcome = outcome
        self.activities: List[Dict[str, Any]] = []
        self.event_task = Mock(name="event_task")
        self.event_task.get_result.return_value = {
            "approval_request_id": "x",
            "approved": outcome == "approve",
        }
        self.timer_task = Mock(name="timer_task")
        self.ctx = self._context()

    def _context(self):
        ctx = Mock()
        ctx.instance_id = "wf-1"
        ctx.is_replaying = False
        ctx.current_utc_datetime.isoformat.return_value = "2026-01-01T00:00:00"
        ctx.call_activity = Mock(side_effect=self._activity)
        ctx.wait_for_external_event = Mock(return_value=self.event_task)
        ctx.create_timer = Mock(return_value=self.timer_task)
        return ctx

    def _activity(self, activity, **kwargs):
        name = activity if isinstance(activity, str) else activity.__name__
        self.activities.append({"name": name, "input": kwargs.get("input")})
        return ("activity", name)

    def run_inputs(self) -> List[Dict[str, Any]]:
        return [
            a["input"] for a in self.activities if a["name"].endswith("run_executor")
        ]

    def published(self) -> List[Dict[str, Any]]:
        return [
            a["input"]["event"]
            for a in self.activities
            if a["name"].endswith("publish_approval_request")
        ]

    def _reply(self, yielded: Any) -> Any:
        if yielded == "WHEN_ANY":
            return self.timer_task if self.outcome == "timeout" else self.event_task
        if isinstance(yielded, tuple) and yielded[1].endswith("run_executor"):
            return self.results.pop(0)
        return None

    def run(self, message: Dict[str, Any]) -> Any:
        entry = _entry()
        self.agent._infra._state_model = entry
        with (
            patch.object(self.agent._infra, "get_state", side_effect=lambda w: entry),
            patch("dapr.ext.workflow.when_any", return_value="WHEN_ANY"),
        ):
            gen = self.agent.agent_workflow(self.ctx, message)
            sent = None
            try:
                while True:
                    sent = self._reply(gen.send(sent))
            except StopIteration as stop:
                return stop.value


def _approval_agent(max_iterations: int = 1):
    agent = _make_agent(_ApprovalExecutor([]))
    agent.execution.max_iterations = max_iterations
    return agent


class TestApprovalLoop:
    def test_approved_call_resumes_same_session_with_decision(self):
        agent = _approval_agent()
        driver = _WorkflowDriver(
            agent,
            [
                _paused(timeout_seconds=30, instructions="Check it", reason="money"),
                {"role": "assistant", "content": "Sent."},
            ],
        )
        result = driver.run({"task": "pay alice", "context": {"tenant": "t"}})

        assert result == {"role": "assistant", "content": "Sent."}
        first, resume = driver.run_inputs()
        assert first["task"] == "pay alice"
        assert first.get("round") is None
        assert resume["task"] is None
        assert resume["round"] == 1
        assert resume["session_id"] == SESSION
        assert resume["context"] == {
            "tenant": "t",
            "tool_decisions": {"t1": {"approved": True, "reason": None}},
        }
        (event,) = driver.published()
        assert event["tool_call_id"] == "t1"
        assert event["step_name"] == "transfer"
        assert event["tool_arguments"] == {"to": "alice"}
        assert event["timeout_seconds"] == 30
        assert event["instructions"] == "Check it"
        driver.ctx.create_timer.assert_called_once()

    @pytest.mark.parametrize("outcome", ["deny", "timeout"])
    def test_denied_or_timed_out_call_is_rejected(self, outcome):
        agent = _approval_agent()
        driver = _WorkflowDriver(
            agent,
            [_paused(timeout_seconds=5), {"role": "assistant", "content": "ok"}],
            outcome=outcome,
        )
        driver.run({"task": "pay"})
        decision = driver.run_inputs()[1]["context"]["tool_decisions"]["t1"]
        assert decision == {
            "approved": False,
            "reason": "approval was not granted or timed out",
        }

    def test_default_timeout_applies_without_hook_details(self):
        agent = _approval_agent()
        agent.execution.approval.default_timeout_seconds = 42
        driver = _WorkflowDriver(
            agent, [_paused(), {"role": "assistant", "content": "ok"}]
        )
        driver.run({"task": "pay"})
        assert driver.published()[0]["timeout_seconds"] == 42

    def test_repeated_pauses_get_new_rounds_and_ids(self):
        agent = _approval_agent(max_iterations=3)
        driver = _WorkflowDriver(
            agent,
            [
                _paused("t1"),
                _paused("t2"),
                {"role": "assistant", "content": "both done"},
            ],
        )
        assert driver.run({"task": "pay twice"})["content"] == "both done"
        inputs = driver.run_inputs()
        assert [i.get("round") for i in inputs] == [None, 1, 2]
        assert list(inputs[2]["context"]["tool_decisions"]) == ["t2"]
        published = driver.published()
        assert [e["tool_call_id"] for e in published] == ["t1", "t2"]
        assert (
            published[0]["approval_request_id"] != published[1]["approval_request_id"]
        )

    def test_approval_rounds_are_capped(self):
        agent = _approval_agent(max_iterations=1)
        driver = _WorkflowDriver(agent, [_paused("t1"), _paused("t2")])
        result = driver.run({"task": "pay"})
        assert len(driver.run_inputs()) == 2
        assert "maximum number of tool approvals" in result["content"]

    def test_session_falls_back_to_caller_session(self):
        agent = _approval_agent()
        driver = _WorkflowDriver(
            agent,
            [_paused(session_id=None), {"role": "assistant", "content": "ok"}],
        )
        driver.run({"task": "pay", "session_id": "caller-session"})
        first, resume = driver.run_inputs()
        assert first["session_id"] == "caller-session"
        assert resume["session_id"] == "caller-session"


class TestPausedExecutorRun:
    def test_round_trip_and_tool_call_shape(self):
        paused = PausedExecutorRun.from_activity_result(_paused(reason="r"))
        assert paused.approval == {"reason": "r"}
        call = paused.tool_call()
        assert call["id"] == "t1"
        assert json.loads(call["function"]["arguments"]) == {"to": "alice"}
        assert paused.require_approval() == RequireApproval(reason="r")

    @pytest.mark.parametrize("value", [None, "x", {"role": "assistant"}, {}])
    def test_plain_results_are_not_paused(self, value):
        assert PausedExecutorRun.from_activity_result(value) is None

    def test_paused_event_requires_call_id(self):
        with pytest.raises(AgentError, match="tool_call_id"):
            PausedExecutorRun.from_event(AgentEvent(type="paused", content={}))


# ---------------------------------------------------------------------------
# Activity: binding, pause bookkeeping, streaming, observer
# ---------------------------------------------------------------------------


def _consume(agent, payload, entry=None, emitter=None, observer=None):
    entry = entry if entry is not None else _entry()
    agent._infra._state_model = entry
    patches = [
        patch.object(agent, "save_state"),
        patch.object(agent._infra, "get_state", side_effect=lambda wid: entry),
        patch.object(agent, "_executor_stream_emitter", return_value=emitter),
    ]
    if observer is not None:
        patches.append(patch.object(agent, "_executor_observer", return_value=observer))
    for p in patches:
        p.start()
    try:
        return asyncio.run(agent._consume_executor(payload)), entry
    finally:
        for p in reversed(patches):
            p.stop()


def _mcp_tool() -> WorkflowContextInjectedTool:
    return mcp_tool_def_to_workflow_tool(
        MCPToolDef(
            name="add",
            description="Add.",
            input_schema={"type": "object", "properties": {"a": {"type": "integer"}}},
            server_name="math",
            call_tool_workflow="dapr.internal.mcp.math.CallTool.add",
        )
    )


class TestBinding:
    def test_binding_carries_profile_tools_hooks_and_store(self):
        executor = _ApprovalExecutor(
            [AgentEvent(type="complete", content={"role": "assistant", "content": "x"})]
        )
        agent = _make_agent(executor)
        agent.tool_executor.register_tool(get_weather)
        agent.tool_executor.register_tool(_mcp_tool())
        hook = lambda ctx: None  # noqa: E731
        agent._hooks = Hooks(before_tool_call=[hook])

        _consume(agent, {"task": "hi", "instance_id": "i"})
        _consume(agent, {"task": "again", "instance_id": "i"})

        first, second = executor.bindings
        assert first.agent_name == "ExecAgent"
        assert "Test Executor Agent" in first.system_prompt
        assert "Exercise the executor branch" in first.system_prompt
        assert first.max_iterations == 1
        names = [t.name for t in first.tools]
        assert get_weather.name in names and "add" in names
        # Workflow-backed (MCP) tools are bridged for use inside the activity.
        assert not any(isinstance(t, WorkflowContextInjectedTool) for t in first.tools)
        assert first.before_tool_call == (hook,)
        assert isinstance(first.session_store, DaprSessionStore)
        # One store per agent, reused across runs.
        assert second.session_store is first.session_store

    def test_no_session_store_without_state_store(self):
        agent = _make_agent(_ApprovalExecutor([]))
        with patch.object(type(agent), "state_store", new=None):
            assert agent._executor_session_store() is None

    def test_prompt_render_failure_is_tolerated(self):
        agent = _make_agent(_ApprovalExecutor([]))
        with patch.object(
            agent.prompting_helper,
            "build_initial_messages",
            side_effect=RuntimeError("bad template"),
        ):
            assert agent._executor_system_prompt() is None

    def test_approval_client_only_for_pausable_executors(self):
        assert _make_agent(_ApprovalExecutor([]))._wf_client is not None
        assert _make_agent(EchoAgentExecutor())._wf_client is None


class TestConsumePaused:
    def test_paused_run_returns_marker_and_records_pending_tool(self):
        executor = _ApprovalExecutor(
            [
                AgentEvent(type="session", content={}, session_id=SESSION),
                AgentEvent(
                    type="tool_call",
                    content={"id": "t1", "name": "transfer", "arguments": {"to": "a"}},
                ),
                AgentEvent(
                    type="paused",
                    content={
                        "tool_call_id": "t1",
                        "name": "transfer",
                        "arguments": {"to": "a"},
                        "approval": {"reason": "money"},
                    },
                    metadata={"cost_usd": 0.01, "usage": {"input_tokens": 3}},
                ),
            ]
        )
        agent = _make_agent(executor)
        result, entry = _consume(agent, {"task": "pay", "instance_id": "i", "round": 2})

        paused = result[EXECUTOR_PAUSED_KEY]
        assert paused["tool_call_id"] == "t1"
        assert paused["session_id"] == SESSION
        assert paused["approval"] == {"reason": "money"}
        assert entry.session_id == SESSION
        assert entry.tool_history[0].status == ToolExecutionStatus.PENDING
        assert entry.executor_usage == [
            {
                "round": 2,
                "session_id": SESSION,
                "cost_usd": 0.01,
                "usage": {"input_tokens": 3},
            }
        ]

    def test_resume_round_does_not_record_task_again(self):
        executor = _ApprovalExecutor(
            [
                AgentEvent(
                    type="complete", content={"role": "assistant", "content": "ok"}
                )
            ]
        )
        agent = _make_agent(executor)
        _, entry = _consume(
            agent,
            {
                "task": None,
                "instance_id": "i",
                "session_id": SESSION,
                "round": 1,
                "context": {"tool_decisions": {"t1": {"approved": True}}},
            },
        )
        assert [m.role for m in entry.messages] == ["assistant"]
        call = executor.calls[0]
        assert call["prompt"] == ""
        assert call["session_id"] == SESSION
        assert call["context"]["tool_decisions"] == {"t1": {"approved": True}}

    def test_failed_tool_result_is_recorded_as_failed(self):
        executor = _ApprovalExecutor(
            [
                AgentEvent(
                    type="tool_call",
                    content={"id": "t1", "name": "x", "arguments": {}},
                ),
                AgentEvent(
                    type="tool_result",
                    content={
                        "tool_call_id": "t1",
                        "result": "denied",
                        "is_error": True,
                    },
                ),
                AgentEvent(
                    type="complete", content={"role": "assistant", "content": "no"}
                ),
            ]
        )
        _, entry = _consume(_make_agent(executor), {"task": "t", "instance_id": "i"})
        assert entry.tool_history[0].status == ToolExecutionStatus.FAILED


class TestStreamingAndObserver:
    def test_events_are_forwarded_to_stream_and_observer(self):
        events = [
            AgentEvent(type="text_delta", content="He"),
            AgentEvent(type="text_delta", content="llo"),
            AgentEvent(
                type="complete",
                content={"role": "assistant", "content": "Hello"},
                metadata={"cost_usd": 0.5, "num_turns": 1},
            ),
        ]
        emitter = MagicMock()
        observer = _RecordingObserver()
        _consume(
            _make_agent(_ApprovalExecutor(events)),
            {"task": "hi", "instance_id": "i"},
            emitter=emitter,
            observer=observer,
        )
        assert [c.args[0] for c in emitter.emit_text_delta.call_args_list] == [
            "He",
            "llo",
        ]
        emitter.complete_turn.assert_called_once_with(
            {"role": "assistant", "content": "Hello"},
            metadata={"cost_usd": 0.5, "num_turns": 1},
        )
        emitter.close.assert_called_once()
        assert observer.events == events
        assert observer.finished == [None]

    def test_pause_emits_turn_paused(self):
        emitter = MagicMock()
        executor = _ApprovalExecutor(
            [AgentEvent(type="paused", content={"tool_call_id": "t1", "name": "x"})]
        )
        _consume(
            _make_agent(executor), {"task": "hi", "instance_id": "i"}, emitter=emitter
        )
        emitter.emit_event.assert_called_once()
        assert emitter.emit_event.call_args.kwargs["event_data"]["reason"] == (
            "tool_approval"
        )

    def test_error_is_streamed_and_reported(self):
        emitter = MagicMock()
        observer = _RecordingObserver()
        executor = _ApprovalExecutor([AgentEvent(type="error", content="boom")])
        with pytest.raises(AgentError, match="boom"):
            _consume(
                _make_agent(executor),
                {"task": "hi", "instance_id": "i"},
                emitter=emitter,
                observer=observer,
            )
        emitter.emit_error.assert_called_once()
        emitter.close.assert_called_once()
        assert isinstance(observer.finished[0], AgentError)

    def test_no_emitter_without_stream_context(self):
        agent = _make_agent(_ApprovalExecutor([]))
        assert agent._executor_stream_emitter("i", 0, _entry()) is None

    def test_emitter_turn_follows_round(self):
        agent = _make_agent(_ApprovalExecutor([]))
        entry = _entry()
        entry.stream_context = {"listener_config": {"type": "in_process"}}
        with patch.object(agent, "_build_stream_emitter", return_value="e") as build:
            assert agent._executor_stream_emitter("i", 2, entry) == "e"
        assert build.call_args.kwargs["turn"] == 3


# ---------------------------------------------------------------------------
# ClaudeAgentExecutor inside DurableAgent (fake SDK client)
# ---------------------------------------------------------------------------


@pytest.fixture
def claude(monkeypatch, tmp_path):
    pytest.importorskip("claude_agent_sdk")
    import dapr_agents.agents.executors.claude as claude_module
    from claude_agent_sdk import InMemorySessionStore

    from tests.executors.claude_fakes import FakeClaudeClient

    FakeClaudeClient.reset()
    monkeypatch.setattr(claude_module, "ClaudeSDKClient", FakeClaudeClient)
    store = InMemorySessionStore()

    def make_agent(**config):
        from dapr_agents.agents.executors import ClaudeAgentExecutorConfig

        executor = claude_module.ClaudeAgentExecutor(
            ClaudeAgentExecutorConfig(cwd=str(tmp_path), **config)
        )
        agent = _make_agent(executor)
        agent._executor_store = store
        return agent

    return make_agent, FakeClaudeClient, store


class TestClaudeInDurableAgent:
    def _gated_agent(self, make_agent):
        agent = make_agent()
        agent.tool_executor.register_tool(get_weather)
        agent._hooks = Hooks(
            before_tool_call=[
                lambda c: (
                    RequireApproval(reason="r")
                    if c.step_name == get_weather.name
                    else None
                )
            ]
        )
        return agent

    def test_agent_config_reaches_claude_options(self, claude):
        from claude_agent_sdk.types import DeferredToolUse

        from tests.executors.claude_fakes import result_message

        make_agent, client, store = claude
        agent = self._gated_agent(make_agent)
        tool_name = f"mcp__dapr__{get_weather.name}"
        client.reset(
            [
                result_message(
                    session_id=SESSION,
                    result="",
                    stop_reason="tool_deferred",
                    deferred_tool_use=DeferredToolUse(
                        id="t1", name=tool_name, input={"city": "P"}
                    ),
                )
            ]
        )
        result, entry = _consume(agent, {"task": "weather?", "instance_id": "i"})

        options = client.last().options
        assert "Test Executor Agent" in options.system_prompt
        assert options.max_turns == 1
        assert options.session_store is store
        assert "dapr" in options.mcp_servers
        # The gate owns the agent's tools; nothing is statically allowed.
        assert options.allowed_tools == []
        assert result[EXECUTOR_PAUSED_KEY]["name"] == get_weather.name
        assert entry.session_id == SESSION

    def test_retried_attempt_resumes_saved_session(self, claude):
        from tests.executors.claude_fakes import result_message

        make_agent, client, store = claude
        agent = make_agent()
        entry = _entry(session_id=SESSION)
        key = {
            "project_key": agent.executor.project_key,
            "session_id": SESSION,
        }
        asyncio.run(store.append(key, [{"type": "user", "uuid": "u1"}]))
        client.reset([result_message(session_id=SESSION, result="done")])

        result, _ = _consume(agent, {"task": "hi", "instance_id": "i"}, entry=entry)

        assert result == {"role": "assistant", "content": "done"}
        options = client.last().options
        assert options.resume == SESSION
        assert options.session_id is None
