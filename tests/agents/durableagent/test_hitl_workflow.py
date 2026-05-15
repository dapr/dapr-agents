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
Workflow-integration tests for the hook-based human-in-the-loop system.

These tests verify the full dispatch pipeline: hook decision – workflow branch (Deny / Skip / Modify / RequireApproval) – correct tool_results handed to save_tool_results. They reuse the mocking pattern from test_durable_agent.py (patch_dapr_check, mock workflow context).
"""

import json
import os
import uuid
from datetime import timedelta
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call

import pytest
from dapr.ext.workflow import DaprWorkflowContext

from dapr_agents.agents.configs import (
    AgentApprovalConfig,
    AgentExecutionConfig,
    AgentMemoryConfig,
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
    ToolExecutionMode,
)
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.agents.schemas import AgentWorkflowEntry
from dapr_agents.hooks import (
    Deny,
    HookContext,
    Hooks,
    Modify,
    Proceed,
    RequireApproval,
    Skip,
)
from dapr_agents.llm import OpenAIChatClient
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.tool.base import AgentTool


@pytest.fixture(autouse=True)
def patch_dapr_check(monkeypatch):
    import dapr.ext.workflow as wf

    mock_runtime = Mock(spec=wf.WorkflowRuntime)
    monkeypatch.setattr(wf, "WorkflowRuntime", lambda: mock_runtime)

    class MockRetryPolicy:
        def __init__(
            self,
            max_number_of_attempts=1,
            first_retry_interval=timedelta(seconds=1),
            max_retry_interval=timedelta(seconds=60),
            backoff_coefficient=2.0,
            retry_timeout: Optional[timedelta] = None,
        ):
            pass

    monkeypatch.setattr(wf, "RetryPolicy", MockRetryPolicy)
    yield mock_runtime


class MockDaprClient:
    def __init__(self, *args, **kwargs):
        self.get_state = MagicMock(return_value=Mock(data=None, json=lambda: {}))
        self.save_state = MagicMock()
        self.delete_state = MagicMock()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def get_metadata(self):
        resp = MagicMock()
        resp.registered_components = []
        resp.application_id = "test-app"
        return resp


# Shared fixtures                                                              #


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    os.environ["OPENAI_API_KEY"] = "test-key"
    mock_client = MockDaprClient()
    monkeypatch.setattr("dapr.clients.DaprClient", lambda *a, **kw: mock_client)
    monkeypatch.setattr(
        "dapr_agents.storage.daprstores.statestore.DaprClient",
        lambda *a, **kw: mock_client,
    )
    yield
    os.environ.pop("OPENAI_API_KEY", None)


@pytest.fixture
def mock_llm():
    m = Mock(spec=OpenAIChatClient)
    m.generate = AsyncMock()
    m.prompt_template = None
    m.__class__.__name__ = "MockLLMClient"
    m.provider = "MockProvider"
    m.api = "MockAPI"
    m.model = "gpt-4o-mock"
    return m


@pytest.fixture
def mock_ctx():
    ctx = DaprWorkflowContext()
    ctx.instance_id = "wf-hitl-test"
    ctx.is_replaying = False
    ctx.call_activity = Mock()
    ctx.wait_for_external_event = Mock()
    ctx.create_timer = Mock()
    ctx.set_custom_status = Mock()
    ctx.current_utc_datetime = Mock()
    ctx.current_utc_datetime.isoformat = Mock(return_value="2024-01-01T00:00:00.000000")
    return ctx


def _make_agent(mock_llm, hooks=None, tools=None, with_approval=True):
    """Build a minimal DurableAgent (no registry, no memory) to keep yield count small."""
    approval = AgentApprovalConfig(
        pubsub_name="test-pubsub",
        topic="test-approvals",
        default_timeout_seconds=60,
    )
    return DurableAgent(
        name="HookTestAgent",
        role="Hook Tester",
        goal="Test hooks",
        instructions=[],
        llm=mock_llm,
        tools=tools or [],
        pubsub=AgentPubSubConfig(
            pubsub_name="testpubsub",
            agent_topic="HookTestAgent",
        ),
        state=AgentStateConfig(store=StateStoreService(store_name="teststatestore")),
        execution=AgentExecutionConfig(
            max_iterations=3,
            approval=approval if with_approval else AgentApprovalConfig(),
            tool_execution_mode=ToolExecutionMode.SEQUENTIAL,
        ),
        hooks=hooks,
    )


def _tool_call(name="DeleteOldData", args=None, call_id="call-1"):
    return {
        "id": call_id,
        "function": {
            "name": name,
            "arguments": json.dumps(args or {}),
        },
    }


# Helper: drive agent_workflow, capturing activity calls                       #
#                                                                              #
# Minimal agent (no registry, no memory) has these yields:                    #
#   Y0  call_activity(record_initial_entry)                                   #
#   Y1  call_activity(call_llm)  – returns assistant_response                 #
#   [for each tool that runs]:                                                 #
#   Y?  call_activity(run_tool) (SEQUENTIAL, one at a time)                   #
#   Y?  call_activity(save_tool_results)                                       #
#   ...repeat loop if more turns...                                            #
#   Yn  call_activity(finalize_workflow)                                       #
#                                                                              #
# For RequireApproval paths, _request_approval inserts two extra yields       #
# between call_llm and run_tool / save_tool_results:                          #
#   Y?  call_activity(publish_approval_request)                               #
#   Y?  dt.when_any([event_task, timer_task])                                 #


def _activity_name(mock_ctx, call_index):
    """Return the first positional arg (the activity fn) for the Nth call_activity call."""
    return mock_ctx.call_activity.call_args_list[call_index][0][0]


def _activity_input(mock_ctx, call_index):
    """Return the input kwarg for the Nth call_activity call."""
    kw = mock_ctx.call_activity.call_args_list[call_index][1]
    return kw.get("input", {})


# Section 1: _request_approval sub-generator                                  #


class TestRequestApprovalGenerator:
    """
    Drive _request_approval as a standalone generator so we can test the
    approve/deny/timeout branches without running the full workflow loop.
    """

    @pytest.fixture
    def agent(self, mock_llm):
        return _make_agent(mock_llm, with_approval=True)

    def _run_approval(self, agent, mock_ctx, tool_call, decision, *, winner):
        """
        Drive _request_approval to completion.

        `winner` is either `event_task` (human responded) or `timer_task` (timeout).
        Returns the generator's return value (True / False).
        """
        event_task = Mock()
        timer_task = Mock()
        mock_ctx.wait_for_external_event.return_value = event_task
        mock_ctx.create_timer.return_value = timer_task

        if winner == "event":
            event_task.get_result.return_value = {
                "approval_request_id": "any",
                "approved": True,
            }
            winning_task = event_task
        else:
            winning_task = timer_task

        gen = agent._request_approval(
            mock_ctx, mock_ctx.instance_id, tool_call, decision
        )

        with patch("durabletask.task.when_any") as mock_when_any:
            when_any_future = Mock()
            mock_when_any.return_value = when_any_future

            next(gen)  # yields call_activity(publish_approval_request)
            gen.send(None)  # yields dt.when_any(...)

            try:
                gen.send(winning_task)
            except StopIteration as exc:
                return exc.value

        raise AssertionError("generator did not stop")

    def test_event_wins_returns_true(self, agent, mock_ctx):
        result = self._run_approval(
            agent,
            mock_ctx,
            _tool_call(),
            RequireApproval(timeout_seconds=30),
            winner="event",
        )
        assert result is True

    def test_timer_wins_returns_false(self, agent, mock_ctx):
        result = self._run_approval(
            agent,
            mock_ctx,
            _tool_call(),
            RequireApproval(timeout_seconds=10),
            winner="timer",
        )
        assert result is False

    def test_decision_timeout_takes_priority_over_default(self, agent, mock_ctx):
        """decision.timeout_seconds overrides the agent-level default."""
        tool_call = _tool_call()
        decision = RequireApproval(timeout_seconds=999)
        gen = agent._request_approval(
            mock_ctx, mock_ctx.instance_id, tool_call, decision
        )

        with patch("durabletask.task.when_any", return_value=Mock()):
            next(gen)
            gen.send(None)
            try:
                gen.send(mock_ctx.create_timer.return_value)
            except StopIteration:
                pass

        # create_timer should have been called with the decision's timeout, not the default (60)
        create_timer_call = mock_ctx.create_timer.call_args
        assert create_timer_call is not None
        actual_td = create_timer_call[0][0]
        assert actual_td == timedelta(seconds=999)

    def test_deterministic_request_id(self, agent, mock_ctx):
        """Same instance_id + tool_call_id always produces the same approval_request_id."""
        tc = _tool_call(call_id="call-abc")
        decision = RequireApproval()

        ids = []
        for _ in range(2):
            gen = agent._request_approval(mock_ctx, "fixed-instance", tc, decision)
            with patch("durabletask.task.when_any", return_value=Mock()):
                next(gen)
            # read the input that was passed to publish_approval_request
            publish_input = mock_ctx.call_activity.call_args[1]["input"]
            ids.append(publish_input["event"]["approval_request_id"])
            mock_ctx.call_activity.reset_mock()

        assert ids[0] == ids[1]
        assert ids[0] == str(uuid.uuid5(uuid.NAMESPACE_DNS, "fixed-instance:call-abc"))

    def test_publish_activity_receives_correct_event_fields(self, agent, mock_ctx):
        """ApprovalRequiredEvent published to the activity has expected shape."""
        tc = _tool_call(name="DropTable", args={"table": "users"}, call_id="c-42")
        decision = RequireApproval(
            timeout_seconds=120,
            instructions="please confirm",
        )
        gen = agent._request_approval(mock_ctx, mock_ctx.instance_id, tc, decision)

        with patch("durabletask.task.when_any", return_value=Mock()):
            next(gen)

        publish_input = mock_ctx.call_activity.call_args[1]["input"]
        event = publish_input["event"]
        assert event["step_name"] == "DropTable"
        assert event["step_kind"] == "tool"
        assert event["tool_arguments"] == {"table": "users"}
        assert event["timeout_seconds"] == 120
        assert event["instructions"] == "please confirm"
        assert publish_input["pubsub_name"] == "test-pubsub"
        assert publish_input["topic"] == "test-approvals"


# Section 2: hook-dispatch in agent_workflow                                   #


class TestHookWorkflowDispatch:
    """
    Drive agent_workflow through hook-dispatch and verify what ends up in the
    save_tool_results input. No live Dapr or LLM required.

    Yield sequence for a minimal agent (no registry, no memory):
      Y0  call_activity(record_initial_entry)
      Y1  call_activity(call_llm)              → send {"tool_calls": [...]}
      [Deny/Skip: hook runs sync, no extra yields]
      Y2  call_activity(save_tool_results)
      Y3  call_activity(call_llm)              → send final response (no tools)
      Y4  call_activity(finalize_workflow)     → StopIteration

    NEVER use side_effect=[list] on call_activity. When the list is exhausted,
    Mock raises StopIteration internally. PEP 479 (Python 3.7+) converts any
    StopIteration raised inside a generator body to RuntimeError. Drive the
    generator with gen.send(value) to control what each yield expression
    evaluates to; call_activity.return_value is left as the default MagicMock.
    """

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _drive_workflow_deny_skip(self, agent, mock_ctx, hook_fn, tool_calls_turn1):
        """
        Drive agent_workflow for one tool-bearing turn (Deny / Skip path).
        Returns the input dict passed to save_tool_results.
        """
        message = {"task": "do something"}
        agent._hooks = Hooks(before_tool_call=[hook_fn])
        gen = agent.agent_workflow(mock_ctx, message)

        try:
            next(gen)  # record_initial_entry
            gen.send(None)  # call_llm
            gen.send(
                {"tool_calls": tool_calls_turn1}
            )  # → hook deny/skip → save_tool_results
            gen.send(None)  # call_llm (final turn)
            gen.send({"role": "assistant", "content": "done"})  # finalize_workflow
            gen.send(None)  # → StopIteration
        except StopIteration:
            pass

        for c in mock_ctx.call_activity.call_args_list:
            if c[0][0] == agent.save_tool_results:
                return c[1]["input"]

        return None

    # ------------------------------------------------------------------ #
    # Deny                                                                 #
    # ------------------------------------------------------------------ #

    def test_deny_synthesizes_denial_message_run_tool_not_called(
        self, mock_llm, mock_ctx
    ):
        """Hook returning Deny → denial ToolMessage in save_tool_results, run_tool skipped."""
        tc = _tool_call(name="DropTable", args={"table": "logs"})

        def hook(ctx: HookContext):
            if ctx.step_name == "DropTable":
                return Deny(reason="schema changes go through DBA review")
            return Proceed()

        agent = _make_agent(mock_llm, hooks=Hooks(before_tool_call=[hook]))
        save_input = self._drive_workflow_deny_skip(agent, mock_ctx, hook, [tc])

        called_fns = [c[0][0] for c in mock_ctx.call_activity.call_args_list]
        assert agent.run_tool not in called_fns

        assert save_input is not None
        results = save_input["tool_results"]
        assert len(results) == 1
        assert results[0]["role"] == "tool"
        assert "not executed" in results[0]["content"]
        assert "DBA review" in results[0]["content"]
        assert results[0]["tool_call_id"] == tc["id"]

    def test_deny_records_hook_decision_in_tool_calls_by_id(self, mock_llm, mock_ctx):
        """Hook-denied tool appears in tool_calls_by_id with hook_decision='denied'."""
        tc = _tool_call(name="DropTable")

        def hook(ctx: HookContext):
            return Deny(reason="blocked")

        agent = _make_agent(mock_llm, hooks=Hooks(before_tool_call=[hook]))
        save_input = self._drive_workflow_deny_skip(agent, mock_ctx, hook, [tc])

        assert save_input is not None
        by_id = save_input["tool_calls_by_id"]
        assert tc["id"] in by_id
        assert by_id[tc["id"]]["hook_decision"] == "denied"

    # ------------------------------------------------------------------ #
    # Skip                                                                 #
    # ------------------------------------------------------------------ #

    def test_skip_uses_hook_result_run_tool_not_called(self, mock_llm, mock_ctx):
        """Hook returning Skip(result=...) → hook result in tool message, run_tool skipped."""
        tc = _tool_call(name="GetCachedValue", args={"key": "x"})

        def hook(ctx: HookContext):
            if ctx.step_name == "GetCachedValue":
                return Skip(result="cached-42")
            return Proceed()

        agent = _make_agent(mock_llm, hooks=Hooks(before_tool_call=[hook]))
        save_input = self._drive_workflow_deny_skip(agent, mock_ctx, hook, [tc])

        called_fns = [c[0][0] for c in mock_ctx.call_activity.call_args_list]
        assert agent.run_tool not in called_fns

        assert save_input is not None
        results = save_input["tool_results"]
        assert len(results) == 1
        assert results[0]["content"] == "cached-42"
        assert results[0]["tool_call_id"] == tc["id"]

    def test_skip_records_hook_decision_in_tool_calls_by_id(self, mock_llm, mock_ctx):
        """Hook-skipped tool appears in tool_calls_by_id with hook_decision='skipped'."""
        tc = _tool_call(name="GetCachedValue")

        def hook(ctx: HookContext):
            return Skip(result="from-cache")

        agent = _make_agent(mock_llm, hooks=Hooks(before_tool_call=[hook]))
        save_input = self._drive_workflow_deny_skip(agent, mock_ctx, hook, [tc])

        assert save_input is not None
        by_id = save_input["tool_calls_by_id"]
        assert tc["id"] in by_id
        assert by_id[tc["id"]]["hook_decision"] == "skipped"

    # ------------------------------------------------------------------ #
    # RequireApproval — timer wins → denial message                        #
    # ------------------------------------------------------------------ #

    def test_require_approval_timer_wins_denial_message(self, mock_llm, mock_ctx):
        """
        When RequireApproval is returned and the timer wins the race, the tool
        is not executed and a denial ToolMessage is handed to save_tool_results.

        Yield sequence (timer path):
          Y0  record_initial_entry
          Y1  call_llm            → {"tool_calls": [tc]}
          Y2  publish_approval_request (inside _request_approval)
          Y3  dt.when_any         → send timer_task (winner)
          Y4  save_tool_results   (timer won → denial → no run_tool)
          Y5  call_llm            → final response
          Y6  finalize_workflow   → StopIteration
        """
        tc = _tool_call(name="DeleteRepo", args={"repo": "myrepo"})

        def hook(ctx: HookContext):
            if ctx.step_name == "DeleteRepo":
                return RequireApproval(timeout_seconds=30)
            return Proceed()

        agent = _make_agent(
            mock_llm, hooks=Hooks(before_tool_call=[hook]), with_approval=True
        )
        message = {"task": "delete the repo"}

        timer_task = Mock(name="timer")
        event_task = Mock(name="event")
        mock_ctx.wait_for_external_event.return_value = event_task
        mock_ctx.create_timer.return_value = timer_task

        gen = agent.agent_workflow(mock_ctx, message)
        save_tool_input = None

        with patch("durabletask.task.when_any", return_value=Mock()):
            next(gen)  # Y0: record_initial_entry
            gen.send(None)  # Y1: call_llm
            gen.send({"tool_calls": [tc]})  # Y2: publish_approval_request
            gen.send(None)  # Y3: when_any
            gen.send(timer_task)  # Y4: timer wins → save_tool_results
            gen.send(None)  # Y5: call_llm (final)
            try:
                gen.send({"role": "assistant", "content": "done"})  # Y6: finalize
                gen.send(None)
            except StopIteration:
                pass

        for c in mock_ctx.call_activity.call_args_list:
            if c[0][0] == agent.save_tool_results:
                save_tool_input = c[1]["input"]
                break

        assert save_tool_input is not None
        results = save_tool_input["tool_results"]
        assert len(results) == 1
        assert results[0]["role"] == "tool"
        assert results[0]["tool_call_id"] == tc["id"]
        called_fns = [c[0][0] for c in mock_ctx.call_activity.call_args_list]
        assert agent.run_tool not in called_fns

    # ------------------------------------------------------------------ #
    # RequireApproval — event wins → tool runs                             #
    # ------------------------------------------------------------------ #

    def test_require_approval_approved_tool_runs(self, mock_llm, mock_ctx):
        """
        When RequireApproval is returned and the external event arrives first
        (approved=True), run_tool is called for that tool call.

        Yield sequence (approval path):
          Y0  record_initial_entry
          Y1  call_llm            → {"tool_calls": [tc]}
          Y2  publish_approval_request
          Y3  dt.when_any         → send event_task (winner)
          Y4  run_tool            → send tool_result
          Y5  save_tool_results
          Y6  call_llm            → final response
          Y7  finalize_workflow   → StopIteration
        """
        tc = _tool_call(
            name="DeleteRepo", args={"repo": "myrepo"}, call_id="c-approved"
        )

        def hook(ctx: HookContext):
            if ctx.step_name == "DeleteRepo":
                return RequireApproval(timeout_seconds=30)
            return Proceed()

        agent = _make_agent(
            mock_llm, hooks=Hooks(before_tool_call=[hook]), with_approval=True
        )
        message = {"task": "delete the repo"}

        event_task = Mock(name="event")
        timer_task = Mock(name="timer")
        mock_ctx.wait_for_external_event.return_value = event_task
        mock_ctx.create_timer.return_value = timer_task

        approval_request_id = str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"{mock_ctx.instance_id}:{tc['id']}")
        )
        event_task.get_result.return_value = {
            "approval_request_id": approval_request_id,
            "approved": True,
        }

        tool_result = {
            "content": "repo deleted",
            "role": "tool",
            "name": "DeleteRepo",
            "tool_call_id": tc["id"],
        }

        gen = agent.agent_workflow(mock_ctx, message)

        with patch("durabletask.task.when_any", return_value=Mock()):
            next(gen)  # Y0: record_initial_entry
            gen.send(None)  # Y1: call_llm
            gen.send({"tool_calls": [tc]})  # Y2: publish_approval_request
            gen.send(None)  # Y3: when_any
            gen.send(event_task)  # Y4: run_tool (event wins → approved)
            gen.send(tool_result)  # Y5: save_tool_results
            gen.send(None)  # Y6: call_llm (final)
            try:
                gen.send({"role": "assistant", "content": "done"})  # Y7: finalize
                gen.send(None)
            except StopIteration:
                pass

        called_fns = [c[0][0] for c in mock_ctx.call_activity.call_args_list]
        assert agent.run_tool in called_fns

    # ------------------------------------------------------------------ #
    # Mixed: denied + proceed in same turn                                 #
    # ------------------------------------------------------------------ #

    def test_mixed_denied_and_proceed_same_turn(self, mock_llm, mock_ctx):
        """
        Two tool calls in one LLM turn: one Denied, one Proceed.
        Only the Proceed tool reaches run_tool; both appear in save_tool_results.

        Yield sequence (SEQUENTIAL mode):
          Y0  record_initial_entry
          Y1  call_llm           → [tc_deny, tc_run]
          Y2  run_tool tc_run    (tc_deny handled sync by Deny branch)
          Y3  save_tool_results
          Y4  call_llm           → final response
          Y5  finalize_workflow  → StopIteration
        """
        tc_deny = _tool_call(name="DropTable", call_id="c-deny")
        tc_run = _tool_call(name="GetWeather", args={"city": "NYC"}, call_id="c-run")

        def hook(ctx: HookContext):
            if ctx.step_name == "DropTable":
                return Deny(reason="blocked")
            return Proceed()

        agent = _make_agent(mock_llm, hooks=Hooks(before_tool_call=[hook]))
        message = {"task": "drop the table and get weather"}

        run_tool_result = {
            "content": "NYC: 72F",
            "role": "tool",
            "name": "GetWeather",
            "tool_call_id": tc_run["id"],
        }

        gen = agent.agent_workflow(mock_ctx, message)
        save_tool_input = None

        try:
            next(gen)  # Y0: record_initial_entry
            gen.send(None)  # Y1: call_llm
            gen.send({"tool_calls": [tc_deny, tc_run]})  # Y2: run_tool tc_run
            gen.send(run_tool_result)  # Y3: save_tool_results
            gen.send(None)  # Y4: call_llm (final)
            gen.send({"role": "assistant", "content": "done"})  # Y5: finalize
            gen.send(None)
        except StopIteration:
            pass

        for c in mock_ctx.call_activity.call_args_list:
            if c[0][0] == agent.save_tool_results:
                save_tool_input = c[1]["input"]
                break

        assert save_tool_input is not None
        results = save_tool_input["tool_results"]
        assert len(results) == 2

        ids = {r["tool_call_id"] for r in results}
        assert tc_deny["id"] in ids
        assert tc_run["id"] in ids

        deny_msg = next(r for r in results if r["tool_call_id"] == tc_deny["id"])
        run_msg = next(r for r in results if r["tool_call_id"] == tc_run["id"])
        assert "not executed" in deny_msg["content"]
        assert run_msg["content"] == "NYC: 72F"

    # ------------------------------------------------------------------ #
    # MCP tool: source forwarded to hook context                           #
    # ------------------------------------------------------------------ #

    def test_mcp_tool_source_forwarded_to_hook_context(self, mock_llm, mock_ctx):
        """
        A tool registered with source='mcp' has ctx.source == 'mcp' when the
        hook fires. This lets hooks gate MCP tools by source without needing
        to list individual tool names.
        """
        received_contexts = []

        def hook(ctx: HookContext):
            received_contexts.append(ctx)
            return Deny(reason="all mcp calls require approval")

        mcp_tool = AgentTool(
            name="DeleteRepo",
            description="delete a repo",
            func=lambda repo: f"deleted {repo}",
            source="mcp",
        )
        agent = _make_agent(
            mock_llm, hooks=Hooks(before_tool_call=[hook]), tools=[mcp_tool]
        )
        tc = _tool_call(name="DeleteRepo", args={"repo": "myrepo"})
        message = {"task": "delete the repo"}

        gen = agent.agent_workflow(mock_ctx, message)

        try:
            next(gen)  # record_initial_entry
            gen.send(None)  # call_llm
            gen.send({"tool_calls": [tc]})  # → Deny → save_tool_results
            gen.send(None)  # call_llm (final)
            gen.send({"role": "assistant", "content": "done"})  # finalize
            gen.send(None)
        except StopIteration:
            pass

        assert len(received_contexts) >= 1
        assert received_contexts[0].source == "mcp"
        assert received_contexts[0].step_kind == "tool"

    # ------------------------------------------------------------------ #
    # Replay determinism                                                   #
    # ------------------------------------------------------------------ #

    def test_replay_determinism_hook_called_twice_identical_decisions(self):
        """
        Hooks must be pure functions. The same HookContext input must always
        produce the same decision. This is the replay-safety contract required
        by Dapr durable task.
        """
        call_count = 0

        def stateless_hook(ctx: HookContext):
            nonlocal call_count
            call_count += 1
            if ctx.step_name == "DropTable":
                return Deny(reason="schema policy")
            return Proceed()

        ctx1 = HookContext(
            step_name="DropTable",
            step_kind="tool",
            source="local",
            payload={"table": "users"},
            tool_call_id="c-1",
        )
        ctx2 = HookContext(
            step_name="DropTable",
            step_kind="tool",
            source="local",
            payload={"table": "users"},
            tool_call_id="c-1",
        )

        d1 = stateless_hook(ctx1)
        d2 = stateless_hook(ctx2)

        assert call_count == 2
        assert type(d1) is type(d2)
        assert isinstance(d1, Deny)
        assert d1.reason == d2.reason

    def test_replay_determinism_proceed_hook(self):
        """Proceed-returning hook is stable across repeated calls."""

        def passthrough(ctx: HookContext):
            return Proceed()

        ctx = HookContext("AnyTool", "tool", "local", {}, "c-1")
        d1 = passthrough(ctx)
        d2 = passthrough(ctx)
        assert isinstance(d1, Proceed)
        assert isinstance(d2, Proceed)
