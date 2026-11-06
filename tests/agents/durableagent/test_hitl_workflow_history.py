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

"""Tests for the history-derived conversation path.

``test_hitl_workflow`` holds the old path's copy; at the contract release
the old copy goes and this module takes its name.
"""

import json
import uuid
from unittest.mock import AsyncMock, Mock, patch
import pytest
from dapr.ext.workflow import DaprWorkflowContext
from dapr_agents.hooks import (
    Deny,
    HookContext,
    Hooks,
    Mutate,
    Proceed,
    RequireApproval,
    Skip,
    ToolHookContext,
)
from dapr_agents.llm import OpenAIChatClient
from dapr_agents.tool.base import AgentTool
from tests.agents.durableagent.test_hitl_workflow import (  # noqa: F401
    patch_dapr_check,
    MockDaprClient,
    setup_env,
    _make_agent,
    _tool_call,
    _activity_name,
    _activity_input,
)

pytestmark = pytest.mark.usefixtures("history_path_only")


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


def _tool_replies(agent, mock_ctx, turn=1):
    """The tool messages handed to the call_llm after ``turn`` (its ``new_messages``)."""
    llm_inputs = [
        c[1]["input"]
        for c in mock_ctx.call_activity.call_args_list
        if c[0][0] == agent._activity_name(agent.call_llm)
    ]
    return llm_inputs[turn]["new_messages"]


class TestHookWorkflowDispatch:
    """
    Drive agent_workflow through hook-dispatch and verify which tool messages
    the next call_llm receives. No live Dapr or LLM required.

    Yield sequence for a minimal agent (no registry, no memory):
      Y0  call_activity(call_llm)              → send {"tool_calls": [...]}
      [Deny/Skip: hook runs sync, no extra yields]
      Y1  call_activity(call_llm)              → send final response (no tools)
      Y2  call_activity(finalize_workflow)     → StopIteration

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
        Returns the tool messages the next call_llm received.
        """
        message = {"task": "do something"}
        agent._hooks = Hooks(before_tool_call=[hook_fn])
        gen = agent.agent_workflow(mock_ctx, message)

        try:
            next(gen)  # call_llm
            gen.send({"tool_calls": tool_calls_turn1})  # -> call_llm
            gen.send({"role": "assistant", "content": "done"})  # finalize_workflow
            gen.send(None)  # → StopIteration
        except StopIteration:
            pass

        return _tool_replies(agent, mock_ctx)

    # ------------------------------------------------------------------ #
    # Deny                                                                 #
    # ------------------------------------------------------------------ #

    def test_deny_synthesizes_denial_message_run_tool_not_called(
        self, mock_llm, mock_ctx
    ):
        """Hook returning Deny → denial ToolMessage for the next call_llm, run_tool skipped."""
        tc = _tool_call(name="DropTable", args={"table": "logs"})

        def hook(ctx: HookContext):
            if ctx.step_name == "DropTable":
                return Deny(reason="schema changes go through DBA review")
            return Proceed()

        agent = _make_agent(mock_llm, hooks=Hooks(before_tool_call=[hook]))
        results = self._drive_workflow_deny_skip(agent, mock_ctx, hook, [tc])

        called_fns = [c[0][0] for c in mock_ctx.call_activity.call_args_list]
        assert agent._activity_name(agent.run_tool) not in called_fns

        assert len(results) == 1
        assert results[0]["role"] == "tool"
        assert "not executed" in results[0]["content"]
        assert "DBA review" in results[0]["content"]
        assert results[0]["tool_call_id"] == tc["id"]

    @pytest.mark.parametrize(
        "other_decision, expected_runs, other_content",
        [
            (Proceed(), [{"table": "tmp"}], "ran"),
            (Mutate(payload={"table": "safe"}), [{"table": "safe"}], "ran"),
            (Skip(result="cached"), [], "cached"),
        ],
        ids=["proceed", "mutate", "skip"],
    )
    def test_deny_applies_only_to_its_call_when_ids_are_shared(
        self, mock_llm, mock_ctx, other_decision, expected_runs, other_content
    ):
        """A decision for one call must not apply to another call with the same tool_call_id."""
        calls = [
            _tool_call(name="DropTable", args={"table": "logs"}, call_id=""),
            _tool_call(name="DropTable", args={"table": "tmp"}, call_id=""),
        ]

        def hook(ctx: HookContext):
            if ctx.payload.get("table") == "logs":
                return Deny(reason="blocked")
            return other_decision

        agent = _make_agent(mock_llm, hooks=Hooks(before_tool_call=[hook]))
        gen = agent.agent_workflow(mock_ctx, {"task": "do something"})
        next(gen)  # call_llm
        gen.send({"tool_calls": calls})  # -> run_tool, or call_llm
        if expected_runs:
            gen.send(
                {
                    "role": "tool",
                    "name": "DropTable",
                    "tool_call_id": "",
                    "content": "ran",
                }
            )  # -> call_llm
        gen.close()

        run_tool_inputs = [
            c[1]["input"]
            for c in mock_ctx.call_activity.call_args_list
            if c[0][0] == agent._activity_name(agent.run_tool)
        ]
        assert [
            json.loads(i["tool_call"]["function"]["arguments"]) for i in run_tool_inputs
        ] == expected_runs
        results = _tool_replies(agent, mock_ctx)
        assert "not executed" in results[0]["content"]
        assert results[1]["content"] == other_content

    def _drive_shared_id_approval(self, agent, mock_ctx, calls, *, approve):
        event_task = Mock(name="event")
        timer_task = Mock(name="timer")
        mock_ctx.wait_for_external_event.return_value = event_task
        mock_ctx.create_timer.return_value = timer_task
        event_task.get_result.return_value = {
            "approval_request_id": str(
                uuid.uuid5(uuid.NAMESPACE_DNS, f"{mock_ctx.instance_id}:shared")
            ),
            "approved": True,
        }

        gen = agent.agent_workflow(mock_ctx, {"task": "do something"})
        with patch("dapr.ext.workflow.when_any", return_value=Mock()):
            next(gen)  # call_llm
            gen.send({"tool_calls": calls})  # publish_approval_request
            gen.send(None)  # when_any
            gen.send(event_task if approve else timer_task)  # run_tool
            gen.send(
                {
                    "role": "tool",
                    "name": "DropTable",
                    "tool_call_id": "shared",
                    "content": "ran",
                }
            )  # -> call_llm
        gen.close()

        return [
            json.loads(c[1]["input"]["tool_call"]["function"]["arguments"])
            for c in mock_ctx.call_activity.call_args_list
            if c[0][0] == agent._activity_name(agent.run_tool)
        ]

    def test_approval_does_not_override_deny_when_ids_are_shared(
        self, mock_llm, mock_ctx
    ):
        """An approved call must not un-deny another call with the same tool_call_id."""
        calls = [
            _tool_call(name="DropTable", args={"table": "logs"}, call_id="shared"),
            _tool_call(name="DropTable", args={"table": "tmp"}, call_id="shared"),
        ]

        def hook(ctx: HookContext):
            if ctx.payload.get("table") == "logs":
                return Deny(reason="blocked")
            return RequireApproval(timeout_seconds=30)

        agent = _make_agent(mock_llm, hooks=Hooks(before_tool_call=[hook]))
        runs = self._drive_shared_id_approval(agent, mock_ctx, calls, approve=True)

        assert runs == [{"table": "tmp"}]

    def test_approval_timeout_denies_only_its_call_when_ids_are_shared(
        self, mock_llm, mock_ctx
    ):
        """A timed-out approval must not deny another call with the same tool_call_id."""
        calls = [
            _tool_call(name="DropTable", args={"table": "logs"}, call_id="shared"),
            _tool_call(name="DropTable", args={"table": "tmp"}, call_id="shared"),
        ]

        def hook(ctx: HookContext):
            if ctx.payload.get("table") == "logs":
                return RequireApproval(timeout_seconds=30)
            return Proceed()

        agent = _make_agent(mock_llm, hooks=Hooks(before_tool_call=[hook]))
        runs = self._drive_shared_id_approval(agent, mock_ctx, calls, approve=False)

        assert runs == [{"table": "tmp"}]

    @pytest.mark.parametrize(
        "order",
        [
            ("deny", "mutate", "approve"),
            ("deny", "approve", "mutate"),
            ("mutate", "deny", "approve"),
            ("mutate", "approve", "deny"),
            ("approve", "deny", "mutate"),
            ("approve", "mutate", "deny"),
        ],
        ids="-".join,
    )
    @pytest.mark.parametrize("approved", [True, False], ids=["approved", "timeout"])
    def test_three_decisions_each_apply_to_their_call_when_ids_are_shared(
        self, mock_llm, mock_ctx, order, approved
    ):
        """Deny, Mutate and an approval sharing one non-empty id each act on their own call."""
        tables = {"deny": "logs", "mutate": "orig", "approve": "tmp"}
        calls = [
            _tool_call(name="DropTable", args={"table": tables[k]}, call_id="shared")
            for k in order
        ]

        def hook(ctx: HookContext):
            table = ctx.payload.get("table")
            if table == "logs":
                return Deny(reason="blocked")
            if table == "orig":
                return Mutate(payload={"table": "safe"})
            return RequireApproval(timeout_seconds=30)

        agent = _make_agent(mock_llm, hooks=Hooks(before_tool_call=[hook]))

        def published_id():
            return mock_ctx.call_activity.call_args_list[-1][1]["input"]["event"][
                "approval_request_id"
            ]

        event_task = Mock(name="event")
        timer_task = Mock(name="timer")
        mock_ctx.wait_for_external_event.return_value = event_task
        mock_ctx.create_timer.return_value = timer_task
        event_task.get_result.side_effect = lambda: {
            "approval_request_id": published_id(),
            "approved": True,
        }

        # calls that run, in call order, with the arguments they run with
        expected_runs = [
            {"table": "safe"} if k == "mutate" else {"table": "tmp"}
            for k in order
            if k == "mutate" or (k == "approve" and approved)
        ]
        tool_result = {
            "role": "tool",
            "name": "DropTable",
            "tool_call_id": "shared",
            "content": "ran",
        }

        gen = agent.agent_workflow(mock_ctx, {"task": "do something"})
        with patch("dapr.ext.workflow.when_any", return_value=Mock()):
            next(gen)  # call_llm
            gen.send({"tool_calls": calls})  # publish_approval_request
            gen.send(published_id())  # when_any
            gen.send(event_task if approved else timer_task)  # first run_tool
            for _ in expected_runs:
                gen.send(tool_result)  # next run_tool, then call_llm
        gen.close()

        run_args = [
            json.loads(c[1]["input"]["tool_call"]["function"]["arguments"])
            for c in mock_ctx.call_activity.call_args_list
            if c[0][0] == agent._activity_name(agent.run_tool)
        ]
        assert run_args == expected_runs
        results = _tool_replies(agent, mock_ctx)
        assert len(results) == 3
        assert all(r["tool_call_id"] == "shared" for r in results)
        for kind, result in zip(order, results):
            if kind == "deny":
                assert "not executed" in result["content"]
                assert "blocked" in result["content"]
            elif kind == "approve" and not approved:
                assert "not executed" in result["content"]
                assert "not granted or timed out" in result["content"]
            else:
                assert result["content"] == "ran"

    def test_deny_in_one_turn_does_not_carry_to_next_turn_with_same_id(
        self, mock_llm, mock_ctx
    ):
        """A Deny for a tool_call_id on turn 1 must not block a call reusing that id on turn 2."""
        turn1 = [_tool_call(name="DropTable", args={"table": "logs"}, call_id="shared")]
        turn2 = [_tool_call(name="DropTable", args={"table": "tmp"}, call_id="shared")]

        def hook(ctx: HookContext):
            if ctx.payload.get("table") == "logs":
                return Deny(reason="blocked")
            return Proceed()

        agent = _make_agent(mock_llm, hooks=Hooks(before_tool_call=[hook]))
        gen = agent.agent_workflow(mock_ctx, {"task": "do something"})
        next(gen)  # call_llm (turn 1)
        gen.send({"tool_calls": turn1})  # -> call_llm
        gen.send({"tool_calls": turn2})  # run_tool
        gen.send(
            {
                "role": "tool",
                "name": "DropTable",
                "tool_call_id": "shared",
                "content": "ran",
            }
        )  # -> call_llm
        gen.close()

        run_args = [
            json.loads(c[1]["input"]["tool_call"]["function"]["arguments"])
            for c in mock_ctx.call_activity.call_args_list
            if c[0][0] == agent._activity_name(agent.run_tool)
        ]
        saves = [_tool_replies(agent, mock_ctx, turn) for turn in (1, 2)]
        assert run_args == [{"table": "tmp"}]
        assert "not executed" in saves[0][0]["content"]
        assert saves[1][0]["content"] == "ran"

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
        results = self._drive_workflow_deny_skip(agent, mock_ctx, hook, [tc])

        called_fns = [c[0][0] for c in mock_ctx.call_activity.call_args_list]
        assert agent._activity_name(agent.run_tool) not in called_fns

        assert len(results) == 1
        assert results[0]["content"] == "cached-42"
        assert results[0]["tool_call_id"] == tc["id"]

    # ------------------------------------------------------------------ #
    # RequireApproval — timer wins → denial message                        #
    # ------------------------------------------------------------------ #

    def test_require_approval_timer_wins_denial_message(self, mock_llm, mock_ctx):
        """
        When RequireApproval is returned and the timer wins the race, the tool
        is not executed and a denial ToolMessage is handed to the next call_llm.

        Yield sequence (timer path):
          Y0  call_llm            → {"tool_calls": [tc]}
          Y1  publish_approval_request (inside _request_approval)
          Y2  dt.when_any         → send timer_task (winner)
          Y3  call_llm            (timer won → denial → no run_tool) → final response
          Y4  finalize_workflow   → StopIteration
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

        with patch("dapr.ext.workflow.when_any", return_value=Mock()):
            next(gen)  # Y0: call_llm
            gen.send({"tool_calls": [tc]})  # Y1: publish_approval_request
            gen.send(None)  # Y2: when_any
            gen.send(timer_task)  # Y3: call_llm
            try:
                gen.send({"role": "assistant", "content": "done"})  # Y4: finalize
                gen.send(None)
            except StopIteration:
                pass

        results = _tool_replies(agent, mock_ctx)
        assert len(results) == 1
        assert results[0]["role"] == "tool"
        assert results[0]["tool_call_id"] == tc["id"]
        called_fns = [c[0][0] for c in mock_ctx.call_activity.call_args_list]
        assert agent._activity_name(agent.run_tool) not in called_fns

    # ------------------------------------------------------------------ #
    # RequireApproval — event wins → tool runs                             #
    # ------------------------------------------------------------------ #

    def test_require_approval_approved_tool_runs(self, mock_llm, mock_ctx):
        """
        When RequireApproval is returned and the external event arrives first
        (approved=True), run_tool is called for that tool call.

        Yield sequence (approval path):
          Y0  call_llm            → {"tool_calls": [tc]}
          Y1  publish_approval_request
          Y2  dt.when_any         → send event_task (winner)
          Y3  run_tool            → send tool_result
          Y4  call_llm            → final response
          Y5  finalize_workflow   → StopIteration
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

        with patch("dapr.ext.workflow.when_any", return_value=Mock()):
            next(gen)  # Y0: call_llm
            gen.send({"tool_calls": [tc]})  # Y1: publish_approval_request
            gen.send(None)  # Y2: when_any
            gen.send(event_task)  # Y3: run_tool (event wins → approved)
            gen.send(tool_result)  # Y4: call_llm
            try:
                gen.send({"role": "assistant", "content": "done"})  # Y5: finalize
                gen.send(None)
            except StopIteration:
                pass

        called_fns = [c[0][0] for c in mock_ctx.call_activity.call_args_list]
        assert agent._activity_name(agent.run_tool) in called_fns

    # ------------------------------------------------------------------ #
    # Mixed: denied + proceed in same turn                                 #
    # ------------------------------------------------------------------ #

    def test_mixed_denied_and_proceed_same_turn(self, mock_llm, mock_ctx):
        """
        Two tool calls in one LLM turn: one Denied, one Proceed.
        Only the Proceed tool reaches run_tool; both replies reach the next call_llm.

        Yield sequence (SEQUENTIAL mode):
          Y0  call_llm           → [tc_deny, tc_run]
          Y1  run_tool tc_run    (tc_deny handled sync by Deny branch)
          Y2  call_llm           → final response
          Y3  finalize_workflow  → StopIteration
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

        try:
            next(gen)  # Y0: call_llm
            gen.send({"tool_calls": [tc_deny, tc_run]})  # Y1: run_tool tc_run
            gen.send(run_tool_result)  # Y2: call_llm
            gen.send({"role": "assistant", "content": "done"})  # Y3: finalize
            gen.send(None)
        except StopIteration:
            pass

        results = _tool_replies(agent, mock_ctx)
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
            next(gen)  # call_llm
            gen.send({"tool_calls": [tc]})  # -> call_llm
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

        ctx = ToolHookContext(
            step_name="AnyTool", source="local", payload={}, tool_call_id="c-1"
        )
        d1 = passthrough(ctx)
        d2 = passthrough(ctx)
        assert isinstance(d1, Proceed)
        assert isinstance(d2, Proceed)
