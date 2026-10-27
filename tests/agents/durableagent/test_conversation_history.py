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

"""The conversation lives only in the workflow history: call_llm gets diffs and
rebuilds the rest from its cache or, on a miss, from the history."""

import json
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest

from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentPubSubConfig,
    AgentStateConfig,
    ToolExecutionMode,
)
from dapr_agents.agents.conversation import Transcript, fold_conversation
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.agents.schemas import ConversationSummary
from dapr_agents.llm import OpenAIChatClient
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.tool.base import AgentTool
from dapr_agents.tool.workflow.ask_user_tool import (
    _make_ask_user_impl,
    pending_inputs_from_custom_status,
)
from dapr_agents.types import (
    AgentError,
    AssistantMessage,
    LLMChatCandidate,
    LLMChatResponse,
)
from dapr_agents.types.message import FunctionCall, ToolCall
from tests.agents.durableagent.test_durable_executor import (  # noqa: F401
    patch_dapr_check,
    setup_env,
)

pytestmark = pytest.mark.usefixtures("history_path_only")

CALL_LLM = "dapr.agents.HistoryAgent.call_llm"


# ---------------------------------------------------------------------------
# History events: only the fields fold_conversation reads
# ---------------------------------------------------------------------------


class _Event:
    def __init__(self, event_id: int = -1, **fields: Any) -> None:
        self.eventId = event_id
        self.taskScheduled = fields.get("taskScheduled")
        self.taskCompleted = fields.get("taskCompleted")
        self.taskFailed = fields.get("taskFailed")

    def HasField(self, name: str) -> bool:
        return getattr(self, name, None) is not None


def _scheduled(event_id: int, name: str, payload: Any) -> _Event:
    task = SimpleNamespace(name=name, input=SimpleNamespace(value=json.dumps(payload)))
    return _Event(event_id, taskScheduled=task)


def _completed(event_id: int, result: Any) -> _Event:
    task = SimpleNamespace(
        taskScheduledId=event_id, result=SimpleNamespace(value=json.dumps(result))
    )
    return _Event(taskCompleted=task)


def _failed(event_id: int) -> _Event:
    return _Event(taskFailed=SimpleNamespace(taskScheduledId=event_id))


# ---------------------------------------------------------------------------
# Transcript + fold
# ---------------------------------------------------------------------------


class TestFold:
    def test_cursor_counts_what_the_fold_rebuilds(self):
        t = Transcript()
        t.add({"role": "user", "content": "hi"})
        in1 = t.payload()
        a1 = {"role": "assistant", "tool_calls": [{"id": "c1"}], "id": "A1"}
        t.advance(a1)
        t.add({"role": "tool", "tool_call_id": "c1", "content": "42"})
        in2 = t.payload()
        a2 = {"role": "assistant", "content": "done", "id": "A2"}
        t.advance(a2)

        history = [
            _scheduled(0, CALL_LLM, in1),
            _scheduled(1, "dapr.agents.HistoryAgent.run_tool", {"x": 1}),
            _completed(0, a1),
            _completed(1, {"role": "tool", "content": "ignored"}),
            _scheduled(2, CALL_LLM, in2),
            _failed(2),  # retried attempt: contributes nothing
            _scheduled(3, CALL_LLM, in2),
            _completed(3, a2),
            _scheduled(4, CALL_LLM, t.payload()),  # in flight: contributes nothing
        ]

        conversation = fold_conversation(history, CALL_LLM)

        assert [m["role"] for m in conversation] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert len(conversation) == t.length
        assert conversation[-1]["id"] == t.last_id

    def test_other_agents_call_llm_is_ignored(self):
        history = [
            _scheduled(0, "dapr.agents.Other.call_llm", Transcript().payload()),
            _completed(0, {"role": "assistant", "id": "x"}),
        ]
        assert fold_conversation(history, CALL_LLM) == []

    def test_fold_reads_real_history_protos(self):
        """The fold's field names match the pinned SDK's HistoryEvent protos.

        Runs in a clean interpreter: this suite's conftest replaces the dapr
        package with mocks, so real protos can't be built in-process.
        """
        script = textwrap.dedent(
            """
            import json
            import dapr.ext.workflow._durabletask.internal.helpers as h
            from dapr_agents.agents.conversation import fold_conversation
            name = "dapr.agents.a.call_llm"
            payload = {"new_messages": [{"role": "user", "content": "hi"}]}
            events = [
                h.new_task_scheduled_event(0, name, json.dumps(payload)),
                h.new_task_completed_event(0, json.dumps({"role": "assistant", "id": "A"})),
                h.new_task_scheduled_event(1, name, json.dumps(payload)),
                h.new_task_failed_event(1, RuntimeError("boom")),
            ]
            print(json.dumps(fold_conversation(events, name)))
            """
        )
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        )
        assert json.loads(out.stdout.strip().splitlines()[-1]) == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "id": "A"},
        ]


# ---------------------------------------------------------------------------
# An agent run with activities executed inline and the history recorded
# ---------------------------------------------------------------------------


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


ADD = AgentTool.from_func(add)


def _response(message: AssistantMessage) -> LLMChatResponse:
    return LLMChatResponse(
        results=[LLMChatCandidate(message=message, finish_reason="stop")]
    )


def _tool_call_turn(call_id: str) -> LLMChatResponse:
    return _response(
        AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id=call_id,
                    type="function",
                    function=FunctionCall(name=ADD.name, arguments='{"a": 2, "b": 3}'),
                )
            ],
        )
    )


class _ScriptedLLM:
    """Plays back responses; answers summary requests on its own."""

    def __init__(self, turns: List[LLMChatResponse]) -> None:
        self.turns = list(turns)
        self.prompts: List[List[Dict[str, Any]]] = []
        self.mock = Mock(spec=OpenAIChatClient)
        self.mock.prompt_template = None
        self.mock.provider = "mock"
        self.mock.api = "mock"
        self.mock.model = "mock"
        self.mock.generate = Mock(side_effect=self._generate)

    def _generate(self, **kwargs: Any) -> Any:
        if kwargs.get("response_format") is ConversationSummary:
            return ConversationSummary(summary="summary")
        self.prompts.append(json.loads(json.dumps(kwargs["messages"], default=str)))
        return self.turns.pop(0)


def _agent(llm: _ScriptedLLM, max_iterations: int = 8) -> DurableAgent:
    return DurableAgent(
        name="HistoryAgent",
        role="Tester",
        goal="Test",
        instructions=["Be brief"],
        llm=llm.mock,
        tools=[ADD],
        pubsub=AgentPubSubConfig(pubsub_name="p", agent_topic="HistoryAgent"),
        state=AgentStateConfig(store=StateStoreService(store_name="s")),
        execution=AgentExecutionConfig(
            max_iterations=max_iterations,
            tool_execution_mode=ToolExecutionMode.SEQUENTIAL,
        ),
    )


class _Sidecar:
    """Runs agent_workflow, executing activities inline and recording history."""

    def __init__(self, agent: DurableAgent, instance_id: str = "wf-1") -> None:
        self.agent = agent
        self.instance_id = instance_id
        self.history: List[_Event] = []
        self.inputs: Dict[int, Any] = {}
        # Committed history as the activity scheduled under each event id saw it.
        self.seen_by: Dict[int, List[_Event]] = {}
        ctx = Mock()
        ctx.instance_id = instance_id
        ctx.is_replaying = False
        ctx.is_patched.return_value = True
        ctx.current_utc_datetime.isoformat.return_value = "2026-10-26T00:00:00"
        ctx.call_activity = self._call_activity
        self.ctx = ctx
        agent._instance_history = self.committed_history

    def committed_history(self, instance_id: str) -> List[_Event]:
        assert instance_id == self.instance_id
        return list(self.history)

    def _call_activity(self, activity: str, *, input: Any = None, retry_policy=None):
        event_id = len(self.inputs)
        self.inputs[event_id] = (activity, input)
        self.history.append(_scheduled(event_id, activity, input))
        self.seen_by[event_id] = list(self.history)
        method = getattr(self.agent, activity.rsplit(".", 1)[-1])
        activity_ctx = SimpleNamespace(workflow_id=self.instance_id)
        result = (
            method(activity_ctx, input) if input is not None else method(activity_ctx)
        )
        self.history.append(_completed(event_id, result))
        return result

    def run(self, task: str) -> Any:
        gen = self.agent.agent_workflow(self.ctx, {"task": task})
        value = None
        try:
            while True:
                value = gen.send(value)
        except StopIteration as stop:
            return stop.value

    def llm_calls(self) -> List[int]:
        return [e for e, (a, _) in self.inputs.items() if a.endswith(".call_llm")]

    def llm_inputs(self) -> List[Dict[str, Any]]:
        return [self.inputs[e][1] for e in self.llm_calls()]


class TestAgentRun:
    def test_each_call_llm_carries_only_its_new_messages(self):
        turns = [_tool_call_turn(f"c{i}") for i in range(4)]
        llm = _ScriptedLLM(turns + [_response(AssistantMessage(content="20"))])
        sidecar = _Sidecar(_agent(llm))

        result = sidecar.run("add 2 and 3, four times")

        assert result["content"] == "20"
        inputs = sidecar.llm_inputs()
        assert [i["history_len"] for i in inputs] == [0, 2, 4, 6, 8]
        assert inputs[0]["new_messages"] == [
            {"role": "user", "content": "add 2 and 3, four times"}
        ]
        for i, payload in enumerate(inputs[1:]):
            (reply,) = payload["new_messages"]
            assert reply["role"] == "tool"
            assert reply["tool_call_id"] == f"c{i}"
            assert reply["content"] == "5"
        # The LLM still sees the whole conversation, built from the cache,
        # with each earlier message's name, as when it came from a stored record.
        last_prompt = [m for m in llm.prompts[-1] if m["role"] != "system"]
        assert [m["role"] for m in last_prompt] == ["user"] + ["assistant", "tool"] * 4
        assert {m["name"] for m in last_prompt if m["role"] == "assistant"} == {
            "HistoryAgent"
        }
        assert {m["name"] for m in last_prompt if m["role"] == "tool"} == {ADD.name}
        assert all("id" not in m for m in last_prompt)

    def test_cold_worker_rebuilds_the_same_prompt_from_history(self):
        llm = _ScriptedLLM(
            [_tool_call_turn("c1"), _response(AssistantMessage(content="5"))]
        )
        agent = _agent(llm)
        sidecar = _Sidecar(agent)
        sidecar.run("add 2 and 3")
        warm_prompt = llm.prompts[1]
        second_call = sidecar.llm_calls()[1]

        cold = _agent(llm)  # empty cache, as on another replica or after a restart
        cold._instance_history = Mock(return_value=sidecar.seen_by[second_call])
        llm.turns.append(_response(AssistantMessage(content="5")))
        cold.call_llm(
            SimpleNamespace(workflow_id="wf-1"), sidecar.inputs[second_call][1]
        )

        cold._instance_history.assert_called_once_with("wf-1")
        assert llm.prompts[-1] == warm_prompt


# ---------------------------------------------------------------------------
# The cache rules
# ---------------------------------------------------------------------------


class TestConversationCache:
    @pytest.fixture
    def agent(self):
        agent = _agent(_ScriptedLLM([]))
        agent._instance_history = Mock(
            return_value=[
                _scheduled(
                    0, CALL_LLM, Transcript(pending=[{"role": "user"}]).payload()
                ),
                _completed(0, {"role": "assistant", "id": "A1"}),
            ]
        )
        return agent

    def test_first_call_needs_no_history(self, agent):
        assert agent._conversation("wf-1", 0, None) == []
        agent._instance_history.assert_not_called()

    def test_hit_when_length_and_last_id_match(self, agent):
        agent._remember("wf-1", [{"role": "user"}, {"role": "assistant", "id": "A1"}])
        assert len(agent._conversation("wf-1", 2, "A1")) == 2
        agent._instance_history.assert_not_called()

    def test_stale_entry_from_a_lost_attempt_is_rebuilt(self, agent):
        """An attempt whose result was not recorded left another id behind."""
        agent._remember("wf-1", [{"role": "user"}, {"role": "assistant", "id": "LOST"}])
        assert agent._conversation("wf-1", 2, "A1")[-1]["id"] == "A1"
        agent._instance_history.assert_called_once_with("wf-1")

    def test_history_that_does_not_match_the_workflow_fails_loudly(self, agent):
        with pytest.raises(AgentError, match="holds 2 conversation messages"):
            agent._conversation("wf-1", 4, "A9")

    def test_redelivered_call_returns_the_recorded_message(self, agent):
        """At-least-once delivery: a repeat of a call that ran here skips the LLM."""
        user = {"role": "user", "content": "hi"}
        answer = {"role": "assistant", "content": "x", "id": "A1"}
        agent._remember("wf-1", [user, answer])
        payload = {"instance_id": "wf-1", "new_messages": [user], "history_len": 0}

        result = agent.call_llm(SimpleNamespace(workflow_id="wf-1"), payload)

        assert result == answer
        agent.llm.generate.assert_not_called()

    def test_a_reused_instance_id_does_not_get_the_earlier_answer(self, agent):
        """A new execution under the same instance id, starting with the same task."""
        user = {"role": "user", "content": "hi"}
        agent._remember("wf-1", [user, {"role": "assistant", "id": "A1"}], "run-1")
        assert agent._recorded_result("wf-1", 0, None, [user], "run-2") is None
        assert agent._recorded_result("wf-1", 0, None, [user], "run-1")["id"] == "A1"

    def test_summary_includes_the_last_turns_tool_replies(self, agent):
        """The last allowed turn ran tools; no call_llm followed to commit the replies."""
        reply = {"role": "tool", "tool_call_id": "c1", "content": "42"}
        payload = {
            "instance_id": "wf-1",
            "new_messages": [reply],
            "history_len": 2,
            "last_id": "A1",
        }
        with patch.object(agent, "_summarize_conversation", return_value={}) as summary:
            agent.summarize(SimpleNamespace(workflow_id="wf-1"), payload)
        messages = summary.call_args.args[1]
        assert len(messages) == 3 and messages[-1] == reply

    def test_a_different_call_is_not_mistaken_for_a_redelivery(self, agent):
        agent._remember(
            "wf-1",
            [{"role": "user", "content": "hi"}, {"role": "assistant", "id": "A1"}],
        )
        other = [{"role": "user", "content": "something else"}]
        assert agent._recorded_result("wf-1", 0, None, other) is None
        assert agent._recorded_result("wf-1", 2, "A1", []) is None


# ---------------------------------------------------------------------------
# ask_user keeps its open prompt in the custom status
# ---------------------------------------------------------------------------


class TestAskUserCustomStatus:
    def test_prompt_is_in_custom_status_only_while_waiting(self):
        ctx = Mock()
        ctx.instance_id = "wf-9"
        ctx.new_uuid = Mock(return_value="req-1")
        stream_ctx = {"root_instance_id": "wf-9", "listener_config": {"type": "pubsub"}}
        gen = _make_ask_user_impl("publish")(
            question="Which city?", ctx=ctx, _stream_context=stream_ctx
        )

        next(gen)  # publish USER_INPUT_REQUESTED
        assert (
            ctx.call_activity.call_args.kwargs["input"]["stream_context"] == stream_ctx
        )
        gen.send(None)  # now waiting for the answer
        (status,) = ctx.set_custom_status.call_args.args
        assert pending_inputs_from_custom_status(status) == [
            {
                "request_id": "req-1",
                "question": "Which city?",
                "target_instance_id": "wf-9",
                "timeout_seconds": 600,
            }
        ]

        gen.send("Paris")  # answer arrives, publish USER_INPUT_RECEIVED
        ctx.set_custom_status.assert_called_with("")

    @pytest.mark.parametrize(
        "status", [None, "", "max_iterations_reached", "[1]", '{"pending_inputs": 3}']
    )
    def test_unrelated_custom_status_has_no_pending_inputs(self, status):
        assert pending_inputs_from_custom_status(status) == []


# ---------------------------------------------------------------------------
# Workflow inputs that replaced reads of the old state record
# ---------------------------------------------------------------------------


def _name(activity: Any) -> str:
    return str(activity).rsplit(".", 1)[-1]


def _drive(gen) -> Any:
    value = None
    try:
        while True:
            value = gen.send(value)
    except StopIteration as stop:
        return stop.value


class TestWorkflowInputs:
    def test_every_stream_event_carries_the_session_stream_context(self):
        from dapr_agents.tool.workflow.agent_tool import agent_to_tool

        helper = agent_to_tool("helper", "Helper agent")
        llm = _ScriptedLLM([])
        agent = _agent(llm)
        agent.tool_executor.register_tool(helper)
        published: List[Dict[str, Any]] = []

        def call_activity(activity, *, input=None, retry_policy=None):
            if _name(activity) == "call_llm":
                if input["history_len"] == 0:
                    return {
                        "role": "assistant",
                        "id": "A1",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": helper.name,
                                    "arguments": '{"task": "help"}',
                                },
                            }
                        ],
                    }
                return {"role": "assistant", "content": "done", "id": "A2"}
            if _name(activity) == "publish_stream_event":
                published.append(input)
            return None

        ctx = Mock()
        ctx.instance_id = "wf-s"
        ctx.is_replaying = False
        ctx.current_utc_datetime.isoformat.return_value = "2026-10-26T00:00:00"
        ctx.call_activity = Mock(side_effect=call_activity)
        ctx.call_child_workflow = Mock(return_value={"content": "helped"})
        incoming = {"root_instance_id": "root", "listener_config": {"type": "pubsub"}}

        _drive(
            agent.agent_workflow(
                ctx,
                {"task": "go", "_message_metadata": {"_stream_context": incoming}},
            )
        )

        assert [p["event_type"] for p in published] == ["turn_paused", "turn_resumed"]
        for p in published:
            assert p["stream_context"]["root_instance_id"] == "root"
            assert p["stream_context"]["call_path"][-1] == "HistoryAgent"

    def test_orchestration_child_summarizes_its_own_conversation(self):
        from dapr_agents.agents.orchestration import AgentOrchestrationStrategy

        agent = _agent(_ScriptedLLM([]))
        agent._orchestration_strategy = AgentOrchestrationStrategy()
        plan = [{"step": 1, "description": "d", "status": "not_started"}]
        llm_inputs: List[Dict[str, Any]] = []
        summarized: List[Dict[str, Any]] = []
        replies = {
            "IterablePlanStep": json.dumps({"objects": plan}),
            "NextStep": json.dumps(
                {"next_agent": "a", "instruction": "go", "step": 1, "substep": None}
            ),
            "ProgressCheckOutput": "{}",
            None: "the summary",
        }

        def call_activity(activity, *, input=None, retry_policy=None):
            name = _name(activity)
            if name == "get_team_members":
                return {
                    "formatted": "- a",
                    "metadata": {"a": {"agent": {"appid": "app-a", "name": "a"}}},
                }
            if name == "call_llm":
                llm_inputs.append(input)
                fmt = input.get("response_format")
                return {
                    "role": "assistant",
                    "content": replies[fmt],
                    "id": f"L{len(llm_inputs)}",
                }
            if name == "validate_step":
                return True
            if name == "evaluate_progress":
                return {"plan": plan, "verdict": "completed"}
            if name == "summarize":
                summarized.append(input)
            return None

        ctx = Mock()
        ctx.instance_id = "parent-1:0001"
        ctx.is_replaying = False
        ctx.current_utc_datetime.isoformat.return_value = "2026-10-26T00:00:00"
        ctx.call_activity = Mock(side_effect=call_activity)
        ctx.call_child_workflow = Mock(return_value={"content": "agent a done"})

        final = _drive(
            agent.orchestration_workflow(ctx, {"task": "t", "instance_id": "parent-1"})
        )

        assert final["content"] == "the summary"
        # plan, next step (after the saved plan), progress check, summary
        assert [i["history_len"] for i in llm_inputs] == [0, 2, 5, 7]
        assert [len(i["new_messages"]) for i in llm_inputs] == [1, 2, 1, 2]
        assert summarized == [
            {
                "instance_id": "parent-1",
                "new_messages": [],
                "history_len": 10,
                "last_id": "L4",
                "execution": "2026-10-26T00:00:00",
            }
        ]
