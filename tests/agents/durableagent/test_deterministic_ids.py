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
from unittest.mock import MagicMock, Mock

import pytest
from dapr.ext.workflow import DaprWorkflowContext

from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    OrchestrationMode,
    ToolExecutionMode,
)
from dapr_agents.agents.durable import DurableAgent, child_workflow_instance_id

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


def _make_ctx(instance_id: str = PARENT_INSTANCE_ID) -> DaprWorkflowContext:
    """A fresh workflow context, as the runtime builds for each replay."""
    ctx = DaprWorkflowContext()
    ctx.instance_id = instance_id
    ctx.is_replaying = False
    ctx.current_utc_datetime = ORCHESTRATION_TIME
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


def _run_agent_turn(agent: DurableAgent, tool_calls: List[Dict[str, Any]]):
    """Drive agent_workflow up to the agent-as-tool dispatch of turn 1."""
    ctx = _make_ctx()
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
