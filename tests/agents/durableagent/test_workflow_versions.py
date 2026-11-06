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

"""The agents' workflows are registered as named versions, and the
history-conversation patch picks each run's workflow body.

See "Changing workflow code" in docs/development/README.md.
"""

from typing import Any, Dict, Tuple
from unittest.mock import MagicMock, patch

import pytest

from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentPubSubConfig,
    AgentStateConfig,
    OrchestrationMode,
)
from dapr_agents.agents.durable import DurableAgent, orchestration_workflow_id
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.types.exceptions import DaprRuntimeVersionNotSupportedError


def _agent(broadcast_topic=None, execution=None) -> DurableAgent:
    return DurableAgent(
        name="Versioned",
        role="Tester",
        goal="Test",
        instructions=["Be brief"],
        pubsub=AgentPubSubConfig(pubsub_name="p", broadcast_topic=broadcast_topic),
        state=AgentStateConfig(store=StateStoreService(store_name="s")),
        execution=execution,
    )


def _registered(agent: DurableAgent) -> Dict[str, Tuple[str, bool, Any]]:
    runtime = MagicMock()
    agent.register_workflows(runtime)
    assert runtime.register_workflow.call_count == 0
    return {
        c.kwargs["name"]: (c.kwargs["version_name"], c.kwargs["is_latest"], c.args[0])
        for c in runtime.register_versioned_workflow.call_args_list
    }


def test_agent_workflow_is_registered_as_latest_v1():
    agent = _agent()
    registered = _registered(agent)

    assert set(registered) == {agent.agent_workflow_name}
    version, latest, fn = registered[agent.agent_workflow_name]
    assert (version, latest) == ("v1", True)
    assert fn.__name__ == agent.agent_workflow_name
    assert fn.__wrapped__ == agent.agent_workflow


def test_broadcast_and_orchestration_are_versioned_when_configured():
    agent = _agent(
        broadcast_topic="team",
        execution=AgentExecutionConfig(orchestration_mode=OrchestrationMode.AGENT),
    )
    registered = _registered(agent)

    assert set(registered) == {
        agent.agent_workflow_name,
        agent.broadcast_workflow_name,
        orchestration_workflow_id(agent.name, infra=agent._infra),
    }
    assert {v[:2] for v in registered.values()} == {("v1", True)}


def test_only_the_last_version_is_latest():
    agent = _agent()
    runtime = MagicMock()

    agent._register_versions(
        runtime, "wf", (("v1", agent.agent_workflow), ("v2", agent.on_broadcast))
    )

    assert [
        (c.kwargs["name"], c.kwargs["version_name"], c.kwargs["is_latest"])
        for c in runtime.register_versioned_workflow.call_args_list
    ] == [("wf", "v1", False), ("wf", "v2", True)]


# ---------------------------------------------------------------------------
# The history-conversation patch
# ---------------------------------------------------------------------------


def _workflow_ctx(patched: bool) -> MagicMock:
    ctx = MagicMock()
    ctx.instance_id = "wf-1"
    ctx.is_replaying = False
    ctx.is_patched.return_value = patched
    ctx.current_utc_datetime.isoformat.return_value = "2026-10-27T00:00:00"
    return ctx


def _first_activity(agent: DurableAgent, ctx: MagicMock) -> str:
    next(agent.agent_workflow(ctx, {"task": "hi"}))
    return ctx.call_activity.call_args.args[0].rsplit(".", 1)[-1]


@pytest.mark.parametrize(
    "patched, first", [(False, "record_initial_entry"), (True, "call_llm")]
)
def test_patch_picks_the_agent_workflow_body(patched, first):
    ctx = _workflow_ctx(patched)
    assert _first_activity(_agent(), ctx) == first
    ctx.is_patched.assert_called_once_with("history-conversation")


def test_patch_is_checked_before_anything_is_scheduled():
    ctx = _workflow_ctx(True)
    calls = MagicMock()
    calls.attach_mock(ctx.is_patched, "is_patched")
    calls.attach_mock(ctx.call_activity, "call_activity")
    _first_activity(_agent(), ctx)
    assert [c[0] for c in calls.mock_calls][:2] == ["is_patched", "call_activity"]


@pytest.mark.parametrize("patched", [False, True])
def test_patch_picks_the_orchestration_body(patched):
    agent = _agent(
        execution=AgentExecutionConfig(orchestration_mode=OrchestrationMode.AGENT)
    )

    def legacy(ctx, message):
        return "legacy"
        yield  # pragma: no cover

    ctx = _workflow_ctx(patched)
    with patch.object(agent, "_orchestration_workflow_legacy", side_effect=legacy):
        gen = agent.orchestration_workflow(ctx, {"task": "hi"})
        try:
            next(gen)
            took_legacy = False
        except StopIteration as stop:
            took_legacy = stop.value == "legacy"
    assert took_legacy is not patched


# An activity scheduled by a run on the old path can execute after the upgrade:
# it gets the old input and must run the old body. Payloads as main sent them.
OLD_INPUTS = [
    (
        "call_llm",
        {"task": "t", "instance_id": "i", "time": "x", "source": None, "turn": 1},
    ),
    ("summarize", {}),
    (
        "finalize_workflow",
        {
            "instance_id": "i",
            "final_output": "o",
            "end_time": "x",
            "triggering_workflow_instance_id": None,
            "final_message": {},
        },
    ),
    ("publish_stream_event", {"instance_id": "i", "event_type": "turn_paused"}),
    (
        "publish_stream_event",
        {
            "instance_id": "i",
            "event_type": "user_input_requested",
            "stream_context": None,
        },
    ),
    (
        "save_plan",
        {"instance_id": "i", "plan_message": {"role": "assistant"}, "time": "x"},
    ),
]


@pytest.mark.parametrize("activity, payload", OLD_INPUTS)
def test_old_activity_inputs_run_the_old_body(activity, payload):
    agent = _agent()
    with patch.object(agent, f"_{activity}_legacy", return_value="legacy") as legacy:
        assert getattr(agent, activity)(MagicMock(), payload) == "legacy"
    legacy.assert_called_once()


def test_old_run_executor_input_runs_the_old_consumer():
    agent = _agent()
    agent.executor = MagicMock()
    with (
        patch.object(
            agent, "_consume_executor_legacy", new=MagicMock(return_value="legacy")
        ) as legacy,
        patch.object(agent, "_run_asyncio_task", side_effect=lambda result: result),
    ):
        old = {"task": "t", "instance_id": "i", "source": None}
        assert agent.run_executor(MagicMock(), old) == "legacy"
    legacy.assert_called_once_with(old)


@pytest.mark.parametrize(
    "version, supported",
    [
        ("1.16.4", False),
        ("1.17.0", True),
        ("1.17.0-rc.1", True),
        ("edge", True),
        (None, True),
    ],
)
def test_register_workflows_needs_dapr_1_17(version, supported):
    agent = _agent()
    agent._dapr_runtime_version = version
    if supported:
        agent.register_workflows(MagicMock())
    else:
        with pytest.raises(DaprRuntimeVersionNotSupportedError, match="1.17"):
            agent.register_workflows(MagicMock())
