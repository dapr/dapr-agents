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

import json
import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from dapr.ext.workflow import WorkflowActivityContext
from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentPubSubConfig,
    AgentStateConfig,
)
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.llm import OpenAIChatClient
from dapr_agents.skills import Skill, SkillResource
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.types import AssistantMessage, LLMChatCandidate, LLMChatResponse
from tests.agents.durableagent.test_durable_agent import MockDaprClient


@pytest.fixture(autouse=True)
def dapr_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-api-key"
    client = MockDaprClient()
    monkeypatch.setattr("dapr.clients.DaprClient", lambda: client)
    monkeypatch.setattr(
        "dapr_agents.storage.daprstores.base.default_dapr_client_factory",
        lambda: client,
    )
    yield
    os.environ.pop("OPENAI_API_KEY", None)


def _agent() -> DurableAgent:
    llm = Mock(spec=OpenAIChatClient)
    llm.generate = AsyncMock()
    llm.prompt_template = None
    llm.__class__.__name__ = "MockLLMClient"
    llm.provider = "MockOpenAIProvider"
    llm.api = "MockOpenAIAPI"
    llm.model = "gpt-4o-mock"
    return DurableAgent(
        name="SkillsMetadataAgent",
        role="Skill assistant",
        goal="Use skills",
        llm=llm,
        skills=[
            Skill(
                name="incident-investigation",
                description="Investigate incidents.",
                instructions="Keep this content out of registry metadata.",
                resources={
                    "reference.md": SkillResource(
                        path="reference.md",
                        content="private resource",
                        media_type="text/markdown",
                    )
                },
            )
        ],
        pubsub=AgentPubSubConfig(
            pubsub_name="testpubsub", agent_topic="SkillsMetadataAgent"
        ),
        state=AgentStateConfig(store=StateStoreService(store_name="teststatestore")),
        execution=AgentExecutionConfig(max_iterations=2),
    )


def test_agent_metadata_contains_static_skill_tool_definitions_only() -> None:
    agent = _agent()

    tool_names = {tool.name for tool in agent.agent_metadata.tools or []}
    serialized = agent.agent_metadata.model_dump_json()

    assert {"load_skill", "read_skill_resource"}.issubset(tool_names)
    assert "Keep this content out of registry metadata." not in serialized
    assert "private resource" not in serialized
    assert "_skill_runtime" not in serialized


def test_parallel_skill_tools_activate_before_resource_read() -> None:
    """Same-turn resource reads wait for matching skill activation."""
    agent = _agent()
    ctx = MagicMock()
    ctx.instance_id = "workflow-1"
    ctx.is_patched.return_value = True
    ctx.current_utc_datetime.isoformat.return_value = "2026-01-01T00:00:00+00:00"
    ctx.call_activity.side_effect = lambda *args, **kwargs: MagicMock(name=args[0])
    ctx.call_child_workflow.return_value = MagicMock()

    skill_name = "incident-investigation"
    # The dependent read deliberately precedes activation in model output.
    tool_calls = [
        {
            "id": "read",
            "function": {
                "name": "read_skill_resource",
                "arguments": json.dumps({"name": skill_name, "path": "reference.md"}),
            },
        },
        {
            "id": "load",
            "function": {
                "name": "load_skill",
                "arguments": json.dumps({"name": skill_name}),
            },
        },
    ]
    load_result = {
        "role": "tool",
        "name": "load_skill",
        "tool_call_id": "load",
        "content": "loaded",
    }
    read_result = {
        "role": "tool",
        "name": "read_skill_resource",
        "tool_call_id": "read",
        "content": "read",
    }

    workflow = agent.agent_workflow(ctx, {"task": "use skills"})
    next(workflow)
    workflow.send(None)
    workflow.send(
        [
            {
                "name": skill_name,
                "description": "Investigate incidents.",
                "_source_id": f"direct:{skill_name}",
            }
        ]
    )

    with patch("dapr.ext.workflow.when_all", side_effect=["load-phase", "read-phase"]):
        workflow.send({"tool_calls": tool_calls})

        run_tool_name = agent._activity_name(agent.run_tool)
        run_tool_calls = [
            call
            for call in ctx.call_activity.call_args_list
            if call.args[0] == run_tool_name
        ]
        assert [call.kwargs["input"]["tool_call"]["id"] for call in run_tool_calls] == [
            "load"
        ]

        workflow.send([load_result])
        run_tool_calls = [
            call
            for call in ctx.call_activity.call_args_list
            if call.args[0] == run_tool_name
        ]
        assert [call.kwargs["input"]["tool_call"]["id"] for call in run_tool_calls] == [
            "load",
            "read",
        ]

        workflow.send([read_result])

        save_tool_results_name = agent._activity_name(agent.save_tool_results)
        save_calls = [
            call
            for call in ctx.call_activity.call_args_list
            if call.args[0] == save_tool_results_name
        ]
        assert [
            result["tool_call_id"]
            for result in save_calls[-1].kwargs["input"]["tool_results"]
        ] == ["read", "load"]

    workflow.close()


def test_skill_runtime_plans_activation_before_dependent_reads() -> None:
    """The planner detects dependencies regardless of tool-call order."""
    agent = _agent()
    assert agent._skill_runtime is not None

    plan = agent._skill_runtime._skill_dispatch_plan(
        [
            {
                "function": {
                    "name": "read_skill_resource",
                    "arguments": json.dumps(
                        {"name": "incident-investigation", "path": "reference.md"}
                    ),
                }
            },
            {
                "function": {
                    "name": "load_skill",
                    "arguments": json.dumps({"name": "incident-investigation"}),
                }
            },
        ]
    )

    assert plan.activation_indexes == frozenset({1})
    assert plan.deferred_resource_indexes == frozenset({0})


def test_catalog_is_not_injected_without_a_run_catalog() -> None:
    agent = _agent()
    messages = agent.prompting_helper.build_initial_messages(chat_history=[])

    assert not any(
        message.get("content", "").startswith("# Available Skills")
        for message in messages
        if isinstance(message, dict)
    )


@pytest.mark.parametrize("patched", [False, True])
def test_agent_workflow_patches_skill_discovery_activity(patched: bool) -> None:
    agent = _agent()
    ctx = MagicMock()
    ctx.instance_id = "workflow-1"
    ctx.is_patched.return_value = patched
    ctx.current_utc_datetime.isoformat.return_value = "2026-01-01T00:00:00+00:00"
    ctx.call_activity.side_effect = lambda *args, **kwargs: MagicMock()
    ctx.call_child_workflow.return_value = MagicMock()

    workflow = agent.agent_workflow(ctx, {"task": "use skills"})
    next(workflow)
    workflow.send(None)

    activity_names = [call.args[0] for call in ctx.call_activity.call_args_list]
    discovery_name = agent._activity_name(agent.discover_skill_catalog)
    if patched:
        assert discovery_name in activity_names
        assert activity_names.index(discovery_name) == 1
    else:
        assert discovery_name not in activity_names
    workflow.close()
    ctx.is_patched.assert_called_once_with("dapr-agents-skills-v1")


def test_call_llm_without_catalog_filters_skill_tools() -> None:
    agent = _agent()
    agent.llm.generate = Mock(
        return_value=LLMChatResponse(
            results=[
                LLMChatCandidate(
                    message=AssistantMessage(content="done"), finish_reason="stop"
                )
            ]
        )
    )
    agent._infra.get_state = Mock(
        return_value=MagicMock(messages=[], stream_context=None)
    )
    agent._reconstruct_conversation_history = Mock(return_value=[])
    agent._sync_system_messages_with_state = Mock()
    agent._process_user_message = Mock()
    agent._save_assistant_message = Mock()
    agent.save_state = Mock()
    agent._get_last_user_message = Mock(return_value=None)
    agent.text_formatter = Mock()
    ctx = Mock(spec=WorkflowActivityContext)
    ctx.workflow_id = "wf-1"
    ctx.task_id = 1

    agent.call_llm(
        ctx,
        {"instance_id": "wf-1", "task": None, "source": "direct"},
    )

    tools = agent.llm.generate.call_args.kwargs.get("tools", [])
    assert {tool.name for tool in tools}.isdisjoint(
        {"load_skill", "read_skill_resource"}
    )
