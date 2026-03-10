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
Unit tests for the agents-as-tools feature in DurableAgent.

Covers:
- DurableAgent instance in tools list auto-converted to AgentWorkflowTool
- register_workflows uses per-agent workflow names via _named wrappers
- agent_workflow_name / broadcast_workflow_name properties
- load_tools discovers all registry agents (excluding orchestrators and self)
"""

from datetime import timedelta
from typing import Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from dapr_agents.agents.configs import AgentExecutionConfig, OrchestrationMode
from dapr_agents.agents.durable import (
    DurableAgent,
    broadcast_workflow_id,
)
from dapr_agents.tool.workflow.agent_tool import (
    AgentWorkflowTool,
    agent_to_tool,
    agent_workflow_id,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_dapr(monkeypatch):
    """Prevent real Dapr connections in unit tests."""
    import dapr.ext.workflow as wf

    mock_runtime = MagicMock()
    monkeypatch.setattr(wf, "WorkflowRuntime", lambda: mock_runtime)

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
    yield mock_runtime


@pytest.fixture(autouse=True)
def patch_dapr_client(monkeypatch):
    mock_client = MagicMock()
    mock_client.get_state.return_value = MagicMock(data=None)
    mock_client.get_metadata.return_value = MagicMock(
        registered_components=[], application_id="test-app"
    )
    monkeypatch.setattr(
        "dapr_agents.storage.daprstores.statestore.DaprClient", lambda: mock_client
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


def _make_agent(name: str, mock_llm, **kwargs) -> DurableAgent:
    return DurableAgent(
        name=name, role=f"{name} role", goal=f"{name} goal", llm=mock_llm, **kwargs
    )


# ---------------------------------------------------------------------------
# tools= list pre-processing
# ---------------------------------------------------------------------------


class TestToolsListPreprocessing:
    def test_durable_agent_instance_converted_to_agent_workflow_tool(self, mock_llm):
        sam = _make_agent("sam", mock_llm)
        frodo = _make_agent("frodo", mock_llm, tools=[sam])
        # get_tool normalises names (case-insensitive, strips spaces/underscores)
        tool = frodo.tool_executor.get_tool("sam")
        assert tool is not None
        assert isinstance(tool, AgentWorkflowTool)

    def test_agent_workflow_tool_name_matches_agent_name(self, mock_llm):
        sam = _make_agent("sam", mock_llm)
        frodo = _make_agent("frodo", mock_llm, tools=[sam])
        tool = frodo.tool_executor.get_tool("sam")
        assert tool.name.lower() == "sam"

    def test_mixed_tools_list(self, mock_llm):
        from dapr_agents.tool.base import AgentTool

        sam = _make_agent("sam", mock_llm)
        mock_tool = MagicMock(spec=AgentTool)
        mock_tool.name = "rope_tool"
        mock_tool.description = "Tie a knot"
        mock_tool._is_async = False

        frodo = _make_agent("frodo", mock_llm, tools=[sam, mock_tool])

        # DurableAgent instance converted immediately
        assert frodo.tool_executor.get_tool("sam") is not None
        # AgentTool registered directly
        assert frodo.tool_executor.get_tool("rope_tool") is not None


# ---------------------------------------------------------------------------
# register_workflows — unique workflow naming
# ---------------------------------------------------------------------------


class TestRegisterWorkflowsNaming:
    def _register_and_get_calls(self, agent: DurableAgent):
        """Run register_workflows with a mock runtime and capture registered names."""
        mock_runtime = MagicMock()
        agent.register_workflows(mock_runtime)
        return mock_runtime

    def test_agent_workflow_registered_with_name_suffix(self, mock_llm):
        frodo = _make_agent("frodo", mock_llm)
        rt = self._register_and_get_calls(frodo)
        registered_names = [
            call.args[0].__name__ for call in rt.register_workflow.call_args_list
        ]
        assert agent_workflow_id("frodo") in registered_names

    def test_broadcast_workflow_registered_with_name_suffix(self, mock_llm):
        sam = _make_agent("sam", mock_llm)
        rt = self._register_and_get_calls(sam)
        registered_names = [
            call.args[0].__name__ for call in rt.register_workflow.call_args_list
        ]
        assert broadcast_workflow_id("sam") in registered_names

    def test_agent_workflow_name_property(self, mock_llm):
        frodo = _make_agent("frodo", mock_llm)
        assert frodo.agent_workflow_name == agent_workflow_id("frodo")

    def test_broadcast_workflow_name_property(self, mock_llm):
        sam = _make_agent("sam", mock_llm)
        assert sam.broadcast_workflow_name == broadcast_workflow_id("sam")

    def test_load_tools_activity_registered(self, mock_llm):
        frodo = _make_agent("frodo", mock_llm)
        rt = self._register_and_get_calls(frodo)
        registered_activities = [
            call.args[0] for call in rt.register_activity.call_args_list
        ]
        assert frodo.load_tools in registered_activities

    def test_two_agents_have_unique_workflow_names(self, mock_llm):
        """Two agents must register under distinct names even on the same runtime."""
        frodo = _make_agent("frodo", mock_llm)
        sam = _make_agent("sam", mock_llm)
        rt = MagicMock()
        frodo.register_workflows(rt)
        sam.register_workflows(rt)

        all_names = [
            call.args[0].__name__ for call in rt.register_workflow.call_args_list
        ]
        assert agent_workflow_id("frodo") in all_names
        assert agent_workflow_id("sam") in all_names
        # No duplicates
        assert len(all_names) == len(set(all_names))


# ---------------------------------------------------------------------------
# load_tools activity
# ---------------------------------------------------------------------------


class TestLoadTools:
    def _make_registry_metadata(self, agents: dict) -> dict:
        """Build the dict returned by _infra.get_agents_metadata."""
        return {
            name: {
                "agent": {
                    "role": info.get("role", ""),
                    "goal": info.get("goal", ""),
                    "appid": info.get("appid", "test-app"),
                }
            }
            for name, info in agents.items()
        }

    def _with_registry(self, agent) -> None:
        """Give the agent a non-None registry so load_tools doesn't short-circuit."""
        agent._infra._registry = MagicMock()

    def test_discovers_registry_agents(self, mock_llm):
        frodo = _make_agent("frodo", mock_llm)
        self._with_registry(frodo)
        ctx = MagicMock()

        metadata = self._make_registry_metadata(
            {
                "sam": {"role": "helper", "goal": "assist"},
            }
        )

        with patch.object(frodo._infra, "get_agents_metadata", return_value=metadata):
            frodo.load_tools(ctx)

        assert frodo.tool_executor.get_tool("sam") is not None

    def test_skips_already_registered_tool(self, mock_llm):
        sam_tool = agent_to_tool("sam", "already registered")
        frodo = _make_agent("frodo", mock_llm, tools=[])
        frodo.tool_executor.register_tool(sam_tool)
        self._with_registry(frodo)
        ctx = MagicMock()

        metadata = self._make_registry_metadata(
            {
                "sam": {"role": "helper", "goal": "assist"},
            }
        )

        with patch.object(frodo._infra, "get_agents_metadata", return_value=metadata):
            result = frodo.load_tools(ctx)

        # sam was already registered, nothing new
        assert "sam" not in result

    def test_returns_empty_list_when_no_registry(self, mock_llm):
        frodo = _make_agent("frodo", mock_llm)
        ctx = MagicMock()
        result = frodo.load_tools(ctx)
        assert result == []

    def test_handles_none_metadata_gracefully(self, mock_llm):
        """Test that load_tools handles None or invalid metadata gracefully."""
        frodo = _make_agent("frodo", mock_llm)
        self._with_registry(frodo)
        ctx = MagicMock()

        # Test with None metadata
        metadata = {"sam": None}
        with patch.object(frodo._infra, "get_agents_metadata", return_value=metadata):
            result = frodo.load_tools(ctx)
            # Should skip None metadata and return empty list
            assert result == []
            assert frodo.tool_executor.get_tool("sam") is None

    def test_handles_invalid_agent_metadata_gracefully(self, mock_llm):
        """Test that load_tools handles invalid agent metadata gracefully."""
        frodo = _make_agent("frodo", mock_llm)
        self._with_registry(frodo)
        ctx = MagicMock()

        # Test with missing "agent" key
        metadata = {"sam": {"some_other_key": "value"}}
        with patch.object(frodo._infra, "get_agents_metadata", return_value=metadata):
            result = frodo.load_tools(ctx)
            # Should skip invalid metadata and return empty list
            assert result == []
            assert frodo.tool_executor.get_tool("sam") is None

        # Test with None agent metadata
        metadata = {"sam": {"agent": None}}
        with patch.object(frodo._infra, "get_agents_metadata", return_value=metadata):
            result = frodo.load_tools(ctx)
            # Should skip None agent metadata and return empty list
            assert result == []
            assert frodo.tool_executor.get_tool("sam") is None

    def test_does_not_include_self_in_discovered_tools(self, mock_llm):
        frodo = _make_agent("frodo", mock_llm)
        self._with_registry(frodo)
        ctx = MagicMock()

        # Registry returns frodo itself (should be excluded by exclude_self=True)
        metadata = self._make_registry_metadata(
            {}
        )  # empty — exclude_self already handled
        with patch.object(frodo._infra, "get_agents_metadata", return_value=metadata):
            result = frodo.load_tools(ctx)

        assert result == []
        assert frodo.tool_executor.get_tool("frodo") is None


# ---------------------------------------------------------------------------
# Orchestrator isolation — agents must not appear in tool_executor
# ---------------------------------------------------------------------------


class TestOrchestratorToolIsolation:
    def _make_orchestrator(self, name: str, mock_llm, **kwargs) -> DurableAgent:
        return DurableAgent(
            name=name,
            role=f"{name} role",
            goal=f"{name} goal",
            llm=mock_llm,
            execution=AgentExecutionConfig(orchestration_mode=OrchestrationMode.AGENT),
            **kwargs,
        )

    def _with_registry(self, agent) -> None:
        agent._infra._registry = MagicMock()

    def _make_registry_metadata(self, agents: dict) -> dict:
        return {
            name: {
                "agent": {
                    "role": info.get("role", ""),
                    "goal": info.get("goal", ""),
                    "appid": info.get("appid", "test-app"),
                }
            }
            for name, info in agents.items()
        }

    def test_load_tools_returns_empty_for_orchestrator(self, mock_llm):
        gandalf = self._make_orchestrator("gandalf", mock_llm)
        self._with_registry(gandalf)
        ctx = MagicMock()

        metadata = self._make_registry_metadata(
            {"frodo": {"role": "ring-bearer", "goal": "destroy the ring"}}
        )
        with patch.object(gandalf._infra, "get_agents_metadata", return_value=metadata):
            result = gandalf.load_tools(ctx)

        assert result == []
        assert gandalf.tool_executor.get_tool("frodo") is None

    def test_orchestrator_does_not_register_agents_as_tools_from_init(self, mock_llm):
        sam = _make_agent("sam", mock_llm)
        gandalf = self._make_orchestrator("gandalf", mock_llm, tools=[sam])

        assert gandalf.tool_executor.get_tool("sam") is None

    def test_orchestrator_is_true_for_orchestration_mode(self, mock_llm):
        gandalf = self._make_orchestrator("gandalf", mock_llm)
        assert gandalf.orchestrator is True
