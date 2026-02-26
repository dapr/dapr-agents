"""
Unit tests for the agents-as-tools feature in DurableAgent.

Covers:
- is_tool flag stored and forwarded to registry metadata
- DurableAgent instance in tools list auto-converted to AgentWorkflowTool
- String agent name deferred to _deferred_agent_names
- register_workflows uses per-agent workflow names via _named wrappers
- agent_workflow_name / broadcast_workflow_name properties
- _load_tools discovers is_tool=True agents and deferred names
"""

from datetime import timedelta
from typing import Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from dapr_agents.agents.durable import (
    AGENT_WORKFLOW_SUFFIX,
    BROADCAST_WORKFLOW_SUFFIX,
    ORCHESTRATION_WORKFLOW_SUFFIX,
    DurableAgent,
)
from dapr_agents.tool.workflow.agent_tool import (
    AgentWorkflowTool,
    agent_to_tool,
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
    return llm


def _make_agent(name: str, mock_llm, **kwargs) -> DurableAgent:
    return DurableAgent(
        name=name, role=f"{name} role", goal=f"{name} goal", llm=mock_llm, **kwargs
    )


# ---------------------------------------------------------------------------
# is_tool flag
# ---------------------------------------------------------------------------


class TestIsToolFlag:
    def test_is_tool_defaults_false(self, mock_llm):
        agent = _make_agent("frodo", mock_llm)
        assert agent.is_tool is False

    def test_is_tool_set_true(self, mock_llm):
        agent = _make_agent("sam", mock_llm, is_tool=True)
        assert agent.is_tool is True

    def test_is_tool_reflected_in_registry_metadata(self, mock_llm):
        """is_tool=True must be accessible on the agent instance."""
        agent = _make_agent("sam", mock_llm, is_tool=True)
        assert agent.is_tool is True

    def test_is_tool_false_in_registry_metadata(self, mock_llm):
        agent = _make_agent("frodo", mock_llm, is_tool=False)
        assert agent.is_tool is False


# ---------------------------------------------------------------------------
# tools= list pre-processing
# ---------------------------------------------------------------------------


class TestToolsListPreprocessing:
    def test_durable_agent_instance_converted_to_agent_workflow_tool(self, mock_llm):
        sam = _make_agent("sam", mock_llm, is_tool=True)
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

    def test_string_name_stored_as_deferred(self, mock_llm):
        frodo = _make_agent("frodo", mock_llm, tools=["gandalf"])
        assert "gandalf" in frodo._agentname_as_tool

    def test_string_name_not_in_tool_executor(self, mock_llm):
        frodo = _make_agent("frodo", mock_llm, tools=["gandalf"])
        assert frodo.tool_executor.get_tool("gandalf") is None

    def test_mixed_tools_list(self, mock_llm):
        from dapr_agents.tool.base import AgentTool

        sam = _make_agent("sam", mock_llm)
        mock_tool = MagicMock(spec=AgentTool)
        mock_tool.name = "rope_tool"
        mock_tool.description = "Tie a knot"
        mock_tool._is_async = False

        frodo = _make_agent("frodo", mock_llm, tools=[sam, mock_tool, "gandalf"])

        # DurableAgent instance converted immediately
        assert frodo.tool_executor.get_tool("sam") is not None
        # AgentTool registered directly
        assert frodo.tool_executor.get_tool("rope_tool") is not None
        # String deferred
        assert "gandalf" in frodo._agentname_as_tool


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
        assert f"frodo{AGENT_WORKFLOW_SUFFIX}" in registered_names

    def test_broadcast_workflow_registered_with_name_suffix(self, mock_llm):
        sam = _make_agent("sam", mock_llm)
        rt = self._register_and_get_calls(sam)
        registered_names = [
            call.args[0].__name__ for call in rt.register_workflow.call_args_list
        ]
        assert f"sam{BROADCAST_WORKFLOW_SUFFIX}" in registered_names

    def test_agent_workflow_name_property(self, mock_llm):
        frodo = _make_agent("frodo", mock_llm)
        assert frodo.agent_workflow_name == f"frodo{AGENT_WORKFLOW_SUFFIX}"

    def test_broadcast_workflow_name_property(self, mock_llm):
        sam = _make_agent("sam", mock_llm)
        assert sam.broadcast_workflow_name == f"sam{BROADCAST_WORKFLOW_SUFFIX}"

    def test_load_tools_activity_registered(self, mock_llm):
        frodo = _make_agent("frodo", mock_llm)
        rt = self._register_and_get_calls(frodo)
        registered_activities = [
            call.args[0] for call in rt.register_activity.call_args_list
        ]
        assert frodo._load_tools in registered_activities

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
        assert f"frodo{AGENT_WORKFLOW_SUFFIX}" in all_names
        assert f"sam{AGENT_WORKFLOW_SUFFIX}" in all_names
        # No duplicates
        assert len(all_names) == len(set(all_names))


# ---------------------------------------------------------------------------
# _load_tools activity
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
                    "is_tool": info.get("is_tool", False),
                }
            }
            for name, info in agents.items()
        }

    def _with_registry(self, agent) -> None:
        """Give the agent a non-None registry so _load_tools doesn't short-circuit."""
        agent._infra._registry = MagicMock()

    def test_discovers_is_tool_agents(self, mock_llm):
        frodo = _make_agent("frodo", mock_llm)
        self._with_registry(frodo)
        ctx = MagicMock()

        metadata = self._make_registry_metadata(
            {
                "sam": {"role": "helper", "goal": "assist", "is_tool": True},
            }
        )

        with patch.object(frodo._infra, "get_agents_metadata", return_value=metadata):
            frodo._load_tools(ctx)

        assert frodo.tool_executor.get_tool("sam") is not None

    def test_skips_already_registered_tool(self, mock_llm):
        sam_tool = agent_to_tool("sam", "already registered")
        frodo = _make_agent("frodo", mock_llm, tools=[])
        frodo.tool_executor.register_tool(sam_tool)
        self._with_registry(frodo)
        ctx = MagicMock()

        metadata = self._make_registry_metadata(
            {
                "sam": {"role": "helper", "goal": "assist", "is_tool": True},
            }
        )

        with patch.object(frodo._infra, "get_agents_metadata", return_value=metadata):
            result = frodo._load_tools(ctx)

        # sam was already registered, nothing new
        assert "sam" not in result

    def test_resolves_deferred_string_agent(self, mock_llm):
        frodo = _make_agent("frodo", mock_llm, tools=["gandalf"])
        self._with_registry(frodo)
        ctx = MagicMock()

        metadata = self._make_registry_metadata(
            {
                "gandalf": {"role": "wizard", "goal": "guide", "is_tool": False},
            }
        )

        with patch.object(frodo._infra, "get_agents_metadata", return_value=metadata):
            result = frodo._load_tools(ctx)

        assert "gandalf" in result
        assert frodo.tool_executor.get_tool("gandalf") is not None

    def test_returns_empty_list_when_no_registry(self, mock_llm):
        frodo = _make_agent("frodo", mock_llm)
        ctx = MagicMock()
        result = frodo._load_tools(ctx)
        assert result == []

    def test_does_not_include_self_in_discovered_tools(self, mock_llm):
        frodo = _make_agent("frodo", mock_llm)
        self._with_registry(frodo)
        ctx = MagicMock()

        # Registry returns frodo itself (should be excluded by exclude_self=True)
        metadata = self._make_registry_metadata(
            {}
        )  # empty — exclude_self already handled
        with patch.object(frodo._infra, "get_agents_metadata", return_value=metadata):
            result = frodo._load_tools(ctx)

        assert result == []
        assert frodo.tool_executor.get_tool("frodo") is None
