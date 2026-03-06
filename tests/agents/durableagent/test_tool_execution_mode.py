"""Tests for ToolExecutionMode configuration and behaviour."""

import os
from datetime import timedelta
from typing import Optional
from unittest.mock import Mock, MagicMock, AsyncMock

import pytest

from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentMemoryConfig,
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
    ToolExecutionMode,
)
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.llm import OpenAIChatClient
from dapr_agents.memory import ConversationDaprStateMemory
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.tool.base import AgentTool


# ---------------------------------------------------------------------------
# Fixtures (mirrors test_durable_agent.py patterns)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_dapr_check(monkeypatch):
    """Mock Dapr dependencies to prevent requiring a running Dapr instance."""
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
            self.max_number_of_attempts = max_number_of_attempts
            self.first_retry_interval = first_retry_interval
            self.max_retry_interval = max_retry_interval
            self.backoff_coefficient = backoff_coefficient
            self.retry_timeout = retry_timeout

    monkeypatch.setattr(wf, "RetryPolicy", MockRetryPolicy)
    yield mock_runtime


class MockDaprClient:
    """Mock DaprClient that supports context manager protocol."""

    def __init__(self, http_timeout_seconds=10):
        self.get_state = MagicMock(return_value=Mock(data=None, json=lambda: {}))
        self.save_state = MagicMock()
        self.delete_state = MagicMock()
        self.query_state = MagicMock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __call__(self, *args, **kwargs):
        return self


# ---------------------------------------------------------------------------
# Enum / Config tests
# ---------------------------------------------------------------------------


class TestToolExecutionModeEnum:
    """Test the ToolExecutionMode enum definition."""

    def test_parallel_value(self):
        assert ToolExecutionMode.PARALLEL == "parallel"

    def test_sequential_value(self):
        assert ToolExecutionMode.SEQUENTIAL == "sequential"

    def test_construct_from_string_parallel(self):
        assert ToolExecutionMode("parallel") is ToolExecutionMode.PARALLEL

    def test_construct_from_string_sequential(self):
        assert ToolExecutionMode("sequential") is ToolExecutionMode.SEQUENTIAL

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ToolExecutionMode("invalid")

    def test_is_str_subclass(self):
        assert isinstance(ToolExecutionMode.PARALLEL, str)


class TestAgentExecutionConfigToolMode:
    """Test ToolExecutionMode integration in AgentExecutionConfig."""

    def test_default_is_parallel(self):
        config = AgentExecutionConfig()
        assert config.tool_execution_mode is ToolExecutionMode.PARALLEL

    def test_set_sequential(self):
        config = AgentExecutionConfig(tool_execution_mode=ToolExecutionMode.SEQUENTIAL)
        assert config.tool_execution_mode is ToolExecutionMode.SEQUENTIAL

    def test_set_parallel_explicit(self):
        config = AgentExecutionConfig(tool_execution_mode=ToolExecutionMode.PARALLEL)
        assert config.tool_execution_mode is ToolExecutionMode.PARALLEL

    def test_other_defaults_unchanged(self):
        config = AgentExecutionConfig(tool_execution_mode=ToolExecutionMode.SEQUENTIAL)
        assert config.max_iterations == 10
        assert config.tool_choice == "auto"
        assert config.orchestration_mode is None


# ---------------------------------------------------------------------------
# DurableAgent integration tests
# ---------------------------------------------------------------------------


class TestDurableAgentToolExecutionMode:
    """Test that DurableAgent respects tool_execution_mode."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        os.environ["OPENAI_API_KEY"] = "test-api-key"
        mock_client = MockDaprClient()
        mock_client.get_state.return_value = Mock(data=None)
        monkeypatch.setattr("dapr.clients.DaprClient", lambda: mock_client)
        monkeypatch.setattr(
            "dapr_agents.storage.daprstores.statestore.DaprClient", lambda: mock_client
        )
        yield
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

    @pytest.fixture
    def mock_llm(self):
        mock = Mock(spec=OpenAIChatClient)
        mock.generate = AsyncMock()
        mock.prompt_template = None
        mock.__class__.__name__ = "MockLLMClient"
        mock.provider = "MockOpenAIProvider"
        mock.api = "MockOpenAIAPI"
        mock.model = "gpt-4o-mock"
        return mock

    @pytest.fixture
    def mock_tool(self):
        tool = Mock(spec=AgentTool)
        tool.name = "test_tool"
        tool.description = "A test tool"
        tool.run = AsyncMock(return_value="test_result")
        tool._is_async = True
        return tool

    def _make_agent(self, mock_llm, execution_config, mock_tool=None):
        """Helper to create a DurableAgent with a given execution config."""
        tools = [mock_tool] if mock_tool else []
        return DurableAgent(
            name="TestAgent",
            role="Test Assistant",
            goal="Test",
            instructions=["Be helpful"],
            llm=mock_llm,
            pubsub=AgentPubSubConfig(
                pubsub_name="testpubsub",
                agent_topic="TestAgent",
            ),
            state=AgentStateConfig(
                store=StateStoreService(store_name="teststatestore")
            ),
            registry=AgentRegistryConfig(
                store=StateStoreService(store_name="testregistry")
            ),
            memory=AgentMemoryConfig(
                store=ConversationDaprStateMemory(
                    store_name="teststatestore",
                    workflow_instance_id="test_session",
                )
            ),
            tools=tools,
            execution=execution_config,
        )

    def test_default_mode_is_parallel(self, mock_llm):
        agent = self._make_agent(mock_llm, AgentExecutionConfig())
        assert agent.execution.tool_execution_mode is ToolExecutionMode.PARALLEL

    def test_sequential_mode_stored(self, mock_llm):
        config = AgentExecutionConfig(tool_execution_mode=ToolExecutionMode.SEQUENTIAL)
        agent = self._make_agent(mock_llm, config)
        assert agent.execution.tool_execution_mode is ToolExecutionMode.SEQUENTIAL

    def test_parallel_mode_stored(self, mock_llm):
        config = AgentExecutionConfig(tool_execution_mode=ToolExecutionMode.PARALLEL)
        agent = self._make_agent(mock_llm, config)
        assert agent.execution.tool_execution_mode is ToolExecutionMode.PARALLEL

    @pytest.mark.parametrize(
        ("mode", "expected_value"),
        [
            (ToolExecutionMode.PARALLEL, "parallel"),
            (ToolExecutionMode.SEQUENTIAL, "sequential"),
        ],
    )
    def test_mode_string_values(self, mode, expected_value):
        assert str(mode) == expected_value
