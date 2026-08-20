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

"""Execution config resolution tests for DurableAgent."""

from unittest.mock import Mock

from pydantic import ValidationError
import pytest

from dapr_agents.agents.constants import (
    AGENT_DEFAULT_MAX_ITERATIONS,
    AGENT_DEFAULT_TOOL_CHOICE,
    AGENT_DEFAULT_TOOL_EXECUTION_MODE,
)
from tests.conftest import MockDaprClient
from dapr_agents.agents.configs import (
    AgentExecutionConfig,
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
)
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.llm import OpenAIChatClient
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.tool.base import AgentTool
from dapr_agents.types.agent import OrchestrationMode, ToolChoice, ToolExecutionMode


class ExecutionConfigTestBase:
    """Shared fixtures and helpers for DurableAgent execution config tests."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Set up environment variables and mocks for testing."""
        for key in (
            "MAX_ITERATIONS",
            "TOOL_CHOICE",
            "TOOL_EXECUTION_MODE",
            "ORCHESTRATION_MODE",
            "MAX_GRPC_INBOUND_MESSAGE_SIZE_BYTES",
        ):
            monkeypatch.delenv(key, raising=False)

        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        monkeypatch.setattr(
            "dapr_agents.agents.base.AgentBase._setup_agent_observability", Mock()
        )
        yield

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        mock = Mock(spec=OpenAIChatClient)
        mock.prompt_template = None
        mock.__class__.__name__ = "MockLLMClient"
        mock.provider = "MockOpenAIProvider"
        mock.api = "MockOpenAIAPI"
        mock.model = "gpt-4o-mock"
        return mock

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool so tool_choice survives resolution."""
        tool = Mock(spec=AgentTool)
        tool.name = "test_tool"
        tool.description = "A test tool"
        tool.run = Mock(return_value="test_result")
        tool._is_async = False
        return tool

    def _patch_dapr_client(self, monkeypatch, mock_client):
        """Patch DaprClient creation to return the provided mock client."""
        monkeypatch.setattr(
            "dapr_agents.agents.base.DaprClient", lambda **kwargs: mock_client
        )
        monkeypatch.setattr(
            "dapr_agents.storage.daprstores.base.default_dapr_client_factory",
            lambda: mock_client,
        )

    def _make_agent(self, mock_llm, execution_config=None, tools=None):
        """Create a DurableAgent with the standard test wiring."""
        return DurableAgent(
            name="TestAgent",
            role="Test Assistant",
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
            execution=execution_config,
            tools=tools,
        )


class TestExecutionConfigDefaults(ExecutionConfigTestBase):
    """Test default execution config resolution through DurableAgent."""

    def test_execution_config_defaults(self, mock_llm, mock_tool, monkeypatch):
        """
        Test defaults are preserved when none of the configuration sources are available.
        """
        mock_client = MockDaprClient()
        self._patch_dapr_client(monkeypatch, mock_client)

        agent = self._make_agent(mock_llm, tools=[mock_tool])

        assert agent.execution.max_iterations == AGENT_DEFAULT_MAX_ITERATIONS
        assert agent.execution.tool_choice == AGENT_DEFAULT_TOOL_CHOICE
        assert agent.execution.tool_execution_mode == AGENT_DEFAULT_TOOL_EXECUTION_MODE
        assert agent.execution.orchestration_mode is None
        assert agent.execution.max_grpc_inbound_message_size_bytes is None


class TestExecutionConfigFromInstantiation(ExecutionConfigTestBase):
    """Test cases for execution config passed during DurableAgent instantiation."""

    def test_execution_config_from_instantiation(
        self, mock_llm, mock_tool, monkeypatch
    ):
        """Test execution config passed during instantiation."""
        mock_client = MockDaprClient()
        self._patch_dapr_client(monkeypatch, mock_client)

        execution_config = AgentExecutionConfig(
            max_iterations=5,
            tool_choice=ToolChoice.REQUIRED,
            tool_execution_mode=ToolExecutionMode.SEQUENTIAL,
            orchestration_mode=OrchestrationMode.AGENT,
            max_grpc_inbound_message_size_bytes=123456,
        )

        agent = self._make_agent(
            mock_llm,
            execution_config=execution_config,
            tools=[mock_tool],
        )

        assert agent.execution.max_iterations == 5
        assert agent.execution.tool_choice == ToolChoice.REQUIRED
        assert agent.execution.tool_execution_mode == ToolExecutionMode.SEQUENTIAL
        assert agent.execution.orchestration_mode == OrchestrationMode.AGENT
        assert agent.execution.max_grpc_inbound_message_size_bytes == 123456

    def test_execution_config_from_instantiation_invalid_values_ignored(
        self, mock_llm, mock_tool, monkeypatch
    ):
        """Test that an invalid max iterations value raises an exception."""
        mock_client = MockDaprClient()
        self._patch_dapr_client(monkeypatch, mock_client)

        execution_config = AgentExecutionConfig(
            max_iterations="invalid",
            tool_choice=42,
            tool_execution_mode="yes",
            orchestration_mode=True,
            max_grpc_inbound_message_size_bytes="large",
        )

        agent = self._make_agent(
            mock_llm,
            execution_config=execution_config,
            tools=[mock_tool],
        )

        assert agent.execution.max_iterations == AGENT_DEFAULT_MAX_ITERATIONS
        assert agent.execution.tool_choice == AGENT_DEFAULT_TOOL_CHOICE
        assert agent.execution.tool_execution_mode == AGENT_DEFAULT_TOOL_EXECUTION_MODE
        assert agent.execution.orchestration_mode is None
        assert agent.execution.max_grpc_inbound_message_size_bytes is None


    def test_execution_config_from_instantiation_permissive_tool_choice(
        self, mock_llm, mock_tool, monkeypatch
    ):
        """Specifically test that a non-standard instantiated tool choice is preserved."""
        mock_client = MockDaprClient()
        self._patch_dapr_client(monkeypatch, mock_client)

        execution_config = AgentExecutionConfig(
            tool_choice="all",
        )

        agent = self._make_agent(
            mock_llm,
            execution_config=execution_config,
            tools=[mock_tool],
        )

        assert agent.execution.max_iterations == AGENT_DEFAULT_MAX_ITERATIONS
        assert agent.execution.tool_choice == "all"
        assert agent.execution.tool_execution_mode == AGENT_DEFAULT_TOOL_EXECUTION_MODE
        assert agent.execution.orchestration_mode is None
        assert agent.execution.max_grpc_inbound_message_size_bytes is None


class TestExecutionConfigFromEnvironment(ExecutionConfigTestBase):
    """Test cases for execution config from environment variables."""

    def test_execution_config_from_env(self, mock_llm, mock_tool, monkeypatch):
        """Test execution config loaded from environment variables."""
        monkeypatch.setenv("MAX_ITERATIONS", "7")
        monkeypatch.setenv("TOOL_CHOICE", "required")
        monkeypatch.setenv("TOOL_EXECUTION_MODE", "sequential")
        monkeypatch.setenv("ORCHESTRATION_MODE", "agent")
        monkeypatch.setenv("MAX_GRPC_INBOUND_MESSAGE_SIZE_BYTES", "654321")

        mock_client = MockDaprClient()
        self._patch_dapr_client(monkeypatch, mock_client)

        agent = self._make_agent(
            mock_llm,
            tools=[mock_tool],
        )

        assert agent.execution.max_iterations == 7
        assert agent.execution.tool_choice == ToolChoice.REQUIRED
        assert agent.execution.tool_execution_mode == ToolExecutionMode.SEQUENTIAL
        assert agent.execution.orchestration_mode == OrchestrationMode.AGENT
        assert agent.execution.max_grpc_inbound_message_size_bytes == 654321

    def test_execution_config_from_env_invalid_values_ignored(
        self, mock_llm, mock_tool, monkeypatch
    ):
        """Test invalid environment variable values are ignored."""
        monkeypatch.setenv("MAX_ITERATIONS", "zero")
        monkeypatch.setenv("TOOL_EXECUTION_MODE", "sideways")
        monkeypatch.setenv("ORCHESTRATION_MODE", "upward")
        monkeypatch.setenv("MAX_GRPC_INBOUND_MESSAGE_SIZE_BYTES", "abc")

        mock_client = MockDaprClient()
        self._patch_dapr_client(monkeypatch, mock_client)

        agent = self._make_agent(
            mock_llm,
            tools=[mock_tool],
        )

        assert agent.execution.max_iterations == AGENT_DEFAULT_MAX_ITERATIONS
        assert agent.execution.tool_choice == AGENT_DEFAULT_TOOL_CHOICE
        assert agent.execution.tool_execution_mode == AGENT_DEFAULT_TOOL_EXECUTION_MODE
        assert agent.execution.orchestration_mode is None
        assert agent.execution.max_grpc_inbound_message_size_bytes is None

    def test_execution_config_from_env_permissive_tool_choice(
        self, mock_llm, mock_tool, monkeypatch
    ):
        """Specifically test that a non-standard environment variable tool choice is preserved."""
        monkeypatch.setenv("TOOL_CHOICE", "tool")

        mock_client = MockDaprClient()
        self._patch_dapr_client(monkeypatch, mock_client)

        agent = self._make_agent(
            mock_llm,
            tools=[mock_tool],
        )

        assert agent.execution.max_iterations == AGENT_DEFAULT_MAX_ITERATIONS
        assert agent.execution.tool_choice == "tool"
        assert agent.execution.tool_execution_mode == AGENT_DEFAULT_TOOL_EXECUTION_MODE
        assert agent.execution.orchestration_mode is None
        assert agent.execution.max_grpc_inbound_message_size_bytes is None


class TestExecutionConfigFromStateStore(ExecutionConfigTestBase):
    """Test cases for execution config from runtime configuration."""

    def test_execution_config_from_statestore(self, mock_llm, mock_tool, monkeypatch):
        """Test execution config loaded from runtime configuration."""
        runtime_config = {
            "MAX_ITERATIONS": "9",
            "TOOL_CHOICE": "none",
        }

        mock_client = MockDaprClient(runtime_config=runtime_config)
        self._patch_dapr_client(monkeypatch, mock_client)

        agent = self._make_agent(mock_llm, tools=[mock_tool])

        assert agent.execution.max_iterations == 9
        assert agent.execution.tool_choice == ToolChoice.NONE
        assert agent.execution.tool_execution_mode == AGENT_DEFAULT_TOOL_EXECUTION_MODE
        assert agent.execution.orchestration_mode is None
        assert agent.execution.max_grpc_inbound_message_size_bytes is None

    def test_execution_config_from_statestore_invalid_values_ignored(
        self, mock_llm, mock_tool, monkeypatch
    ):
        """Test invalid runtime values are ignored."""
        runtime_config = {
            "MAX_ITERATIONS": "two",
        }

        mock_client = MockDaprClient(runtime_config=runtime_config)
        self._patch_dapr_client(monkeypatch, mock_client)

        agent = self._make_agent(mock_llm, tools=[mock_tool])

        assert agent.execution.max_iterations == AGENT_DEFAULT_MAX_ITERATIONS
        assert agent.execution.tool_choice == AGENT_DEFAULT_TOOL_CHOICE
        assert agent.execution.tool_execution_mode == AGENT_DEFAULT_TOOL_EXECUTION_MODE
        assert agent.execution.orchestration_mode is None
        assert agent.execution.max_grpc_inbound_message_size_bytes is None

    def test_execution_config_from_statestore_permissive_tool_choice(
        self, mock_llm, mock_tool, monkeypatch
    ):
        """Specifically test that a non-standard runtime tool choice is preserved."""
        runtime_config = {
            "TOOL_CHOICE": "no",
        }

        mock_client = MockDaprClient(runtime_config=runtime_config)
        self._patch_dapr_client(monkeypatch, mock_client)

        agent = self._make_agent(mock_llm, tools=[mock_tool])

        assert agent.execution.max_iterations == AGENT_DEFAULT_MAX_ITERATIONS
        assert agent.execution.tool_choice == "no"
        assert agent.execution.tool_execution_mode == AGENT_DEFAULT_TOOL_EXECUTION_MODE
        assert agent.execution.orchestration_mode is None
        assert agent.execution.max_grpc_inbound_message_size_bytes is None


class TestExecutionConfigPrecedence(ExecutionConfigTestBase):
    """Test execution config precedence with across sources."""

    def test_execution_config_statestore_over_instantiation(
        self, mock_llm, mock_tool, monkeypatch
    ):
        """Test  runtime > instantiation precedence."""
        runtime_config = {
            "MAX_ITERATIONS": "8",
            "TOOL_CHOICE": "required",
        }
        mock_client = MockDaprClient(runtime_config=runtime_config)
        self._patch_dapr_client(monkeypatch, mock_client)

        execution_config = AgentExecutionConfig(
            max_iterations=4,
            tool_choice=ToolChoice.AUTO,
            tool_execution_mode=ToolExecutionMode.PARALLEL,
        )

        agent = self._make_agent(
            mock_llm,
            execution_config=execution_config,
            tools=[mock_tool],
        )

        assert agent.execution.max_iterations == 8
        assert agent.execution.tool_choice == ToolChoice.REQUIRED
        assert agent.execution.tool_execution_mode == ToolExecutionMode.PARALLEL
        assert agent.execution.orchestration_mode is None
        assert agent.execution.max_grpc_inbound_message_size_bytes is None

    def test_execution_config_instantiation_over_env(
        self, mock_llm, mock_tool, monkeypatch
    ):
        """Test instantiation > environment variable precedence."""
        monkeypatch.setenv("MAX_ITERATIONS", "2")
        monkeypatch.setenv("TOOL_CHOICE", "auto")
        monkeypatch.setenv("TOOL_EXECUTION_MODE", "sequential")

        mock_client = MockDaprClient()
        self._patch_dapr_client(monkeypatch, mock_client)

        execution_config = AgentExecutionConfig(
            max_iterations=5,
            tool_choice=ToolChoice.NONE,
            tool_execution_mode=ToolExecutionMode.PARALLEL,
        )

        agent = self._make_agent(
            mock_llm,
            execution_config=execution_config,
            tools=[mock_tool],
        )

        assert agent.execution.max_iterations == 5
        assert agent.execution.tool_choice == ToolChoice.NONE
        assert agent.execution.tool_execution_mode == ToolExecutionMode.PARALLEL
        assert agent.execution.orchestration_mode is None
        assert agent.execution.max_grpc_inbound_message_size_bytes is None

    def test_execution_config_statestore_over_env(
        self, mock_llm, mock_tool, monkeypatch
    ):
        """Test runtime > environment variable precedence."""
        monkeypatch.setenv("MAX_ITERATIONS", "3")
        monkeypatch.setenv("TOOL_CHOICE", "auto")
        monkeypatch.setenv("TOOL_EXECUTION_MODE", "parallel")

        mock_client = MockDaprClient()
        self._patch_dapr_client(monkeypatch, mock_client)

        execution_config = AgentExecutionConfig(
            max_iterations=6,
            tool_choice=ToolChoice.NONE,
            tool_execution_mode=ToolExecutionMode.SEQUENTIAL,
        )

        agent = self._make_agent(
            mock_llm,
            execution_config=execution_config,
            tools=[mock_tool],
        )

        assert agent.execution.max_iterations == 6
        assert agent.execution.tool_choice == ToolChoice.NONE
        assert agent.execution.tool_execution_mode == ToolExecutionMode.SEQUENTIAL
        assert agent.execution.orchestration_mode is None
        assert agent.execution.max_grpc_inbound_message_size_bytes is None

    def test_execution_config_full_precedence(self, mock_llm, mock_tool, monkeypatch):
        """Test runtime > instantiation > environment variable precedence."""
        monkeypatch.setenv("MAX_ITERATIONS", "1")
        monkeypatch.setenv("TOOL_CHOICE", "none")
        monkeypatch.setenv("TOOL_EXECUTION_MODE", "sequential")

        runtime_config = {
            "MAX_ITERATIONS": "2",
            "TOOL_CHOICE": "auto",
        }
        mock_client = MockDaprClient(runtime_config=runtime_config)
        self._patch_dapr_client(monkeypatch, mock_client)

        execution_config = AgentExecutionConfig(
            max_iterations=3,
            tool_choice=ToolChoice.REQUIRED,
            tool_execution_mode=ToolExecutionMode.PARALLEL,
        )

        agent = self._make_agent(
            mock_llm,
            execution_config=execution_config,
            tools=[mock_tool],
        )

        assert agent.execution.max_iterations == 2
        assert agent.execution.tool_choice == ToolChoice.AUTO
        assert agent.execution.tool_execution_mode == ToolExecutionMode.PARALLEL
        assert agent.execution.orchestration_mode is None
        assert agent.execution.max_grpc_inbound_message_size_bytes is None

    def test_execution_config_full_precedence_partial_overlap(
        self, mock_llm, mock_tool, monkeypatch
    ):
        """Test partial overlap where each source contributes different fields."""
        monkeypatch.setenv("MAX_ITERATIONS", "4")
        monkeypatch.setenv("TOOL_CHOICE", "none")
        monkeypatch.setenv("TOOL_EXECUTION_MODE", "sequential")
        monkeypatch.setenv("ORCHESTRATION_MODE", "random")
        monkeypatch.setenv("MAX_GRPC_INBOUND_MESSAGE_SIZE_BYTES", "111111")

        runtime_config = {
            "MAX_ITERATIONS": "8",
            "TOOL_CHOICE": "required",
        }
        mock_client = MockDaprClient(runtime_config=runtime_config)
        self._patch_dapr_client(monkeypatch, mock_client)

        execution_config = AgentExecutionConfig(
            max_iterations=5,
            tool_choice=ToolChoice.AUTO,
            tool_execution_mode=None,
            orchestration_mode=None,
            max_grpc_inbound_message_size_bytes=222222,
        )

        agent = self._make_agent(
            mock_llm,
            execution_config=execution_config,
            tools=[mock_tool],
        )

        assert agent.execution.max_iterations == 8
        assert agent.execution.tool_choice == ToolChoice.REQUIRED
        assert agent.execution.tool_execution_mode == ToolExecutionMode.SEQUENTIAL
        assert agent.execution.orchestration_mode == OrchestrationMode.RANDOM
        assert agent.execution.max_grpc_inbound_message_size_bytes == 222222
