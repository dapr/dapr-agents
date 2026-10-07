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

"""Tests for DurableAgent execution configuration."""

from unittest.mock import MagicMock, Mock

import pytest

from dapr_agents.agents.constants import (
    AGENT_DEFAULT_MAX_ITERATIONS,
    AGENT_DEFAULT_TOOL_CHOICE,
    AGENT_DEFAULT_TOOL_EXECUTION_MODE,
)
from tests.conftest import MockDaprClient
from dapr_agents.agents.configs import (
    AgentApprovalConfig,
    AgentExecutionConfig,
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentStateConfig,
    BuiltinTool,
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
            "DAPR_AGENTS_MAX_ITERATIONS",
            "DAPR_AGENTS_TOOL_CHOICE",
            "DAPR_AGENTS_TOOL_EXECUTION_MODE",
            "DAPR_GRPC_MAX_INBOUND_MESSAGE_SIZE_BYTES",
        ):
            monkeypatch.delenv(key, raising=False)

        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

        # Mock DaprClient with no runtime config
        mock_client = MockDaprClient()
        monkeypatch.setattr(
            "dapr_agents.agents.base.DaprClient", lambda **kwargs: mock_client
        )
        monkeypatch.setattr(
            "dapr_agents.storage.daprstores.base.default_dapr_client_factory",
            lambda: mock_client,
        )

        # Mock metadata models to avoid schema validation failures during initialization
        monkeypatch.setattr(
            "dapr_agents.agents.base.AgentMetadata",
            lambda **kwargs: MagicMock(**kwargs),
        )
        monkeypatch.setattr(
            "dapr_agents.agents.base.AgentMetadataSchema",
            lambda **kwargs: MagicMock(**kwargs),
        )
        monkeypatch.setattr(
            "dapr_agents.agents.base.AgentBase.register_agentic_system", Mock()
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
        """
        Patch DaprClient creation to return the provided mock client.
        Used by runtime config tests to inject custom runtime config values.
        """
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

    def test_execution_config_defaults(self, mock_llm, mock_tool):
        """
        Test that execution defaults are preserved when none of the configuration sources are available.
        """
        agent = self._make_agent(mock_llm, tools=[mock_tool])

        assert agent.execution.max_iterations == AGENT_DEFAULT_MAX_ITERATIONS
        assert agent.execution.tool_choice == AGENT_DEFAULT_TOOL_CHOICE
        assert agent.execution.tool_execution_mode == AGENT_DEFAULT_TOOL_EXECUTION_MODE
        assert agent.execution.orchestration_mode is None
        assert isinstance(
            agent.execution.approval, AgentApprovalConfig
        )  # test_hitl.py covers approval config defaults
        assert agent.execution.max_approval_rounds is None
        assert agent.execution.max_grpc_inbound_message_size_bytes is None
        assert agent.execution.streaming is False
        assert agent.execution.stream_listener is None
        assert not agent.execution.builtin_tools


class TestExecutionConfigFromInstantiation(ExecutionConfigTestBase):
    """Test cases for execution config passed during DurableAgent instantiation."""

    def test_execution_config_from_instantiation_all_fields(self, mock_llm, mock_tool):
        """Test execution config passed during instantiation."""
        execution_config = AgentExecutionConfig(
            max_iterations=5,
            tool_choice=ToolChoice.REQUIRED,
            tool_execution_mode=ToolExecutionMode.SEQUENTIAL,
            orchestration_mode=OrchestrationMode.AGENT,
            max_grpc_inbound_message_size_bytes=123456,
            approval=AgentApprovalConfig(),
            max_approval_rounds=3,
            streaming=False,
            stream_listener=None,
            builtin_tools=[BuiltinTool.ASK_USER],
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
        assert isinstance(agent.execution.approval, AgentApprovalConfig)
        assert agent.execution.max_approval_rounds == 3
        assert agent.execution.max_grpc_inbound_message_size_bytes == 123456
        assert agent.execution.streaming is False
        assert agent.execution.stream_listener is None
        assert agent.execution.builtin_tools == [BuiltinTool.ASK_USER]

    @pytest.mark.parametrize(
        ("execution_config", "expected_match"),
        [
            (
                AgentExecutionConfig(max_iterations="invalid"),
                "max_iterations",
            ),
            (
                AgentExecutionConfig(tool_execution_mode="yes"),
                "tool_execution_mode",
            ),
            (
                AgentExecutionConfig(orchestration_mode=True),
                "orchestration_mode",
            ),
            (
                AgentExecutionConfig(max_grpc_inbound_message_size_bytes="large"),
                "max_grpc_inbound_message_size_bytes",
            ),
            (
                AgentExecutionConfig(max_grpc_inbound_message_size_bytes=0),
                "max_grpc_inbound_message_size_bytes",
            ),
            (
                AgentExecutionConfig(max_grpc_inbound_message_size_bytes=-1),
                "max_grpc_inbound_message_size_bytes",
            ),
        ],
    )
    def test_execution_config_from_instantiation_raises_for_invalid_values(
        self, mock_llm, mock_tool, execution_config, expected_match
    ):
        """Test that invalid instantiated config values raise for strict fields."""
        with pytest.raises(ValueError, match=expected_match):
            self._make_agent(
                mock_llm,
                execution_config=execution_config,
                tools=[mock_tool],
            )

    def test_execution_config_from_instantiation_does_not_mutate_input_config(self):
        """Test that execution config resolution copies and leaves the caller's config untouched."""
        execution_config = AgentExecutionConfig(
            tool_choice="auto",
            tool_execution_mode="parallel",
        )

        resolved_config = AgentExecutionConfig._from_instantiation(execution_config)

        assert execution_config is not resolved_config
        assert execution_config.tool_choice == "auto"
        assert execution_config.tool_execution_mode == "parallel"
        assert resolved_config.tool_choice == ToolChoice.AUTO
        assert resolved_config.tool_execution_mode == ToolExecutionMode.PARALLEL

    @pytest.mark.parametrize(
        ("tool_choice", "tool_execution_mode"),
        [
            ("auto", "parallel"),
            ("AUTO", "PARALLEL"),
            ("Auto", "Parallel"),
        ],
    )
    def test_execution_config_from_instantiation_accepts_case_insensitive_values(
        self, tool_choice, tool_execution_mode
    ):
        """Test that execution config accepts case-insensitive instantiated values."""
        execution_config = AgentExecutionConfig(
            tool_choice=tool_choice,
            tool_execution_mode=tool_execution_mode,
        )

        resolved_config = AgentExecutionConfig._from_instantiation(execution_config)

        assert resolved_config.tool_choice == ToolChoice.AUTO
        assert resolved_config.tool_execution_mode == ToolExecutionMode.PARALLEL

    def test_execution_config_from_instantiation_accepts_non_standard_tool_choices(
        self, mock_llm, mock_tool
    ):
        """Test that non-standard instantiated tool choices are permitted."""
        execution_config = AgentExecutionConfig(
            tool_choice="all",
        )

        agent = self._make_agent(
            mock_llm,
            execution_config=execution_config,
            tools=[mock_tool],
        )

        # Tool choice should be preserved
        assert agent.execution.tool_choice == "all"

    @pytest.mark.parametrize("tool_choice", ["", " \t\r\n"])
    def test_execution_config_from_instantiation_rejects_empty_and_whitespace_only_tool_choices(
        self, mock_llm, mock_tool, tool_choice
    ):
        """Test that empty and whitespace-only instantiated tool choices are rejected."""
        execution_config = AgentExecutionConfig(tool_choice=tool_choice)

        with pytest.raises(ValueError, match="tool_choice"):
            self._make_agent(
                mock_llm,
                execution_config=execution_config,
                tools=[mock_tool],
            )


class TestExecutionConfigFromEnvironment(ExecutionConfigTestBase):
    """Test cases for execution config from environment variables."""

    def test_execution_config_from_env(self, mock_llm, mock_tool, monkeypatch):
        """Test execution config loaded from environment variables."""
        monkeypatch.setenv("DAPR_AGENTS_MAX_ITERATIONS", "7")
        monkeypatch.setenv("DAPR_AGENTS_TOOL_CHOICE", "required")
        monkeypatch.setenv("DAPR_AGENTS_TOOL_EXECUTION_MODE", "sequential")
        monkeypatch.setenv("DAPR_GRPC_MAX_INBOUND_MESSAGE_SIZE_BYTES", "654321")

        agent = self._make_agent(
            mock_llm,
            tools=[mock_tool],
        )

        assert agent.execution.max_iterations == 7
        assert agent.execution.tool_choice == ToolChoice.REQUIRED
        assert agent.execution.tool_execution_mode == ToolExecutionMode.SEQUENTIAL
        assert agent.execution.orchestration_mode is None
        assert agent.execution.max_grpc_inbound_message_size_bytes == 654321

    def test_execution_config_from_env_accepts_lowercase_keys(
        self, mock_llm, mock_tool, monkeypatch
    ):
        """Test that execution config accepts lowercase environment variable names."""
        monkeypatch.setenv("dapr_agents_max_iterations", "7")
        monkeypatch.setenv("dapr_agents_tool_choice", "required")
        monkeypatch.setenv("dapr_agents_tool_execution_mode", "sequential")
        monkeypatch.setenv("dapr_grpc_max_inbound_message_size_bytes", "654321")

        agent = self._make_agent(mock_llm, tools=[mock_tool])

        assert agent.execution.max_iterations == 7
        assert agent.execution.tool_choice == ToolChoice.REQUIRED
        assert agent.execution.tool_execution_mode == ToolExecutionMode.SEQUENTIAL
        assert agent.execution.max_grpc_inbound_message_size_bytes == 654321

    @pytest.mark.parametrize(
        ("tool_choice", "tool_execution_mode"),
        [
            ("auto", "parallel"),
            ("AUTO", "PARALLEL"),
            ("Auto", "Parallel"),
        ],
    )
    def test_execution_config_from_env_accepts_case_insensitive_values(
        self, tool_choice, tool_execution_mode, monkeypatch
    ):
        """Test that execution config accepts case-insensitive environment variables."""
        monkeypatch.setenv("DAPR_AGENTS_TOOL_CHOICE", tool_choice)
        monkeypatch.setenv("DAPR_AGENTS_TOOL_EXECUTION_MODE", tool_execution_mode)

        resolved_config = AgentExecutionConfig._from_env()

        assert resolved_config.tool_choice == ToolChoice.AUTO
        assert resolved_config.tool_execution_mode == ToolExecutionMode.PARALLEL

    def test_execution_config_from_env_ignores_invalid_values(
        self, mock_llm, mock_tool, monkeypatch
    ):
        """Test that invalid environment variable values are ignored."""
        monkeypatch.setenv("DAPR_AGENTS_MAX_ITERATIONS", "zero")
        monkeypatch.setenv("DAPR_AGENTS_TOOL_EXECUTION_MODE", "sideways")
        monkeypatch.setenv("DAPR_GRPC_MAX_INBOUND_MESSAGE_SIZE_BYTES", "abc")

        agent = self._make_agent(
            mock_llm,
            tools=[mock_tool],
        )

        # All invalid values should be ignored and defaults should be used
        assert agent.execution.max_iterations == AGENT_DEFAULT_MAX_ITERATIONS
        assert agent.execution.tool_execution_mode == AGENT_DEFAULT_TOOL_EXECUTION_MODE
        assert agent.execution.max_grpc_inbound_message_size_bytes is None

    def test_execution_config_from_env_accepts_non_standard_tool_choices(
        self, mock_llm, mock_tool, monkeypatch
    ):
        """Test that non-standard environment variable tool choices are permitted."""
        monkeypatch.setenv("DAPR_AGENTS_TOOL_CHOICE", "tool")

        agent = self._make_agent(
            mock_llm,
            tools=[mock_tool],
        )

        # Tool choice should be preserved
        assert agent.execution.tool_choice == "tool"

        # All other fields should resolve to defaults since they were not provided
        assert agent.execution.max_iterations == AGENT_DEFAULT_MAX_ITERATIONS
        assert agent.execution.tool_execution_mode == AGENT_DEFAULT_TOOL_EXECUTION_MODE
        assert agent.execution.orchestration_mode is None
        assert agent.execution.max_grpc_inbound_message_size_bytes is None

    @pytest.mark.parametrize("tool_choice", ["", " \t\r\n"])
    def test_execution_config_from_env_ignores_empty_and_whitespace_only_tool_choices(
        self, mock_llm, mock_tool, monkeypatch, tool_choice
    ):
        """Test that empty and whitespace-only tool choices from environment variables are ignored."""
        monkeypatch.setenv("DAPR_AGENTS_TOOL_CHOICE", tool_choice)

        agent = self._make_agent(mock_llm, tools=[mock_tool])

        assert agent.execution.tool_choice == AGENT_DEFAULT_TOOL_CHOICE


class TestExecutionConfigFromStateStore(ExecutionConfigTestBase):
    """Test cases for execution config from runtime configuration."""

    def test_execution_config_from_statestore(self, mock_llm, mock_tool, monkeypatch):
        """Test execution config loaded from runtime configuration."""
        runtime_config = {
            "MAX_ITERATIONS": "9",
            "TOOL_CHOICE": "any",
        }

        mock_client = MockDaprClient(runtime_config=runtime_config)
        self._patch_dapr_client(monkeypatch, mock_client)

        agent = self._make_agent(mock_llm, tools=[mock_tool])

        assert agent.execution.max_iterations == 9
        assert agent.execution.tool_choice == ToolChoice.ANY
        assert agent.execution.tool_execution_mode == ToolExecutionMode.PARALLEL
        assert agent.execution.orchestration_mode is None
        assert agent.execution.max_grpc_inbound_message_size_bytes is None

    def test_execution_config_from_statestore_accepts_lowercase_keys(
        self, mock_llm, mock_tool, monkeypatch
    ):
        """Test that execution config accepts lowercase runtime config keys."""
        runtime_config = {
            "max_iterations": "9",
            "tool_choice": "none",
        }

        mock_client = MockDaprClient(runtime_config=runtime_config)
        self._patch_dapr_client(monkeypatch, mock_client)

        agent = self._make_agent(mock_llm, tools=[mock_tool])

        assert agent.execution.max_iterations == 9
        assert agent.execution.tool_choice == ToolChoice.NONE

    @pytest.mark.parametrize(
        ("tool_choice"),
        [
            ("auto"),
            ("AUTO"),
            ("Auto"),
        ],
    )
    def test_execution_config_from_statestore_accepts_case_insensitive_values(
        self, tool_choice
    ):
        """Test that execution config accepts case-insensitive runtime config values."""
        runtime_config = {
            "TOOL_CHOICE": tool_choice,
        }

        resolved_config = AgentExecutionConfig._from_statestore(runtime_config)

        assert resolved_config.tool_choice == ToolChoice.AUTO

    def test_execution_config_from_statestore_ignores_invalid_values(
        self, mock_llm, mock_tool, monkeypatch
    ):
        """Test that invalid runtime config values are ignored."""
        runtime_config = {
            "MAX_ITERATIONS": "two",
        }

        mock_client = MockDaprClient(runtime_config=runtime_config)
        self._patch_dapr_client(monkeypatch, mock_client)

        agent = self._make_agent(mock_llm, tools=[mock_tool])

        # All invalid values should be ignored and defaults should be used
        assert agent.execution.max_iterations == AGENT_DEFAULT_MAX_ITERATIONS
        assert agent.execution.tool_choice == AGENT_DEFAULT_TOOL_CHOICE
        assert agent.execution.tool_execution_mode == AGENT_DEFAULT_TOOL_EXECUTION_MODE
        assert agent.execution.orchestration_mode is None
        assert agent.execution.max_grpc_inbound_message_size_bytes is None

    def test_execution_config_from_statestore_accepts_non_standard_tool_choices(
        self, mock_llm, mock_tool, monkeypatch
    ):
        """Test that non-standard runtime config tool choices are permitted."""
        runtime_config = {
            "TOOL_CHOICE": "no",
        }

        mock_client = MockDaprClient(runtime_config=runtime_config)
        self._patch_dapr_client(monkeypatch, mock_client)

        agent = self._make_agent(mock_llm, tools=[mock_tool])

        # Tool choice should be preserved
        assert agent.execution.tool_choice == "no"

        # All other fields should resolve to defaults since they were not provided
        assert agent.execution.max_iterations == AGENT_DEFAULT_MAX_ITERATIONS
        assert agent.execution.tool_execution_mode == AGENT_DEFAULT_TOOL_EXECUTION_MODE
        assert agent.execution.orchestration_mode is None
        assert agent.execution.max_grpc_inbound_message_size_bytes is None

    @pytest.mark.parametrize("tool_choice", ["", " \t\r\n"])
    def test_execution_config_from_statestore_ignores_empty_and_whitespace_only_tool_choices(
        self, mock_llm, mock_tool, monkeypatch, tool_choice
    ):
        """Test that empty and whitespace-only runtime config tool choices are ignored."""
        runtime_config = {"TOOL_CHOICE": tool_choice}

        mock_client = MockDaprClient(runtime_config=runtime_config)
        self._patch_dapr_client(monkeypatch, mock_client)

        agent = self._make_agent(mock_llm, tools=[mock_tool])

        assert agent.execution.tool_choice == AGENT_DEFAULT_TOOL_CHOICE


class TestExecutionConfigPrecedence(ExecutionConfigTestBase):
    """Test execution config precedence with across sources."""

    def test_execution_config_statestore_over_instantiation(
        self, mock_llm, mock_tool, monkeypatch
    ):
        """Test runtime > instantiation precedence."""
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
            orchestration_mode=None,
        )

        agent = self._make_agent(
            mock_llm,
            execution_config=execution_config,
            tools=[mock_tool],
        )

        # Runtime config should override instantiation for max_iterations and tool_choice
        assert agent.execution.max_iterations == 8
        assert agent.execution.tool_choice == ToolChoice.REQUIRED

        # Instantiation should provide tool_execution_mode and orchestration_mode
        assert agent.execution.tool_execution_mode == ToolExecutionMode.PARALLEL
        assert agent.execution.orchestration_mode is None

        # All other fields should resolve to defaults since they were not provided
        assert agent.execution.max_grpc_inbound_message_size_bytes is None

    def test_execution_config_partial_instantiation_preserves_env(
        self, mock_llm, mock_tool, monkeypatch
    ):
        """Test that fields omitted from instantiation do not override environment variables."""
        monkeypatch.setenv("DAPR_AGENTS_MAX_ITERATIONS", "3")
        monkeypatch.setenv("DAPR_AGENTS_TOOL_CHOICE", "auto")
        monkeypatch.setenv("DAPR_AGENTS_TOOL_EXECUTION_MODE", "sequential")
        monkeypatch.setenv("DAPR_GRPC_MAX_INBOUND_MESSAGE_SIZE_BYTES", "4000000")

        execution_config = AgentExecutionConfig(
            tool_choice=ToolChoice.REQUIRED,
            orchestration_mode=OrchestrationMode.AGENT,
        )

        agent = self._make_agent(
            mock_llm,
            execution_config=execution_config,
            tools=[mock_tool],
        )

        assert agent.execution.max_iterations == 3
        assert agent.execution.tool_choice == ToolChoice.REQUIRED
        assert agent.execution.tool_execution_mode == ToolExecutionMode.SEQUENTIAL
        assert agent.execution.orchestration_mode == OrchestrationMode.AGENT
        assert agent.execution.max_grpc_inbound_message_size_bytes == 4000000

    def test_execution_config_instantiation_over_env(
        self, mock_llm, mock_tool, monkeypatch
    ):
        """Test instantiation > environment variable precedence."""
        monkeypatch.setenv("DAPR_AGENTS_MAX_ITERATIONS", "2")
        monkeypatch.setenv("DAPR_AGENTS_TOOL_CHOICE", "auto")
        monkeypatch.setenv("DAPR_AGENTS_TOOL_EXECUTION_MODE", "sequential")

        execution_config = AgentExecutionConfig(
            max_iterations=5,
            tool_choice=ToolChoice.NONE,
            tool_execution_mode=ToolExecutionMode.PARALLEL,
            orchestration_mode=OrchestrationMode.RANDOM,
        )

        agent = self._make_agent(
            mock_llm,
            execution_config=execution_config,
            tools=[mock_tool],
        )

        # Instantiation should override environment variables
        # for max_iterations, tool_choice, and tool_execution_mode
        assert agent.execution.max_iterations == 5
        assert agent.execution.tool_choice == ToolChoice.NONE
        assert agent.execution.tool_execution_mode == ToolExecutionMode.PARALLEL

        # Instantiation should provide orchestration_mode
        assert agent.execution.orchestration_mode == OrchestrationMode.RANDOM

        # All other fields should resolve to defaults since they were not provided
        assert agent.execution.max_grpc_inbound_message_size_bytes is None

    def test_execution_config_statestore_over_env(
        self, mock_llm, mock_tool, monkeypatch
    ):
        """Test runtime > environment variable precedence."""
        monkeypatch.setenv("DAPR_AGENTS_MAX_ITERATIONS", "3")
        monkeypatch.setenv("DAPR_AGENTS_TOOL_CHOICE", "required")
        monkeypatch.setenv("DAPR_AGENTS_TOOL_EXECUTION_MODE", "sequential")
        monkeypatch.setenv("DAPR_GRPC_MAX_INBOUND_MESSAGE_SIZE_BYTES", "604")

        runtime_config = {
            "MAX_ITERATIONS": "6",
            "TOOL_CHOICE": "any",
        }
        mock_client = MockDaprClient(runtime_config=runtime_config)
        self._patch_dapr_client(monkeypatch, mock_client)

        agent = self._make_agent(
            mock_llm,
            tools=[mock_tool],
        )

        # Runtime config should override environment variables
        # for max_iterations and tool_choice
        assert agent.execution.max_iterations == 6
        assert agent.execution.tool_choice == ToolChoice.ANY

        # Environment variables should provide
        # tool_execution_mode and max_grpc_inbound_message_size_bytes
        assert agent.execution.tool_execution_mode == ToolExecutionMode.SEQUENTIAL
        assert agent.execution.max_grpc_inbound_message_size_bytes == 604

        # All other fields should resolve to defaults since they were not provided
        assert agent.execution.orchestration_mode is None

    def test_execution_config_full_precedence(self, mock_llm, mock_tool, monkeypatch):
        """Test runtime > instantiation > environment variable precedence."""
        monkeypatch.setenv("DAPR_AGENTS_MAX_ITERATIONS", "1")
        monkeypatch.setenv("DAPR_AGENTS_TOOL_CHOICE", "none")
        monkeypatch.setenv("DAPR_AGENTS_TOOL_EXECUTION_MODE", "sequential")
        monkeypatch.setenv("DAPR_GRPC_MAX_INBOUND_MESSAGE_SIZE_BYTES", "604")

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
            orchestration_mode=OrchestrationMode.AGENT,
            max_grpc_inbound_message_size_bytes=121212,
        )

        agent = self._make_agent(
            mock_llm,
            execution_config=execution_config,
            tools=[mock_tool],
        )

        # Runtime config should override instantiation and environment variables
        # for max_iterations and tool_choice
        assert agent.execution.max_iterations == 2
        assert agent.execution.tool_choice == ToolChoice.AUTO

        # Instantiation should override environment variables
        # for tool_execution_mode and max_grpc_inbound_message_size_bytes
        assert agent.execution.tool_execution_mode == ToolExecutionMode.PARALLEL
        assert agent.execution.max_grpc_inbound_message_size_bytes == 121212

        # Instantiation should provide orchestration_mode
        assert agent.execution.orchestration_mode == OrchestrationMode.AGENT

    def test_execution_config_full_precedence_partial_overlap(
        self, mock_llm, mock_tool, monkeypatch
    ):
        """Test partial overlap where each source contributes different fields."""
        monkeypatch.setenv("DAPR_AGENTS_MAX_ITERATIONS", "4")
        monkeypatch.setenv("DAPR_AGENTS_TOOL_CHOICE", "none")
        monkeypatch.setenv("DAPR_GRPC_MAX_INBOUND_MESSAGE_SIZE_BYTES", "111111")

        runtime_config = {
            "MAX_ITERATIONS": "8",
            "TOOL_CHOICE": "required",
        }
        mock_client = MockDaprClient(runtime_config=runtime_config)
        self._patch_dapr_client(monkeypatch, mock_client)

        execution_config = AgentExecutionConfig(
            max_iterations=5,
            tool_choice=ToolChoice.AUTO,
            tool_execution_mode=ToolExecutionMode.SEQUENTIAL,
            max_grpc_inbound_message_size_bytes=None,
        )

        agent = self._make_agent(
            mock_llm,
            execution_config=execution_config,
            tools=[mock_tool],
        )

        # Runtime config should override instantiation for max_iterations and tool_choice
        assert agent.execution.max_iterations == 8
        assert agent.execution.tool_choice == ToolChoice.REQUIRED

        # Tool execution mode resolves from instantiation since runtime config does not provide it
        assert agent.execution.tool_execution_mode == ToolExecutionMode.SEQUENTIAL

        # max_grpc_inbound_message_size_bytes should resolve from environment since neither runtime nor instantiation provide it
        assert agent.execution.max_grpc_inbound_message_size_bytes == 111111

        # All other fields should resolve to defaults since they were not provided
        assert agent.execution.orchestration_mode is None
