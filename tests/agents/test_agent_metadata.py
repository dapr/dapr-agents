"""
Unit tests for agent metadata building logic.

These tests validate the metadata structure without requiring Dapr.
They test the pure logic of _build_agent_metadata(), _extract_component_mappings(),
and _extract_tool_definitions().
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch

from dapr_agents.agents.standalone import Agent
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.agents.configs import (
    AgentRegistryConfig,
    AgentStateConfig,
    AgentPubSubConfig,
)
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.tool.base import AgentTool
from dapr_agents.registry.metadata import AgentMetadata


def sample_tool_function(input_text: str) -> str:
    """A sample tool function for testing."""
    return f"Processed: {input_text}"


def custom_tool_callable(x: int, y: int) -> int:
    """A custom callable tool."""
    return x + y


class TestAgentMetadataBuilding:
    """Test agent metadata building without Dapr client."""

    def test_build_metadata_basic_fields(self):
        """Test that _build_agent_metadata creates correct basic fields."""
        # Don't provide agent_registry_config so no registration happens
        agent = Agent(
            name="TestAgent",
            role="Tester",
            goal="Test metadata building",
        )

        metadata = agent._build_agent_metadata()

        assert metadata is not None
        assert isinstance(metadata, AgentMetadata)
        assert metadata.name == "TestAgent"
        assert metadata.role == "Tester"
        assert metadata.goal == "Test metadata building"
        assert metadata.agent_framework == "dapr-agents"
        assert metadata.agent_class == "Agent"
        assert metadata.agent_category == "agent"
        assert metadata.agent_id is not None
        assert metadata.sub_agents == []

    def test_build_metadata_with_instructions(self):
        """Test metadata with instructions."""
        agent = Agent(
            name="InstructedAgent",
            role="Assistant",
            goal="Follow instructions",
            instructions=["Be helpful", "Be concise", "Be accurate"],
        )

        metadata = agent._build_agent_metadata()

        assert metadata.instructions == ["Be helpful", "Be concise", "Be accurate"]

    def test_build_metadata_durable_agent_category(self):
        """Test that DurableAgent has correct category."""
        # Create with all required configs but don't trigger registration
        pubsub_config = AgentPubSubConfig(pubsub_name="test-pubsub")

        agent = DurableAgent(
            name="DurableTestAgent",
            role="Durable",
            goal="Test durable category",
            pubsub_config=pubsub_config,
        )

        metadata = agent._build_agent_metadata()

        assert metadata.agent_class == "DurableAgent"
        assert metadata.agent_category == "durable-agent"


class TestComponentMappingsExtraction:
    """Test component mappings extraction logic."""

    def test_extract_components_with_state_store(self):
        """Test component extraction with state store."""
        state_store = StateStoreService(store_name="my-state-store")
        state_config = AgentStateConfig(store=state_store)

        agent = Agent(
            name="StatefulAgent",
            role="Tester",
            state_config=state_config,
        )

        components = agent._extract_component_mappings()

        assert "workflow" in components.state_stores
        assert components.state_stores["workflow"].name == "my-state-store"
        assert (
            components.state_stores["workflow"].usage
            == "Durable workflow state storage"
        )

    @patch("dapr.clients.DaprClient")
    def test_extract_components_with_agent_registry(self, mock_client_cls):
        """Test component extraction with agent registry store."""
        # Mock DaprClient as a context manager
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client_cls.return_value.__exit__.return_value = None

        # Mock the state operations
        mock_response = Mock()
        mock_response.data = json.dumps({}).encode("utf-8")
        mock_response.etag = "test-etag"
        mock_client.get_state.return_value = mock_response

        agent_registry_store = StateStoreService(store_name="agent-registry-store")
        agent_registry_config = AgentRegistryConfig(store=agent_registry_store)

        agent = Agent(
            name="RegistryAgent",
            role="Tester",
            agent_registry_config=agent_registry_config,
        )

        components = agent._extract_component_mappings()

        assert "agent_registry" in components.state_stores
        assert components.state_stores["agent_registry"].name == "agent-registry-store"
        assert (
            components.state_stores["agent_registry"].usage
            == "Agent metadata discovery registry"
        )

    def test_extract_components_with_team_registry(self):
        """Test component extraction with team registry store."""
        team_registry_store = StateStoreService(store_name="team-registry-store")
        registry_config = AgentRegistryConfig(store=team_registry_store)

        agent = Agent(
            name="TeamAgent",
            role="Tester",
            registry_config=registry_config,
        )

        components = agent._extract_component_mappings()

        assert "team_registry" in components.state_stores
        assert components.state_stores["team_registry"].name == "team-registry-store"
        assert (
            components.state_stores["team_registry"].usage
            == "Team pub/sub addressing registry"
        )

    @patch("dapr.clients.DaprClient")
    def test_extract_components_with_both_registries(self, mock_client_cls):
        """Test component extraction with both agent and team registry."""
        # Mock DaprClient as a context manager
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client_cls.return_value.__exit__.return_value = None

        # Mock the state operations
        mock_response = Mock()
        mock_response.data = json.dumps({}).encode("utf-8")
        mock_response.etag = "test-etag"
        mock_client.get_state.return_value = mock_response

        agent_registry_store = StateStoreService(store_name="agent-store")
        team_registry_store = StateStoreService(store_name="team-store")

        agent_registry_config = AgentRegistryConfig(store=agent_registry_store)
        registry_config = AgentRegistryConfig(store=team_registry_store)

        agent = Agent(
            name="DualRegistryAgent",
            role="Tester",
            agent_registry_config=agent_registry_config,
            registry_config=registry_config,
        )

        components = agent._extract_component_mappings()

        assert "agent_registry" in components.state_stores
        assert components.state_stores["agent_registry"].name == "agent-store"

        assert "team_registry" in components.state_stores
        assert components.state_stores["team_registry"].name == "team-store"

    def test_extract_components_with_pubsub(self):
        """Test component extraction with pub/sub config for DurableAgent."""
        pubsub_config = AgentPubSubConfig(
            pubsub_name="my-pubsub",
            agent_topic="agent-topic",
        )

        agent = DurableAgent(
            name="PubSubAgent",
            role="Tester",
            pubsub_config=pubsub_config,
        )

        components = agent._extract_component_mappings()

        assert "default" in components.pubsub_components
        assert components.pubsub_components["default"].name == "my-pubsub"
        assert components.pubsub_components["default"].topic_name == "agent-topic"


class TestToolDefinitionsExtraction:
    """Test tool definitions extraction logic."""

    def test_extract_tool_from_agent_tool(self):
        """Test extracting tool definition from AgentTool."""
        tool = AgentTool.from_func(sample_tool_function)

        agent = Agent(
            name="ToolAgent",
            role="Tester",
            tools=[tool],
        )

        tool_defs = agent._extract_tool_definitions()

        assert len(tool_defs) == 1
        assert tool_defs[0].name == "SampleToolFunction"
        assert "sample tool function" in tool_defs[0].description.lower()
        assert tool_defs[0].tool_type == "function"

    def test_extract_tool_from_callable(self):
        """Test extracting tool definition from plain callable wrapped as AgentTool."""
        tool = AgentTool.from_func(custom_tool_callable)
        agent = Agent(
            name="CallableToolAgent",
            role="Tester",
            tools=[tool],
        )

        tool_defs = agent._extract_tool_definitions()

        assert len(tool_defs) == 1
        assert tool_defs[0].name == "CustomToolCallable"
        assert "custom callable" in tool_defs[0].description.lower()
        assert tool_defs[0].tool_type == "function"

    def test_extract_multiple_tools(self):
        """Test extracting multiple tool definitions."""
        tool1 = AgentTool.from_func(sample_tool_function)
        tool2 = AgentTool.from_func(custom_tool_callable)

        agent = Agent(
            name="MultiToolAgent",
            role="Tester",
            tools=[tool1, tool2],
        )

        tool_defs = agent._extract_tool_definitions()

        assert len(tool_defs) == 2
        assert tool_defs[0].name == "SampleToolFunction"
        assert tool_defs[1].name == "CustomToolCallable"

    def test_extract_tools_empty_list(self):
        """Test extracting tools when no tools provided."""
        agent = Agent(
            name="NoToolsAgent",
            role="Tester",
        )

        tool_defs = agent._extract_tool_definitions()

        assert tool_defs == []


class TestMetadataIntegration:
    """Test full metadata building with all components."""

    @patch("dapr.clients.DaprClient")
    def test_complete_metadata_structure(self, mock_client_cls):
        """Test complete metadata with all features."""
        # Mock DaprClient as a context manager
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client_cls.return_value.__exit__.return_value = None

        # Mock the state operations
        mock_response = Mock()
        mock_response.data = json.dumps({}).encode("utf-8")
        mock_response.etag = "test-etag"
        mock_client.get_state.return_value = mock_response

        # Setup all configs
        state_store = StateStoreService(store_name="state-store")
        agent_registry_store = StateStoreService(store_name="agent-store")
        team_registry_store = StateStoreService(store_name="team-store")

        state_config = AgentStateConfig(store=state_store)
        agent_registry_config = AgentRegistryConfig(store=agent_registry_store)
        registry_config = AgentRegistryConfig(store=team_registry_store)

        tool = AgentTool.from_func(sample_tool_function)

        agent = Agent(
            name="CompleteAgent",
            role="Full Featured Agent",
            goal="Test all features",
            instructions=["Instruction 1", "Instruction 2"],
            state_config=state_config,
            agent_registry_config=agent_registry_config,
            registry_config=registry_config,
            tools=[tool],
        )

        metadata = agent._build_agent_metadata()

        # Verify all fields
        assert metadata.name == "CompleteAgent"
        assert metadata.role == "Full Featured Agent"
        assert metadata.goal == "Test all features"
        assert metadata.instructions == ["Instruction 1", "Instruction 2"]
        assert metadata.agent_class == "Agent"
        assert metadata.agent_category == "agent"

        # Verify components
        assert len(metadata.components.state_stores) == 3
        assert "workflow" in metadata.components.state_stores
        assert "agent_registry" in metadata.components.state_stores
        assert "team_registry" in metadata.components.state_stores

        # Verify tools
        assert len(metadata.tools) == 1
        assert metadata.tools[0].name == "SampleToolFunction"
        assert metadata.tools[0].tool_type == "function"
