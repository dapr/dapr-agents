"""Tests for hot-reload configuration and deregistration on stop."""

import logging
import pytest
from unittest.mock import Mock, MagicMock, patch

from dapr_agents.agents.base import AgentBase
from dapr_agents.agents.configs import AgentConfigurationConfig
from .mocks.llm_client import MockLLMClient


class ConcreteAgentBase(AgentBase):
    """Concrete implementation of AgentBase for testing."""

    def run(self, input_data):
        return f"Processed: {input_data}"


class TestApplyConfigUpdate:
    """Tests for _apply_config_update."""

    @pytest.fixture
    def mock_llm_client(self):
        return MockLLMClient()

    @pytest.fixture
    def basic_agent(self, mock_llm_client):
        return ConcreteAgentBase(
            name="TestAgent",
            role="Original Role",
            goal="Original Goal",
            instructions=["Original instruction"],
            llm=mock_llm_client,
        )

    def test_update_role(self, basic_agent):
        basic_agent._apply_config_update("role", "New Role")
        assert basic_agent.profile.role == "New Role"
        assert basic_agent.prompting_helper.role == "New Role"

    def test_update_agent_role_alias(self, basic_agent):
        basic_agent._apply_config_update("agent_role", "Alias Role")
        assert basic_agent.profile.role == "Alias Role"

    def test_update_goal(self, basic_agent):
        basic_agent._apply_config_update("goal", "New Goal")
        assert basic_agent.profile.goal == "New Goal"
        assert basic_agent.prompting_helper.goal == "New Goal"

    def test_update_agent_goal_alias(self, basic_agent):
        basic_agent._apply_config_update("agent_goal", "Alias Goal")
        assert basic_agent.profile.goal == "Alias Goal"

    def test_update_instructions_string(self, basic_agent):
        basic_agent._apply_config_update("instructions", "Single instruction")
        assert basic_agent.profile.instructions == ["Single instruction"]
        assert basic_agent.prompting_helper.instructions == ["Single instruction"]

    def test_update_instructions_list(self, basic_agent):
        basic_agent._apply_config_update("agent_instructions", ["First", "Second"])
        assert basic_agent.profile.instructions == ["First", "Second"]

    def test_update_system_prompt(self, basic_agent):
        basic_agent._apply_config_update("system_prompt", "New system prompt")
        assert basic_agent.profile.system_prompt == "New system prompt"
        assert basic_agent.prompting_helper.system_prompt == "New system prompt"

    def test_update_llm_model(self, basic_agent):
        basic_agent._apply_config_update("llm_model", "gpt-4o-mini")
        assert basic_agent.llm.model == "gpt-4o-mini"

    def test_update_llm_provider_readonly_no_error(self, basic_agent):
        """provider is a read-only @property on OpenAIChatClient.
        The update should be silently skipped without raising."""
        original = basic_agent.llm.provider
        basic_agent._apply_config_update("llm_provider", "azure")
        assert basic_agent.llm.provider == original

    def test_unrecognized_key_returns_early(self, basic_agent, caplog):
        with patch.object(basic_agent, "register_agentic_system") as mock_reg:
            # Even if registry_state were set, unrecognized keys should not trigger re-registration
            basic_agent._apply_config_update("unknown_key", "value")
            mock_reg.assert_not_called()

    def test_sensitive_key_redacted_in_logs(self, basic_agent, caplog):
        with caplog.at_level(logging.INFO):
            basic_agent._apply_config_update("openai_api_key", "sk-secret-123")
        assert "sk-secret-123" not in caplog.text
        assert "***" in caplog.text

    def test_hyphenated_key_normalized(self, basic_agent):
        basic_agent._apply_config_update("agent-role", "Hyphen Role")
        assert basic_agent.profile.role == "Hyphen Role"


class TestApplyConfigUpdateReregistration:
    """Tests for re-registration after config updates."""

    @pytest.fixture
    def mock_llm_client(self):
        return MockLLMClient()

    @pytest.fixture
    def agent_with_registry(self, mock_llm_client):
        agent = ConcreteAgentBase(
            name="RegAgent",
            role="Role",
            goal="Goal",
            instructions=["Instr"],
            llm=mock_llm_client,
        )
        # Simulate having a registry and pre-built metadata
        mock_registry = Mock()
        mock_registry.store = Mock()
        mock_registry.store.store_name = "agent-registry"
        agent._infra._registry = mock_registry
        agent._infra.registry_state = mock_registry.store
        agent.agent_metadata = {
            "agent": {
                "role": "Role",
                "goal": "Goal",
                "instructions": ["Instr"],
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-4o",
            },
        }
        return agent

    def test_triggers_reregistration(self, agent_with_registry):
        with patch.object(agent_with_registry, "register_agentic_system") as mock_reg:
            agent_with_registry._apply_config_update("role", "Updated Role")
            mock_reg.assert_called_once()

    def test_syncs_llm_metadata(self, agent_with_registry):
        with patch.object(agent_with_registry, "register_agentic_system"):
            agent_with_registry._apply_config_update("llm_model", "gpt-4o-mini")
            assert agent_with_registry.agent_metadata["llm"]["model"] == "gpt-4o-mini"

    def test_syncs_llm_model_metadata(self, agent_with_registry):
        with patch.object(agent_with_registry, "register_agentic_system"):
            agent_with_registry._apply_config_update("llm_model", "gpt-3.5-turbo")
            assert agent_with_registry.agent_metadata["llm"]["model"] == "gpt-3.5-turbo"

    def test_syncs_profile_metadata(self, agent_with_registry):
        with patch.object(agent_with_registry, "register_agentic_system"):
            agent_with_registry._apply_config_update("goal", "New Goal")
            assert agent_with_registry.agent_metadata["agent"]["goal"] == "New Goal"

    def test_reregistration_failure_is_warning(self, agent_with_registry, caplog):
        with patch.object(
            agent_with_registry,
            "register_agentic_system",
            side_effect=Exception("store error"),
        ):
            with caplog.at_level(logging.WARNING):
                agent_with_registry._apply_config_update("role", "Fail Role")
        assert "Failed to re-register" in caplog.text


class TestConfigHandler:
    """Tests for _config_handler."""

    @pytest.fixture
    def mock_llm_client(self):
        return MockLLMClient()

    @pytest.fixture
    def basic_agent(self, mock_llm_client):
        return ConcreteAgentBase(
            name="TestAgent",
            role="Role",
            goal="Goal",
            llm=mock_llm_client,
        )

    def _make_config_response(self, items_dict):
        """Build a mock ConfigurationResponse with the given key-value pairs."""
        response = Mock()
        items = {}
        for key, value in items_dict.items():
            item = Mock()
            item.value = value
            items[key] = item
        response.items = items
        return response

    def test_plain_value(self, basic_agent):
        response = self._make_config_response({"role": "Handler Role"})
        basic_agent._config_handler("sub-1", response)
        assert basic_agent.profile.role == "Handler Role"

    def test_json_dict_value(self, basic_agent):
        response = self._make_config_response(
            {"config": '{"role": "JSON Role", "goal": "JSON Goal"}'}
        )
        basic_agent._config_handler("sub-1", response)
        assert basic_agent.profile.role == "JSON Role"
        assert basic_agent.profile.goal == "JSON Goal"

    def test_json_non_dict_falls_through(self, basic_agent):
        """A JSON string that is not a dict should be treated as a plain value."""
        response = self._make_config_response({"role": '"just a string"'})
        basic_agent._config_handler("sub-1", response)
        # The raw JSON string (with quotes) gets applied as role
        assert basic_agent.profile.role == '"just a string"'

    def test_handler_error_is_logged(self, basic_agent, caplog):
        response = Mock()
        response.items = Mock(side_effect=AttributeError("bad response"))
        with caplog.at_level(logging.ERROR):
            basic_agent._config_handler("sub-1", response)
        assert "Error in configuration handler" in caplog.text


class TestSetupConfigurationSubscription:
    """Tests for _setup_configuration_subscription."""

    @pytest.fixture
    def mock_llm_client(self):
        return MockLLMClient()

    def test_subscribes_with_correct_params(self, mock_llm_client):
        agent = ConcreteAgentBase(
            name="ConfigAgent",
            llm=mock_llm_client,
            configuration=AgentConfigurationConfig(
                store_name="configstore",
                keys=["agent_role", "agent_goal"],
            ),
        )
        mock_client = MagicMock()
        mock_client.subscribe_configuration.return_value = "sub-123"

        with patch("dapr_agents.agents.base.DaprClient", return_value=mock_client):
            agent._setup_configuration_subscription()

        mock_client.subscribe_configuration.assert_called_once_with(
            store_name="configstore",
            keys=["agent_role", "agent_goal"],
            handler=agent._config_handler,
            config_metadata={},
        )
        assert agent._subscription_id == "sub-123"

    def test_defaults_keys_to_agent_name(self, mock_llm_client):
        agent = ConcreteAgentBase(
            name="MyAgent",
            llm=mock_llm_client,
            configuration=AgentConfigurationConfig(store_name="configstore"),
        )
        mock_client = MagicMock()
        mock_client.subscribe_configuration.return_value = "sub-456"

        with patch("dapr_agents.agents.base.DaprClient", return_value=mock_client):
            agent._setup_configuration_subscription()

        call_kwargs = mock_client.subscribe_configuration.call_args
        assert call_kwargs.kwargs["keys"] == ["MyAgent"]

    def test_subscription_error_is_logged(self, mock_llm_client, caplog):
        agent = ConcreteAgentBase(
            name="ErrorAgent",
            llm=mock_llm_client,
            configuration=AgentConfigurationConfig(
                store_name="configstore", keys=["k"]
            ),
        )
        mock_client = MagicMock()
        mock_client.subscribe_configuration.side_effect = RuntimeError(
            "connection refused"
        )

        with patch("dapr_agents.agents.base.DaprClient", return_value=mock_client):
            with caplog.at_level(logging.ERROR):
                agent._setup_configuration_subscription()

        assert "failed to subscribe" in caplog.text


class TestStop:
    """Tests for stop() — deregistration and config cleanup."""

    @pytest.fixture
    def mock_llm_client(self):
        return MockLLMClient()

    def test_deregisters_from_registry(self, mock_llm_client):
        agent = ConcreteAgentBase(name="StopAgent", llm=mock_llm_client)
        mock_registry = Mock()
        mock_registry.store = Mock()
        mock_registry.store.store_name = "agent-registry"
        agent._infra._registry = mock_registry
        agent._infra.registry_state = mock_registry.store

        with patch.object(agent, "deregister_agentic_system") as mock_dereg:
            agent.stop()
            mock_dereg.assert_called_once()

    def test_deregistration_error_is_swallowed(self, mock_llm_client, caplog):
        agent = ConcreteAgentBase(name="StopAgent", llm=mock_llm_client)
        mock_registry = Mock()
        mock_registry.store = Mock()
        mock_registry.store.store_name = "agent-registry"
        agent._infra._registry = mock_registry
        agent._infra.registry_state = mock_registry.store

        with patch.object(
            agent,
            "deregister_agentic_system",
            side_effect=Exception("store unavailable"),
        ):
            with caplog.at_level(logging.DEBUG):
                agent.stop()  # Should not raise
        assert "Error deregistering" in caplog.text

    def test_unsubscribes_configuration(self, mock_llm_client):
        agent = ConcreteAgentBase(
            name="StopAgent",
            llm=mock_llm_client,
            configuration=AgentConfigurationConfig(
                store_name="configstore", keys=["k"]
            ),
        )
        mock_client = MagicMock()
        agent._config_client = mock_client
        agent._subscription_id = "sub-999"

        agent.stop()

        mock_client.unsubscribe_configuration.assert_called_once_with(
            store_name="configstore",
            configuration_id="sub-999",
        )
        mock_client.close.assert_called_once()
        assert agent._config_client is None

    def test_stop_minimal_agent_no_error(self, mock_llm_client):
        """stop() on an agent with no registry or config should not raise."""
        agent = ConcreteAgentBase(name="MinAgent", llm=mock_llm_client)
        agent.stop()  # Should complete without error
