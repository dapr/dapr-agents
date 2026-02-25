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

    def test_unrecognized_key_returns_early(self, basic_agent):
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
            config_metadata={"pgNotifyChannel": "config"},
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


class TestCoerceConfigValue:
    """Tests for _coerce_config_value type coercion."""

    def test_str_passthrough(self):
        assert AgentBase._coerce_config_value("hello", str) == "hello"

    def test_str_from_int(self):
        assert AgentBase._coerce_config_value(42, str) == "42"

    def test_int_from_string(self):
        assert AgentBase._coerce_config_value("42", int) == 42

    def test_int_from_float_string(self):
        assert AgentBase._coerce_config_value("10.0", int) == 10

    def test_int_already_int(self):
        assert AgentBase._coerce_config_value(7, int) == 7

    def test_int_invalid_raises(self):
        with pytest.raises((ValueError, TypeError)):
            AgentBase._coerce_config_value("not_a_number", int)

    def test_bool_true_variants(self):
        for v in ("true", "True", "1", "yes"):
            assert AgentBase._coerce_config_value(v, bool) is True

    def test_bool_false_variants(self):
        for v in ("false", "False", "0", "no"):
            assert AgentBase._coerce_config_value(v, bool) is False

    def test_bool_invalid_raises(self):
        with pytest.raises(ValueError):
            AgentBase._coerce_config_value("maybe", bool)

    def test_list_from_json(self):
        result = AgentBase._coerce_config_value('["a", "b"]', list)
        assert result == ["a", "b"]

    def test_list_wraps_single_string(self):
        result = AgentBase._coerce_config_value("single", list)
        assert result == ["single"]

    def test_list_already_list(self):
        result = AgentBase._coerce_config_value(["already"], list)
        assert result == ["already"]

    def test_dict_from_json(self):
        result = AgentBase._coerce_config_value('{"key": "val"}', dict)
        assert result == {"key": "val"}

    def test_dict_already_dict(self):
        result = AgentBase._coerce_config_value({"key": "val"}, dict)
        assert result == {"key": "val"}

    def test_dict_non_dict_json_raises(self):
        with pytest.raises(ValueError):
            AgentBase._coerce_config_value("[1, 2]", dict)


class TestLoadInitialConfiguration:
    """Tests for _load_initial_configuration."""

    @pytest.fixture
    def mock_llm_client(self):
        return MockLLMClient()

    @staticmethod
    def _make_config_response(items_dict):
        """Build a mock ConfigurationResponse with the given key-value pairs."""
        response = Mock()
        items = {}
        for key, value in items_dict.items():
            item = Mock()
            item.value = value
            items[key] = item
        response.items = items
        return response

    def test_loads_and_applies_initial_values(self, mock_llm_client):
        agent = ConcreteAgentBase(
            name="InitAgent",
            role="Default",
            llm=mock_llm_client,
            configuration=AgentConfigurationConfig(
                store_name="configstore", keys=["role"]
            ),
        )
        mock_client = MagicMock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client.get_configuration.return_value = self._make_config_response(
            {"role": "Loaded Role"}
        )

        with patch("dapr_agents.agents.base.DaprClient", return_value=mock_client):
            agent._load_initial_configuration(["role"])

        assert agent.profile.role == "Loaded Role"

    def test_get_configuration_does_not_pass_metadata(self, mock_llm_client):
        """Metadata (e.g. pgNotifyChannel) must NOT be passed to get_configuration."""
        agent = ConcreteAgentBase(
            name="MetaAgent",
            role="Default",
            llm=mock_llm_client,
            configuration=AgentConfigurationConfig(
                store_name="configstore",
                keys=["role"],
                metadata={"pgNotifyChannel": "config"},
            ),
        )
        mock_client = MagicMock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        response = Mock()
        response.items = {}
        mock_client.get_configuration.return_value = response

        with patch("dapr_agents.agents.base.DaprClient", return_value=mock_client):
            agent._load_initial_configuration(["role"])

        # Verify get_configuration was called without config_metadata
        call_kwargs = mock_client.get_configuration.call_args
        assert (
            "config_metadata" not in call_kwargs.kwargs
            or call_kwargs.kwargs.get("config_metadata") is None
        )

    def test_initial_load_failure_is_warning(self, mock_llm_client, caplog):
        agent = ConcreteAgentBase(
            name="FailAgent",
            llm=mock_llm_client,
            configuration=AgentConfigurationConfig(
                store_name="configstore", keys=["role"]
            ),
        )
        mock_client = MagicMock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client.get_configuration.side_effect = RuntimeError("connection refused")

        with patch("dapr_agents.agents.base.DaprClient", return_value=mock_client):
            with caplog.at_level(logging.WARNING):
                agent._load_initial_configuration(["role"])

        assert "could not load initial configuration" in caplog.text.lower()

    def test_empty_response_no_error(self, mock_llm_client):
        agent = ConcreteAgentBase(
            name="EmptyAgent",
            role="Default",
            llm=mock_llm_client,
            configuration=AgentConfigurationConfig(
                store_name="configstore", keys=["role"]
            ),
        )
        mock_client = MagicMock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        response = Mock()
        response.items = {}
        mock_client.get_configuration.return_value = response

        with patch("dapr_agents.agents.base.DaprClient", return_value=mock_client):
            agent._load_initial_configuration(["role"])

        # Profile unchanged
        assert agent.profile.role == "Default"

    def test_get_called_before_subscribe(self, mock_llm_client):
        """Verify that get_configuration is called before subscribe_configuration."""
        call_order = []
        mock_client = MagicMock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)

        response = Mock()
        response.items = {}
        mock_client.get_configuration.side_effect = lambda **kw: (
            call_order.append("get") or response
        )
        mock_client.subscribe_configuration.side_effect = lambda **kw: (
            call_order.append("subscribe") or "sub-id"
        )

        agent = ConcreteAgentBase(
            name="OrderAgent",
            llm=mock_llm_client,
            configuration=AgentConfigurationConfig(
                store_name="configstore", keys=["k"]
            ),
        )

        with patch("dapr_agents.agents.base.DaprClient", return_value=mock_client):
            agent._setup_configuration_subscription()

        assert call_order == ["get", "subscribe"]


class TestConfigChangeCallback:
    """Tests for user-provided on_config_change callback."""

    @pytest.fixture
    def mock_llm_client(self):
        return MockLLMClient()

    def test_callback_invoked(self, mock_llm_client):
        callback = Mock()
        config = AgentConfigurationConfig(
            store_name="configstore",
            keys=["role"],
            on_config_change=callback,
        )
        agent = ConcreteAgentBase(
            name="CBAgent", llm=mock_llm_client, configuration=config
        )
        agent._apply_config_update("role", "New Role")
        callback.assert_called_once_with("role", "New Role")

    def test_callback_error_swallowed(self, mock_llm_client, caplog):
        callback = Mock(side_effect=RuntimeError("boom"))
        config = AgentConfigurationConfig(
            store_name="configstore",
            keys=["role"],
            on_config_change=callback,
        )
        agent = ConcreteAgentBase(
            name="CBAgent", llm=mock_llm_client, configuration=config
        )
        with caplog.at_level(logging.WARNING):
            agent._apply_config_update("role", "New Role")
        assert "callback failed" in caplog.text.lower()
        # Value still applied despite callback failure
        assert agent.profile.role == "New Role"

    def test_no_callback_configured(self, mock_llm_client):
        """No callback set should not raise."""
        agent = ConcreteAgentBase(name="NoCB", llm=mock_llm_client)
        agent._apply_config_update("role", "Updated")
        assert agent.profile.role == "Updated"


class TestNewHotReloadableFields:
    """Tests for newly supported hot-reloadable fields."""

    @pytest.fixture
    def mock_llm_client(self):
        return MockLLMClient()

    @pytest.fixture
    def basic_agent(self, mock_llm_client):
        return ConcreteAgentBase(
            name="TestAgent",
            role="Role",
            goal="Goal",
            instructions=["Instr"],
            style_guidelines=["Be brief"],
            llm=mock_llm_client,
        )

    def test_update_style_guidelines_string(self, basic_agent):
        basic_agent._apply_config_update("style_guidelines", "Be concise")
        assert basic_agent.profile.style_guidelines == ["Be concise"]

    def test_update_style_guidelines_json_list(self, basic_agent):
        basic_agent._apply_config_update(
            "style_guidelines", '["Be concise", "Use examples"]'
        )
        assert basic_agent.profile.style_guidelines == ["Be concise", "Use examples"]

    def test_update_style_guidelines_alias(self, basic_agent):
        basic_agent._apply_config_update("agent_style_guidelines", "Formal")
        assert basic_agent.profile.style_guidelines == ["Formal"]

    def test_update_max_iterations(self, basic_agent):
        basic_agent._apply_config_update("max_iterations", "5")
        assert basic_agent.execution.max_iterations == 5

    def test_update_max_iterations_invalid_rejected(self, basic_agent, caplog):
        original = basic_agent.execution.max_iterations
        with caplog.at_level(logging.WARNING):
            basic_agent._apply_config_update("max_iterations", "0")
        assert "validation failed" in caplog.text.lower()
        # Value should not have been applied
        assert basic_agent.execution.max_iterations == original

    def test_update_tool_choice(self, basic_agent):
        basic_agent._apply_config_update("tool_choice", "none")
        assert basic_agent.execution.tool_choice == "none"

    def test_update_max_iterations_non_numeric_rejected(self, basic_agent, caplog):
        original = basic_agent.execution.max_iterations
        with caplog.at_level(logging.WARNING):
            basic_agent._apply_config_update("max_iterations", "abc")
        assert "invalid value" in caplog.text.lower()
        assert basic_agent.execution.max_iterations == original
