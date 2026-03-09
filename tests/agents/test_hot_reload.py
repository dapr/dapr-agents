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

"""Tests for hot-reload configuration and deregistration on stop."""

import logging
import pytest
from unittest.mock import Mock, MagicMock, patch

from dapr_agents.agents.base import AgentBase
from dapr_agents.agents.configs import (
    AgentMetadata,
    AgentMetadataSchema,
    AgentObservabilityConfig,
    LLMMetadata,
    RuntimeSubscriptionConfig,
)
from dapr_agents.observability.instrumentor import DaprAgentsInstrumentor
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
        basic_agent._apply_config_update("agent_role", "New Role")
        assert basic_agent.profile.role == "New Role"
        assert basic_agent.prompting_helper.role == "New Role"

    def test_update_goal(self, basic_agent):
        basic_agent._apply_config_update("agent_goal", "New Goal")
        assert basic_agent.profile.goal == "New Goal"
        assert basic_agent.prompting_helper.goal == "New Goal"

    def test_update_instructions_string(self, basic_agent):
        basic_agent._apply_config_update("agent_instructions", "Single instruction")
        assert basic_agent.profile.instructions == ["Single instruction"]
        assert basic_agent.prompting_helper.instructions == ["Single instruction"]

    def test_update_instructions_list(self, basic_agent):
        basic_agent._apply_config_update("agent_instructions", ["First", "Second"])
        assert basic_agent.profile.instructions == ["First", "Second"]

    def test_update_system_prompt(self, basic_agent):
        basic_agent._apply_config_update("agent_system_prompt", "New system prompt")
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

    def test_unprefixed_profile_key_is_unrecognized(self, basic_agent):
        """Short-form profile keys (e.g. 'role') are no longer supported."""
        original = basic_agent.profile.role
        basic_agent._apply_config_update("role", "Should Not Apply")
        assert basic_agent.profile.role == original

    def test_sensitive_key_redacted_in_logs(self, basic_agent, caplog):
        with caplog.at_level(logging.INFO):
            basic_agent._apply_config_update("llm_api_key", "sk-secret-123")
        assert "sk-secret-123" not in caplog.text
        assert "***" in caplog.text

    def test_hyphenated_key_normalized(self, basic_agent):
        basic_agent._apply_config_update("agent-role", "Hyphen Role")
        assert basic_agent.profile.role == "Hyphen Role"

    def test_invalid_type_for_valid_key_rejected(self, basic_agent, caplog):
        """Passing a non-coercible type (e.g. dict for an int key) should be
        logged as a warning and the original value preserved."""
        original = basic_agent.execution.max_iterations
        with caplog.at_level(logging.WARNING):
            basic_agent._apply_config_update("max_iterations", {"bad": "type"})
        assert "invalid value" in caplog.text.lower()
        assert basic_agent.execution.max_iterations == original


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
        agent.agent_metadata = AgentMetadataSchema(
            version="0.0.0-test",
            name="RegAgent",
            registered_at="2026-01-01T00:00:00Z",
            agent=AgentMetadata(
                appid="reg-agent",
                type="standalone",
                role="Role",
                goal="Goal",
                instructions=["Instr"],
            ),
            llm=LLMMetadata(
                client="openai",
                provider="openai",
                model="gpt-4o",
            ),
        )
        return agent

    def test_triggers_reregistration(self, agent_with_registry):
        with patch.object(agent_with_registry, "register_agentic_system") as mock_reg:
            agent_with_registry._apply_config_update("agent_role", "Updated Role")
            mock_reg.assert_called_once()

    def test_syncs_llm_metadata(self, agent_with_registry):
        with patch.object(agent_with_registry, "register_agentic_system"):
            agent_with_registry._apply_config_update("llm_model", "gpt-4o-mini")
            assert agent_with_registry.agent_metadata.llm.model == "gpt-4o-mini"

    def test_syncs_llm_model_metadata(self, agent_with_registry):
        with patch.object(agent_with_registry, "register_agentic_system"):
            agent_with_registry._apply_config_update("llm_model", "gpt-3.5-turbo")
            assert agent_with_registry.agent_metadata.llm.model == "gpt-3.5-turbo"

    def test_syncs_profile_metadata(self, agent_with_registry):
        with patch.object(agent_with_registry, "register_agentic_system"):
            agent_with_registry._apply_config_update("agent_goal", "New Goal")
            assert agent_with_registry.agent_metadata.agent.goal == "New Goal"

    def test_reregistration_failure_is_warning(self, agent_with_registry, caplog):
        with patch.object(
            agent_with_registry,
            "register_agentic_system",
            side_effect=Exception("store error"),
        ):
            with caplog.at_level(logging.WARNING):
                agent_with_registry._apply_config_update("agent_role", "Fail Role")
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
        response = self._make_config_response({"agent_role": "Handler Role"})
        basic_agent._config_handler("sub-1", response)
        assert basic_agent.profile.role == "Handler Role"

    def test_json_dict_value(self, basic_agent):
        response = self._make_config_response(
            {"config": '{"agent_role": "JSON Role", "agent_goal": "JSON Goal"}'}
        )
        basic_agent._config_handler("sub-1", response)
        assert basic_agent.profile.role == "JSON Role"
        assert basic_agent.profile.goal == "JSON Goal"

    def test_json_non_dict_falls_through(self, basic_agent):
        """A JSON string that is not a dict should be treated as a plain value."""
        response = self._make_config_response({"agent_role": '"just a string"'})
        basic_agent._config_handler("sub-1", response)
        # The raw JSON string (with quotes) gets applied as role
        assert basic_agent.profile.role == '"just a string"'

    def test_handler_error_is_logged(self, basic_agent, caplog):
        response = Mock()
        response.items.items.side_effect = AttributeError("bad response")
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
            configuration=RuntimeSubscriptionConfig(
                store_name="runtime-config",
                keys=["agent_role", "agent_goal"],
            ),
        )
        mock_client = MagicMock()
        mock_client.subscribe_configuration.return_value = "sub-123"

        with patch("dapr_agents.agents.base.DaprClient", return_value=mock_client):
            agent._setup_configuration_subscription()

        mock_client.subscribe_configuration.assert_called_once_with(
            store_name="runtime-config",
            keys=["agent_role", "agent_goal"],
            handler=agent._config_handler,
            config_metadata={"pgNotifyChannel": "config"},
        )
        assert agent._subscription_id == "sub-123"

    def test_defaults_keys_to_agent_name(self, mock_llm_client):
        agent = ConcreteAgentBase(
            name="MyAgent",
            llm=mock_llm_client,
            configuration=RuntimeSubscriptionConfig(store_name="runtime-config"),
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
            configuration=RuntimeSubscriptionConfig(
                store_name="runtime-config", keys=["k"]
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
            configuration=RuntimeSubscriptionConfig(
                store_name="runtime-config", keys=["k"]
            ),
        )
        mock_client = MagicMock()
        agent._config_client = mock_client
        agent._subscription_id = "sub-999"

        agent.stop()

        mock_client.unsubscribe_configuration.assert_called_once_with(
            store_name="runtime-config",
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
            configuration=RuntimeSubscriptionConfig(
                store_name="runtime-config", keys=["agent_role"]
            ),
        )
        mock_client = MagicMock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client.get_configuration.return_value = self._make_config_response(
            {"agent_role": "Loaded Role"}
        )

        with patch("dapr_agents.agents.base.DaprClient", return_value=mock_client):
            agent._load_initial_configuration(["agent_role"])

        assert agent.profile.role == "Loaded Role"

    def test_get_configuration_does_not_pass_metadata(self, mock_llm_client):
        """Metadata (e.g. pgNotifyChannel) must NOT be passed to get_configuration."""
        agent = ConcreteAgentBase(
            name="MetaAgent",
            role="Default",
            llm=mock_llm_client,
            configuration=RuntimeSubscriptionConfig(
                store_name="runtime-config",
                keys=["agent_role"],
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
            agent._load_initial_configuration(["agent_role"])

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
            configuration=RuntimeSubscriptionConfig(
                store_name="runtime-config", keys=["agent_role"]
            ),
        )
        mock_client = MagicMock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client.get_configuration.side_effect = RuntimeError("connection refused")

        with patch("dapr_agents.agents.base.DaprClient", return_value=mock_client):
            with caplog.at_level(logging.WARNING):
                agent._load_initial_configuration(["agent_role"])

        assert "could not load initial configuration" in caplog.text.lower()

    def test_empty_response_no_error(self, mock_llm_client):
        agent = ConcreteAgentBase(
            name="EmptyAgent",
            role="Default",
            llm=mock_llm_client,
            configuration=RuntimeSubscriptionConfig(
                store_name="runtime-config", keys=["agent_role"]
            ),
        )
        mock_client = MagicMock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        response = Mock()
        response.items = {}
        mock_client.get_configuration.return_value = response

        with patch("dapr_agents.agents.base.DaprClient", return_value=mock_client):
            agent._load_initial_configuration(["agent_role"])

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
            configuration=RuntimeSubscriptionConfig(
                store_name="runtime-config", keys=["k"]
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
        config = RuntimeSubscriptionConfig(
            store_name="runtime-config",
            keys=["agent_role"],
            on_config_change=callback,
        )
        agent = ConcreteAgentBase(
            name="CBAgent", llm=mock_llm_client, configuration=config
        )
        agent._apply_config_update("agent_role", "New Role")
        callback.assert_called_once_with("agent_role", "New Role")

    def test_callback_error_swallowed(self, mock_llm_client, caplog):
        callback = Mock(side_effect=RuntimeError("boom"))
        config = RuntimeSubscriptionConfig(
            store_name="runtime-config",
            keys=["agent_role"],
            on_config_change=callback,
        )
        agent = ConcreteAgentBase(
            name="CBAgent", llm=mock_llm_client, configuration=config
        )
        with caplog.at_level(logging.WARNING):
            agent._apply_config_update("agent_role", "New Role")
        assert "callback failed" in caplog.text.lower()
        # Value still applied despite callback failure
        assert agent.profile.role == "New Role"

    def test_no_callback_configured(self, mock_llm_client):
        """No callback set should not raise."""
        agent = ConcreteAgentBase(name="NoCB", llm=mock_llm_client)
        agent._apply_config_update("agent_role", "Updated")
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
        basic_agent._apply_config_update("agent_style_guidelines", "Be concise")
        assert basic_agent.profile.style_guidelines == ["Be concise"]

    def test_update_style_guidelines_json_list(self, basic_agent):
        basic_agent._apply_config_update(
            "agent_style_guidelines", '["Be concise", "Use examples"]'
        )
        assert basic_agent.profile.style_guidelines == ["Be concise", "Use examples"]

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


class TestOtelHotReload:
    """Tests for OTel hot-reload support."""

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

    def test_apply_config_update_returns_true_for_otel_key(self, basic_agent):
        """OTel config keys should return True to signal reload needed."""
        result = basic_agent._apply_config_update(
            "otel_exporter_otlp_endpoint", "http://localhost:4317"
        )
        assert result is True

    def test_apply_config_update_returns_false_for_non_otel_key(self, basic_agent):
        """Non-OTel config keys should return False."""
        result = basic_agent._apply_config_update("agent_role", "New Role")
        assert result is False

    def test_apply_config_update_returns_false_for_unknown_key(self, basic_agent):
        """Unknown keys should return False."""
        result = basic_agent._apply_config_update("unknown_key", "value")
        assert result is False

    def test_otel_sdk_disabled_inverts_enabled(self, basic_agent):
        """OTEL_SDK_DISABLED=true should set enabled=False."""
        basic_agent._apply_config_update("otel_sdk_disabled", "true")
        assert basic_agent._agent_observability.enabled is False

        basic_agent._apply_config_update("otel_sdk_disabled", "false")
        assert basic_agent._agent_observability.enabled is True

    def test_otel_endpoint_setter(self, basic_agent):
        basic_agent._apply_config_update(
            "otel_exporter_otlp_endpoint", "http://collector:4317"
        )
        assert basic_agent._agent_observability.endpoint == "http://collector:4317"

    def test_otel_headers_sensitive(self, basic_agent, caplog):
        """OTEL_EXPORTER_OTLP_HEADERS should be marked sensitive."""
        with caplog.at_level(logging.INFO):
            basic_agent._apply_config_update(
                "otel_exporter_otlp_headers", "Bearer secret-token"
            )
        assert "secret-token" not in caplog.text
        assert "***" in caplog.text

    def test_otel_service_name_setter(self, basic_agent):
        basic_agent._apply_config_update("otel_service_name", "my-agent-service")
        assert basic_agent._agent_observability.service_name == "my-agent-service"

    def test_otel_tracing_enabled_setter(self, basic_agent):
        basic_agent._apply_config_update("otel_tracing_enabled", "true")
        assert basic_agent._agent_observability.tracing_enabled is True

    def test_otel_logging_enabled_setter(self, basic_agent):
        basic_agent._apply_config_update("otel_logging_enabled", "true")
        assert basic_agent._agent_observability.logging_enabled is True

    def test_otel_traces_exporter_valid(self, basic_agent):
        basic_agent._apply_config_update("otel_traces_exporter", "otlp_grpc")
        from dapr_agents.agents.configs import AgentTracingExporter

        assert (
            basic_agent._agent_observability.tracing_exporter
            == AgentTracingExporter.OTLP_GRPC
        )

    def test_otel_traces_exporter_invalid_rejected(self, basic_agent, caplog):
        with caplog.at_level(logging.WARNING):
            basic_agent._apply_config_update("otel_traces_exporter", "invalid_exporter")
        assert "validation failed" in caplog.text.lower()

    def test_otel_logs_exporter_valid(self, basic_agent):
        basic_agent._apply_config_update("otel_logs_exporter", "console")
        from dapr_agents.agents.configs import AgentLoggingExporter

        assert (
            basic_agent._agent_observability.logging_exporter
            == AgentLoggingExporter.CONSOLE
        )

    def test_otel_logs_exporter_invalid_rejected(self, basic_agent, caplog):
        with caplog.at_level(logging.WARNING):
            basic_agent._apply_config_update("otel_logs_exporter", "invalid_exporter")
        assert "validation failed" in caplog.text.lower()

    def test_batched_reload_called_once_for_multiple_otel_keys(self, basic_agent):
        """Multiple OTel keys in one config response should trigger reload exactly once."""
        response = self._make_config_response(
            {
                "config": '{"otel_exporter_otlp_endpoint": "http://new:4317", "otel_tracing_enabled": "true"}'
            }
        )
        with patch.object(basic_agent, "_reload_observability") as mock_reload:
            basic_agent._config_handler("sub-1", response)
            mock_reload.assert_called_once()

    def test_no_reload_for_non_otel_keys(self, basic_agent):
        """Non-OTel keys should not trigger OTel reload."""
        response = self._make_config_response({"agent_role": "New Role"})
        with patch.object(basic_agent, "_reload_observability") as mock_reload:
            basic_agent._config_handler("sub-1", response)
            mock_reload.assert_not_called()

    def test_mixed_keys_triggers_reload(self, basic_agent):
        """A mix of OTel and non-OTel keys should trigger reload."""
        response = self._make_config_response(
            {"config": '{"agent_role": "New Role", "otel_tracing_enabled": "true"}'}
        )
        with patch.object(basic_agent, "_reload_observability") as mock_reload:
            basic_agent._config_handler("sub-1", response)
            mock_reload.assert_called_once()

    def test_reload_does_not_call_global_setters(self, basic_agent):
        """Hot-reload should NOT call set_tracer_provider / set_logger_provider."""
        basic_agent._agent_observability.enabled = True
        basic_agent._agent_observability.tracing_enabled = True
        basic_agent._agent_observability.logging_enabled = True
        basic_agent._agent_observability.endpoint = "http://localhost:4317"
        basic_agent._agent_observability.tracing_exporter = "console"
        basic_agent._agent_observability.logging_exporter = "console"
        basic_agent.instrumentor = Mock()

        with (
            patch("dapr_agents.agents.base.trace.set_tracer_provider") as mock_set_tp,
            patch("dapr_agents.agents.base._logs.set_logger_provider") as mock_set_lp,
            patch("dapr_agents.agents.base.TracerProvider"),
            patch("dapr_agents.agents.base.LoggerProvider"),
            patch("dapr_agents.agents.base.BatchSpanProcessor"),
            patch("dapr_agents.agents.base.BatchLogRecordProcessor"),
            patch("dapr_agents.agents.base.ConsoleSpanExporter"),
            patch("dapr_agents.agents.base.ConsoleLogRecordExporter"),
        ):
            basic_agent._reload_observability()

        mock_set_tp.assert_not_called()
        mock_set_lp.assert_not_called()
        basic_agent.instrumentor.update_providers.assert_called_once()


class TestInstrumentorUpdateProviders:
    """Tests for DaprAgentsInstrumentor.update_providers."""

    def test_update_providers_mutates_wrapper_tracers(self):
        instrumentor = DaprAgentsInstrumentor()
        # Create mock wrappers with _tracer attributes
        w1 = Mock()
        w1._tracer = "old_tracer"
        w2 = Mock()
        w2._tracer = "old_tracer"
        instrumentor._wrappers = [w1, w2]
        instrumentor._tracer = "old_tracer"

        mock_provider = Mock()
        with patch("dapr_agents.observability.instrumentor.trace_api") as mock_trace:
            mock_trace.get_tracer.return_value = "new_tracer"
            instrumentor.update_providers(tracer_provider=mock_provider)

        assert w1._tracer == "new_tracer"
        assert w2._tracer == "new_tracer"
        assert instrumentor._tracer == "new_tracer"
        assert DaprAgentsInstrumentor._global_tracer == "new_tracer"

    def test_update_providers_updates_logger(self):
        instrumentor = DaprAgentsInstrumentor()
        instrumentor._wrappers = []

        mock_provider = Mock()
        with patch("dapr_agents.observability.instrumentor.logs_api") as mock_logs:
            mock_logs.get_logger.return_value = "new_logger"
            instrumentor.update_providers(logger_provider=mock_provider)

        assert instrumentor._logger == "new_logger"
        assert DaprAgentsInstrumentor._global_logger == "new_logger"

    def test_track_wrapper_appends_and_returns(self):
        instrumentor = DaprAgentsInstrumentor()
        wrapper = Mock()
        result = instrumentor._track_wrapper(wrapper)
        assert result is wrapper
        assert wrapper in instrumentor._wrappers


class TestShutdownOtelProviders:
    """Tests for _shutdown_otel_providers."""

    def test_flush_and_shutdown_tracer(self):
        mock_tp = Mock()
        AgentBase._shutdown_otel_providers(mock_tp, None)
        mock_tp.force_flush.assert_called_once_with(timeout_millis=5000)
        mock_tp.shutdown.assert_called_once()

    def test_flush_and_shutdown_logger(self):
        mock_lp = Mock()
        AgentBase._shutdown_otel_providers(None, mock_lp)
        mock_lp.force_flush.assert_called_once_with(timeout_millis=5000)
        mock_lp.shutdown.assert_called_once()

    def test_both_providers_shutdown(self):
        mock_tp = Mock()
        mock_lp = Mock()
        AgentBase._shutdown_otel_providers(mock_tp, mock_lp)
        mock_tp.force_flush.assert_called_once()
        mock_tp.shutdown.assert_called_once()
        mock_lp.force_flush.assert_called_once()
        mock_lp.shutdown.assert_called_once()

    def test_resilient_to_flush_error(self):
        mock_tp = Mock()
        mock_tp.force_flush.side_effect = RuntimeError("flush failed")
        # Should not raise
        AgentBase._shutdown_otel_providers(mock_tp, None)

    def test_none_providers_no_error(self):
        # Should not raise
        AgentBase._shutdown_otel_providers(None, None)
