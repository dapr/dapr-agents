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

"""Tests for DurableAgent observability configuration."""

import logging
import os
import pytest
from unittest.mock import MagicMock, Mock, patch

from tests.conftest import MockDaprClient
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.agents.configs import (
    AgentPubSubConfig,
    AgentStateConfig,
    AgentRegistryConfig,
    AgentObservabilityConfig,
    AgentTracingExporter,
    AgentLoggingExporter,
)
from dapr_agents.llm import OpenAIChatClient
from dapr_agents.storage.daprstores.stateservice import StateStoreService


class ObservabilityConfigTestBase:
    """Shared fixtures and helpers for observability configuration tests."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Set up environment variables and mocks for testing."""
        for key in list(os.environ.keys()):
            if key.startswith("OTEL_"):
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

        # Mock the observability setup to avoid actual OTel initialization
        monkeypatch.setattr(
            "dapr_agents.agents.base.AgentBase._setup_agent_observability", Mock()
        )

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


class TestObservabilityConfigFromInstantiation(ObservabilityConfigTestBase):
    """Test cases for observability config provided during instantiation."""

    def test_observability_config_from_instantiation_does_not_mutate_input_config(self):
        """Test that observability config resolution copies and leaves the caller's config untouched."""
        observability_config = AgentObservabilityConfig(
            logging_exporter="otlp_grpc",
            tracing_exporter="zipkin",
        )

        resolved_config = AgentObservabilityConfig._from_instantiation(
            observability_config
        )

        assert observability_config is not resolved_config
        assert observability_config.logging_exporter == "otlp_grpc"
        assert observability_config.tracing_exporter == "zipkin"
        assert resolved_config.logging_exporter == AgentLoggingExporter.OTLP_GRPC
        assert resolved_config.tracing_exporter == AgentTracingExporter.ZIPKIN

    @pytest.mark.parametrize(
        ("logging_exporter", "tracing_exporter"),
        [
            ("otlp_grpc", "zipkin"),
            ("OTLP_GRPC", "ZIPKIN"),
            ("Otlp_Grpc", "Zipkin"),
        ],
    )
    def test_observability_config_from_instantiation_accepts_case_insensitive_values(
        self, logging_exporter, tracing_exporter
    ):
        """Test that observability config accepts case-insensitive instantiated values."""
        observability_config = AgentObservabilityConfig(
            logging_exporter=logging_exporter,
            tracing_exporter=tracing_exporter,
        )

        resolved_config = AgentObservabilityConfig._from_instantiation(
            observability_config
        )

        assert resolved_config.logging_exporter == AgentLoggingExporter.OTLP_GRPC
        assert resolved_config.tracing_exporter == AgentTracingExporter.ZIPKIN

    def test_observability_config_from_instantiation_all_fields(self, mock_llm):
        """Test observability config passed during instantiation with all fields."""
        obs_config = AgentObservabilityConfig(
            enabled=True,
            headers={"Authorization": "Bearer token123"},
            auth_token="token123",
            endpoint="http://otel-collector:4317",
            service_name="test-service",
            logging_enabled=True,
            logging_exporter=AgentLoggingExporter.OTLP_GRPC,
            tracing_enabled=True,
            tracing_exporter=AgentTracingExporter.OTLP_GRPC,
        )

        agent = DurableAgent(
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
            agent_observability=obs_config,
        )

        resolved_config = agent._agent_observability

        assert resolved_config.enabled is True
        assert resolved_config.headers == {"Authorization": "Bearer token123"}
        assert resolved_config.auth_token == "token123"
        assert resolved_config.endpoint == "http://otel-collector:4317"
        assert resolved_config.service_name == "test-service"
        assert resolved_config.logging_enabled is True
        assert resolved_config.logging_exporter == AgentLoggingExporter.OTLP_GRPC
        assert resolved_config.tracing_enabled is True
        assert resolved_config.tracing_exporter == AgentTracingExporter.OTLP_GRPC

    def test_observability_config_from_instantiation_partial_fields(self, mock_llm):
        """Test observability config with only some fields set."""
        obs_config = AgentObservabilityConfig(
            enabled=True,
            tracing_enabled=True,
            tracing_exporter=AgentTracingExporter.ZIPKIN,
        )

        agent = DurableAgent(
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
            agent_observability=obs_config,
        )

        resolved_config = agent._agent_observability

        assert resolved_config.enabled is True
        assert resolved_config.tracing_enabled is True
        assert resolved_config.tracing_exporter == AgentTracingExporter.ZIPKIN
        # logging_enabled comes from statestore default (False)
        assert resolved_config.logging_enabled is False
        # logging_exporter comes from statestore default (console)
        assert resolved_config.logging_exporter == AgentLoggingExporter.CONSOLE
        assert resolved_config.endpoint is None

    def test_observability_config_disabled_from_instantiation(self, mock_llm):
        """Test observability config explicitly disabled."""
        obs_config = AgentObservabilityConfig(enabled=False)

        agent = DurableAgent(
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
            agent_observability=obs_config,
        )

        resolved_config = agent._agent_observability

        assert resolved_config.enabled is False

    @pytest.mark.parametrize(
        ("auth_token", "endpoint", "service_name"),
        [
            ("", "", ""),
            (" \t\r\n", " \t\r\n", " \t\r\n"),
        ],
    )
    def test_observability_config_from_instantiation_ignores_empty_and_whitespace_only_strings(
        self, mock_llm
    ):
        """Test that observability config treats empty and whitespace-only strings as unset."""
        obs_config = AgentObservabilityConfig(
            auth_token="",
            endpoint="",
            service_name="",
        )

        agent = DurableAgent(
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
            agent_observability=obs_config,
        )

        resolved_config = agent._agent_observability

        # Should resolve to defaults
        assert resolved_config.auth_token is None
        assert resolved_config.endpoint is None
        assert resolved_config.service_name is None


class TestObservabilityConfigFromEnvironment(ObservabilityConfigTestBase):
    """Test cases for observability config from environment variables."""

    def test_observability_config_from_env_all_fields(self, mock_llm, monkeypatch):
        """Test observability config loaded from environment variables."""
        monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
        monkeypatch.setenv(
            "OTEL_EXPORTER_OTLP_HEADERS", "Authorization=Bearer env-token"
        )
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://env-collector:4318")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "env-service")
        monkeypatch.setenv("OTEL_LOGGING_ENABLED", "true")
        monkeypatch.setenv("OTEL_LOGS_EXPORTER", "otlp_http")
        monkeypatch.setenv("OTEL_TRACING_ENABLED", "true")
        monkeypatch.setenv("OTEL_TRACES_EXPORTER", "console")

        agent = DurableAgent(
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
        )

        resolved_config = agent._agent_observability

        assert resolved_config.enabled is True
        assert resolved_config.headers == {"Authorization": "Bearer env-token"}
        assert resolved_config.endpoint == "http://env-collector:4318"
        assert resolved_config.service_name == "env-service"
        assert resolved_config.logging_enabled is True
        assert resolved_config.logging_exporter == AgentLoggingExporter.OTLP_HTTP
        assert resolved_config.tracing_enabled is True
        assert resolved_config.tracing_exporter == AgentTracingExporter.CONSOLE

    def test_observability_config_from_env_accepts_lowercase_keys(
        self, mock_llm, monkeypatch
    ):
        """Test that observability config accepts lowercase environment variable names."""
        monkeypatch.setenv("otel_sdk_disabled", "false")
        monkeypatch.setenv(
            "otel_exporter_otlp_headers", "Authorization=Bearer env-token"
        )
        monkeypatch.setenv("otel_exporter_otlp_endpoint", "http://env-collector:4318")
        monkeypatch.setenv("otel_service_name", "env-service")
        monkeypatch.setenv("otel_logging_enabled", "true")
        monkeypatch.setenv("otel_logs_exporter", "otlp_http")
        monkeypatch.setenv("otel_tracing_enabled", "true")
        monkeypatch.setenv("otel_traces_exporter", "console")

        agent = DurableAgent(
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
        )

        assert agent._agent_observability.enabled is True
        assert agent._agent_observability.headers == {
            "Authorization": "Bearer env-token"
        }
        assert agent._agent_observability.endpoint == "http://env-collector:4318"
        assert agent._agent_observability.service_name == "env-service"
        assert agent._agent_observability.logging_enabled is True
        assert (
            agent._agent_observability.logging_exporter
            == AgentLoggingExporter.OTLP_HTTP
        )
        assert agent._agent_observability.tracing_enabled is True
        assert (
            agent._agent_observability.tracing_exporter == AgentTracingExporter.CONSOLE
        )

    @pytest.mark.parametrize(
        ("logging_exporter", "tracing_exporter"),
        [
            ("otlp_grpc", "zipkin"),
            ("OTLP_GRPC", "ZIPKIN"),
            ("Otlp_Grpc", "Zipkin"),
        ],
    )
    def test_observability_config_from_env_accepts_case_insensitive_values(
        self, logging_exporter, tracing_exporter, monkeypatch
    ):
        """Test that observability config accepts case-insensitive environment variables."""
        monkeypatch.setenv("OTEL_LOGS_EXPORTER", logging_exporter)
        monkeypatch.setenv("OTEL_TRACES_EXPORTER", tracing_exporter)

        resolved_config = AgentObservabilityConfig._from_env()

        assert resolved_config.logging_exporter == AgentLoggingExporter.OTLP_GRPC
        assert resolved_config.tracing_exporter == AgentTracingExporter.ZIPKIN

    def test_observability_config_from_env_partial_fields(self, mock_llm, monkeypatch):
        """Test observability config with only some env variables set."""
        monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "partial-service")

        agent = DurableAgent(
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
        )

        resolved_config = agent._agent_observability

        assert resolved_config.enabled is True
        assert resolved_config.service_name == "partial-service"
        assert resolved_config.endpoint is None
        # logging_enabled comes from statestore default (False)
        assert resolved_config.logging_enabled is False

    def test_observability_config_from_env_disabled(self, mock_llm, monkeypatch):
        """Test observability explicitly disabled via environment."""
        monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "disabled-service")

        agent = DurableAgent(
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
        )

        resolved_config = agent._agent_observability

        assert resolved_config.enabled is False

    def test_observability_config_from_env_invalid_exporter(
        self, mock_llm, monkeypatch
    ):
        """Test observability config with invalid exporter defaults to console."""
        monkeypatch.setenv("OTEL_TRACES_EXPORTER", "invalid_exporter")
        monkeypatch.setenv("OTEL_LOGS_EXPORTER", "another_invalid")

        agent = DurableAgent(
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
        )

        resolved_config = agent._agent_observability

        # Should default to CONSOLE for invalid values
        assert resolved_config.tracing_exporter == AgentTracingExporter.CONSOLE
        assert resolved_config.logging_exporter == AgentLoggingExporter.CONSOLE

    # TODO: remove in future release
    def test_observability_config_from_env_deprecated_warns_and_preserves_behavior(
        self, monkeypatch
    ):
        """Contract test for the deprecated public ``from_env`` method."""
        monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
        monkeypatch.setenv(
            "OTEL_EXPORTER_OTLP_HEADERS", "Authorization=Bearer env-token"
        )
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://env-collector:4318")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "env-service")
        monkeypatch.setenv("OTEL_LOGGING_ENABLED", "true")
        monkeypatch.setenv("OTEL_LOGS_EXPORTER", "otlp_http")
        monkeypatch.setenv("OTEL_TRACING_ENABLED", "true")
        monkeypatch.setenv("OTEL_TRACES_EXPORTER", "console")

        expected_config = AgentObservabilityConfig._from_env()
        with pytest.warns(DeprecationWarning):
            resolved_config = AgentObservabilityConfig.from_env()

        assert isinstance(resolved_config, AgentObservabilityConfig)
        assert resolved_config == expected_config


class TestObservabilityConfigFromStateStore(ObservabilityConfigTestBase):
    """Test cases for observability config from default statestore."""

    def test_observability_config_from_statestore_all_fields(
        self, mock_llm, monkeypatch
    ):
        """Test observability config loaded from statestore."""
        runtime_config = {
            "OTEL_SDK_DISABLED": "false",
            "OTEL_EXPORTER_OTLP_HEADERS": "statestore-token",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://statestore-collector:4317",
            "OTEL_SERVICE_NAME": "statestore-service",
            "OTEL_LOGGING_ENABLED": "true",
            "OTEL_LOGS_EXPORTER": "otlp_grpc",
            "OTEL_TRACING_ENABLED": "true",
            "OTEL_TRACES_EXPORTER": "zipkin",
        }

        mock_client = MockDaprClient(runtime_config=runtime_config)
        self._patch_dapr_client(monkeypatch, mock_client)

        agent = DurableAgent(
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
        )

        resolved_config = agent._agent_observability

        assert resolved_config.enabled is True
        assert resolved_config.auth_token == "statestore-token"
        assert resolved_config.endpoint == "http://statestore-collector:4317"  # noqa: E501
        assert resolved_config.service_name == "statestore-service"
        assert resolved_config.logging_enabled is True
        assert resolved_config.logging_exporter == AgentLoggingExporter.OTLP_GRPC
        assert resolved_config.tracing_enabled is True
        assert resolved_config.tracing_exporter == AgentTracingExporter.ZIPKIN

    def test_observability_config_from_statestore_accepts_lowercase_keys(
        self, mock_llm, monkeypatch
    ):
        """Test that observability config accepts lowercase runtime config keys."""
        runtime_config = {
            "otel_sdk_disabled": "false",
            "otel_exporter_otlp_headers": "statestore-token",
            "otel_exporter_otlp_endpoint": "http://statestore-collector:4317",
            "otel_service_name": "statestore-service",
            "otel_logging_enabled": "true",
            "otel_logs_exporter": "otlp_grpc",
            "otel_tracing_enabled": "true",
            "otel_traces_exporter": "zipkin",
        }

        mock_client = MockDaprClient(runtime_config=runtime_config)
        self._patch_dapr_client(monkeypatch, mock_client)

        agent = DurableAgent(
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
        )

        resolved_config = agent._agent_observability

        assert resolved_config.enabled is True
        assert resolved_config.auth_token == "statestore-token"
        assert resolved_config.endpoint == "http://statestore-collector:4317"
        assert resolved_config.service_name == "statestore-service"
        assert resolved_config.logging_enabled is True
        assert resolved_config.logging_exporter == AgentLoggingExporter.OTLP_GRPC
        assert resolved_config.tracing_enabled is True
        assert resolved_config.tracing_exporter == AgentTracingExporter.ZIPKIN

    @pytest.mark.parametrize(
        ("logging_exporter", "tracing_exporter"),
        [
            ("otlp_grpc", "zipkin"),
            ("OTLP_GRPC", "ZIPKIN"),
            ("Otlp_Grpc", "Zipkin"),
        ],
    )
    def test_observability_config_from_statestore_accepts_case_insensitive_values(
        self, logging_exporter, tracing_exporter
    ):
        """Test that observability config accepts case-insensitive runtime config values."""
        runtime_config = {
            "OTEL_LOGS_EXPORTER": logging_exporter,
            "OTEL_TRACES_EXPORTER": tracing_exporter,
        }

        resolved_config = AgentObservabilityConfig._from_statestore(runtime_config)

        assert resolved_config.logging_exporter == AgentLoggingExporter.OTLP_GRPC
        assert resolved_config.tracing_exporter == AgentTracingExporter.ZIPKIN

    def test_observability_config_from_statestore_partial_fields(
        self, mock_llm, monkeypatch
    ):
        """Test observability config from statestore with partial fields."""
        runtime_config = {
            "OTEL_SDK_DISABLED": "false",
            "OTEL_SERVICE_NAME": "partial-statestore-service",
        }

        mock_client = MockDaprClient(runtime_config=runtime_config)
        self._patch_dapr_client(monkeypatch, mock_client)

        agent = DurableAgent(
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
        )

        resolved_config = agent._agent_observability

        assert resolved_config.enabled is True
        assert resolved_config.service_name == "partial-statestore-service"
        assert resolved_config.endpoint is None

    def test_observability_config_from_statestore_disabled(self, mock_llm, monkeypatch):
        """Test observability disabled from statestore."""
        runtime_config = {
            "OTEL_SDK_DISABLED": "true",
            "OTEL_SERVICE_NAME": "disabled-statestore-service",
        }

        mock_client = MockDaprClient(runtime_config=runtime_config)
        self._patch_dapr_client(monkeypatch, mock_client)

        agent = DurableAgent(
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
        )

        resolved_config = agent._agent_observability

        assert resolved_config.enabled is False

    def test_observability_config_statestore_invalid_exporter(
        self, mock_llm, monkeypatch
    ):
        """Test observability config from statestore with invalid exporters."""
        runtime_config = {
            "OTEL_SDK_DISABLED": "false",
            "OTEL_TRACES_EXPORTER": "invalid_type",
            "OTEL_LOGS_EXPORTER": "wrong_value",
        }

        mock_client = MockDaprClient(runtime_config=runtime_config)
        self._patch_dapr_client(monkeypatch, mock_client)

        agent = DurableAgent(
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
        )

        resolved_config = agent._agent_observability

        # Should default to CONSOLE for invalid values
        assert resolved_config.tracing_exporter == AgentTracingExporter.CONSOLE
        assert resolved_config.logging_exporter == AgentLoggingExporter.CONSOLE


class TestObservabilityConfigPrecedence(ObservabilityConfigTestBase):
    """Test cases for observability config precedence and merging."""

    def test_precedence_instantiation_over_env(self, mock_llm, monkeypatch):
        """Test instantiation config takes precedence over environment."""
        # Set environment variables
        monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "env-service")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://env-endpoint:4317")
        monkeypatch.setenv("OTEL_TRACES_EXPORTER", "console")

        # Create observability config for instantiation
        obs_config = AgentObservabilityConfig(
            enabled=False,  # Different from env
            service_name="instantiation-service",  # Different from env
            tracing_exporter=AgentTracingExporter.ZIPKIN,  # Different from env
        )

        agent = DurableAgent(
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
            agent_observability=obs_config,
        )

        resolved_config = agent._agent_observability

        # Instantiation should win
        assert resolved_config.enabled is False
        assert resolved_config.service_name == "instantiation-service"
        assert resolved_config.tracing_exporter == AgentTracingExporter.ZIPKIN
        # Endpoint should come from env since instantiation didn't specify it
        assert resolved_config.endpoint == "http://env-endpoint:4317"

    def test_precedence_env_over_statestore(self, mock_llm, monkeypatch):
        """Test environment config takes precedence over statestore."""
        # Set statestore config
        runtime_config = {
            "OTEL_SDK_DISABLED": "false",
            "OTEL_SERVICE_NAME": "statestore-service",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://statestore-endpoint:4317",
            "OTEL_LOGS_EXPORTER": "console",
        }

        mock_client = MockDaprClient(runtime_config=runtime_config)
        self._patch_dapr_client(monkeypatch, mock_client)

        # Set environment variables (should override statestore)
        monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "env-service")
        monkeypatch.setenv("OTEL_LOGS_EXPORTER", "otlp_grpc")

        agent = DurableAgent(
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
        )

        resolved_config = agent._agent_observability

        # Environment should win
        assert resolved_config.enabled is False
        assert resolved_config.service_name == "env-service"
        assert resolved_config.logging_exporter == AgentLoggingExporter.OTLP_GRPC
        # Endpoint comes from statestore since env didn't specify it
        assert resolved_config.endpoint == "http://statestore-endpoint:4317"

    def test_precedence_full_hierarchy(self, mock_llm, monkeypatch):
        """Test full precedence hierarchy: instantiation > env > statestore."""
        # Set statestore config (lowest priority)
        runtime_config = {
            "OTEL_SDK_DISABLED": "false",
            "OTEL_SERVICE_NAME": "statestore-service",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://statestore-endpoint:4317",
            "OTEL_LOGGING_ENABLED": "true",
            "OTEL_LOGS_EXPORTER": "console",
            "OTEL_TRACING_ENABLED": "true",
            "OTEL_TRACES_EXPORTER": "console",
        }

        mock_client = MockDaprClient(runtime_config=runtime_config)
        self._patch_dapr_client(monkeypatch, mock_client)

        # Set environment variables (middle priority)
        monkeypatch.setenv("OTEL_SERVICE_NAME", "env-service")
        monkeypatch.setenv("OTEL_LOGS_EXPORTER", "otlp_grpc")

        # Create observability config for instantiation (highest priority)
        obs_config = AgentObservabilityConfig(
            service_name="instantiation-service",
            tracing_exporter=AgentTracingExporter.ZIPKIN,
        )

        agent = DurableAgent(
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
            agent_observability=obs_config,
        )

        resolved_config = agent._agent_observability

        # Instantiation wins for service_name and tracing_exporter
        assert resolved_config.service_name == "instantiation-service"
        assert resolved_config.tracing_exporter == AgentTracingExporter.ZIPKIN

        # Env wins for logging_exporter (not in instantiation)
        assert resolved_config.logging_exporter == AgentLoggingExporter.OTLP_GRPC

        # Statestore values used where not specified elsewhere
        assert resolved_config.enabled is True
        assert resolved_config.endpoint == "http://statestore-endpoint:4317"
        assert resolved_config.logging_enabled is True
        assert resolved_config.tracing_enabled is True

    def test_merge_configs_with_headers(self, mock_llm, monkeypatch):
        """Test merging configs with headers properly combines them."""
        # Set environment with headers (creates Authorization header)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "Authorization=env-token")

        # Create observability config with additional headers
        obs_config = AgentObservabilityConfig(
            headers={
                "X-Custom-Header": "custom-value",
                "Authorization": "Bearer instantiation-token",  # Should override env
            },
        )

        agent = DurableAgent(
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
            agent_observability=obs_config,
        )

        resolved_config = agent._agent_observability

        # Headers should be merged with instantiation taking precedence
        assert "X-Custom-Header" in resolved_config.headers
        assert resolved_config.headers["X-Custom-Header"] == "custom-value"
        assert resolved_config.headers["Authorization"] == "Bearer instantiation-token"

    def test_no_config_sources_returns_defaults(self, mock_llm, monkeypatch):
        """Test that when no config is provided, defaults are used."""
        agent = DurableAgent(
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
        )

        resolved_config = agent._agent_observability

        # Values come from statestore defaults (False for booleans, console for exporters)
        assert resolved_config.enabled is False
        assert resolved_config.headers == {}
        assert resolved_config.auth_token is None
        assert resolved_config.endpoint is None
        assert resolved_config.service_name is None
        assert resolved_config.logging_enabled is False
        assert resolved_config.logging_exporter == AgentLoggingExporter.CONSOLE
        assert resolved_config.tracing_enabled is False
        assert resolved_config.tracing_exporter == AgentTracingExporter.CONSOLE


class TestObservabilityConfigResolutionSecrets:
    """Test that observability secrets are not exposed in logs during config resolution."""

    def test_auth_token_from_instantiation_is_not_exposed_in_logs(self, caplog):
        auth_token = "instantiation-auth-token"
        config = AgentObservabilityConfig(auth_token=auth_token)

        with caplog.at_level(logging.DEBUG):
            AgentObservabilityConfig._resolve_config(config=config)

        assert auth_token not in caplog.text

    def test_headers_from_instantiation_are_not_exposed_in_logs(self, caplog):
        auth_token = "instantiation-auth-token"
        custom_value = "instantiation-custom-value"
        config = AgentObservabilityConfig(
            headers={
                "Authorization": f"Bearer {auth_token}",
                "X-Custom-Header": custom_value,
            },
        )

        with caplog.at_level(logging.DEBUG):
            AgentObservabilityConfig._resolve_config(config=config)

        assert auth_token not in caplog.text
        assert custom_value not in caplog.text

    def test_headers_from_env_are_not_exposed_in_logs(self, caplog, monkeypatch):
        auth_token = "env-auth-token"
        custom_value = "env-custom-value"
        monkeypatch.setenv(
            "OTEL_EXPORTER_OTLP_HEADERS",
            f"Authorization=Bearer {auth_token},X-Custom-Header={custom_value}",
        )

        with caplog.at_level(logging.DEBUG):
            AgentObservabilityConfig._resolve_config()

        assert auth_token not in caplog.text
        assert custom_value not in caplog.text

    def test_headers_from_statestore_are_not_exposed_in_logs(self, caplog):
        auth_token = "statestore-auth-token"
        custom_value = "statestore-custom-value"
        runtime_config = {
            "OTEL_EXPORTER_OTLP_HEADERS": f"Authorization=Bearer {auth_token},X-Custom-Header={custom_value}"
        }

        with caplog.at_level(logging.DEBUG):
            AgentObservabilityConfig._resolve_config(runtime_config=runtime_config)

        assert auth_token not in caplog.text
        assert custom_value not in caplog.text
