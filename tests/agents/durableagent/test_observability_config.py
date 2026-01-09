"""Test cases for observability configuration in agents."""
import os
import pytest
from unittest.mock import Mock

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


class TestObservabilityConfigFromInstantiation:
    """Test cases for observability config provided during instantiation."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Set up environment variables and mocks for testing."""
        # Clear any OTEL environment variables
        for key in list(os.environ.keys()):
            if key.startswith("OTEL_"):
                monkeypatch.delenv(key, raising=False)

        os.environ["OPENAI_API_KEY"] = "test-api-key"

        # Mock DaprClient with no runtime config
        mock_client = MockDaprClient()
        monkeypatch.setattr(
            "dapr_agents.agents.base.DaprClient", lambda **kwargs: mock_client
        )
        monkeypatch.setattr(
            "dapr_agents.storage.daprstores.statestore.DaprClient", lambda: mock_client
        )

        # Mock the observability setup to avoid actual OTel initialization
        monkeypatch.setattr(
            "dapr_agents.agents.base.AgentBase._setup_agent_observability", Mock()
        )

        yield
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

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

        resolved_config = agent._resolve_observability_config()

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

        resolved_config = agent._resolve_observability_config()

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

        resolved_config = agent._resolve_observability_config()
        assert resolved_config.enabled is False


class TestObservabilityConfigFromEnvironment:
    """Test cases for observability config from environment variables."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Set up environment variables and mocks for testing."""
        # Clear any existing OTEL environment variables
        for key in list(os.environ.keys()):
            if key.startswith("OTEL_"):
                monkeypatch.delenv(key, raising=False)

        os.environ["OPENAI_API_KEY"] = "test-api-key"

        # Mock DaprClient with no runtime config
        mock_client = MockDaprClient()
        monkeypatch.setattr(
            "dapr_agents.agents.base.DaprClient", lambda **kwargs: mock_client
        )
        monkeypatch.setattr(
            "dapr_agents.storage.daprstores.statestore.DaprClient", lambda: mock_client
        )

        # Mock the observability setup to avoid actual OTel initialization
        monkeypatch.setattr(
            "dapr_agents.agents.base.AgentBase._setup_agent_observability", Mock()
        )

        yield
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

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

    def test_observability_config_from_env_all_fields(self, mock_llm, monkeypatch):
        """Test observability config loaded from environment variables."""
        monkeypatch.setenv("OTEL_ENABLED", "true")
        monkeypatch.setenv("OTEL_TOKEN", "Bearer env-token")
        monkeypatch.setenv("OTEL_ENDPOINT", "http://env-collector:4318")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "env-service")
        monkeypatch.setenv("OTEL_LOGGING_ENABLED", "true")
        monkeypatch.setenv("OTEL_LOGGING_EXPORTER", "otlp_http")
        monkeypatch.setenv("OTEL_TRACING_ENABLED", "true")
        monkeypatch.setenv("OTEL_TRACING_EXPORTER", "console")

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

        resolved_config = agent._resolve_observability_config()

        assert resolved_config.enabled is True
        assert resolved_config.headers == {"Authorization": "Bearer env-token"}
        assert resolved_config.endpoint == "http://env-collector:4318"
        assert resolved_config.service_name == "env-service"
        assert resolved_config.logging_enabled is True
        assert resolved_config.logging_exporter == AgentLoggingExporter.OTLP_HTTP
        assert resolved_config.tracing_enabled is True
        assert resolved_config.tracing_exporter == AgentTracingExporter.CONSOLE

    def test_observability_config_from_env_partial_fields(self, mock_llm, monkeypatch):
        """Test observability config with only some env variables set."""
        monkeypatch.setenv("OTEL_ENABLED", "true")
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

        resolved_config = agent._resolve_observability_config()

        assert resolved_config.enabled is True
        assert resolved_config.service_name == "partial-service"
        assert resolved_config.endpoint is None
        # logging_enabled comes from statestore default (False)
        assert resolved_config.logging_enabled is False

    def test_observability_config_from_env_disabled(self, mock_llm, monkeypatch):
        """Test observability explicitly disabled via environment."""
        monkeypatch.setenv("OTEL_ENABLED", "false")
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

        resolved_config = agent._resolve_observability_config()
        assert resolved_config.enabled is False

    def test_observability_config_from_env_invalid_exporter(
        self, mock_llm, monkeypatch
    ):
        """Test observability config with invalid exporter defaults to console."""
        monkeypatch.setenv("OTEL_TRACING_EXPORTER", "invalid_exporter")
        monkeypatch.setenv("OTEL_LOGGING_EXPORTER", "another_invalid")

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

        resolved_config = agent._resolve_observability_config()

        # Should default to CONSOLE for invalid values
        assert resolved_config.tracing_exporter == AgentTracingExporter.CONSOLE
        assert resolved_config.logging_exporter == AgentLoggingExporter.CONSOLE


class TestObservabilityConfigFromStatestore:
    """Test cases for observability config from default statestore."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Set up environment variables and mocks for testing."""
        # Clear any OTEL environment variables
        for key in list(os.environ.keys()):
            if key.startswith("OTEL_"):
                monkeypatch.delenv(key, raising=False)

        os.environ["OPENAI_API_KEY"] = "test-api-key"

        # Mock the observability setup to avoid actual OTel initialization
        monkeypatch.setattr(
            "dapr_agents.agents.base.AgentBase._setup_agent_observability", Mock()
        )

        yield
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

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

    def test_observability_config_from_statestore_all_fields(
        self, mock_llm, monkeypatch
    ):
        """Test observability config loaded from statestore."""
        runtime_config = {
            "OTEL_ENABLED": "true",
            "OTEL_TOKEN": "statestore-token",
            "OTEL_ENDPOINT": "http://statestore-collector:4317",
            "OTEL_SERVICE_NAME": "statestore-service",
            "OTEL_LOGGING_ENABLED": "true",
            "OTEL_LOGGING_EXPORTER": "otlp_grpc",
            "OTEL_TRACING_ENABLED": "true",
            "OTEL_TRACING_EXPORTER": "zipkin",
        }

        mock_client = MockDaprClient(runtime_config=runtime_config)
        monkeypatch.setattr(
            "dapr_agents.agents.base.DaprClient", lambda **kwargs: mock_client
        )
        monkeypatch.setattr(
            "dapr_agents.storage.daprstores.statestore.DaprClient", lambda: mock_client
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
        )

        resolved_config = agent._resolve_observability_config()

        assert resolved_config.enabled is True
        assert resolved_config.auth_token == "statestore-token"
        assert resolved_config.endpoint == "http://statestore-collector:4317"
        assert resolved_config.service_name == "statestore-service"
        assert resolved_config.logging_enabled is True
        assert resolved_config.logging_exporter == AgentLoggingExporter.OTLP_GRPC
        assert resolved_config.tracing_enabled is True
        assert resolved_config.tracing_exporter == AgentTracingExporter.ZIPKIN

    def test_observability_config_from_statestore_partial_fields(
        self, mock_llm, monkeypatch
    ):
        """Test observability config from statestore with partial fields."""
        runtime_config = {
            "OTEL_ENABLED": "true",
            "OTEL_SERVICE_NAME": "partial-statestore-service",
        }

        mock_client = MockDaprClient(runtime_config=runtime_config)
        monkeypatch.setattr(
            "dapr_agents.agents.base.DaprClient", lambda **kwargs: mock_client
        )
        monkeypatch.setattr(
            "dapr_agents.storage.daprstores.statestore.DaprClient", lambda: mock_client
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
        )

        resolved_config = agent._resolve_observability_config()

        assert resolved_config.enabled is True
        assert resolved_config.service_name == "partial-statestore-service"
        assert resolved_config.endpoint is None

    def test_observability_config_from_statestore_disabled(self, mock_llm, monkeypatch):
        """Test observability disabled from statestore."""
        runtime_config = {
            "OTEL_ENABLED": "false",
            "OTEL_SERVICE_NAME": "disabled-statestore-service",
        }

        mock_client = MockDaprClient(runtime_config=runtime_config)
        monkeypatch.setattr(
            "dapr_agents.agents.base.DaprClient", lambda **kwargs: mock_client
        )
        monkeypatch.setattr(
            "dapr_agents.storage.daprstores.statestore.DaprClient", lambda: mock_client
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
        )

        resolved_config = agent._resolve_observability_config()
        assert resolved_config.enabled is False

    def test_observability_config_statestore_invalid_exporter(
        self, mock_llm, monkeypatch
    ):
        """Test observability config from statestore with invalid exporters."""
        runtime_config = {
            "OTEL_ENABLED": "true",
            "OTEL_TRACING_EXPORTER": "invalid_type",
            "OTEL_LOGGING_EXPORTER": "wrong_value",
        }

        mock_client = MockDaprClient(runtime_config=runtime_config)
        monkeypatch.setattr(
            "dapr_agents.agents.base.DaprClient", lambda **kwargs: mock_client
        )
        monkeypatch.setattr(
            "dapr_agents.storage.daprstores.statestore.DaprClient", lambda: mock_client
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
        )

        resolved_config = agent._resolve_observability_config()

        # Should default to CONSOLE for invalid values
        assert resolved_config.tracing_exporter == AgentTracingExporter.CONSOLE
        assert resolved_config.logging_exporter == AgentLoggingExporter.CONSOLE


class TestObservabilityConfigPrecedence:
    """Test cases for observability config precedence and merging."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Set up environment variables and mocks for testing."""
        # Clear any OTEL environment variables
        for key in list(os.environ.keys()):
            if key.startswith("OTEL_"):
                monkeypatch.delenv(key, raising=False)

        os.environ["OPENAI_API_KEY"] = "test-api-key"

        # Mock the observability setup to avoid actual OTel initialization
        monkeypatch.setattr(
            "dapr_agents.agents.base.AgentBase._setup_agent_observability", Mock()
        )

        yield
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

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

    def test_precedence_instantiation_over_env(self, mock_llm, monkeypatch):
        """Test instantiation config takes precedence over environment."""
        # Set environment variables
        monkeypatch.setenv("OTEL_ENABLED", "true")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "env-service")
        monkeypatch.setenv("OTEL_ENDPOINT", "http://env-endpoint:4317")
        monkeypatch.setenv("OTEL_TRACING_EXPORTER", "console")

        # Mock DaprClient with no runtime config
        mock_client = MockDaprClient()
        monkeypatch.setattr(
            "dapr_agents.agents.base.DaprClient", lambda **kwargs: mock_client
        )
        monkeypatch.setattr(
            "dapr_agents.storage.daprstores.statestore.DaprClient", lambda: mock_client
        )

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

        resolved_config = agent._resolve_observability_config()

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
            "OTEL_ENABLED": "true",
            "OTEL_SERVICE_NAME": "statestore-service",
            "OTEL_ENDPOINT": "http://statestore-endpoint:4317",
            "OTEL_LOGGING_EXPORTER": "console",
        }

        mock_client = MockDaprClient(runtime_config=runtime_config)
        monkeypatch.setattr(
            "dapr_agents.agents.base.DaprClient", lambda **kwargs: mock_client
        )
        monkeypatch.setattr(
            "dapr_agents.storage.daprstores.statestore.DaprClient", lambda: mock_client
        )

        # Set environment variables (should override statestore)
        monkeypatch.setenv("OTEL_ENABLED", "false")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "env-service")
        monkeypatch.setenv("OTEL_LOGGING_EXPORTER", "otlp_grpc")

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

        resolved_config = agent._resolve_observability_config()

        # Environment should win
        assert resolved_config.enabled is False
        assert resolved_config.service_name == "env-service"
        assert resolved_config.logging_exporter == AgentLoggingExporter.OTLP_GRPC
        # Endpoint should come from statestore since env didn't specify it
        assert resolved_config.endpoint == "http://statestore-endpoint:4317"

    def test_precedence_full_hierarchy(self, mock_llm, monkeypatch):
        """Test full precedence hierarchy: instantiation > env > statestore."""
        # Set statestore config (lowest priority)
        runtime_config = {
            "OTEL_ENABLED": "true",
            "OTEL_SERVICE_NAME": "statestore-service",
            "OTEL_ENDPOINT": "http://statestore-endpoint:4317",
            "OTEL_LOGGING_ENABLED": "true",
            "OTEL_LOGGING_EXPORTER": "console",
            "OTEL_TRACING_ENABLED": "true",
            "OTEL_TRACING_EXPORTER": "console",
        }

        mock_client = MockDaprClient(runtime_config=runtime_config)
        monkeypatch.setattr(
            "dapr_agents.agents.base.DaprClient", lambda **kwargs: mock_client
        )
        monkeypatch.setattr(
            "dapr_agents.storage.daprstores.statestore.DaprClient", lambda: mock_client
        )

        # Set environment variables (middle priority)
        monkeypatch.setenv("OTEL_SERVICE_NAME", "env-service")
        monkeypatch.setenv("OTEL_LOGGING_EXPORTER", "otlp_grpc")

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

        resolved_config = agent._resolve_observability_config()

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
        # Mock DaprClient with no runtime config
        mock_client = MockDaprClient()
        monkeypatch.setattr(
            "dapr_agents.agents.base.DaprClient", lambda **kwargs: mock_client
        )
        monkeypatch.setattr(
            "dapr_agents.storage.daprstores.statestore.DaprClient", lambda: mock_client
        )

        # Set environment with token (creates Authorization header)
        monkeypatch.setenv("OTEL_TOKEN", "env-token")

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

        resolved_config = agent._resolve_observability_config()

        # Headers should be merged with instantiation taking precedence
        assert "X-Custom-Header" in resolved_config.headers
        assert resolved_config.headers["X-Custom-Header"] == "custom-value"
        assert resolved_config.headers["Authorization"] == "Bearer instantiation-token"

    def test_no_config_sources_returns_defaults(self, mock_llm, monkeypatch):
        """Test that when no config is provided, defaults are used."""
        # Mock DaprClient with no runtime config
        mock_client = MockDaprClient()
        monkeypatch.setattr(
            "dapr_agents.agents.base.DaprClient", lambda **kwargs: mock_client
        )
        monkeypatch.setattr(
            "dapr_agents.storage.daprstores.statestore.DaprClient", lambda: mock_client
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
        )

        resolved_config = agent._resolve_observability_config()

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


class TestObservabilityConfigMergeLogic:
    """Test cases for the merge logic specifically."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Set up environment variables and mocks for testing."""
        # Clear any OTEL environment variables
        for key in list(os.environ.keys()):
            if key.startswith("OTEL_"):
                monkeypatch.delenv(key, raising=False)

        os.environ["OPENAI_API_KEY"] = "test-api-key"

        # Mock DaprClient
        mock_client = MockDaprClient()
        monkeypatch.setattr(
            "dapr_agents.agents.base.DaprClient", lambda **kwargs: mock_client
        )
        monkeypatch.setattr(
            "dapr_agents.storage.daprstores.statestore.DaprClient", lambda: mock_client
        )

        # Mock the observability setup to avoid actual OTel initialization
        monkeypatch.setattr(
            "dapr_agents.agents.base.AgentBase._setup_agent_observability", Mock()
        )

        yield
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

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

    def test_merge_none_values_dont_override(self, mock_llm):
        """Test that None values in override don't override base values."""
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

        base = AgentObservabilityConfig(
            enabled=True,
            service_name="base-service",
            endpoint="http://base-endpoint:4317",
        )

        override = AgentObservabilityConfig(
            enabled=None,  # Should not override
            service_name="override-service",
            endpoint=None,  # Should not override
        )

        merged = agent._merge_observability_configs(base, override)

        assert merged.enabled is True  # From base
        assert merged.service_name == "override-service"  # From override
        assert merged.endpoint == "http://base-endpoint:4317"  # From base

    def test_merge_boolean_fields_correctly(self, mock_llm):
        """Test that boolean fields merge correctly with None handling."""
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

        base = AgentObservabilityConfig(
            enabled=True,
            logging_enabled=True,
            tracing_enabled=False,
        )

        override = AgentObservabilityConfig(
            enabled=False,
            logging_enabled=None,
            tracing_enabled=True,
        )

        merged = agent._merge_observability_configs(base, override)

        assert merged.enabled is False  # Override wins
        assert merged.logging_enabled is True  # Base wins (override is None)
        assert merged.tracing_enabled is True  # Override wins

    def test_merge_empty_configs(self, mock_llm):
        """Test merging two empty configs."""
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

        base = AgentObservabilityConfig()
        override = AgentObservabilityConfig()

        merged = agent._merge_observability_configs(base, override)

        assert merged.enabled is None
        assert merged.headers == {}
        assert merged.auth_token is None
        assert merged.endpoint is None
        assert merged.service_name is None
