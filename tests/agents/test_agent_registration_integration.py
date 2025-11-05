"""
Integration tests for agent registration with real Dapr sidecar and Redis using testcontainers.

These tests use Docker containers for both Redis and Dapr, with proper networking.
Tests are automatically skipped if dependencies (dapr, testcontainers, docker) are not available.

Test Isolation and Clean Slate
-------------------------------
Each test starts with a clean registry (both local cache and Dapr state store) to ensure
test isolation and predictable results. This allows tests to validate specific scenarios
without interference from previous test runs.

Some tests simulate agent restarts by clearing only the local cache while keeping the
state store intact, which mirrors real-world restart scenarios where agents persist in
the registry.

IMPORTANT: Agent Registry Persistence in Production
----------------------------------------------------
In production environments, agents are NOT garbage collected or deleted from the registry.
Once an agent is registered, it persists indefinitely to maintain agent history.

Future enhancements will include:
- Creation/update timestamps to track agent lifecycle
- Support for different session IDs (memory contexts) per agent
- Agent versioning to track changes over time

Run with: pytest tests/agents/test_agent_registration_integration.py -v
"""

import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request

import pytest

# Skip tests if dependencies are not available
pytest.importorskip("dapr")
pytest.importorskip("testcontainers")

# Remove mock Dapr modules before importing real ones
# The agents/conftest.py may have added mocks, we need to remove them for integration tests
dapr_modules_to_remove = [k for k in list(sys.modules.keys()) if k.startswith("dapr")]
for module_name in dapr_modules_to_remove:
    del sys.modules[module_name]

# Also remove dapr_agents modules that may have imported mocked dapr
dapr_agents_modules = [
    k for k in list(sys.modules.keys()) if k.startswith("dapr_agents")
]
for module_name in dapr_agents_modules:
    del sys.modules[module_name]

# Now import real Dapr modules (no mocking needed for integration tests)
from dapr.clients import DaprClient
from testcontainers.core.container import DockerContainer
from testcontainers.core.network import Network
from testcontainers.core.waiting_utils import wait_for_logs

# Import dapr_agents modules
from dapr_agents.agents.standalone import Agent
from dapr_agents.agents.durable import DurableAgent
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.agents.configs import (
    AgentRegistryConfig,
    AgentStateConfig,
)
from dapr_agents.tool.base import AgentTool
from dapr_agents.registry.registry import Registry

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


def sample_tool_function(input_text: str) -> str:
    """A sample tool function for testing."""
    return f"Processed: {input_text}"


# TODO: move dapr testcontainers logic to a separate file to reuse in other tests


def wait_for_dapr_health(host: str, port: int, timeout: int = 60) -> bool:
    """
    Wait for Dapr sidecar to become healthy by checking the HTTP health endpoint.

    Args:
        host: The host where Dapr is running
        port: The HTTP port (typically 3500)
        timeout: Maximum time to wait in seconds

    Returns:
        True if Dapr becomes healthy, False otherwise
    """
    health_url = f"http://{host}:{port}/v1.0/healthz/outbound"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if 200 <= response.status < 300:
                    print(f"✓ Dapr health check passed on {health_url}")
                    return True
        except Exception:
            pass

        time.sleep(1)

    print(f"✗ Dapr health check timed out after {timeout}s on {health_url}")
    return False


@pytest.fixture(scope="module")
def docker_network():
    """Create a Docker network for container-to-container communication."""
    with Network() as network:
        yield network


@pytest.fixture(scope="module")
def redis_container(docker_network):
    """Start Redis container on the shared network."""
    container = (
        DockerContainer("redis:7-alpine")
        .with_network(docker_network)
        .with_network_aliases("redis")
        .with_exposed_ports(
            6379
        )  # TODO: switch to random port by making configuration dynamic at runtime to avoid port conflicts
    )
    container.start()
    wait_for_logs(container, "Ready to accept connections", timeout=30)
    try:
        yield container
    finally:
        container.stop()


@pytest.fixture(scope="module")
def placement_container(docker_network):
    """Start Dapr placement service container for actor/workflow support."""
    container = (
        DockerContainer("daprio/dapr:latest")
        .with_network(docker_network)
        .with_network_aliases("placement")
        .with_command(["./placement", "-port", "50005"])
        .with_exposed_ports(50005)
    )
    container.start()

    # Give placement service time to start
    time.sleep(2)

    try:
        yield container
    finally:
        container.stop()


@pytest.fixture(scope="module")
def dapr_container(redis_container, placement_container, docker_network):
    """Start Dapr sidecar container with Redis state store configuration."""
    # Create temporary components directory
    temp_dir = tempfile.mkdtemp()
    components_path = os.path.join(temp_dir, "components")
    os.makedirs(components_path, exist_ok=True)

    # Write Redis state store component configuration
    # KEY: Use 'redis:6379' (network alias), NOT localhost!
    state_store_config = """
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: statestore
spec:
  type: state.redis
  version: v1
  metadata:
  - name: redisHost
    value: redis:6379
  - name: redisPassword
    value: ""
  - name: actorStateStore
    value: "false"
"""
    with open(os.path.join(components_path, "statestore.yaml"), "w") as f:
        f.write(state_store_config)

    # Write Redis pubsub component configuration
    pubsub_config = """
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: pubsub
spec:
  type: pubsub.redis
  version: v1
  metadata:
  - name: redisHost
    value: redis:6379
  - name: redisPassword
    value: ""
"""
    with open(os.path.join(components_path, "pubsub.yaml"), "w") as f:
        f.write(pubsub_config)

    # Create Dapr container with placement service for actor/workflow support
    container = DockerContainer("daprio/daprd:latest")
    container = container.with_network(docker_network)
    container = container.with_volume_mapping(components_path, "/components", mode="ro")
    container = container.with_command(
        [
            "./daprd",
            "-app-id",
            "test-app",
            "-dapr-http-port",
            "3500",
            "-dapr-grpc-port",
            "50001",
            "-components-path",
            "/components",
            "-placement-host-address",
            "placement:50005",
            "-log-level",
            "info",
        ]
    )
    container = container.with_exposed_ports(3500, 50001)

    container.start()

    # Get the exposed ports
    http_host = "127.0.0.1"
    http_port = container.get_exposed_port(3500)
    grpc_port = container.get_exposed_port(50001)

    # Wait for Dapr to become healthy
    if not wait_for_dapr_health(http_host, http_port, timeout=60):
        container.stop()
        pytest.fail("Dapr container failed to become healthy")

    # Set environment variables for Dapr SDK
    os.environ["DAPR_HTTP_PORT"] = str(http_port)
    os.environ["DAPR_GRPC_PORT"] = str(grpc_port)
    os.environ["DAPR_RUNTIME_HOST"] = http_host

    yield container

    # Cleanup environment variables
    os.environ.pop("DAPR_HTTP_PORT", None)
    os.environ.pop("DAPR_GRPC_PORT", None)
    os.environ.pop("DAPR_RUNTIME_HOST", None)

    container.stop()

    # Cleanup temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def reset_registry(request):
    """
    Reset the agent registry before and after each test.

    This ensures each test starts with a clean slate in both the local cache
    and the Dapr state store. Tests can opt out of the initial cleanup by
    marking themselves with @pytest.mark.no_cleanup.
    """
    # Clear local registry cache
    Registry.clear_registered_names()

    # Clear the state store unless test is marked with no_cleanup
    if "no_cleanup" not in request.keywords:
        # Get Dapr client to clear the registry
        try:
            client = get_dapr_client()
            client.delete_state(store_name="statestore", key="agent_registry")
            client.close()
        except Exception:
            # If we can't connect to Dapr yet (fixture not ready), skip cleanup
            pass

    yield

    # Always clear local cache after test
    Registry.clear_registered_names()


@pytest.fixture(autouse=True)
def patch_dapr_client(monkeypatch):
    """
    Patch DaprClient to use the correct address and skip health checks.

    This ensures all DaprClient instances created during tests (including
    those created internally by Agent, MemoryStore, etc.) use the exposed
    port from the testcontainer.
    """
    from dapr.clients import DaprClient
    from dapr.clients.grpc.client import DaprGrpcClient
    from dapr.clients.health import DaprHealth

    # Skip health check
    monkeypatch.setattr(DaprHealth, "wait_until_ready", lambda: None)

    # Save the original __init__
    original_init = DaprGrpcClient.__init__

    # Create patched __init__
    def patched_init(self, address=None, *args, **kwargs):
        # If no address provided, use the one from environment variables
        if address is None:
            dapr_host = os.environ.get("DAPR_RUNTIME_HOST", "localhost")
            dapr_grpc_port = os.environ.get("DAPR_GRPC_PORT", "50001")
            address = f"{dapr_host}:{dapr_grpc_port}"
        # Call original __init__ with the address
        original_init(self, address, *args, **kwargs)

    monkeypatch.setattr(DaprGrpcClient, "__init__", patched_init)


def get_dapr_client() -> DaprClient:
    """Create a Dapr client using environment variables set by the fixture."""
    dapr_host = os.environ.get("DAPR_RUNTIME_HOST", "localhost")
    dapr_grpc_port = os.environ.get("DAPR_GRPC_PORT", "50001")
    dapr_address = f"{dapr_host}:{dapr_grpc_port}"
    return DaprClient(address=dapr_address)


def test_agent_metadata_persisted(dapr_container):
    """
    Test that agent metadata is correctly persisted to Redis via Dapr.

    Starts with a clean registry and validates that all metadata fields
    are correctly stored in the Dapr state store in both team and agent registries.
    """
    client = get_dapr_client()

    try:
        # Create config objects for the new API (separate configs for each registry)
        state_store = StateStoreService(store_name="statestore")
        team_registry_store = StateStoreService(store_name="statestore")
        agent_registry_store = StateStoreService(store_name="statestore")

        state_config = AgentStateConfig(store=state_store)
        registry_config = AgentRegistryConfig(
            store=team_registry_store
        )  # Team registry
        agent_registry_config = AgentRegistryConfig(
            store=agent_registry_store
        )  # Agent registry

        tool = AgentTool.from_func(sample_tool_function)

        _ = Agent(
            name="IntegrationTestAgent",
            role="Integration Tester",
            goal="Test metadata persistence",
            state_config=state_config,
            registry_config=registry_config,
            agent_registry_config=agent_registry_config,
            tools=[tool],
        )

        # Give Dapr a moment to persist
        time.sleep(1)

        # Verify team registry (for pub/sub addressing)
        response_team = client.get_state(store_name="statestore", key="agents:default")
        assert response_team.data, "Team registry data should exist"
        team_data = json.loads(response_team.data)
        assert "IntegrationTestAgent" in team_data

        # Verify agent registry (for metadata discovery)
        response_agent = client.get_state(
            store_name="statestore", key="agents-registry"
        )
        assert response_agent.data, "Agent registry data should exist"

        registry_data = json.loads(response_agent.data)
        assert "IntegrationTestAgent" in registry_data

        agent_metadata = registry_data["IntegrationTestAgent"]

        # Verify core metadata fields
        assert agent_metadata["name"] == "IntegrationTestAgent"
        assert agent_metadata["role"] == "Integration Tester"
        assert agent_metadata["goal"] == "Test metadata persistence"
        assert agent_metadata["agent_framework"] == "dapr-agents"
        assert agent_metadata["agent_class"] == "Agent"
        assert agent_metadata["agent_category"] == "agent"

        # Verify tool definitions
        assert "tools" in agent_metadata
        assert len(agent_metadata["tools"]) == 1
        tool_def = agent_metadata["tools"][0]
        assert (
            tool_def["name"] == "SampleToolFunction"
        )  # AgentTool converts to PascalCase
        assert tool_def["tool_type"] == "function"
        assert "sample tool function" in tool_def["description"].lower()

        # Verify component mappings
        assert "components" in agent_metadata
        components = agent_metadata["components"]
        assert "state_stores" in components

        # Verify separate registry stores
        assert "agent_registry" in components["state_stores"]
        assert components["state_stores"]["agent_registry"]["name"] == "statestore"
        assert "team_registry" in components["state_stores"]
        assert components["state_stores"]["team_registry"]["name"] == "statestore"

    finally:
        client.close()


def test_idempotent_reregistration(dapr_container):
    """
    Test that re-registering the same agent with same metadata is idempotent.

    Starts with a clean registry, registers an agent, then simulates a restart
    by clearing the local cache and registering the same agent again with identical
    metadata. This should succeed without error.
    """
    client = get_dapr_client()

    try:
        state_store = StateStoreService(store_name="statestore")
        agent_registry_store = StateStoreService(store_name="statestore")

        state_config = AgentStateConfig(store=state_store)
        agent_registry_config = AgentRegistryConfig(store=agent_registry_store)

        # Register agent first time
        _ = Agent(
            name="IdempotentAgent",
            role="Tester",
            goal="Test idempotency",
            state_config=state_config,
            agent_registry_config=agent_registry_config,
        )

        time.sleep(1)

        # Get initial registry state from agent registry
        response1 = client.get_state(store_name="statestore", key="agents-registry")
        registry1 = json.loads(response1.data)

        # Clear local registry cache to simulate a restart
        Registry.clear_registered_names()

        # Register same agent again (simulating restart with same metadata)
        _ = Agent(
            name="IdempotentAgent",
            role="Tester",
            goal="Test idempotency",
            state_config=AgentStateConfig(
                store=StateStoreService(store_name="statestore")
            ),
            agent_registry_config=AgentRegistryConfig(
                store=StateStoreService(store_name="statestore")
            ),
        )

        time.sleep(1)

        # Get registry state after second registration
        response2 = client.get_state(store_name="statestore", key="agents-registry")
        registry2 = json.loads(response2.data)

        # Should still have the entry
        assert "IdempotentAgent" in registry2

        # Metadata should be identical (no error on idempotent registration)
        # Note: In the future, timestamps might differ but core metadata should match
        assert (
            registry1["IdempotentAgent"]["name"] == registry2["IdempotentAgent"]["name"]
        )
        assert (
            registry1["IdempotentAgent"]["role"] == registry2["IdempotentAgent"]["role"]
        )
        assert (
            registry1["IdempotentAgent"]["goal"] == registry2["IdempotentAgent"]["goal"]
        )

    finally:
        client.close()


def test_registry_error_handling(dapr_container):
    """
    Test Registry error handling for edge cases.
    """
    from dapr_agents.registry.registry import Registry

    # Test: Registry without client should raise error during init
    with pytest.raises(ValueError, match="Dapr client is required"):
        Registry(client=None, store_name="test", store_key="test")

    # Test: Registry without store_name should raise error
    client = get_dapr_client()
    try:
        with pytest.raises(ValueError, match="State store name is required"):
            Registry(client=client, store_name=None, store_key="test")

        # Test: Registry without store_key should raise error
        with pytest.raises(ValueError, match="State store key is required"):
            Registry(client=client, store_name="test", store_key=None)
    finally:
        client.close()
