"""Basic integration tests for the test framework."""

import pytest
from tests.utils import DevelopmentScenario


@pytest.mark.integration
def test_development_scenario_detection(development_scenario):
    """Test that development scenario is detected correctly."""
    assert isinstance(development_scenario, DevelopmentScenario)
    assert development_scenario in list(DevelopmentScenario)


@pytest.mark.integration
def test_test_environment_setup(test_environment):
    """Test that test environment is set up correctly."""
    assert "scenario" in test_environment
    assert "env_loaded" in test_environment
    assert "api_keys" in test_environment
    assert "development_repos" in test_environment
    assert "requirements_met" in test_environment


@pytest.mark.integration
def test_api_keys_detection(api_keys):
    """Test that API keys are detected correctly."""
    assert isinstance(api_keys, dict)
    # Should at least check for common providers
    expected_providers = ["openai", "anthropic", "gemini", "azure_openai"]
    for provider in expected_providers:
        assert provider in api_keys
        assert isinstance(api_keys[provider], bool)


@pytest.mark.integration
def test_component_manager_functionality(component_manager, development_scenario):
    """Test that component manager works correctly."""
    # Test getting components directory
    components_dir = component_manager.get_components_for_scenario(development_scenario)
    assert isinstance(components_dir, str)
    # Check that the path contains expected scenario-related terms
    assert (
        "local_dev" in components_dir
        or "partial_dev" in components_dir
        or "production" in components_dir
    )

    # Test listing available components
    base_components = component_manager.list_available_components()
    assert isinstance(base_components, list)


@pytest.mark.integration
def test_version_manager_functionality(version_manager):
    """Test that version manager works correctly."""
    # Test getting all versions
    versions = version_manager.get_all_versions()
    assert isinstance(versions, dict)
    assert "python" in versions
    assert "dapr" in versions
    assert "python_sdk" in versions
    assert "dapr_agents" in versions

    # Test compatibility checking
    compatibility = version_manager.check_compatibility(versions)
    assert isinstance(compatibility, dict)


@pytest.mark.integration
@pytest.mark.slow
def test_dapr_runtime_management(dapr_runtime):
    """Test that Dapr runtime can be managed correctly."""
    # Test that Dapr runtime is available
    assert dapr_runtime is not None

    # Test getting endpoints
    endpoints = dapr_runtime.get_endpoints()
    assert isinstance(endpoints, dict)
    assert "http" in endpoints
    assert "grpc" in endpoints

    # Test listing components
    components = dapr_runtime.list_components()
    assert isinstance(components, list)
    # Should have at least some components loaded
    assert len(components) > 0


@pytest.mark.integration
def test_sample_tools_fixture(sample_tools):
    """Test that sample tools are provided correctly."""
    assert isinstance(sample_tools, list)
    assert len(sample_tools) > 0

    # Check that tools have required attributes
    for tool in sample_tools:
        assert hasattr(tool, "name")
        assert hasattr(tool, "__call__")


@pytest.mark.integration
def test_sample_messages_fixture(sample_messages):
    """Test that sample messages are provided correctly."""
    assert isinstance(sample_messages, list)
    assert len(sample_messages) > 0

    # Check message format
    for message in sample_messages:
        assert isinstance(message, dict)
        assert "role" in message
        assert "content" in message
        assert message["role"] in ["user", "assistant", "system"]


@pytest.mark.integration
@pytest.mark.slow
def test_dapr_chat_client_creation(dapr_chat_client):
    """Test that DaprChatClient can be created correctly."""
    assert dapr_chat_client is not None
    # Test that client has required attributes
    assert hasattr(dapr_chat_client, "generate")
    assert hasattr(dapr_chat_client, "client")  # Internal DaprInferenceClient
    assert hasattr(dapr_chat_client, "config")  # Client configuration

    # Test that the internal client is properly initialized
    assert dapr_chat_client.client is not None
    assert hasattr(dapr_chat_client.client, "dapr_client")  # Internal DaprClient
