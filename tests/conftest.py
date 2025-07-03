"""Pytest configuration and shared fixtures for Dapr Agents testing."""

import pytest
import logging
from pathlib import Path
from typing import Dict, Generator

from tests.utils import (
    ScenarioManager,
    DevelopmentScenario,
    EnvManager,
    ComponentManager,
    DaprManager,
    VersionManager,
)

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def scenario_manager():
    """Scenario manager for detecting development environment."""
    return ScenarioManager()


@pytest.fixture(scope="session")
def development_scenario(scenario_manager):
    """Detect and configure development scenario."""
    scenario = scenario_manager.detect_scenario()
    logger.info(f"Detected development scenario: {scenario.value}")
    logger.info(f"Description: {scenario_manager.get_scenario_description(scenario)}")
    return scenario


@pytest.fixture(scope="session")
def env_manager():
    """Environment manager for test setup."""
    return EnvManager()


@pytest.fixture(scope="session")
def component_manager():
    """Component manager for test configurations."""
    return ComponentManager()


@pytest.fixture(scope="session")
def version_manager():
    """Version manager for compatibility checking."""
    return VersionManager()


@pytest.fixture(scope="session")
def test_environment(env_manager, development_scenario):
    """Setup test environment based on development scenario."""
    setup_result = env_manager.setup_for_scenario(development_scenario)

    if not setup_result["requirements_met"]:
        pytest.skip(
            f"Test environment requirements not met for scenario {development_scenario.value}"
        )

    if setup_result["warnings"]:
        for warning in setup_result["warnings"]:
            logger.warning(f"Test environment warning: {warning}")

    return setup_result


@pytest.fixture(scope="session")
def dapr_runtime(
    development_scenario, component_manager, test_environment
) -> Generator[DaprManager, None, None]:
    """Session-scoped Dapr runtime based on development scenario."""
    # Use CLI for production-like scenarios, local binary for development
    use_cli = development_scenario in [
        DevelopmentScenario.AGENT_ONLY,
        DevelopmentScenario.PRODUCTION,
    ]
    manager = DaprManager(use_cli=use_cli)

    # Setup CLI if needed
    if use_cli and not manager._is_dapr_cli_installed():
        logger.info("Setting up Dapr CLI for production-like testing...")
        if not manager.setup_dapr_cli(interactive=False):
            pytest.skip("Could not setup Dapr CLI for production testing")

    # Get components directory for scenario
    components_dir = component_manager.get_components_for_scenario(development_scenario)

    # Ensure components exist for scenario
    component_manager.create_scenario_components(development_scenario)

    # Start Dapr runtime
    app_id = f"test-{development_scenario.value.replace('_', '-')}"
    success = manager.start_dapr(components_dir, app_id, development_scenario)

    if not success:
        pytest.skip(
            f"Could not start Dapr runtime for scenario {development_scenario.value}"
        )

    # Validate that required components are loaded
    components = manager.list_components()
    logger.info(f"Dapr started with {len(components)} components: {components}")

    yield manager

    # Cleanup
    manager.stop_dapr()


@pytest.fixture(scope="session")
def api_keys(env_manager):
    """Available API keys for testing."""
    return env_manager.check_api_keys()


@pytest.fixture
def skip_if_no_api_key(api_keys):
    """Decorator to skip tests if required API key is missing."""

    def _skip_if_no_key(provider: str):
        if not api_keys.get(provider, False):
            pytest.skip(f"API key for {provider} not available")

    return _skip_if_no_key


@pytest.fixture
def dapr_endpoints(dapr_runtime):
    """Dapr endpoints for testing."""
    return dapr_runtime.get_endpoints()


@pytest.fixture
def sample_tools():
    """Common tool definitions for testing."""
    from dapr_agents.tool.base import tool

    @tool
    def get_weather(location: str) -> str:
        """Get weather information for a location."""
        return f"Weather in {location}: Sunny, 72°F"

    @tool
    def calculate(expression: str) -> str:
        """Calculate a mathematical expression."""
        try:
            # Simple safe evaluation for testing
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def get_time() -> str:
        """Get current time."""
        from datetime import datetime

        return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    return [get_weather, calculate, get_time]


@pytest.fixture
def sample_messages():
    """Sample conversation messages for testing."""
    return [
        {"role": "user", "content": "What's the weather like in San Francisco?"},
        {"role": "user", "content": "Calculate 2 + 2"},
        {"role": "user", "content": "What time is it?"},
        {
            "role": "user",
            "content": "Tell me about the weather in New York and what time it is",
        },
    ]


@pytest.fixture
def dapr_chat_client(dapr_endpoints):
    """DaprChatClient configured for testing."""
    from dapr_agents.llm.dapr import DaprChatClient

    # Create client without endpoint parameters since DaprChatClient
    # uses internal DaprClient() that handles connection automatically
    return DaprChatClient()


# Pytest markers for test categorization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (require Dapr runtime)"
    )
    config.addinivalue_line("markers", "e2e: End-to-end tests (full workflows)")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow tests (may take minutes)")
    config.addinivalue_line("markers", "requires_api_key: Tests requiring API keys")
    config.addinivalue_line(
        "markers", "provider_specific: Tests for specific providers"
    )
    config.addinivalue_line(
        "markers", "local_full: Tests requiring full local development"
    )
    config.addinivalue_line(
        "markers", "local_partial: Tests requiring partial local development"
    )
    config.addinivalue_line("markers", "agent_only: Tests for agent-only development")
    config.addinivalue_line("markers", "production: Tests for production scenarios")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on scenario and markers."""
    # Get current scenario
    scenario_manager = ScenarioManager()
    current_scenario = scenario_manager.detect_scenario()

    # Skip tests that don't match current scenario
    scenario_markers = {
        DevelopmentScenario.LOCAL_FULL: "local_full",
        DevelopmentScenario.LOCAL_PARTIAL: "local_partial",
        DevelopmentScenario.AGENT_ONLY: "agent_only",
        DevelopmentScenario.PRODUCTION: "production",
    }

    for item in items:
        # Check if test has scenario-specific markers
        for scenario, marker_name in scenario_markers.items():
            if item.get_closest_marker(marker_name):
                if current_scenario != scenario:
                    item.add_marker(
                        pytest.mark.skip(
                            reason=f"Test requires {scenario.value} scenario, current: {current_scenario.value}"
                        )
                    )


@pytest.fixture(autouse=True)
def test_logging(request):
    """Setup test-specific logging."""
    test_name = request.node.name
    logger.info(f"Starting test: {test_name}")

    yield

    logger.info(f"Completed test: {test_name}")


# Session-scoped fixtures for resource management
@pytest.fixture(scope="session", autouse=True)
def session_setup(development_scenario, version_manager):
    """Session setup and validation."""
    logger.info("=== Test Session Starting ===")
    logger.info(f"Development Scenario: {development_scenario.value}")

    # Validate environment
    validation = version_manager.validate_test_environment()
    if not validation["all_compatible"]:
        logger.warning("Environment compatibility issues detected:")
        for warning in validation["warnings"]:
            logger.warning(f"  - {warning}")
        for error in validation["errors"]:
            logger.error(f"  - {error}")

    yield

    logger.info("=== Test Session Ending ===")


@pytest.fixture(scope="session")
def test_data_dir():
    """Test data directory."""
    return Path(__file__).parent / "fixtures" / "data"


@pytest.fixture(scope="session")
def temp_components_dir(tmp_path_factory):
    """Temporary components directory for tests."""
    return tmp_path_factory.mktemp("components")


# Helper functions for test scenarios
def get_available_providers(api_keys: Dict[str, bool]) -> list:
    """Get list of available providers based on API keys."""
    providers = ["echo"]  # Echo is always available

    for provider, available in api_keys.items():
        if available:
            providers.append(provider)

    return providers


def require_scenario(scenario: DevelopmentScenario):
    """Decorator to require specific development scenario."""

    def decorator(func):
        return pytest.mark.skipif(
            ScenarioManager().detect_scenario() != scenario,
            reason=f"Test requires {scenario.value} development scenario",
        )(func)

    return decorator


# Export commonly used items
__all__ = [
    "development_scenario",
    "dapr_runtime",
    "dapr_chat_client",
    "sample_tools",
    "sample_messages",
    "api_keys",
    "skip_if_no_api_key",
    "get_available_providers",
    "require_scenario",
]
