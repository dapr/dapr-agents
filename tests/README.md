# Testing Guide

This directory contains tests for the dapr-agents framework.

## Test Types

The test suite uses pytest markers to distinguish between fast unit tests and slower integration tests.

### Unit Tests (Default)

Fast tests that use mocked dependencies. Run by default with pytest.

**Characteristics:**
- ✅ Fast execution (~6 seconds for 178 tests)
- ✅ No external dependencies (Docker, Dapr, Redis)
- ✅ Use mocks for Dapr SDK and external services
- ✅ Isolated and deterministic
- ✅ Run automatically in CI/CD on every commit

**Running:**
```bash
# Run all unit tests (excludes integration tests)
pytest -m "not integration"

# Run with verbose output
pytest -m "not integration" -v

# Run all tests (unit + integration)
pytest
```

### Integration Tests

Tests that require Docker and real Dapr containers. Marked with `@pytest.mark.integration`.

**Characteristics:**
- 🐳 Require Docker daemon running
- 🕐 Slower execution (~10 seconds with container startup)
- 🔧 Use testcontainers to spin up:
  - Redis (state store)
  - Dapr sidecar
  - Dapr placement service
- ✅ Test real Dapr API interactions
- ✅ Validate actual state persistence

**Running:**
```bash
# Run integration tests only
pytest -m integration -v

# Run specific integration test file
pytest tests/agents/test_agent_registration_integration.py -v

# Run with detailed output
pytest -m integration -xvs
```

## Pytest Configuration

Test behavior is configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "integration: marks tests as integration tests (require Docker, Dapr containers, slow)",
]
filterwarnings = [
    "ignore::RuntimeWarning:unittest.mock",
    "ignore::RuntimeWarning:inspect",
    "ignore::DeprecationWarning",
]
```

**Why warnings are suppressed:**
- **RuntimeWarning**: Production code bridges sync Dapr workflow activities with async agent methods using `asyncio.run()`. AsyncMock triggers warnings in these sync→async bridge points, but the code executes correctly.
- **DeprecationWarning**: Python stdlib deprecations, not our code.

## Test Organization

### Agent Registration Tests

- **`tests/agents/test_agent_metadata.py`** (Unit Tests)
  - Tests metadata building logic without Dapr
  - Fast execution (~3 seconds, 13 tests)
  - Tests `_build_agent_metadata()`, `_extract_component_mappings()`, `_extract_tool_definitions()`
  - Uses `@patch("dapr.clients.DaprClient")` to mock Dapr SDK
  - No external dependencies

- **`tests/agents/test_agent_registration_integration.py`** (Integration Tests)
  - Tests with real Dapr containers using testcontainers
  - ~10 second execution (includes Docker container startup, 3 tests)
  - Tests full metadata persistence, idempotent re-registration, dual registry system
  - Validates actual state store operations
  - Marked with `@pytest.mark.integration`
  - Requires Docker daemon running

### Other Test Files

All test files follow the same pattern:
- **Unit tests**: Default, fast, mocked dependencies
- **Integration tests**: Marked with `@pytest.mark.integration`, require Docker

Use `pytest --co -q` to list all tests and their markers.

## Running Tests

### Quick Development Cycle (Unit Tests Only - Recommended)

Fast feedback loop for local development:

```bash
# Run unit tests only (default, ~6 seconds)
pytest -m "not integration" -v

# With coverage report
pytest -m "not integration" --cov=dapr_agents --cov-report=term-missing

# Watch mode (requires pytest-watch)
ptw -- -m "not integration"
```

**When to use:** Always during development. Fast, no Docker needed.

### Full Test Suite (CI/CD)

Complete validation including integration tests:

```bash
# Run all tests (unit + integration, ~14 seconds)
pytest -v

# With coverage
pytest --cov=dapr_agents --cov-report=html

# Only integration tests (~10 seconds)
pytest -m integration -v
```

**When to use:** Before commits, in CI/CD pipelines, validating Dapr interactions.

### Specific Test Files

```bash
# Run specific unit test file
pytest tests/agents/test_agent_metadata.py -v

# Run specific integration test file (requires Docker)
pytest tests/agents/test_agent_registration_integration.py -v

# Run specific test class
pytest tests/agents/test_agent_metadata.py::TestAgentMetadataBuilding -v

# Run specific test function
pytest tests/agents/test_agent_metadata.py::TestAgentMetadataBuilding::test_build_metadata_basic_fields -v

# List all tests without running
pytest --co -q

# List only integration tests
pytest --co -q -m integration
```

## Test Markers

Tests use pytest markers for categorization:

| Marker | Purpose | Speed | Dependencies | Usage |
|--------|---------|-------|--------------|-------|
| **(none)** | Unit test | Fast (~6s total) | None | Default, always run |
| `@pytest.mark.integration` | Integration test | Slow (~10s per file) | Docker, Dapr | Opt-in with `-m integration` |

### Marking Integration Tests

Add the marker to individual tests or entire test files:

```python
import pytest

# Mark a single test
@pytest.mark.integration
def test_with_real_dapr():
    # Uses real Dapr containers
    pass

# Mark entire test file (at top of file)
pytestmark = pytest.mark.integration

def test_one():
    pass

def test_two():
    pass
```

## Mocking Strategy

### Unit Tests (Default)

Mock all external dependencies:

```python
from unittest.mock import Mock, MagicMock, patch

@patch("dapr.clients.DaprClient")
def test_agent_metadata(mock_client_cls):
    """Unit test with mocked Dapr client."""
    # Mock the context manager behavior
    mock_client = MagicMock()
    mock_client_cls.return_value.__enter__.return_value = mock_client
    mock_client_cls.return_value.__exit__.return_value = None
    
    # Mock state responses
    mock_response = Mock()
    mock_response.data = json.dumps({}).encode("utf-8")
    mock_response.etag = "test-etag"
    mock_client.get_state.return_value = mock_response
    
    # Test agent logic without real Dapr
    agent = Agent(name="TestAgent", role="Tester")
    metadata = agent._build_agent_metadata()
    
    assert metadata.name == "TestAgent"
```

**Key mocking patterns:**
- Mock `DaprClient` as a context manager (`__enter__`, `__exit__`)
- Mock state store responses with `.data` and `.etag` attributes
- Use `Mock()` instead of `AsyncMock()` to avoid unawaited coroutine warnings
- Mock is configured in `tests/conftest.py` with `autouse` fixture for Dapr health checks

### Integration Tests (`@pytest.mark.integration`)

Use real Dapr runtime with testcontainers:

```python
import pytest
from testcontainers.redis import RedisContainer

@pytest.mark.integration
def test_real_dapr_persistence(dapr_container):
    """Integration test with real Dapr + Redis."""
    client = DaprClient()
    
    # Real Dapr state store operations
    agent = Agent(
        name="RealAgent",
        role="Tester",
        state_config=AgentStateConfig(store=StateStoreService("statestore")),
        agent_registry_config=AgentRegistryConfig(store=StateStoreService("statestore"))
    )
    
    # Verify actual persistence
    response = client.get_state(store_name="statestore", key="agents-registry")
    assert response.data
    
    client.close()
```

**Key integration patterns:**
- Use `testcontainers` for Redis, Dapr sidecar, Dapr placement
- Create real `DaprClient()` instances (no mocking)
- Validate actual state store operations
- Test cross-service interactions
- Fixtures manage container lifecycle in `tests/agents/test_agent_registration_integration.py`

## Coverage

Target coverage for the registry module: **>70%**

```bash
# Check coverage for registry module
pytest tests/agents/test_agent_metadata.py tests/agents/test_agent_registration_integration.py \
  --cov=dapr_agents/registry --cov-report=term-missing

# Full coverage report
pytest --cov=dapr_agents --cov-report=html
open htmlcov/index.html  # View in browser
```

## Best Practices

1. **Write unit tests first** - They run fast and catch most bugs
2. **Add integration tests for critical paths** - Verify real Dapr interaction
3. **Use descriptive test names** - Test name should describe what is being tested
4. **Keep tests isolated** - Each test should be independent
5. **Clean up resources** - Use fixtures and teardown to clean up
6. **Mark slow tests** - Use `@pytest.mark.integration` for slow tests

## Troubleshooting

### Unit Tests Hang or Timeout

**Symptom:** Tests like `test_agent_metadata.py` are slow or hang.

**Cause:** Unit tests attempting to connect to real Dapr sidecar.

**Solution:**
- Verify `tests/conftest.py` has the `skip_dapr_health_check_for_unit_tests` fixture
- Check that Dapr SDK is properly mocked in `conftest.py`
- Run with `-xvs` to see where it's hanging

### Integration Tests Fail with "Connection Refused"

**Symptom:** `Connection refused` errors on ports 3500, 50001, 6379, 50005.

**Cause:** Docker not running or ports unavailable.

**Solution:**
- Ensure Docker daemon is running: `docker ps`
- Check port availability: `lsof -i :3500,50001,6379,50005`
- Try running with verbose output: `pytest -m integration -xvs`
- Check testcontainer logs in test output

### Integration Tests Timeout

**Symptom:** Tests timeout waiting for containers to start.

**Cause:** Insufficient Docker resources or slow container startup.

**Solution:**
- Increase Docker resources (memory/CPU) in Docker Desktop settings
- Check container health: `docker ps` then `docker logs <container-id>`
- Verify Dapr health: `curl http://localhost:3500/v1.0/healthz/outbound`
- Wait longer or increase timeout in test fixtures

### Import Errors or Module Not Found

**Symptom:** `ModuleNotFoundError` or import failures.

**Cause:** Missing test dependencies.

**Solution:**
```bash
# Install all test dependencies
pip install -e ".[test]"

# For integration tests, also ensure Docker is available
docker --version
```

### Warnings About Unawaited Coroutines

**Symptom:** `RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never awaited`

**Cause:** AsyncMock used in sync context (production code bridges sync/async).

**Solution:** These warnings are suppressed in `pyproject.toml` as they don't indicate bugs. If you see them:
- Ensure you're using latest `pyproject.toml`
- Verify `filterwarnings` configuration is present
- This is expected behavior for Dapr workflow activities that bridge sync/async code

### Test Failures After Git Pull

**Symptom:** Tests that previously passed now fail.

**Cause:** Dependencies or configuration changed.

**Solution:**
```bash
# Reinstall dependencies
pip install -e ".[test]"

# Clear pytest cache
rm -rf .pytest_cache

# Run tests with verbose output
pytest -xvs
```
