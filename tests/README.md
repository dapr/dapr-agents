# Testing Guide

This directory contains tests for the dapr-agents framework.

## Test Types

### Unit Tests

Fast tests that use mocked dependencies. Run by default with pytest.

```bash
# Run all unit tests (excludes integration tests)
pytest -m "not integration"

# Run all tests including unit tests
pytest
```

### Integration Tests

Tests that require Docker and real Dapr containers. These tests:
- Use testcontainers to spin up Redis, Dapr sidecar, and Dapr placement service
- Take ~10 seconds to run (with container startup)
- Require Docker to be running
- Are excluded by default when running unit tests

```bash
# Run integration tests only
pytest -m integration -v

# Run specific integration test file
pytest tests/agents/test_agent_registration_integration.py -v
```

## Test Organization

### Agent Registration Tests

- **`tests/agents/test_agent_metadata.py`**
  - Unit tests for metadata building logic
  - Fast execution (~90 seconds)
  - Tests `_build_agent_metadata()`, `_extract_component_mappings()`, `_extract_tool_definitions()`
  - No Dapr required - pure logic testing

- **`tests/agents/test_agent_registration_integration.py`**
  - Integration tests with real Dapr containers
  - ~10 second execution (includes Docker container startup)
  - Tests full metadata persistence, idempotent re-registration, error handling with real Dapr
  - Requires Docker

## Running Tests

### Quick Development Cycle (Unit Tests Only)

```bash
# Fast - run unit tests only
pytest -m "not integration" -v

# With coverage
pytest -m "not integration" --cov=dapr_agents --cov-report=term-missing
```

### Full Test Suite (CI/CD)

```bash
# Run all tests including integration
pytest -v

# With coverage
pytest --cov=dapr_agents --cov-report=html
```

### Specific Test Files

```bash
# Run specific unit test file
pytest tests/agents/test_agent_metadata.py -v

# Run specific integration test (requires Docker)
pytest tests/agents/test_agent_registration_integration.py -v

# Run specific test class
pytest tests/agents/test_agent_metadata.py::TestAgentMetadataStructure -v

# Run specific test function
pytest tests/agents/test_agent_metadata.py::TestAgentMetadataStructure::test_agent_metadata_basic_fields -v
```

## Test Markers

- `@pytest.mark.integration` - Marks tests as integration tests (slow, require Docker)

## Mocking Strategy

### Unit Tests
- Mock `DaprClient` as a context manager
- Mock state transactions and responses
- Fast, isolated, no external dependencies

### Integration Tests
- Use `testcontainers` for real Redis and Dapr containers
- Validate actual Dapr API interactions
- Ensure real-world compatibility

## Coverage

Target coverage for the registry module: **>70%**

```bash
# Check coverage for specific module
pytest tests/agents/test_agent_metadata.py --cov=dapr_agents/registry --cov-report=term-missing
```

## Best Practices

1. **Write unit tests first** - They run fast and catch most bugs
2. **Add integration tests for critical paths** - Verify real Dapr interaction
3. **Use descriptive test names** - Test name should describe what is being tested
4. **Keep tests isolated** - Each test should be independent
5. **Clean up resources** - Use fixtures and teardown to clean up
6. **Mark slow tests** - Use `@pytest.mark.integration` for slow tests

## Troubleshooting

### Integration Tests Fail with "Connection Refused"

- Ensure Docker is running
- Check that ports 3500, 50001, 6379, 50005 are available
- Try running with verbose output: `pytest -m integration -xvs`

### Integration Tests Timeout

- Increase Docker resources (memory/CPU)
- Check Docker logs: `docker ps` then `docker logs <container-id>`
- Verify Dapr health endpoint: `curl http://localhost:3500/v1.0/healthz/outbound`

### Unit Tests Fail

- Ensure all dependencies are installed: `pip install -e ".[test]"`
- Check if mocks are set up correctly
- Run with `-xvs` for detailed output
