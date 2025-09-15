# Development Guide

## Dependencies

This project uses modern Python packaging with `pyproject.toml`. Dependencies are managed as follows:

- Main dependencies are in `[project.dependencies]`
- Test dependencies are in `[project.optional-dependencies.test]`
- Development dependencies are in `[project.optional-dependencies.dev]`

### Working within a virtual environment
Create your python virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

### Generating Requirements Files

If you need to generate requirements files (e.g., for deployment or specific environments):

#### Option 1 - Using pip-tools:
```bash
# Install dev tools
pip install -e ".[dev]"

# Generate requirements.txt
pip-compile pyproject.toml

# Generate dev-requirements.txt
pip-compile pyproject.toml # --extra dev
```

#### Option 2 - Using uv:
```bash
# Generate lock file with all dependencies
uv lock --all-extras

# Install everything from lock file
uv sync --all-extras
```

### Installing Dependencies

#### Option 1 - Using pip:
```bash
# Install main package with test dependencies
pip install -e ".[test]"

# Install main package with development dependencies
pip install -e ".[dev]"

# Install main package with all optional dependencies
pip install -e ".[test,dev]"
```

#### Option 2 - Using uv:
```bash
# Install main package with test dependencies
uv sync --extra=test

# Install main package with development dependencies
uv sync --extra=dev

# Install main package with all optional dependencies
uv sync --all-extras

# Install in editable mode with all extras
uv sync --all-extras --editable
```

## Command Mapping

| pip/pip-tools command | uv equivalent |
|----------------------|---------------|
| `pip-compile pyproject.toml` | `uv lock` |
| `pip-compile --all-extras` | `uv lock` (automatic) |
| `pip install -r requirements.txt` | `uv sync` |
| `pip install -e .` | `uv sync --editable` |
| `pip install -e ".[dev]"` | `uv sync --extra=dev` |
| `pip install -e ".[test,dev]"` | `uv sync --all-extras` |

## Testing

The project uses pytest for testing. To run tests:

```bash
# Run all tests
tox -e pytest

# Run specific test file
tox -e pytest tests/test_random_orchestrator.py

# Run tests with coverage
tox -e pytest --cov=dapr_agents
```

## Code Quality

The project uses several tools to maintain code quality:

```bash
# Run linting
tox -e flake8

# Run code formatting
tox -e ruff

# Run type checking
tox -e type
```

## Development Workflow

### Option 1 - Using pip:
1. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   # Alternatively, you can use uv with:
   # uv sync --extra=dev
   ```

2. Run tests before making changes:
   ```bash
   tox -e pytest
   ```

3. Make your changes

4. Run code quality checks:
   ```bash
   tox -e flake8
   tox -e ruff
   tox -e type
   ```

5. Run tests again:
   ```bash
   tox -e pytest
   ```

6. Submit your changes

## Design/Behavioral Decisions

### DurableAgent Durability
   Scenarios:
   1. every time we run the same app instance any inflight workflow will be resumed.
   1.1 caveat here is wf will continue but you will not get the result.
   2. every time i have a .run() or invoke new workflow via curl or pubsub then a new workflow instance id will be created. If there is an inflight workflow already then it will be resumed, and the new one will be created.
   3. Trace ID = workflow ID and make the tracing pick up from where it left off.
