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

pip install -r requirements.txt
```

#### Option 2 - Using uv:
```bash
# Install everything from lock file
uv sync --all-extras

# Generate lock file with all dependencies
uv lock
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

## Local Development

Sometimes you need to work with local changes from other Dapr repositories.

### Using Local Python Dapr Package Changes
If you need to work with additional Python Dapr packages during local development,
for example, those from [python-sdk](https://github.com/dapr/python-sdk) or [durabletask-python](https://github.com/dapr/durabletask-python), 
then you can follow the same steps above and then install your local versions.
Adjust the paths as needed for your setup.

Example with pip:
```bash
pip install -e ../durabletask-python \
   -e ../python-sdk \
   -e ../python-sdk/ext/dapr-ext-fastapi \
   -e ../python-sdk/ext/dapr-ext-workflow \ 
```

Or using uv:

```bash
uv pip install -e ../durabletask-python \
   -e ../python-sdk \
   -e ../python-sdk/ext/dapr-ext-fastapi \
   -e ../python-sdk/ext/dapr-ext-workflow
```

### Using Local Dapr Runtime Changes
If you need to make changes relating to Dapr runtime during local development,
for example, those from [dapr](https://github.com/dapr/dapr) or [components-contrib](https://github.com/dapr/components-contrib),
then follow these steps.

Working with local components-contrib changes:
1. Make your changes in components-contrib.
2. In `dapr/dapr`, update the root `go.mod` file to point to your local `components-contrib` repository. The override block is near the bottom to uncomment and adjust the path as needed.

#### Using the Dapr CLI with local dapr/dapr changes:
```bash
cd /cmd/daprd
go build -tags=allcomponents -v
cp daprd  ~/.dapr/bin/daprd
```

Once copied, the binary at `~/.dapr/bin/daprd` becomes the version used by `dapr run`.
Adjust this path as needed for your local setup.
Running `dapr version` should show the runtime as `edge`, which confirms your local build is being used.


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

### Integration Tests

> Note: we do not use `pytest-docker-compose` intentionally here because it is not compatible with Python2, 
and requires an old version of pyyaml < version6, but the rest of our project requires >6 for this pkg.

Requires Dapr CLI to be installed.

```
# Install test dependencies
pip install -e .[test]

# Set API key (required)
export OPENAI_API_KEY=your_key_here

# Run all integration tests
pytest tests/integration/quickstarts/ -v -m integration

# Run specific test file
pytest tests/integration/quickstarts/test_01_dapr_agents_fundamentals.py -v

# Run specific test func
pytest -m integration -v integration/quickstarts/test_01_dapr_agents_fundamentals.py::TestHelloWorldQuickstart::test_chain_tasks

# Run with coverage
pytest tests/integration/quickstarts/ -v -m integration --cov=dapr_agents
```

> Note: Parallel execution can be enabled with pytest-xdist using -n auto or -n <num>. Example: `pytest -n auto -m integration`.

To use an existing venv to speed up local development time, then you can update the quickstarts to set `create_venv` to `True` as a parameter in `run_quickstart_script`. Alternatively, you can set the env var setting: `USE_EXISTING_VENV=true`.

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

### Internal class structuring/setup
When to use Pydantic vs dataclasses:
- Use Pydantic for:
  - Data crossing trust boundaries or is persisted: API payloads, pub/sub messages, persisted state (workflow state, timeline messages, trigger/broadcast schemas, tool records, etc.).
  - Schemas requiring coercion, validation, or versioned migrations.
- Use dataclasses for:
  - Agent construction knobs you pass in code (ie agent config classes).
  - Dependency injection of services/stores/policies and behavior hooks.

Mental model:
- Think “config vs data”:
  - Config you wire at construction time → dataclasses.
  - Data the system processes/persists at runtime → Pydantic.


### Using the env-template resolver

For Dapr component files that reference environment variables (e.g., `{{OPENAI_API_KEY}}` or `${{OPENAI_API_KEY}}`), use the helper to render a temporary resources folder and pass it to Dapr:
```bash
dapr run --resources-path $(quickstarts/resolve_env_templates.py quickstarts/01-dapr-agents-fundamentals/components) -- python 03_durable_agent.py
```

The helper scans only `.yaml`/`.yml` files (non-recursive), replaces placeholders with matching env var values, writes processed files to a temp directory, and prints that directory path.
