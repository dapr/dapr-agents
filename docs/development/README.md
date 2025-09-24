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
uv lock --all-extras
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


## Contributing to Dapr Agents Quickstarts

### Add runnable README tests with Mechanical Markdown

We annotate README code blocks with lightweight metadata so they can be executed and validated automatically using [mechanical-markdown](https://github.com/dapr/mechanical-markdown).

- **Install**:
   ```bash
   pip install mechanical-markdown
   ```

- **Annotate steps**: Wrap executable blocks with STEP markers (as in `01-hello-world/README.md`). Example:
   ```markdown
   <!-- STEP
   name: Run a simple workflow example
   expected_stderr_lines:
   - "Creating virtual environment"
   expected_stdout_lines:
   - "Result:"
   output_match_mode: substring
   -->
   ```
   
   ```bash
   dapr run --app-id dapr-agent-wf -- python 04_chain_tasks.py
   ```
   
   ```markdown
   <!-- END_STEP -->
   ```

- **Run locally**:

   ```bash
   # cd to the directory you want to run mechanical markdown on
   cd quickstarts/01-hello-world/

   # run mechanical markdown on the README.md file
   mm.py README.md
   ```

   Tip: Use dry-run first to see which commands will execute:
   ```bash
   mm.py --dry-run README.md
   ```

   Before running, export your `.env` so keys are available to the shell used by `mm.py`:
   ```bash
   # export the environment variables from the .env file if it is in the current directory
   export $(grep -v '^#' .env | xargs)
   ```

- **Use the validate.sh script**:
   This script will run the mechanical markdown script against the quickstarts directory you want to validate. It is a simpler wrapper around the mechanical markdown script.

   ```bash
   # cd to the quickstarts directory
   cd quickstarts

   # run the validate.sh script against the quickstarts directory you want to validate
   ./validate.sh 01-hello-world
   ```

### Troubleshooting

- Remember to setup the virtual environment and install the dependencies before running the mechanical markdown script.
- You can also add the setup of the virtual environment and installation of the dependencies to the mechanical markdown script execution scope. For example:
   ```markdown
   <!-- STEP
   name: Setup virtual environment
   expected_stderr_lines:
   - "Creating virtual environment"
   -->
   ```
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   Your script run:
   ```bash
   source .venv/bin/activate
   python my_script.py
   ```

   ```markdown
   <!-- END_STEP -->
   ```
- If you have errors related to order of execution, you can use the `match_order` keyword to specify the order of execution. For example:
   ```markdown
   <!-- STEP
   name: Run my script
   match_order: none
   -->
   ```
- Remember that mechanical markdown is YAML so follow YAML syntax. One issue at times is with strings and quotes inside of strings. You can use single quotes inside of double quotes and vice versa. For example:
   ```markdown
   <!-- STEP
   name: Run my script
   expected_stdout_lines:
   - '"Hello, world!"'
   -->
   ```

### Using the env-template resolver

For Dapr component files that reference environment variables (e.g., `{{OPENAI_API_KEY}}` or `${{OPENAI_API_KEY}}`), use the helper to render a temporary resources folder and pass it to Dapr:
```bash
dapr run --resources-path $(quickstarts/resolve_env_templates.py quickstarts/01-hello-world/components) -- python 03_durable_agent.py
```

The helper scans only `.yaml`/`.yml` files (non-recursive), replaces placeholders with matching env var values, writes processed files to a temp directory, and prints that directory path.

### CI note

Mechanical Markdown steps are not executed in CI yet due to the absence of a repository OpenAI API key. Please validate locally as shown above. Once CI secrets are provisioned, these steps can be enabled for automated verification.

