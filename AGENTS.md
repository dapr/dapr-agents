# Dapr Agents - Developer Guide

## Project Context

**Full Setup**: See [./docs/development/README.md](./docs/development/README.md)

## Quick Commands

- **Setup**: `uv venv && source .venv/bin/activate && uv sync --group test`
- **Before commit (REQUIRED)**: `uv run ruff format && uv run flake8 dapr_agents tests --ignore=E501,F401,W503,E203,E704 && uv run mypy --config-file mypy.ini && uv run pytest tests -m "not integration"`
- **Individual checks**:
  - Auto-format: `uv run ruff format`
  - Lint: `uv run flake8 dapr_agents tests --ignore=E501,F401,W503,E203,E704`
  - Type check: `uv run mypy --config-file mypy.ini`
  - Unit tests: `uv run pytest tests -m "not integration"`
- **Testing**:
  - Integration tests (requires API keys): `uv run pytest tests -m integration`

## Code Standards

**Branch Names**: `feat/*`, `fix/*`, `bugfix/*`, `hotfix/*`

**Commit Format**: `<type>[scope]: <description>`
- Types: `feat`, `fix`, `docs`, `chore`, `test`, `refactor`, `perf`, `ci`, `style`, `build`
- Example: `feat(agents): add new orchestrator` or `fix(llm): handle timeout errors`

**Versioning**:
Dapr Agents use semantic versioning for releasing. Prefer making changes that allow backward compatability.

**Code Quality** (enforced by CI):
- **Python version**: >=3.11
- **Formatting**: ruff (auto-format, no exceptions)
- **Linting**: flake8 (ignores: E501, F401, W503, E203)
- **Type Checking**: mypy (config: `./mypy.ini`)
- **All checks MUST pass** before merge
- **General**:
  - **Typing**: Enforce strong typing (e.g., `variable: Dict[str, List[str]] = ..`)
  - **Pydantic Models**: For validated/serialized data (APIs, configs, external input)
  - **Dataclasses**: For internal data structures without validation
  - **Logging**: Use module-level logger: `logger = logging.getLogger(__name__)`
  - **ConfigDict**: Add `model_config = ConfigDict(arbitrary_types_allowed=True)` when Pydantic models contain non-serializable types
  - **Custom Exceptions**: Define domain-specific exceptions (inherit from `Exception`); use descriptive names ending in `Error`
  - **Async Execution**: 
    - Check for running loop with `asyncio.get_running_loop()` before creating new loops
    - Use `asyncio.run_coroutine_threadsafe()` when submitting coroutines to a running loop from another thread
    - Create fresh event loops only when no loop is running
  - **Error Handling**: 
    - Always catch specific exceptions before general ones
    - Log errors with context before raising: `logger.error("Operation failed: %s", error, exc_info=True)`
    - Use `span.record_exception(e)` for observability when available
  - **Function Signatures**: Return `None` explicitly for functions with no return value (e.g., `def setup() -> None:`)
  - **Module Exports**: Define `__all__` in `__init__.py` to explicitly control public API

## Testing

**Test Structure**: `./tests/` mirrors `./dapr_agents/` structure
- `agents/`, `llm/`, `workflow/` - Unit tests
- `quickstarts/` - E2E integration tests (requires API keys: `OPENAI_API_KEY`, etc.)

**CI** (`./.github/workflows/build.yaml`): ruff → flake8 → mypy → pytest
- Matrix: Python 3.11, 3.12, 3.13, 3.14
- Failures block merge

## Pull Request Rules

**REQUIRED Before PR**:
1. Run `uv run ruff format && uv run flake8 dapr_agents tests --ignore=E501,F401,W503,E203,E704 && uv run mypy --config-file mypy.ini && uv run pytest tests -m "not integration"` locally - all checks must pass
2. Use conventional commit format for PR title
3. Update docs in `dapr/docs` repo for: API changes, new features, breaking changes, config options
4. Include "AGENTS.md Notes" in the PR with suggestions to make this prompt better

**PR Will Fail If**:
- Commits are not signed
- Not formatted with ruff
- Flake8 errors exist
- Mypy type errors present
- Any tests fail
- No corresponding docs PR (when required)
