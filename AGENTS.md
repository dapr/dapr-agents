# Dapr Agents - Developer Guide

## Project Context

**Full Setup**: See [./docs/development/README.md](./docs/development/README.md)

## Quick Commands

- **Setup**: `uv venv && source .venv/bin/activate && uv sync`
- **Before commit (REQUIRED)**: `tox`
- **Individual checks**:
  - Auto-format: `tox -e ruff`
  - Lint: `tox -e flake8`
  - Type check: `tox -e type`
  - Unit tests: `tox -e pytest`
- **Testing**:
  - Unit tests only: `pytest tests -m "not integration"`
  - Integration tests (requires API keys): `pytest tests -m integration`

## Code Standards

**Branch Names**: `feat/*`, `fix/*`, `bugfix/*`, `hotfix/*`

**Commit Format**: `<type>[scope]: <description>`
- Types: `feat`, `fix`, `docs`, `chore`, `test`, `refactor`, `perf`, `ci`
- Example: `feat(agents): add new orchestrator` or `fix(llm): handle timeout errors`

**Code Quality** (enforced by CI):
- **Formatting**: ruff (auto-format, no exceptions)
- **Linting**: flake8 (ignores: E501, F401, W503, E203)
- **Type Checking**: mypy (config: `./mypy.ini`)
- **All checks MUST pass** before merge

## Testing

**Test Structure**: `./tests/` mirrors `./dapr_agents/` structure
- `agents/`, `llm/`, `workflow/` - Unit tests
- `quickstarts/` - E2E integration tests (requires API keys: `OPENAI_API_KEY`, etc.)

**Run Before Commit**: `tox` (tests Python 3.11, 3.12, 3.13)

**CI** (`./.github/workflows/build.yaml`): ruff → flake8 → mypy → pytest
- Matrix: Python 3.11, 3.12, 3.13
- Failures block merge

## Pull Request Rules

**REQUIRED Before PR**:
1. Run `tox` locally - all checks must pass
2. Use conventional commit format for PR title
3. Update docs in `dapr/docs` repo for: API changes, new features, breaking changes, config options
4. Include "AGENTS.md Notes" in the PR with suggestions to make this prompt better

**PR Will Fail If**:
- Not formatted with ruff
- Flake8 errors exist
- Mypy type errors present
- Any tests fail
- No corresponding docs PR (when required)
