# Test targets
.PHONY: test
test:
	@echo "Running tests..."
	python -m pytest tests/ -v --tb=short

.PHONY: test-cov
test-cov:
	@echo "Running tests with coverage..."
	python -m pytest tests/ -v --cov=dapr_agents --cov-report=term-missing --cov-report=html

.PHONY: test-install
test-install:
	@echo "Installing test dependencies..."
	pip install -e .[test]

.PHONY: test-all
test-all: test-install test-cov
	@echo "All tests completed!"

# Pre-commit hook targets
.PHONY: hooks-install
hooks-install:
	@echo "Installing pre-push hooks..."
	pre-commit install --hook-type pre-push

.PHONY: hooks-uninstall
hooks-uninstall:
	@echo "Uninstalling pre-push hooks..."
	pre-commit uninstall --hook-type pre-push

.PHONY: hooks-run
hooks-run:
	@echo "Running all pre-push hooks..."
	pre-commit run --all-files --hook-stage pre-push

.PHONY: hooks-run-all
hooks-run-all:
	@echo "Running all pre-push hooks plus integration tests..."
	@echo "Step 1/2: Running pre-push hooks (format, lint, type check, unit tests)..."
	pre-commit run --all-files --hook-stage pre-push
	@echo "Step 2/2: Running integration tests..."
	uv run pytest tests -m integration -v

.PHONY: format
format:
	@echo "Formatting code with ruff..."
	uv run ruff format dapr_agents tests

.PHONY: lint
lint:
	@echo "Linting with flake8..."
	uv run flake8 dapr_agents tests --ignore=E501,F401,W503,E203,E704

.PHONY: typecheck
typecheck:
	@echo "Type checking with mypy..."
	uv run mypy --config-file mypy.ini

.PHONY: test-unit
test-unit:
	@echo "Running unit tests..."
	uv run pytest tests -m "not integration" -v
