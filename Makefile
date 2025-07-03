# Makefile for Dapr Agents Development

.PHONY: help install install-dev install-test clean test test-unit test-integration test-all lint format check validate-quickstarts validate-quickstarts-local validate-quickstart

# Default target
help:
	@echo "Dapr Agents Development Commands:"
	@echo ""
	@echo "Setup Commands:"
	@echo "  install          Install package with basic dependencies"
	@echo "  install-dev      Install for local development (includes local python-sdk)"
	@echo "  install-test     Install for testing (includes local python-sdk)"
	@echo "  install-ci       Install for CI/CD testing (no local dependencies)"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only (fast)"
	@echo "  test-integration Run integration tests (requires Dapr)"
	@echo "  test-fast        Run fast tests (unit + echo integration)"
	@echo "  test-providers   Run provider-specific tests (requires API keys)"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  lint             Run all linters (mypy, flake8)"
	@echo "  format           Format code (black, isort)"
	@echo "  check            Run all checks (lint + format check)"
	@echo ""
	@echo "Utility Commands:"
	@echo "  clean            Clean up build artifacts and caches"
	@echo "  dapr-start       Start Dapr runtime for testing"
	@echo "  dapr-stop        Stop Dapr runtime"
	@echo ""
	@echo "Quickstart Commands:"
	@echo "  validate-quickstarts       Validate all quickstarts (isolated venvs)"
	@echo "  validate-quickstarts-local Validate quickstarts (current env)"
	@echo "  validate-quickstart QS=<name> Validate single quickstart"

# Installation targets
install:
	pip install .

install-dev:
	pip install -r requirements-dev-local.txt

install-test:
	pip install -r requirements-test-local.txt

install-ci:
	pip install .[test-all]

# Testing targets
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v -m "unit"

test-integration:
	pytest tests/integration/ -v -m "integration"

test-fast:
	pytest tests/unit/ tests/integration/ -v -m "not slow and not requires_api_key"

test-providers:
	pytest tests/integration/providers/ -v -m "requires_api_key"

test-performance:
	pytest tests/performance/ -v -m "performance"

# Code quality targets
lint:
	mypy dapr_agents/
	flake8 dapr_agents/ tests/

format:
	black dapr_agents/ tests/
	isort dapr_agents/ tests/

format-check:
	black --check dapr_agents/ tests/
	isort --check-only dapr_agents/ tests/

check: lint format-check

# Utility targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

dapr-start:
	python tools/run_dapr_dev.py --app-id test-app --components ./tests/components/minimal &

dapr-stop:
	pkill -f "daprd.*test-app" || true

# Development workflow shortcuts
dev-setup: install-dev
	@echo "Development environment ready!"
	@echo "Run 'make test-fast' to verify setup"

ci-setup: install-ci
	@echo "CI environment ready!"
	@echo "Run 'make test-unit' for fast CI tests"

# Pre-commit workflow
pre-commit: format lint test-fast
	@echo "Pre-commit checks passed!"

# Quickstart validation
QUICKSTART_DIRS := $(shell find quickstarts -mindepth 1 -maxdepth 1 -type d -exec basename {} \;)

validate-quickstarts:
	@echo "🧪 Validating all quickstarts with isolated environments..."
	@python tools/validate_quickstarts.py --isolated

validate-quickstarts-local:
	@echo "🧪 Validating all quickstarts with current environment..."
	@python tools/validate_quickstarts.py

validate-quickstart:
	@if [ -z "$(QS)" ]; then \
		echo "❌ Usage: make validate-quickstart QS=<quickstart-name>"; \
		echo "   Example: make validate-quickstart QS=01-hello-world"; \
		exit 1; \
	fi
	@python tools/validate_quickstarts.py --quickstart $(QS)