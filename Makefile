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
	pip install install -e .[test]

.PHONY: test-all
test-all: test-install test-cov
	@echo "All tests completed!"