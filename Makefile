# Get all directories within quickstarts
QUICKSTART_DIRS := $(shell find quickstarts -mindepth 1 -maxdepth 1 -type d -exec basename {} \;)

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

# Main target to validate all quickstarts
.PHONY: validate-quickstarts
validate-quickstarts:
	@echo "Validating all quickstart directories..."
	@for dir in $(QUICKSTART_DIRS); do \
		echo "\n=== Validating $$dir ==="; \
		( \
			cd quickstarts && \
			cd "$$dir" && \
			if [ -f requirements.txt ]; then \
				echo "Creating virtual environment for $$dir..." && \
				python3 -m venv .venv && \
				echo "Activating virtual environment and installing requirements..." && \
				. .venv/bin/activate && \
				pip install -r requirements.txt && \
				USING_VENV=true; \
			else \
				echo "No requirements.txt found in $$dir, skipping virtual environment setup"; \
				USING_VENV=false; \
			fi && \
			cd .. && \
			echo "Running validation script for $$dir..." && \
			./validate.sh "$$dir"; \
			RESULT=$$?; \
			if [ "$$USING_VENV" = "true" ]; then \
				cd "$$dir" && \
				echo "Deactivating and cleaning up virtual environment..." && \
				deactivate && \
				rm -rf .venv; \
			fi; \
			exit $$RESULT; \
		); \
		if [ $$? -ne 0 ]; then \
			echo "Validation failed for $$dir. Stopping all validations."; \
			exit 1; \
		fi; \
		sleep 1; \
	done
	@echo "\nAll validations completed successfully!"

# Useful for local development, with a single, controlled venv
.PHONY: validate-quickstarts-local
validate-quickstarts-local:
	@echo "Validating all quickstart directories..."
	@for dir in $(QUICKSTART_DIRS); do \
		echo "\n=== Validating $$dir ==="; \
		(cd quickstarts && ./validate.sh $$dir); \
		if [ $$? -ne 0 ]; then \
			echo "Validation failed for $$dir. Stopping all validations."; \
			exit 1; \
		fi; \
		sleep 2; \
	done
	@echo "\nAll validations completed successfully!"