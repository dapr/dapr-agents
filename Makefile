# Default target
.PHONY: all
all: validate-quickstarts

# Get all directories within quickstarts
QUICKSTART_DIRS := $(shell find quickstarts -mindepth 1 -maxdepth 1 -type d -exec basename {} \;)

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
			exit $$RESULT \
		) || echo "Validation failed for $$dir"; \
	done
	@echo "\nAll validations completed!"