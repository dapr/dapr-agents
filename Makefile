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
		(cd quickstarts && ./validate.sh $$dir) || echo "Validation failed for $$dir"; \
	done
	@echo "\nAll validations completed!"