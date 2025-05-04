PYTHON ?= python3
PIP ?= pip3
PROJECT_NAME = vehicle_ml
TEST_DIR = tests

.PHONY: all

install-dev:
	$(PIP) install -e .[dev]

format: ## Format code with black and ruff
	black $(PROJECT_NAME) $(TEST_DIR)
	ruff format $(PROJECT_NAME) $(TEST_DIR)

help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
