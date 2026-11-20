.DEFAULT_GOAL := help

ARGS ?=

.PHONY: setup dataset dataset-headless clean-data help

setup: ## Create/sync the environment (uv sync)
	uv sync

dataset: ## Generate data/cartpole_dqn.pkl with the live window (ARGS="--seed 1")
	uv run python generate_dataset.py $(ARGS)

dataset-headless: ## Generate data/cartpole_dqn.pkl without the live window
	uv run python generate_dataset.py --no-live $(ARGS)

clean-data: ## Delete generated data (rm -rf data)
	rm -rf data

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'
