.DEFAULT_GOAL := help

ARGS ?=

.PHONY: setup dataset dataset-headless train eval experiments summary test clean-data clean-runs help

setup: ## Create/sync the environment (uv sync)
	uv sync

dataset: ## Generate data/cartpole_dqn.pkl with the live window (ARGS="--seed 1")
	uv run python generate_dataset.py $(ARGS)

dataset-headless: ## Generate data/cartpole_dqn.pkl without the live window
	uv run python generate_dataset.py --no-live $(ARGS)

train: ## Train one DT (ARGS="--K 5 --seed 1", "--bc", "--hide-velocity", "--device cuda")
	uv run python train_dt.py $(ARGS)

eval: ## Target-return sweep for one run (RUN=runs/dt_K20_full_s0)
	uv run python eval_dt.py $(or $(RUN),runs/dt_K20_full_s0) $(ARGS)

experiments: ## Full grid: DT K=1/5/20 + BC, full/no-velocity obs, 5 seeds
	uv run python run_experiments.py $(ARGS)

summary: ## Rebuild runs/summary plots and table from finished runs
	uv run python run_experiments.py --summary-only $(ARGS)

test: ## Run tests
	uv run pytest -q

clean-data: ## Delete generated data (rm -rf data)
	rm -rf data

clean-runs: ## Delete trained models and results (rm -rf runs)
	rm -rf runs

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'
