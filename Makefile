# ══════════════════════════════════════════════════════════════════════════════
#  GraphPulse — Makefile
#  Real-time graph-based fraud detection platform
# ══════════════════════════════════════════════════════════════════════════════

PYTHON  := python
PYTEST  := pytest
RUFF    := ruff
MYPY    := mypy
DVC     := dvc
DOCKER  := docker
COMPOSE := docker compose -f docker-compose.yml

.DEFAULT_GOAL := help

.PHONY: help install dev-install \
        lint lint-fix typecheck \
        test smoke \
        download-ieee download-elliptic \
        dvc-repro \
        train-lgbm train-catboost train-graphsage train-tgn train-hgt train-all \
        export-onnx bench explain replay \
        up down logs db-migrate mlflow-ui \
        k3s-apply

# ── Help ──────────────────────────────────────────────────────────────────────
help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Install ───────────────────────────────────────────────────────────────────
install:  ## Install production dependencies
	$(PYTHON) -m pip install -e .

dev-install:  ## Install all dependencies including dev extras
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

# ── Lint & Type-check ─────────────────────────────────────────────────────────
lint:  ## Run ruff linter (check only)
	$(RUFF) check ml/ apps/ services/ scripts/

lint-fix:  ## Run ruff linter with auto-fix
	$(RUFF) check --fix ml/ apps/ services/ scripts/
	$(RUFF) format ml/ apps/ services/ scripts/

typecheck:  ## Run mypy static type checker
	$(MYPY) ml/ apps/ services/

# ── Tests ─────────────────────────────────────────────────────────────────────
test:  ## Run full test suite with coverage
	$(PYTEST)

smoke:  ## Run smoke integration test (fast sanity check)
	$(PYTEST) tests/integration/test_smoke_train.py -v

# ── Data acquisition ──────────────────────────────────────────────────────────
download-ieee:  ## Download IEEE-CIS Fraud Detection dataset (requires KAGGLE_*)
	$(PYTHON) scripts/download_ieee.py

download-elliptic:  ## Download Elliptic Bitcoin transaction dataset
	$(PYTHON) scripts/download_elliptic.py

# ── DVC pipeline ──────────────────────────────────────────────────────────────
dvc-repro:  ## Reproduce the full DVC pipeline (features → graph → train → bench)
	$(DVC) repro

# ── Model training ────────────────────────────────────────────────────────────
train-lgbm:  ## Train LightGBM tabular baseline (~5 min, CPU)
	$(PYTHON) -m ml.train.cli fit --config configs/lgbm.yaml

train-catboost:  ## Train CatBoost tabular model (~8 min, CPU)
	$(PYTHON) -m ml.train.cli fit --config configs/catboost.yaml

train-graphsage:  ## Train GraphSAGE inductive GNN (~30 min, GPU)
	$(PYTHON) -m ml.train.cli fit --config configs/graphsage.yaml

train-tgn:  ## Train Temporal Graph Network (~2 hr, RTX 2070 Super)
	$(PYTHON) -m ml.train.cli fit --config configs/tgn.yaml

train-hgt:  ## Train Heterogeneous Graph Transformer (~2.5 hr, RTX 2070 Super)
	$(PYTHON) -m ml.train.cli fit --config configs/hgt.yaml

train-all:  ## Train LightGBM + CatBoost + TGN sequentially
	$(MAKE) train-lgbm
	$(MAKE) train-catboost
	$(MAKE) train-tgn

# ── Export & Benchmark ────────────────────────────────────────────────────────
export-onnx:  ## Export LightGBM model to ONNX for low-latency inference
	$(PYTHON) scripts/export_onnx.py --model lgbm

bench:  ## Benchmark inference latency across all exported models
	$(PYTHON) scripts/bench.py

explain:  ## Generate SHAP explanations for the LightGBM model
	$(PYTHON) -m ml.explain.shap_wrapper --model lgbm

# ── Streaming simulation ───────────────────────────────────────────────────────
replay:  ## Replay historical transactions as Kafka stream at 100 TPS
	$(PYTHON) scripts/replay_stream.py --tps 100

# ── Docker Compose ────────────────────────────────────────────────────────────
up:  ## Start the full GraphPulse stack (all services)
	$(COMPOSE) up -d --build
	@echo ""
	@echo "  Services ready:"
	@echo "  ┌──────────────────────────────────────────────────────┐"
	@echo "  │  API              http://localhost:8000/docs          │"
	@echo "  │  Redpanda Console http://localhost:8080               │"
	@echo "  │  MLflow           http://localhost:5001               │"
	@echo "  │  Grafana          http://localhost:3001               │"
	@echo "  │  Prometheus       http://localhost:9090               │"
	@echo "  │  MinIO            http://localhost:9001               │"
	@echo "  └──────────────────────────────────────────────────────┘"

down:  ## Stop and remove all containers (preserves volumes)
	$(COMPOSE) down

logs:  ## Tail logs from all running services
	$(COMPOSE) logs -f

db-migrate:  ## Run Alembic database migrations
	$(COMPOSE) exec api alembic upgrade head

mlflow-ui:  ## Open MLflow tracking UI in the browser
	open http://localhost:5001

# ── Kubernetes (k3s) ──────────────────────────────────────────────────────────
k3s-apply:  ## Apply all Kubernetes manifests to the local k3s cluster
	kubectl apply -R -f infra/k8s/
