# GraphPulse — How to Run

Real-time graph-based fraud detection: GNN (TGN / HGT / GraphSAGE) + LightGBM + Redpanda streaming + online learning.

---

## 1. Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11+ |
| Docker + Docker Compose | Docker 25+ |
| CUDA Toolkit | 12.1 (for GPU training) |
| GPU | RTX 2070 Super (8 GB VRAM) or better |
| Kaggle account | Required for IEEE-CIS dataset |

Install CUDA drivers from https://developer.nvidia.com/cuda-12-1-0-download-archive before proceeding.

---

## 2. Clone & Install

```bash
git clone <repo-url> graphpulse
cd graphpulse

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install the project in editable mode with dev dependencies
make dev-install
```

### PyG (PyTorch Geometric) — special install step

PyG requires a CUDA-specific wheel index. After `make dev-install`, run:

```bash
pip install torch-geometric \
    torch-scatter torch-sparse \
    --extra-index-url https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

> For CPU-only machines replace `cu121` with `cpu`.

---

## 3. Configure Environment

```bash
cp .env.example .env
```

Open `.env` and fill in your Kaggle credentials:

```
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

All other defaults work out of the box for local Docker development.

---

## 4. Start the Stack

```bash
make up
```

This builds and starts all services. Once healthy, the following are available:

| Service | URL |
|---|---|
| REST API (FastAPI) | http://localhost:8000/docs |
| Redpanda Console | http://localhost:8080 |
| MLflow Tracking | http://localhost:5001 |
| Grafana Dashboards | http://localhost:3001 |
| Prometheus Metrics | http://localhost:9090 |
| MinIO Console | http://localhost:9001 |

---

## 5. Create Kafka Topics

```bash
docker exec redpanda rpk topic create graphpulse.transactions graphpulse.scores
```

---

## 6. Download IEEE-CIS Fraud Detection Dataset

```bash
make download-ieee
```

Downloads train_transaction.csv, train_identity.csv, and test files from Kaggle into `data/raw/ieee_cis/`. Requires valid `KAGGLE_*` env vars.

---

## 7. Build Tabular Features

```bash
python -m ml.data.tabular_dataset build-features
```

Runs feature engineering (aggregations, encoding, time-based splits) and writes Parquet files to `data/features/`.

---

## 8. Build Transaction Graph

```bash
python -m ml.data.graph_builder build
```

Constructs a heterogeneous transaction–card–email–device graph and serialises it to `data/graph/` (PyG `HeteroData` format).

---

## 9. Train LightGBM Baseline

```bash
make train-lgbm
```

Expected runtime: ~5 minutes on CPU. Logs metrics (AUC-ROC, AP, F1) to MLflow and writes the model to `artifacts/lgbm/`.

---

## 10. Train TGN (Temporal Graph Network)

```bash
make train-tgn
```

Expected runtime: ~2 hours on RTX 2070 Super. Uses PyG's `TGNMemory` module with edge-level supervision. Checkpoint saved to `artifacts/tgn/`.

---

## 11. Export to ONNX

```bash
make export-onnx
```

Exports the LightGBM model to ONNX format (`artifacts/lgbm/model.onnx`) for sub-millisecond CPU inference in the FastAPI service.

---

## 12. Benchmark Latency

```bash
make bench
```

Measures p50 / p95 / p99 inference latency across all exported models and writes results to `reports/latency_bench.json`.

---

## 13. Generate SHAP Explanations

```bash
make explain
```

Computes SHAP values for the LightGBM model on the test set and saves summary plots + per-transaction explanation CSVs to `reports/explanations/`.

---

## 14. Replay Transaction Stream

```bash
make replay
```

Reads historical transactions from `data/raw/ieee_cis/` and publishes them to `graphpulse.transactions` at 100 transactions/second. The consumer picks them up, calls the API for scoring, and publishes fraud scores to `graphpulse.scores`.

---

## 15. Monitor

- **Grafana** http://localhost:3001 — pre-built dashboards for fraud rate, model latency, Kafka lag, and system metrics.
- **Redpanda Console** http://localhost:8080 — inspect topics, consumer groups, and message throughput in real time.

---

## GPU Memory Reference

| Model | Approximate VRAM | Recommended batch_size |
|---|---|---|
| GraphSAGE | ~1.5 GB | 2048 |
| TGN | ~2–3 GB | 512 |
| HGT | ~3 GB | 256 |

If you encounter OOM errors, reduce `batch_size` in the relevant config (e.g., `configs/tgn.yaml`) and re-run training.

---

## Tip: Full DVC Pipeline

To reproduce all stages from scratch in one command:

```bash
make dvc-repro
```

DVC tracks data and model checksums so only changed stages are re-run.
