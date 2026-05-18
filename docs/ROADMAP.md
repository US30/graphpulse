# GraphPulse — Project Roadmap

## 8-Week Build Plan

| Week | Milestone | Status |
|---|---|---|
| 1 | Scaffold + Docker stack (Redpanda, Redis, Postgres, MinIO, MLflow, Prometheus, Grafana) + Feast smoke test | ✅ Complete |
| 2 | IEEE-CIS ingest + Feast offline/online features + EDA notebook | Planned |
| 3 | LightGBM + CatBoost baselines + calibration + SHAP report | Planned |
| 4 | TGN training + ablation vs static GraphSAGE / HGT | Planned |
| 5 | River online learner + ADWIN drift detection (shadow mode) | Planned |
| 6 | Kafka producer/consumer + FastAPI scorer + Grafana p99 latency dashboard | Planned |
| 7 | Stress test (1 k TPS replay) + ONNX/TorchScript export + benchmark JSON | Planned |
| 8 | Next.js live dashboard + README + ABLATION_REPORT.md + demo video | Planned |

---

## Week-by-Week Detail

### Week 1 — Scaffold ✅
- [x] `pyproject.toml` with all dependencies (torch-geometric, lightgbm, catboost, river, feast, kafka, shap, mlflow, prometheus-client)
- [x] `docker-compose.yml` — 12 services
- [x] `Makefile` — 28 targets
- [x] `dvc.yaml` — 8 pipeline stages
- [x] ML models: `lgbm.py`, `catboost.py`, `graphsage.py`, `tgn.py`, `hgt.py`
- [x] Data pipeline: `tabular_dataset.py`, `graph_builder.py`, `temporal_sampler.py`
- [x] Training CLI: `ml/train/cli.py`, `gnn_trainer.py`
- [x] Eval: `metrics.py`, `latency_bench.py`
- [x] Online learning: `river_learner.py`, `adwin_drift.py`
- [x] Explainability: `shap_wrapper.py`, `gnn_explainer.py`
- [x] Feature store: `feast_repo.py`
- [x] Apps: FastAPI (`apps/api/`), Kafka producer/consumer
- [x] Infra: Dockerfiles, Prometheus config, Grafana dashboard, Redpanda config
- [x] Tests: unit + integration smoke
- [x] Scripts: `download_ieee.py`, `download_elliptic.py`, `replay_stream.py`, `export_onnx.py`, `bench.py`
- [x] Notebooks: 01_eda through 05_explain (skeleton)
- [x] CI: GitHub Actions 5-job pipeline
- [x] Docs: ARCHITECTURE.md, ABLATION_REPORT.md, ROADMAP.md

### Week 2 — Data & Features
- [ ] Download IEEE-CIS from Kaggle (590 k rows)
- [ ] Run `make ingest` end-to-end (features.parquet)
- [ ] Materialise Feast feature views to Redis
- [ ] Complete `notebooks/01_eda.ipynb` — fraud rate by category, time series, Vesta feature correlations
- [ ] DVC `dvc repro download,build_features` — track with `dvc push`

### Week 3 — Tabular Baselines
- [ ] `make train-lgbm` — full LightGBM run, log to MLflow
- [ ] `make train-catboost` — full CatBoost run
- [ ] Calibration curves (reliability diagrams)
- [ ] SHAP summary plots + force plots in `notebooks/02_lgbm.ipynb`
- [ ] Optuna HPO: `make hpo-lgbm` (50 trials)

### Week 4 — GNN Training
- [ ] `make graph` — build homo + hetero PyG graphs
- [ ] `make train-graphsage` — GraphSAGE ablation baseline
- [ ] `make train-tgn` — Temporal GNN full training
- [ ] `make train-hgt` — Heterogeneous GNN
- [ ] Q1/Q2/Q3 ablation results logged to MLflow
- [ ] `notebooks/03_tgn.ipynb` — training curves, confusion matrices, PR curves

### Week 5 — Online Learning + Drift
- [ ] River shadow learner integrated into consumer pipeline
- [ ] Simulate distribution shift using PaySim subset
- [ ] Q4 ablation: ADWIN recovery experiment
- [ ] `notebooks/04_online_drift.ipynb` — rolling PR-AUC + ADWIN alpha_t evolution

### Week 6 — Streaming Pipeline
- [ ] `make replay` — 500 TPS IEEE-CIS replay into Redpanda
- [ ] Consumer polls + routes to FastAPI /score
- [ ] Grafana `fraud_metrics.json` dashboard wired to live data
- [ ] Prometheus alerts: fraud_rate > 5%, p99 latency > 50ms

### Week 7 — Export + Benchmark
- [ ] `make export` — LightGBM → ONNX, GraphSAGE → TorchScript
- [ ] `make bench` — p50/p95/p99 latency report → `reports/bench.json`
- [ ] 1 k TPS stress test with `replay_stream.py --tps 1000`
- [ ] Verify p99 < 50 ms end-to-end (producer → consumer → /score)

### Week 8 — Polish + Demo
- [ ] Next.js live dashboard (`apps/ui/`) — real-time fraud feed, score histogram
- [ ] `notebooks/05_explain.ipynb` — SHAP waterfall + GNNExplainer subgraph vis
- [ ] `docs/ABLATION_REPORT.md` — fill in real numbers
- [ ] Demo video: Docker up, replay stream, Grafana dashboard, SHAP explanation
- [ ] GitHub repo finalised with CI green

---

## Future Work (Post-8-Week)

- **Triton Inference Server**: serve TGN on Triton for dynamic batching + multi-GPU
- **Payload sampling**: log 1% of transactions to S3 for offline retraining trigger
- **Evidently**: nightly PSI/KS drift report on feature distributions
- **GraphSAINT**: mini-batch GNN training for 30 M+ node graphs (scaling to full PaySim)
- **Multi-graph**: combine IEEE-CIS + Elliptic + PaySim as a heterogeneous multi-relational graph
- **Conformal calibration**: split conformal prediction intervals on fraud scores
