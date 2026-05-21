# GraphPulse

Real-time streaming fraud detection on transaction graphs — TGN/HGT/LightGBM + Redpanda + River ADWIN + SHAP + GNNExplainer.

## Owner
Utkarsh Sinha (sinha.utkarshsinha30@gmail.com) — MTech Data Science student building resume portfolio.

## Purpose
Fourth resume project. Fills graph ML, streaming, online learning, drift detection, tabular boosting, fintech/MLE recruiter signal gaps.

## Use Case
Detect financial fraud in real time by modelling card transactions as a heterogeneous graph and combining TGN ensemble with LightGBM, online learning via River, and SHAP + GNNExplainer rationales.

## ML Stack
- **Tabular**: LightGBM, CatBoost (IEEE-CIS, 590k rows)
- **Graph**: Temporal Graph Network (TGN), HGT, GraphSAGE via PyTorch Geometric
- **Online**: River HoeffdingAdaptiveTree + ADWIN drift detector (shadow mode)
- **Explainability**: SHAP TreeExplainer + GNNExplainer subgraph rationales
- **Streaming**: Redpanda (Kafka API) producer/consumer + kafka-python
- **Feature store**: Feast (Redis online, Parquet offline)
- **Serving**: FastAPI + ONNX Runtime (LightGBM) + TorchScript (TGN)
- **MLOps**: MLflow + DVC + Optuna HPO

## Infra Stack
Redpanda + Redis + PostgreSQL + MinIO + MLflow + Prometheus + Grafana + Docker Compose (12 services) + GitHub Actions CI + DVC + Helm/k3s

## GPU Budget (RTX 2070 Super 8 GB)
| Model | VRAM | Train time |
|---|---|---|
| LightGBM | CPU | ~5 min |
| CatBoost | CPU | ~8 min |
| GraphSAGE | ~2 GB | ~20 min |
| TGN | ~3 GB | ~90 min |
| HGT | ~3 GB | ~90 min |

## Critical API Contract
- `_get_model_class(key) -> (ModelClass, ConfigClass)` in `ml/train/cli.py`
- `_build_config(ConfigClass, yaml_dict)` filters to known dataclass fields
- All models: `ModelClass(config: DataclassConfig)` — never `**kwargs`
- Data: `IEEECISDataset.load_raw() -> df`, `.build_features(df) -> df`, `.get_splits(df) -> (X_train, X_val, y_train, y_val)`
- `SyntheticFraudDataset.generate(n_samples, fraud_rate) -> (X, y)` — returns tuple, not dict

## 8-Week Roadmap
| Week | Milestone |
|---|---|
| 1 | Scaffold (commit 0f1d8f1, 2026-05-16) |
| 2 | IEEE-CIS ingest + Feast offline/online features + EDA |
| 3 | LightGBM + CatBoost baselines + calibration + SHAP |
| 4 | TGN training + ablation vs static GraphSAGE / HGT |
| 5 | River online learner + ADWIN drift detection (shadow model) |
| 6 | Kafka producer/consumer + FastAPI scorer + Grafana p99 latency |
| 7 | Stress test (1k TPS) + ONNX/TorchScript export + bench report |
| 8 | Next.js live dashboard + README + ABLATION_REPORT + demo video |

## Datasets
- IEEE-CIS Fraud Detection (590k rows, Kaggle free)
- Elliptic Bitcoin (203k node graph, Kaggle free)
- PaySim Synthetic (for stream replay)

## Ablation Questions
1. TGN vs GraphSAGE: does temporal memory help?
2. HGT vs GraphSAGE: does heterogeneous structure help?
3. Graph vs tabular: does relational structure add value over LightGBM?
4. ADWIN drift adaptation: can online learner recover under distribution shift?

## Portfolio Context
Project 4 of 4. Others: KhetSAR (satellite EO), DocuMind (multimodal doc RAG), PulsePredict (time-series). All share infra patterns.
