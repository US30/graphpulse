# GraphPulse — System Architecture

## Overview

GraphPulse is a real-time streaming fraud detection system. It combines:

1. **Tabular ensemble** (LightGBM + CatBoost) for fast, interpretable baseline scoring
2. **Temporal GNN** (TGN) for relational fraud patterns across card/address networks
3. **Online learner** (River ADWIN) as a shadow model for concept-drift tracking
4. **Streaming pipeline** (Redpanda / Kafka API) for sub-50ms end-to-end latency
5. **Explainability** (SHAP + GNNExplainer) for per-prediction audit rationales

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INGESTION                                     │
│                                                                         │
│  IEEE-CIS CSV ──► scripts/download_ieee.py                              │
│       │                                                                 │
│       ▼                                                                 │
│  ml/data/tabular_dataset.py ──► data/features/features.parquet         │
│       │                                                                 │
│       ▼                                                                 │
│  ml/data/graph_builder.py ──► data/graph/homo_graph.pt                 │
│                               data/graph/hetero_graph.pt               │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           TRAINING                                      │
│                                                                         │
│  ml/train/cli.py fit --config configs/lgbm.yaml                        │
│       │                                                                 │
│       ├── LGBMFraudDetector ──────────────────► artifacts/lgbm/        │
│       ├── CatBoostFraudDetector ──────────────► artifacts/catboost/    │
│       ├── GraphSAGEClassifier + GNNTrainer ───► artifacts/graphsage/   │
│       ├── TGNFraudClassifier + GNNTrainer ────► artifacts/tgn/         │
│       └── HGTClassifier + GNNTrainer ─────────► artifacts/hgt/        │
│                                                                         │
│  All runs tracked in MLflow (http://localhost:5001)                    │
│  HPO via Optuna + MLflowCallback                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        STREAMING PIPELINE                               │
│                                                                         │
│  apps/producer/main.py                                                  │
│       │  (kafka-python, synthetic or IEEE-CIS replay)                  │
│       ▼                                                                 │
│  Redpanda topic: graphpulse.transactions                               │
│       │                                                                 │
│       ▼                                                                 │
│  apps/consumer/main.py                                                  │
│       │  (polls topic, calls /score API)                               │
│       ▼                                                                 │
│  apps/api/main.py (FastAPI)                                            │
│       ├── _score_transaction()  — loads lgbm/pkl artifact              │
│       ├── Prometheus metrics (latency, fraud rate, model)              │
│       └── Redis SHAP cache (TTL 1 hr)                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        ONLINE LEARNING                                  │
│                                                                         │
│  ml/online/river_learner.py                                             │
│       │  HoeffdingAdaptiveTree + ADWIN drift detector                  │
│       │  test-then-train on every scored transaction                   │
│       └── shadow mode (does not serve primary traffic)                 │
│                                                                         │
│  ml/online/adwin_drift.py                                               │
│       └── dual ADWIN: score_drift + error_drift                        │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        EXPLAINABILITY                                   │
│                                                                         │
│  ml/explain/shap_wrapper.py                                             │
│       └── TreeExplainer → top-5 feature attributions per transaction   │
│                                                                         │
│  ml/explain/gnn_explainer.py                                            │
│       └── GNNExplainer → subgraph node/edge masks                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Service Map (Docker Compose)

| Service | Port | Purpose |
|---|---|---|
| `postgres` | 5432 | Transaction decisions + MLflow backend |
| `redis` | 6379 | SHAP explanation cache + Feast online store |
| `redpanda` | 9092/9644 | Kafka-compatible streaming broker |
| `redpanda-console` | 8080 | Redpanda web UI |
| `minio` | 9000/9001 | MLflow artifact store (S3-compatible) |
| `mlflow` | 5001 | Experiment tracking |
| `feast` | 6566 | Feature store materialisation |
| `api` | 8000 | FastAPI fraud scoring service |
| `producer` | — | Kafka transaction producer (background) |
| `consumer` | — | Kafka consumer → scorer pipeline |
| `prometheus` | 9090 | Metrics scraping |
| `grafana` | 3000 | Dashboards (fraud rate, latency, drift) |

---

## Model Architecture Details

### LightGBM Baseline
- `LGBMClassifier(n_estimators=2000, num_leaves=63, scale_pos_weight=20)`
- Early stopping on validation AUC (patience=100)
- Features: IEEE-CIS V1–V339 (sparse, ~300 Vesta features), card/address encodings, log-transformed transaction amount, hour/day cyclical features
- Export: ONNX via `skl2onnx` for sub-2ms p99 serving

### TGN (Temporal Graph Network)
- `TGNMemory(num_nodes, raw_msg_dim, memory_dim, time_dim, message_module=IdentityMessage, aggregator_module=LastAggregator)`
- `TransformerConv` GNN layers (2 layers, hidden=256)
- `BCEWithLogitsLoss(pos_weight=20)` for class imbalance
- Trained on temporal edge stream: each transaction is an edge event between card and address nodes
- Memory updated per mini-batch via `update_state` → `detach_memory` pattern

### GraphSAGE (Static Baseline)
- Stacked `SAGEConv` layers with BatchNorm + Dropout
- Homogeneous graph: transactions are edges, cards/addresses are nodes
- Used as ablation baseline vs TGN to quantify temporal memory benefit

### HGT (Heterogeneous)
- `HGTConv` with `group="sum"` aggregation
- Node types: `transaction`, `card`, `address`
- Edge types: `(transaction, uses, card)`, `(transaction, ships_to, address)`
- Used to ablate heterogeneous structure vs homogeneous GraphSAGE

### River Shadow Learner
- `HoeffdingAdaptiveTreeClassifier` (max_depth=6, grace_period=200)
- Wrapped in `Pipeline(StandardScaler → HATC)` via River compose
- ADWIN drift detector on prediction error stream
- On drift: pipeline reinitialised (soft reset) without stopping production traffic

---

## Feature Engineering Pipeline

```
IEEECISDataset.load_raw()
    ├── merge train_transaction.csv + train_identity.csv on TransactionID
    └── returns merged DataFrame

IEEECISDataset.build_features(df)
    ├── drop columns with >50% missing
    ├── label-encode string columns (mapping stored in _label_encoders)
    ├── fill numeric NaN with -999 (LightGBM sentinel)
    └── add: TransactionAmt_log, hour_of_day, day_of_week

IEEECISDataset.get_splits(df)
    └── time-based 80/20 split on TransactionDT
```

---

## Feast Feature Store

```
feast_repo.py
├── card_entity (join_key: card_id)
├── address_entity (join_key: addr_id)
├── card_features_fv (TTL 7 days, FileSource → data/feast/card_features.parquet)
└── transaction_features_fv (TTL 1 day, FileSource → data/feast/transaction_features.parquet)
```

Online serving via Redis; offline via Parquet. Materialise with `make feast-materialize`.

---

## Metrics

| Metric | Description |
|---|---|
| `roc_auc` | ROC-AUC (area under ROC curve) |
| `pr_auc` | PR-AUC (primary metric for imbalanced fraud) |
| `f1` | F1 at 0.5 threshold |
| `ks_statistic` | Kolmogorov–Smirnov separation |
| `brier_score` | Calibration quality |
| `precision_at_k` | Precision at top-100 ranked predictions |
| `recall_at_k` | Recall at top-100 ranked predictions |

---

## Deployment

Production serving uses:
- LightGBM exported to ONNX via `skl2onnx` → `onnxruntime` (p99 < 2 ms)
- TGN saved as TorchScript weights → loaded on GPU for graph-aware scoring
- FastAPI handles routing: tabular path for latency-critical, GNN path for high-risk re-scoring
- Helm chart in `infra/helm/` for k3s/Kubernetes deployment
