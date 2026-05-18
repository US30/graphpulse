from __future__ import annotations

import argparse
import dataclasses
import logging
import shutil
import yaml
from pathlib import Path

import mlflow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _get_model_class(model_key: str) -> tuple:
    """Return (ModelClass, ConfigClass) for the given model key."""
    if model_key == "lgbm":
        from ml.models.lgbm import LGBMFraudDetector, LGBMConfig
        return LGBMFraudDetector, LGBMConfig
    elif model_key == "catboost":
        from ml.models.catboost import CatBoostFraudDetector, CatBoostConfig
        return CatBoostFraudDetector, CatBoostConfig
    elif model_key == "graphsage":
        from ml.models.graphsage import GraphSAGEClassifier, GraphSAGEConfig
        return GraphSAGEClassifier, GraphSAGEConfig
    elif model_key == "tgn":
        from ml.models.tgn import TGNFraudClassifier, TGNConfig
        return TGNFraudClassifier, TGNConfig
    elif model_key == "hgt":
        from ml.models.hgt import HGTClassifier, HGTConfig
        return HGTClassifier, HGTConfig
    else:
        raise ValueError(
            f"Unknown model: {model_key}. Supported: lgbm, catboost, graphsage, tgn, hgt"
        )


def _build_config(ConfigClass, d: dict):
    """Instantiate ConfigClass from a dict, filtering to known fields only."""
    known = {f.name for f in dataclasses.fields(ConfigClass)}
    return ConfigClass(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Helpers: data loading
# ---------------------------------------------------------------------------

def _load_tabular_splits(config: dict):
    """Load train/val splits for tabular models.

    Falls back to synthetic data when the raw files are missing.
    Returns (X_train, y_train, X_val, y_val).
    """
    try:
        from ml.data.tabular_dataset import IEEECISDataset, TabularConfig
        tab_cfg = TabularConfig(
            data_dir=config.get("data_dir", "data/raw/ieee_cis"),
            train_cutoff=config.get("train_cutoff", 0.8),
            target_col=config.get("target_col", "isFraud"),
        )
        dataset = IEEECISDataset(tab_cfg)
        df_raw = dataset.load_raw()
        df_feat = dataset.build_features(df_raw)
        X_train, X_val, y_train, y_val = dataset.get_splits(df_feat)
        return X_train, y_train, X_val, y_val
    except FileNotFoundError:
        logger.warning(
            "Raw data not found — falling back to SyntheticFraudDataset for smoke test."
        )
        from ml.data.tabular_dataset import SyntheticFraudDataset
        n_samples = config.get("smoke_n_samples", 5000)
        fraud_rate = config.get("smoke_fraud_rate", 0.02)
        X, y = SyntheticFraudDataset.generate(n_samples=n_samples, fraud_rate=fraud_rate)
        split = int(len(X) * 0.8)
        return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


def _load_graph_data(config: dict):
    """Load PyG graph data for GNN models.

    Falls back to a synthetic PyG Data object when graph files are missing.
    """
    import torch

    graph_dir = Path(config.get("graph_dir", "data/graph"))
    graph_file = graph_dir / "transaction_graph.pt"

    try:
        if graph_file.exists():
            data = torch.load(graph_file, weights_only=False)
            logger.info("Loaded graph from %s", graph_file)
            return data
        # Build from features parquet
        from ml.data.graph_builder import TransactionGraphBuilder, GraphConfig
        import pandas as pd
        feat_path = Path(config.get("features_path", "data/features/features.parquet"))
        df = pd.read_parquet(feat_path)
        builder = TransactionGraphBuilder(GraphConfig(output_dir=str(graph_dir)))
        data = builder.build_homogeneous(df)
        logger.info("Built graph via TransactionGraphBuilder")
        return data
    except Exception:
        logger.warning(
            "Graph data not found — using synthetic PyG Data for smoke test."
        )
        from torch_geometric.data import Data

        n_nodes = config.get("smoke_n_nodes", 500)
        n_edges = config.get("smoke_n_edges", 2000)
        n_features = config.get("in_channels", 64)

        x = torch.randn(n_nodes, n_features)
        edge_index = torch.randint(0, n_nodes, (2, n_edges))
        edge_attr = torch.randn(n_edges, 4)
        y = (torch.rand(n_nodes) < 0.02).long()
        # Simulate edge timestamps for time-based splits
        edge_time = torch.sort(torch.rand(n_edges)).values

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            edge_time=edge_time,
        )
        return data


# ---------------------------------------------------------------------------
# Fit command
# ---------------------------------------------------------------------------

def fit(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open() as fh:
        config: dict = yaml.safe_load(fh)

    model_key: str = config["model"]
    logger.info("Training model: %s", model_key)

    # ------------------------------------------------------------------
    # Instantiate model via CRITICAL RULE:
    #   _get_model_class → (ModelClass, ConfigClass)
    #   _build_config    → ConfigClass instance
    #   ModelClass(config)
    # ------------------------------------------------------------------
    ModelClass, ConfigClass = _get_model_class(model_key)
    model_config = _build_config(ConfigClass, config)
    model = ModelClass(model_config)

    output_dir = Path(args.output_dir) / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # MLflow tracking
    # ------------------------------------------------------------------
    experiment_name = args.experiment_name or config.get(
        "experiment_name", f"graphpulse_{model_key}"
    )
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{model_key}_run") as run:
        # Log flat config as params (MLflow only accepts str values)
        flat_params = {
            k: str(v)
            for k, v in config.items()
            if not isinstance(v, (dict, list))
        }
        mlflow.log_params(flat_params)
        mlflow.log_param("config_path", str(config_path.resolve()))

        # ----------------------------------------------------------------
        # Tabular path: LightGBM / CatBoost
        # ----------------------------------------------------------------
        if model_key in ("lgbm", "catboost"):
            from ml.eval.metrics import compute_metrics

            X_train, y_train, X_val, y_val = _load_tabular_splits(config)
            logger.info(
                "Data loaded — train: %d rows, val: %d rows", len(X_train), len(X_val)
            )

            model.fit(X_train, y_train, X_val, y_val)

            y_score = model.predict_proba(X_val)
            import numpy as np
            y_score_np = (
                y_score[:, 1] if hasattr(y_score, "shape") and y_score.ndim == 2
                else y_score
            )
            metrics = compute_metrics(
                y_true=y_val.values if hasattr(y_val, "values") else y_val,
                y_score=y_score_np,
            )

            mlflow.log_metrics(
                {
                    "val_roc_auc": metrics.roc_auc,
                    "val_pr_auc": metrics.pr_auc,
                    "val_f1": metrics.f1,
                    "val_ks": metrics.ks_statistic,
                    "val_brier": metrics.brier_score,
                    "val_precision_at_100": metrics.precision_at_k,
                    "val_recall_at_100": metrics.recall_at_k,
                }
            )

            logger.info(
                "val_roc_auc=%.4f  val_pr_auc=%.4f  val_f1=%.4f",
                metrics.roc_auc,
                metrics.pr_auc,
                metrics.f1,
            )

            # Save model artifact
            model.save(output_dir / "model.joblib")
            mlflow.log_artifacts(str(output_dir), artifact_path=model_key)

        # ----------------------------------------------------------------
        # GNN path: GraphSAGE / TGN / HGT
        # ----------------------------------------------------------------
        elif model_key in ("graphsage", "tgn", "hgt"):
            from ml.train.gnn_trainer import GNNTrainer

            graph_data = _load_graph_data(config)
            logger.info("Graph data ready — nodes: %d", graph_data.num_nodes)

            trainer = GNNTrainer(model, model_config)
            gnn_metrics = trainer.train(graph_data)

            mlflow.log_metrics(gnn_metrics)
            logger.info("GNN training complete — metrics: %s", gnn_metrics)

            # Save GNN checkpoint
            import torch
            ckpt_path = output_dir / "model.pt"
            torch.save(model.state_dict(), ckpt_path)
            mlflow.log_artifacts(str(output_dir), artifact_path=model_key)

        else:
            raise ValueError(f"No training branch for model_key={model_key}")

        # Copy config into artifact dir for reproducibility
        shutil.copy(config_path, output_dir / "config.yaml")

        logger.info(
            "Artifacts saved to %s | MLflow run_id=%s", output_dir, run.info.run_id
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="graphpulse-train",
        description="GraphPulse ML training CLI",
    )
    sub = parser.add_subparsers(title="commands", dest="command", required=True)

    fit_parser = sub.add_parser("fit", help="Train a fraud detection model")
    fit_parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML training config (must contain 'model' key)",
    )
    fit_parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Root directory for saved model artifacts (default: artifacts/)",
    )
    fit_parser.add_argument(
        "--experiment-name",
        default=None,
        help="MLflow experiment name (overrides config value if set)",
    )
    fit_parser.set_defaults(func=fit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
