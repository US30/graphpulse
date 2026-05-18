"""
Export trained GraphPulse models to ONNX / TorchScript for production serving.

Usage:
    # Export LightGBM to ONNX
    python scripts/export_onnx.py lgbm \
        --model-path artifacts/lgbm/model.joblib \
        --output artifacts/lgbm/model.onnx \
        --sample-data data/features/features.parquet

    # Export GraphSAGE to TorchScript
    python scripts/export_onnx.py graphsage \
        --model-path artifacts/graphsage/model.pt \
        --output artifacts/graphsage/model.ts
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LGBM → ONNX
# ---------------------------------------------------------------------------

def export_lgbm(model_path: Path, output_path: Path, sample_data: Path | None) -> None:
    import joblib
    import numpy as np

    payload = joblib.load(model_path)
    lgbm_clf = payload["model"] if isinstance(payload, dict) else payload

    if sample_data and sample_data.exists():
        import pandas as pd
        df = pd.read_parquet(sample_data).head(10)
        target_col = "isFraud" if "isFraud" in df.columns else None
        if target_col:
            X_sample = df.drop(columns=[target_col])
        else:
            X_sample = df
    else:
        n_features = lgbm_clf.n_features_in_ if hasattr(lgbm_clf, "n_features_in_") else 50
        X_sample = np.random.randn(5, n_features)
        import pandas as pd
        X_sample = pd.DataFrame(X_sample, columns=[f"f{i}" for i in range(n_features)])

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        n_features = X_sample.shape[1]
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(lgbm_clf, initial_types=initial_type, target_opset=15)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        logger.info("LightGBM ONNX saved → %s", output_path)

    except ImportError:
        fallback = output_path.with_suffix(".joblib")
        joblib.dump(lgbm_clf, fallback)
        logger.warning("skl2onnx not available — joblib fallback saved to %s", fallback)


# ---------------------------------------------------------------------------
# GraphSAGE / TGN → TorchScript
# ---------------------------------------------------------------------------

def export_graphsage(model_path: Path, output_path: Path) -> None:
    import torch
    from ml.models.graphsage import GraphSAGEClassifier, GraphSAGEConfig

    state_dict = torch.load(model_path, weights_only=True, map_location="cpu")

    cfg = GraphSAGEConfig()
    model = GraphSAGEClassifier(cfg)
    model.load_state_dict(state_dict)
    model.eval()

    try:
        scripted = torch.jit.script(model)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scripted.save(str(output_path))
        logger.info("GraphSAGE TorchScript saved → %s", output_path)
    except Exception as exc:
        logger.error("TorchScript export failed: %s. Saving state_dict instead.", exc)
        torch.save(state_dict, output_path.with_suffix(".pt"))


def export_tgn(model_path: Path, output_path: Path) -> None:
    import torch
    from ml.models.tgn import TGNFraudClassifier, TGNConfig

    state_dict = torch.load(model_path, weights_only=True, map_location="cpu")
    cfg = TGNConfig()
    model = TGNFraudClassifier(cfg)
    model.load_state_dict(state_dict)
    model.eval()

    torch.save(state_dict, output_path)
    logger.info("TGN weights saved → %s (TorchScript not supported for TGN memory)", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="GraphPulse model export utility")
    sub = parser.add_subparsers(dest="model", required=True)

    lgbm_p = sub.add_parser("lgbm", help="Export LightGBM → ONNX")
    lgbm_p.add_argument("--model-path", required=True, help="Path to model.joblib")
    lgbm_p.add_argument("--output", default="artifacts/lgbm/model.onnx", help="Output ONNX path")
    lgbm_p.add_argument("--sample-data", default=None, help="Parquet sample for feature shape inference")

    sage_p = sub.add_parser("graphsage", help="Export GraphSAGE → TorchScript")
    sage_p.add_argument("--model-path", required=True, help="Path to model.pt state_dict")
    sage_p.add_argument("--output", default="artifacts/graphsage/model.ts")

    tgn_p = sub.add_parser("tgn", help="Export TGN weights")
    tgn_p.add_argument("--model-path", required=True, help="Path to model.pt state_dict")
    tgn_p.add_argument("--output", default="artifacts/tgn/model_export.pt")

    args = parser.parse_args()

    if args.model == "lgbm":
        export_lgbm(
            Path(args.model_path),
            Path(args.output),
            Path(args.sample_data) if args.sample_data else None,
        )
    elif args.model == "graphsage":
        export_graphsage(Path(args.model_path), Path(args.output))
    elif args.model == "tgn":
        export_tgn(Path(args.model_path), Path(args.output))


if __name__ == "__main__":
    main()
