from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LightGBM benchmark
# ---------------------------------------------------------------------------

def benchmark_lgbm(
    model_path: Path,
    X: pd.DataFrame,
    n_runs: int = 1000,
) -> dict:
    """Measure single-row inference latency for LGBMFraudDetector.

    Parameters
    ----------
    model_path: Directory from which the model is loaded.
    X:          Feature DataFrame (any number of rows; benchmark uses row 0).
    n_runs:     Number of timed inference calls.

    Returns
    -------
    {"model": "lgbm", "p50_ms", "p99_ms", "mean_ms", "n_runs"}
    """
    from ml.models.lgbm import LGBMFraudDetector

    model = LGBMFraudDetector.load(model_path)
    single_row = X.iloc[[0]]

    # Warm-up
    for _ in range(10):
        model.predict_proba(single_row)

    latencies_ms: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict_proba(single_row)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies_ms)
    result = {
        "model": "lgbm",
        "p50_ms": float(np.percentile(arr, 50)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "n_runs": n_runs,
    }
    logger.info("LGBM benchmark: %s", result)
    return result


# ---------------------------------------------------------------------------
# ONNX benchmark
# ---------------------------------------------------------------------------

def benchmark_onnx(
    onnx_path: Path,
    X: pd.DataFrame,
    n_runs: int = 1000,
) -> dict:
    """Measure single-row inference latency for an ONNX-exported model.

    Parameters
    ----------
    onnx_path: Path to the .onnx model file.
    X:         Feature DataFrame.
    n_runs:    Number of timed inference calls.

    Returns
    -------
    {"model": "lgbm_onnx", "p50_ms", "p99_ms", "mean_ms", "n_runs"}
    """
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1  # single-threaded for fair latency
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_options)
    input_name = session.get_inputs()[0].name

    single_row = X.iloc[[0]].values.astype(np.float32)

    # Warm-up
    for _ in range(10):
        session.run(None, {input_name: single_row})

    latencies_ms: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: single_row})
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies_ms)
    result = {
        "model": "lgbm_onnx",
        "p50_ms": float(np.percentile(arr, 50)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "n_runs": n_runs,
    }
    logger.info("ONNX benchmark: %s", result)
    return result


# ---------------------------------------------------------------------------
# TGN benchmark
# ---------------------------------------------------------------------------

def benchmark_tgn(
    model_path: Path,
    graph_data,
    n_runs: int = 100,
) -> dict:
    """Measure mini-batch forward-pass latency for TGNFraudClassifier.

    Parameters
    ----------
    model_path: Directory from which the model checkpoint is loaded.
    graph_data: A torch_geometric.data.Data object (or None → synthetic).
    n_runs:     Number of timed forward passes.

    Returns
    -------
    {"model": "tgn", "p50_ms", "p99_ms", "mean_ms", "n_runs"}
    """
    import torch
    from ml.models.tgn import TGNFraudClassifier

    ckpt_file = model_path / "model.pt"
    model = TGNFraudClassifier.load(model_path) if hasattr(
        TGNFraudClassifier, "load"
    ) else TGNFraudClassifier.__new__(TGNFraudClassifier)

    if ckpt_file.exists():
        state = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        try:
            model.load_state_dict(state)
        except Exception:
            logger.warning("Could not load TGN state dict — benchmarking with random weights.")

    model.eval()

    if graph_data is None:
        from torch_geometric.data import Data
        n_nodes, n_edges = 256, 512
        graph_data = Data(
            x=torch.randn(n_nodes, 64),
            edge_index=torch.randint(0, n_nodes, (2, n_edges)),
        )

    x = graph_data.x
    edge_index = graph_data.edge_index

    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            model(x, edge_index)

    latencies_ms: list[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x, edge_index)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies_ms)
    result = {
        "model": "tgn",
        "p50_ms": float(np.percentile(arr, 50)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "n_runs": n_runs,
    }
    logger.info("TGN benchmark: %s", result)
    return result


# ---------------------------------------------------------------------------
# Aggregate runner
# ---------------------------------------------------------------------------

def run_all_benchmarks(
    artifacts_dir: Path,
    output_path: Path,
) -> dict:
    """Run all available benchmarks and save a JSON report.

    Skips individual benchmarks gracefully when artifacts are missing.

    Parameters
    ----------
    artifacts_dir: Root directory containing model sub-directories.
    output_path:   Where to write the JSON report.

    Returns
    -------
    dict mapping model name → benchmark result dict.
    """
    results: dict[str, dict] = {}

    # ---- LightGBM ----
    lgbm_dir = artifacts_dir / "lgbm"
    if lgbm_dir.exists():
        try:
            # Create a minimal dummy DataFrame for benchmarking
            dummy_X = pd.DataFrame(
                np.random.randn(1, 50),
                columns=[f"f{i}" for i in range(50)],
            )
            res = benchmark_lgbm(lgbm_dir, dummy_X)
            results["lgbm"] = res
        except Exception as exc:
            logger.warning("LGBM benchmark skipped: %s", exc)

    # ---- ONNX ----
    onnx_path = artifacts_dir / "lgbm" / "model.onnx"
    if onnx_path.exists():
        try:
            dummy_X = pd.DataFrame(
                np.random.randn(1, 50),
                columns=[f"f{i}" for i in range(50)],
            )
            res = benchmark_onnx(onnx_path, dummy_X)
            results["lgbm_onnx"] = res
        except Exception as exc:
            logger.warning("ONNX benchmark skipped: %s", exc)

    # ---- TGN ----
    tgn_dir = artifacts_dir / "tgn"
    if tgn_dir.exists():
        try:
            res = benchmark_tgn(tgn_dir, graph_data=None)
            results["tgn"] = res
        except Exception as exc:
            logger.warning("TGN benchmark skipped: %s", exc)

    if not results:
        logger.warning(
            "No benchmarks ran — check that artifacts exist under %s", artifacts_dir
        )

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Latency report saved to %s", output_path)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="graphpulse-latency-bench",
        description="Measure p50/p99 inference latency for each GraphPulse model.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Root directory containing trained model artifacts.",
    )
    parser.add_argument(
        "--output",
        default="reports/latency_bench.json",
        help="Output path for the JSON benchmark report.",
    )
    args = parser.parse_args()

    run_all_benchmarks(
        artifacts_dir=Path(args.artifacts_dir),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
