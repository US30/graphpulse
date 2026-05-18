"""
GraphPulse latency benchmark: measure p50/p95/p99 inference latency across models.

Usage:
    python scripts/bench.py \
        --artifacts artifacts/ \
        --n-warmup 50 \
        --n-iters 1000 \
        --output reports/bench.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _latency_stats(times_sec: list[float]) -> dict:
    arr = np.array(times_sec) * 1000  # ms
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "n_iters": len(times_sec),
    }


def bench_lgbm(model_path: Path, n_warmup: int, n_iters: int, n_features: int = 50) -> dict:
    import joblib
    import pandas as pd

    payload = joblib.load(model_path)
    model = payload["model"] if isinstance(payload, dict) else payload

    X = pd.DataFrame(
        np.random.randn(1, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )

    for _ in range(n_warmup):
        model.predict_proba(X)

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        model.predict_proba(X)
        times.append(time.perf_counter() - t0)

    stats = _latency_stats(times)
    stats["model"] = "lgbm"
    logger.info("LightGBM — p99=%.2f ms", stats["p99_ms"])
    return stats


def bench_onnx(model_path: Path, n_warmup: int, n_iters: int, n_features: int = 50) -> dict:
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed — skipping ONNX bench")
        return {"model": "lgbm_onnx", "skipped": True}

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(model_path), sess_options=sess_options)
    input_name = sess.get_inputs()[0].name

    X = np.random.randn(1, n_features).astype(np.float32)

    for _ in range(n_warmup):
        sess.run(None, {input_name: X})

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        sess.run(None, {input_name: X})
        times.append(time.perf_counter() - t0)

    stats = _latency_stats(times)
    stats["model"] = "lgbm_onnx"
    logger.info("LightGBM ONNX — p99=%.2f ms", stats["p99_ms"])
    return stats


def bench_graphsage(model_path: Path, n_warmup: int, n_iters: int, n_nodes: int = 100) -> dict:
    try:
        import torch
        from torch_geometric.data import Data
        from ml.models.graphsage import GraphSAGEClassifier, GraphSAGEConfig
    except ImportError as exc:
        logger.warning("PyG not available (%s) — skipping GraphSAGE bench", exc)
        return {"model": "graphsage", "skipped": True}

    state_dict = torch.load(model_path, weights_only=True, map_location="cpu")
    cfg = GraphSAGEConfig()
    model = GraphSAGEClassifier(cfg)
    model.load_state_dict(state_dict)
    model.eval()

    data = Data(
        x=torch.randn(n_nodes, cfg.in_channels),
        edge_index=torch.randint(0, n_nodes, (2, n_nodes * 4)),
    )

    with torch.no_grad():
        for _ in range(n_warmup):
            model(data)

    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            t0 = time.perf_counter()
            model(data)
            times.append(time.perf_counter() - t0)

    stats = _latency_stats(times)
    stats["model"] = "graphsage"
    stats["n_nodes"] = n_nodes
    logger.info("GraphSAGE — p99=%.2f ms (n_nodes=%d)", stats["p99_ms"], n_nodes)
    return stats


def bench_api_http(base_url: str, n_warmup: int, n_iters: int) -> dict:
    try:
        import requests
    except ImportError:
        logger.warning("requests not installed — skipping API bench")
        return {"model": "api_http", "skipped": True}

    payload = {
        "transaction_id": "bench-tx",
        "features": {f"V{i}": float(np.random.randn()) for i in range(1, 51)},
        "timestamp_unix": time.time(),
        "model": "lgbm",
    }

    for _ in range(n_warmup):
        try:
            requests.post(f"{base_url}/score", json=payload, timeout=5)
        except Exception:
            pass

    times = []
    errors = 0
    for _ in range(n_iters):
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{base_url}/score", json=payload, timeout=5)
            r.raise_for_status()
        except Exception:
            errors += 1
        times.append(time.perf_counter() - t0)

    stats = _latency_stats(times)
    stats["model"] = "api_http"
    stats["errors"] = errors
    logger.info("API HTTP — p99=%.2f ms (errors=%d)", stats["p99_ms"], errors)
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="GraphPulse latency benchmark")
    parser.add_argument("--artifacts", default="artifacts/", help="Artifacts directory")
    parser.add_argument("--n-warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--n-iters", type=int, default=1000, help="Benchmark iterations")
    parser.add_argument("--output", default="reports/bench.json", help="JSON output path")
    parser.add_argument(
        "--api-url", default=None, help="Optional base URL for HTTP API bench (e.g. http://localhost:8000)"
    )
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    results = []

    # LightGBM joblib
    lgbm_path = artifacts / "lgbm" / "model.joblib"
    if lgbm_path.exists():
        results.append(bench_lgbm(lgbm_path, args.n_warmup, args.n_iters))
    else:
        logger.info("No LightGBM artifact at %s — skipping", lgbm_path)

    # LightGBM ONNX
    onnx_path = artifacts / "lgbm" / "model.onnx"
    if onnx_path.exists():
        results.append(bench_onnx(onnx_path, args.n_warmup, args.n_iters))

    # GraphSAGE
    sage_path = artifacts / "graphsage" / "model.pt"
    if sage_path.exists():
        results.append(bench_graphsage(sage_path, args.n_warmup, args.n_iters))

    # HTTP API
    if args.api_url:
        results.append(bench_api_http(args.api_url, n_warmup=10, n_iters=200))

    if not results:
        logger.warning("No artifacts found — nothing to benchmark. Train models first.")
        return

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Benchmark results saved → %s", output)
    print("\n=== Benchmark Summary ===")
    for r in results:
        if r.get("skipped"):
            print(f"  {r['model']}: SKIPPED")
        else:
            print(
                f"  {r['model']}: p50={r['p50_ms']:.2f}ms  p95={r['p95_ms']:.2f}ms  p99={r['p99_ms']:.2f}ms"
            )


if __name__ == "__main__":
    main()
