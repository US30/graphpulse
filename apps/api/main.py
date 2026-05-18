from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import time
import logging
import os
import json
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
SCORE_COUNT = Counter(
    "fraud_score_requests_total",
    "Total fraud scoring requests",
    ["model", "decision"],
)
SCORE_LATENCY = Histogram(
    "fraud_score_latency_seconds",
    "Latency of fraud scoring requests in seconds",
    ["model"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
FRAUD_RATE = Counter(
    "fraud_decisions_total",
    "Total fraud decisions made",
    ["decision"],
)

# ---------------------------------------------------------------------------
# Global model registry
# ---------------------------------------------------------------------------
_loaded_models: dict = {}
_available_models: list[str] = []

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
DEFAULT_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))

# Redis for SHAP cache (optional; degrades gracefully if unavailable)
_redis_client = None
try:
    import redis

    _redis_client = redis.Redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        decode_responses=True,
        socket_connect_timeout=2,
    )
    _redis_client.ping()
    logger.info("Redis connected — SHAP explanations will be cached.")
except Exception as exc:
    logger.warning("Redis not available (%s); SHAP cache disabled.", exc)
    _redis_client = None

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ScoringRequest(BaseModel):
    transaction_id: str
    features: dict[str, float]
    timestamp_unix: float
    model: str = "lgbm"


class ScoringResponse(BaseModel):
    transaction_id: str
    fraud_score: float
    is_fraud: bool
    model: str
    latency_ms: float
    threshold: float


class BatchScoringRequest(BaseModel):
    transactions: list[ScoringRequest]


class BatchScoringResponse(BaseModel):
    results: list[ScoringResponse]
    total_latency_ms: float


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GraphPulse Fraud Scoring API",
    description="Real-time streaming graph fraud detection — scoring service.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup_event() -> None:
    """Scan artifacts/ and hot-load LightGBM model if present."""
    global _loaded_models, _available_models

    if not ARTIFACTS_DIR.exists():
        logger.warning(
            "Artifacts directory '%s' not found. Running in fallback mode.",
            ARTIFACTS_DIR,
        )
        return

    for artifact_path in ARTIFACTS_DIR.glob("*.pkl"):
        model_name = artifact_path.stem
        try:
            import joblib

            model = joblib.load(artifact_path)
            _loaded_models[model_name] = {"type": "pkl", "model": model}
            _available_models.append(model_name)
            logger.info("Loaded model '%s' from %s", model_name, artifact_path)
        except Exception as exc:
            logger.error("Failed to load '%s': %s", artifact_path, exc)

    for artifact_path in ARTIFACTS_DIR.glob("*.txt"):
        model_name = artifact_path.stem
        if model_name in _loaded_models:
            continue
        try:
            import lightgbm as lgb

            model = lgb.Booster(model_file=str(artifact_path))
            _loaded_models[model_name] = {"type": "lgbm_txt", "model": model}
            _available_models.append(model_name)
            logger.info(
                "Loaded LightGBM model '%s' from %s", model_name, artifact_path
            )
        except Exception as exc:
            logger.error("Failed to load LightGBM '%s': %s", artifact_path, exc)

    if not _available_models:
        logger.warning(
            "No model artifacts found in '%s'. Fallback scoring active.", ARTIFACTS_DIR
        )


# ---------------------------------------------------------------------------
# Internal scoring helpers
# ---------------------------------------------------------------------------


def _score_transaction(req: ScoringRequest) -> tuple[float, str]:
    """Return (fraud_score, model_used). Falls back to 0.5 if no model loaded."""
    model_key = req.model
    entry = _loaded_models.get(model_key)

    if entry is None:
        # Try any available model before falling back
        if _loaded_models:
            model_key = next(iter(_loaded_models))
            entry = _loaded_models[model_key]
        else:
            return 0.5, "fallback"

    feature_values = list(req.features.values())

    try:
        model_type = entry["type"]
        model = entry["model"]

        if model_type == "lgbm_txt":
            import lightgbm as lgb
            import numpy as np

            X = np.array(feature_values).reshape(1, -1)
            score = float(model.predict(X)[0])
        elif model_type == "pkl":
            import numpy as np

            X = np.array(feature_values).reshape(1, -1)
            if hasattr(model, "predict_proba"):
                score = float(model.predict_proba(X)[0, 1])
            else:
                score = float(model.predict(X)[0])
        else:
            score = 0.5
    except Exception as exc:
        logger.error("Scoring error for model '%s': %s", model_key, exc)
        score = 0.5
        model_key = "fallback"

    return score, model_key


def _compute_shap(req: ScoringRequest, model_key: str) -> Optional[dict]:
    """Compute SHAP top-5 features and cache in Redis if available."""
    entry = _loaded_models.get(model_key)
    if entry is None:
        return None

    try:
        import shap
        import numpy as np

        feature_names = list(req.features.keys())
        feature_values = np.array(list(req.features.values())).reshape(1, -1)
        model = entry["model"]

        if entry["type"] == "lgbm_txt":
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, "estimators_") or hasattr(model, "booster_"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, feature_values)

        shap_values = explainer.shap_values(feature_values)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_flat = shap_values[0].tolist()
        importance = sorted(
            zip(feature_names, shap_flat), key=lambda x: abs(x[1]), reverse=True
        )
        top5 = {k: round(v, 6) for k, v in importance[:5]}
        return top5
    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)
        return None


def _cache_shap(transaction_id: str, shap_top5: dict) -> None:
    if _redis_client is None or shap_top5 is None:
        return
    try:
        _redis_client.setex(
            f"shap:{transaction_id}", 3600, json.dumps(shap_top5)
        )
    except Exception as exc:
        logger.warning("Redis SHAP cache write failed: %s", exc)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "models_loaded": _available_models}


@app.post("/score", response_model=ScoringResponse)
async def score(req: ScoringRequest, background_tasks: BackgroundTasks) -> ScoringResponse:
    """Score a single transaction for fraud."""
    t0 = time.perf_counter()

    fraud_score, model_used = _score_transaction(req)
    is_fraud = fraud_score >= DEFAULT_THRESHOLD

    latency_s = time.perf_counter() - t0
    latency_ms = latency_s * 1000.0

    decision = "fraud" if is_fraud else "legit"
    SCORE_COUNT.labels(model=model_used, decision=decision).inc()
    SCORE_LATENCY.labels(model=model_used).observe(latency_s)
    FRAUD_RATE.labels(decision=decision).inc()

    # Async SHAP + cache in background so latency isn't blocked
    background_tasks.add_task(
        _async_shap_and_cache, req, model_used
    )

    return ScoringResponse(
        transaction_id=req.transaction_id,
        fraud_score=round(fraud_score, 6),
        is_fraud=is_fraud,
        model=model_used,
        latency_ms=round(latency_ms, 3),
        threshold=DEFAULT_THRESHOLD,
    )


def _async_shap_and_cache(req: ScoringRequest, model_key: str) -> None:
    shap_top5 = _compute_shap(req, model_key)
    _cache_shap(req.transaction_id, shap_top5)


@app.post("/score/batch", response_model=BatchScoringResponse)
async def score_batch(req: BatchScoringRequest) -> BatchScoringResponse:
    """Score a batch of up to 1000 transactions."""
    if len(req.transactions) > 1000:
        raise HTTPException(
            status_code=422,
            detail="Batch size must not exceed 1000 transactions.",
        )

    t_batch_start = time.perf_counter()
    results: list[ScoringResponse] = []

    for tx in req.transactions:
        t0 = time.perf_counter()
        fraud_score, model_used = _score_transaction(tx)
        is_fraud = fraud_score >= DEFAULT_THRESHOLD
        latency_s = time.perf_counter() - t0
        latency_ms = latency_s * 1000.0

        decision = "fraud" if is_fraud else "legit"
        SCORE_COUNT.labels(model=model_used, decision=decision).inc()
        SCORE_LATENCY.labels(model=model_used).observe(latency_s)
        FRAUD_RATE.labels(decision=decision).inc()

        results.append(
            ScoringResponse(
                transaction_id=tx.transaction_id,
                fraud_score=round(fraud_score, 6),
                is_fraud=is_fraud,
                model=model_used,
                latency_ms=round(latency_ms, 3),
                threshold=DEFAULT_THRESHOLD,
            )
        )

    total_latency_ms = (time.perf_counter() - t_batch_start) * 1000.0
    return BatchScoringResponse(
        results=results, total_latency_ms=round(total_latency_ms, 3)
    )


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics scrape endpoint."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.get("/explain/{transaction_id}")
async def explain(transaction_id: str) -> dict:
    """Return cached SHAP top-5 features for the last scored transaction."""
    if _redis_client is None:
        raise HTTPException(
            status_code=503,
            detail="SHAP explanation cache (Redis) is not available.",
        )
    try:
        cached = _redis_client.get(f"shap:{transaction_id}")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Redis error: {exc}")

    if cached is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No SHAP explanation found for transaction '{transaction_id}'. "
                "Score the transaction first."
            ),
        )

    return {
        "transaction_id": transaction_id,
        "shap_top5": json.loads(cached),
    }
