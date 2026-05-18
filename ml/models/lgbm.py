from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib


@dataclass
class LGBMConfig:
    n_estimators: int = 2000
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    scale_pos_weight: float = 20.0  # class imbalance (fraud ~5%)
    n_jobs: int = -1
    random_state: int = 42
    early_stopping_rounds: int = 100
    eval_metric: str = "auc"


class LGBMFraudDetector:
    """LightGBM-based fraud detector wrapper with early stopping and ONNX export."""

    def __init__(self, config: LGBMConfig) -> None:
        self.config = config
        self._model = lgb.LGBMClassifier(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            num_leaves=config.num_leaves,
            max_depth=config.max_depth,
            min_child_samples=config.min_child_samples,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            reg_alpha=config.reg_alpha,
            reg_lambda=config.reg_lambda,
            scale_pos_weight=config.scale_pos_weight,
            n_jobs=config.n_jobs,
            random_state=config.random_state,
            metric=config.eval_metric,
        )
        self._fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> "LGBMFraudDetector":
        """Fit the model with early stopping on validation set."""
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=self.config.early_stopping_rounds, verbose=False
            ),
            lgb.log_evaluation(period=-1),  # verbose=-1 equivalent
        ]
        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability estimates of shape [n, 2]."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba.")
        return self._model.predict_proba(X)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions using the given threshold."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def feature_importance(self) -> pd.DataFrame:
        """Return DataFrame with feature names and importance scores, sorted descending."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before retrieving feature importance.")
        booster = self._model.booster_
        names = booster.feature_name()
        importances = booster.feature_importance(importance_type="gain")
        df = pd.DataFrame({"feature": names, "importance": importances})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, path: Path) -> None:
        """Persist the model to disk using joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"config": self.config, "model": self._model, "fitted": self._fitted}, path)

    @classmethod
    def load(cls, path: Path) -> "LGBMFraudDetector":
        """Load a persisted model from disk and wrap in a new instance."""
        path = Path(path)
        payload = joblib.load(path)
        instance = cls(payload["config"])
        instance._model = payload["model"]
        instance._fitted = payload["fitted"]
        return instance

    def export_onnx(self, path: Path, X_sample: pd.DataFrame) -> None:
        """Export model to ONNX via onnxmltools + skl2onnx if available, else joblib fallback."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            import onnxmltools  # noqa: F401 — verify availability

            n_features = X_sample.shape[1]
            initial_type = [("float_input", FloatTensorType([None, n_features]))]
            onnx_model = convert_sklearn(
                self._model,
                initial_types=initial_type,
                target_opset=15,
            )
            with open(path, "wb") as f:
                f.write(onnx_model.SerializeToString())
        except ImportError:
            # Graceful fallback: save joblib artifact alongside the intended path
            fallback_path = path.with_suffix(".joblib")
            self.save(fallback_path)
            import warnings
            warnings.warn(
                f"onnxmltools/skl2onnx not available. Saved joblib fallback to {fallback_path}.",
                RuntimeWarning,
                stacklevel=2,
            )
