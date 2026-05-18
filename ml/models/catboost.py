from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
import joblib


@dataclass
class CatBoostConfig:
    iterations: int = 2000
    learning_rate: float = 0.05
    depth: int = 6
    l2_leaf_reg: float = 3.0
    random_strength: float = 1.0
    bagging_temperature: float = 1.0
    scale_pos_weight: float = 20.0
    eval_metric: str = "AUC"
    early_stopping_rounds: int = 100
    random_seed: int = 42
    verbose: int = 0


class CatBoostFraudDetector:
    """CatBoost-based fraud detector with native categorical feature support."""

    def __init__(self, config: CatBoostConfig) -> None:
        self.config = config
        self._model = CatBoostClassifier(
            iterations=config.iterations,
            learning_rate=config.learning_rate,
            depth=config.depth,
            l2_leaf_reg=config.l2_leaf_reg,
            random_strength=config.random_strength,
            bagging_temperature=config.bagging_temperature,
            scale_pos_weight=config.scale_pos_weight,
            eval_metric=config.eval_metric,
            early_stopping_rounds=config.early_stopping_rounds,
            random_seed=config.random_seed,
            verbose=config.verbose,
        )
        self._fitted = False

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
        cat_features: list[str] | None = None,
    ) -> "CatBoostFraudDetector":
        """Fit the model with validation set for early stopping.

        Parameters
        ----------
        cat_features:
            List of column names (or indices if arrays) for categorical features.
            CatBoost handles them natively without encoding.
        """
        cat_idx: list[int] | None = None
        if cat_features is not None and isinstance(X_train, pd.DataFrame):
            cols = list(X_train.columns)
            cat_idx = [cols.index(c) for c in cat_features if c in cols]

        train_pool = Pool(data=X_train, label=y_train, cat_features=cat_idx)
        val_pool = Pool(data=X_val, label=y_val, cat_features=cat_idx)

        self._model.fit(train_pool, eval_set=val_pool)
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return probability estimates of shape [n, 2]."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba.")
        return self._model.predict_proba(X)

    def predict(
        self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        """Return binary predictions using the given threshold."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def save(self, path: Path) -> None:
        """Persist the model to disk using joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"config": self.config, "model": self._model, "fitted": self._fitted}, path)

    @classmethod
    def load(cls, path: Path) -> "CatBoostFraudDetector":
        """Load a persisted model from disk and wrap in a new instance."""
        path = Path(path)
        payload = joblib.load(path)
        instance = cls(payload["config"])
        instance._model = payload["model"]
        instance._fitted = payload["fitted"]
        return instance
