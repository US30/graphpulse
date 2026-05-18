from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from river import compose, drift, metrics, preprocessing, tree

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class OnlineLearnerConfig:
    """Configuration for the River-based online fraud learner."""

    max_depth: int = 6
    grace_period: int = 200
    drift_detector: str = "adwin"          # adwin | eddm | hddm
    feature_names: Optional[list] = None  # informational only; not enforced


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------

class RiverFraudLearner:
    """Online fraud detector using a Hoeffding Adaptive Tree via River.

    Operates in shadow mode: it learns from every labelled transaction
    while the primary (batch) model handles real-time scoring.  The
    learner exposes ``predict_one`` for direct use in A/B comparisons.
    """

    def __init__(self, config: OnlineLearnerConfig) -> None:
        self.config = config

        # Pipeline: standardise → Hoeffding Adaptive Tree
        self.pipeline = compose.Pipeline(
            preprocessing.StandardScaler(),
            tree.HoeffdingAdaptiveTreeClassifier(
                max_depth=config.max_depth,
                grace_period=config.grace_period,
            ),
        )

        # Concept-drift detector
        self.drift_detector = self._build_drift_detector(config.drift_detector)

        # Cumulative ROC-AUC
        self.metric = metrics.ROCAUC()

        self.n_seen: int = 0
        self.n_drifts: int = 0
        self._last_drift_detected: bool = False

    # ------------------------------------------------------------------
    @staticmethod
    def _build_drift_detector(name: str):
        name = name.lower()
        if name == "adwin":
            return drift.ADWIN()
        elif name == "eddm":
            return drift.EDDM()
        elif name == "hddm":
            return drift.HDDM_W()
        else:
            logger.warning(
                "Unknown drift detector '%s'; falling back to ADWIN.", name
            )
            return drift.ADWIN()

    # ------------------------------------------------------------------
    def predict_one(self, x: dict) -> float:
        """Return the predicted fraud probability for a single transaction."""
        prob_dict: dict = self.pipeline.predict_proba_one(x)
        # River classifiers return {0: p_legit, 1: p_fraud}
        return float(prob_dict.get(1, 0.0))

    # ------------------------------------------------------------------
    def learn_one(self, x: dict, y: int) -> float:
        """Update the model with one (feature_dict, label) pair.

        Parameters
        ----------
        x: Feature dictionary for one transaction.
        y: Ground-truth label (1 = fraud, 0 = legitimate).

        Returns
        -------
        Predicted fraud probability *before* the update (prequential eval).
        """
        # Predict before learning (prequential / test-then-train)
        y_pred_proba = self.predict_one(x)
        y_pred = int(y_pred_proba >= 0.5)

        # Drift detection — feed binary error signal (1 = wrong, 0 = correct)
        error = int(y_pred != y)
        self.drift_detector.update(error)

        self._last_drift_detected = False
        if self.drift_detector.drift_detected:
            logger.info(
                "Concept drift detected at n_seen=%d — resetting pipeline.",
                self.n_seen,
            )
            self.n_drifts += 1
            self._last_drift_detected = True
            # Soft reset: reinitialise the pipeline while keeping config
            self.pipeline = compose.Pipeline(
                preprocessing.StandardScaler(),
                tree.HoeffdingAdaptiveTreeClassifier(
                    max_depth=self.config.max_depth,
                    grace_period=self.config.grace_period,
                ),
            )

        # Train
        self.pipeline.learn_one(x, y)
        self.metric.update(y, y_pred_proba)
        self.n_seen += 1

        return y_pred_proba

    # ------------------------------------------------------------------
    def shadow_evaluate(self, y_true: int, y_pred_proba: float) -> dict:
        """Summarise current learner performance for shadow monitoring.

        Call this *after* ``learn_one`` with the same true label and the
        probability returned by that call.

        Returns
        -------
        dict with keys: roc_auc, n_seen, n_drifts, drift_detected.
        """
        return {
            "roc_auc": self.metric.get(),
            "n_seen": self.n_seen,
            "n_drifts": self.n_drifts,
            "drift_detected": self._last_drift_detected,
        }

    # ------------------------------------------------------------------
    def save(self, path) -> None:
        """Persist the learner to disk via joblib."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("RiverFraudLearner saved to %s", path)

    @classmethod
    def load(cls, path) -> "RiverFraudLearner":
        """Load a persisted learner from disk."""
        import joblib

        learner = joblib.load(Path(path))
        logger.info("RiverFraudLearner loaded from %s", path)
        return learner
