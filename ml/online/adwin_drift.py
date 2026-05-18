from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from river.drift import ADWIN

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DriftConfig:
    """Configuration for the ADWIN-based batch drift monitor."""

    delta: float = 0.002   # ADWIN sensitivity (smaller = more sensitive)
    window_size: int = 1000


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class BatchDriftMonitor:
    """Detects distribution shift in batch model output scores and errors.

    Maintains two independent ADWIN detectors:
    - ``score_adwin``:  tracks raw fraud-probability scores.
    - ``error_adwin``:  tracks absolute prediction error |score - label|.

    When either detector signals drift, an event is appended to
    ``drift_log`` and returned in the ``update`` response.
    """

    def __init__(self, config: DriftConfig) -> None:
        self.config = config
        self.score_adwin = ADWIN(delta=config.delta)
        self.error_adwin = ADWIN(delta=config.delta)
        self.score_history: deque[float] = deque(maxlen=config.window_size)
        self.drift_log: list[dict] = []
        self._n_seen: int = 0

    # ------------------------------------------------------------------
    def update(self, y_score: float, y_true: int) -> dict:
        """Feed one (score, label) pair to both ADWIN detectors.

        Parameters
        ----------
        y_score: Model's predicted fraud probability (float in [0, 1]).
        y_true:  Ground-truth label (1 = fraud, 0 = legitimate).

        Returns
        -------
        dict with keys:
          - score_drift (bool)
          - error_drift (bool)
          - current_mean_score (float)
        """
        y_score = float(y_score)
        error = abs(y_score - int(y_true))

        self.score_history.append(y_score)
        self._n_seen += 1

        self.score_adwin.update(y_score)
        self.error_adwin.update(error)

        score_drift = bool(self.score_adwin.drift_detected)
        error_drift = bool(self.error_adwin.drift_detected)

        if score_drift:
            event = self._build_event("score_drift")
            self.drift_log.append(event)
            logger.warning("Score drift detected at n_seen=%d — event: %s", self._n_seen, event)

        if error_drift:
            event = self._build_event("error_drift")
            self.drift_log.append(event)
            logger.warning("Error drift detected at n_seen=%d — event: %s", self._n_seen, event)

        current_mean = float(np.mean(self.score_history)) if self.score_history else 0.0

        return {
            "score_drift": score_drift,
            "error_drift": error_drift,
            "current_mean_score": current_mean,
        }

    # ------------------------------------------------------------------
    def _build_event(self, drift_type: str) -> dict:
        """Construct a drift event dict.

        ADWIN exposes ``estimation`` (mean of the current window) and
        previously also ``total`` (mean before the change point).  We
        approximate 'mean_before' using the older half of ``score_history``.
        """
        history = list(self.score_history)
        if len(history) >= 2:
            mid = len(history) // 2
            mean_before = float(np.mean(history[:mid]))
            mean_after = float(np.mean(history[mid:]))
        else:
            mean_before = float(np.mean(history)) if history else 0.0
            mean_after = mean_before

        return {
            "type": drift_type,
            "n_seen": self._n_seen,
            "mean_before": round(mean_before, 6),
            "mean_after": round(mean_after, 6),
        }

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Return a high-level summary of drift events.

        Returns
        -------
        dict with:
          - n_drifts (int): total number of drift events recorded.
          - drift_events (list): up to the last 10 events.
        """
        return {
            "n_drifts": len(self.drift_log),
            "drift_events": self.drift_log[-10:],
        }
