import numpy as np
import pytest

from ml.eval.metrics import compute_metrics, FraudMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perfect_scores(n: int = 200, fraud_rate: float = 0.05):
    rng = np.random.default_rng(0)
    y_true = (rng.uniform(size=n) < fraud_rate).astype(int)
    # Perfect classifier: score == label
    y_score = y_true.astype(float)
    return y_true, y_score


def _random_scores(n: int = 1000, fraud_rate: float = 0.05, seed: int = 42):
    rng = np.random.default_rng(seed)
    y_true = (rng.uniform(size=n) < fraud_rate).astype(int)
    y_score = rng.uniform(size=n)
    return y_true, y_score


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_classifier(self):
        y_true, y_score = _perfect_scores()
        m = compute_metrics(y_true, y_score)
        assert m.roc_auc == pytest.approx(1.0, abs=1e-6)
        assert m.pr_auc == pytest.approx(1.0, abs=1e-6)

    def test_random_classifier(self):
        rng = np.random.default_rng(7)
        roc_aucs = []
        for seed in range(10):
            y_true, y_score = _random_scores(n=2000, seed=seed)
            m = compute_metrics(y_true, y_score)
            roc_aucs.append(m.roc_auc)
        mean_auc = np.mean(roc_aucs)
        assert abs(mean_auc - 0.5) <= 0.15

    def test_metrics_fields_non_negative(self):
        y_true, y_score = _random_scores(n=500)
        m = compute_metrics(y_true, y_score)
        for field in vars(m).values():
            if isinstance(field, (int, float, np.floating)):
                assert field >= 0.0, f"Field {field} is negative"

    def test_precision_at_k(self):
        n = 100
        k = 10
        rng = np.random.default_rng(3)
        y_true = np.zeros(n, dtype=int)
        # Top-k scored samples are fraud
        y_score = rng.uniform(size=n)
        top_k_idx = np.argsort(y_score)[-k:]
        y_true[top_k_idx] = 1
        m = compute_metrics(y_true, y_score, k=k)
        assert m.precision_at_k == pytest.approx(1.0, abs=1e-6)

    def test_brier_score_range(self):
        for seed in range(5):
            y_true, y_score = _random_scores(n=500, seed=seed)
            m = compute_metrics(y_true, y_score)
            assert 0.0 <= m.brier_score <= 1.0

    def test_ks_statistic_range(self):
        for seed in range(5):
            y_true, y_score = _random_scores(n=500, seed=seed)
            m = compute_metrics(y_true, y_score)
            assert 0.0 <= m.ks_statistic <= 1.0

    def test_imbalanced_data(self):
        """PR-AUC should be well below 1.0 for random scores on imbalanced data."""
        rng = np.random.default_rng(17)
        n = 2000
        y_true = (rng.uniform(size=n) < 0.05).astype(int)  # 95/5 split
        y_score = rng.uniform(size=n)  # random, no signal
        m = compute_metrics(y_true, y_score)
        # Random model on 5% fraud should have PR-AUC close to the fraud rate
        assert m.pr_auc < 0.8
