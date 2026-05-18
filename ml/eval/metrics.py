from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FraudMetrics:
    """Comprehensive evaluation metrics for a fraud detection model."""

    roc_auc: float
    pr_auc: float          # area under precision-recall curve = average_precision_score
    f1: float
    precision_at_k: float  # precision among top-k scored transactions (k=100)
    recall_at_k: float
    brier_score: float
    ks_statistic: float    # KS = max(|TPR - FPR|) along ROC
    n_positives: int
    n_negatives: int


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    top_k: int = 100,
) -> FraudMetrics:
    """Compute all FraudMetrics from ground-truth labels and predicted scores.

    Parameters
    ----------
    y_true:    Binary labels (0 / 1), shape [N].
    y_score:   Predicted fraud probabilities, shape [N].
    threshold: Decision threshold for F1 / confusion matrix.
    top_k:     Number of top-scored transactions for precision/recall@k.

    Returns
    -------
    FraudMetrics dataclass.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_true and y_score must be 1-D arrays.")
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")

    n_positives = int(y_true.sum())
    n_negatives = int(len(y_true) - n_positives)

    # ROC-AUC
    if n_positives == 0 or n_negatives == 0:
        roc_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true, y_score))

    # PR-AUC (average precision)
    pr_auc = float(average_precision_score(y_true, y_score))

    # F1 at threshold
    y_pred = (y_score >= threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Brier score
    brier = float(brier_score_loss(y_true, y_score))

    # KS statistic: max(|TPR - FPR|) along ROC curve
    if n_positives == 0 or n_negatives == 0:
        ks = float("nan")
    else:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ks = float(np.max(np.abs(tpr - fpr)))

    # Precision / recall @ top-k
    k = min(top_k, len(y_score))
    top_k_idx = np.argsort(y_score)[::-1][:k]
    top_k_labels = y_true[top_k_idx]
    precision_at_k = float(top_k_labels.sum() / k) if k > 0 else 0.0
    recall_at_k = (
        float(top_k_labels.sum() / n_positives) if n_positives > 0 else 0.0
    )

    return FraudMetrics(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        f1=f1,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        brier_score=brier,
        ks_statistic=ks,
        n_positives=n_positives,
        n_negatives=n_negatives,
    )


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

def calibration_curve_data(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute data for a reliability (calibration) diagram.

    Parameters
    ----------
    y_true:  Binary labels.
    y_score: Predicted probabilities.
    n_bins:  Number of equal-width bins.

    Returns
    -------
    (mean_predicted, fraction_of_positives) — each of length n_bins,
    omitting empty bins.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_score, bins[1:-1])  # 0-indexed bin membership

    mean_predicted_list: list[float] = []
    fraction_positive_list: list[float] = []

    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if mask.sum() == 0:
            continue
        mean_predicted_list.append(float(y_score[mask].mean()))
        fraction_positive_list.append(float(y_true[mask].mean()))

    return np.array(mean_predicted_list), np.array(fraction_positive_list)


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_metrics(metrics: FraudMetrics) -> None:
    """Print metrics as a formatted table.

    Uses ``rich`` when available; falls back to plain text.
    """
    rows = [
        ("ROC-AUC", f"{metrics.roc_auc:.4f}"),
        ("PR-AUC", f"{metrics.pr_auc:.4f}"),
        ("F1 Score", f"{metrics.f1:.4f}"),
        ("KS Statistic", f"{metrics.ks_statistic:.4f}"),
        ("Brier Score", f"{metrics.brier_score:.4f}"),
        ("Precision@100", f"{metrics.precision_at_k:.4f}"),
        ("Recall@100", f"{metrics.recall_at_k:.4f}"),
        ("Positives (fraud)", str(metrics.n_positives)),
        ("Negatives (legit)", str(metrics.n_negatives)),
    ]

    try:
        from rich.table import Table
        from rich.console import Console

        table = Table(title="Fraud Detection Evaluation Metrics", show_lines=True)
        table.add_column("Metric", style="bold cyan", no_wrap=True)
        table.add_column("Value", style="bold green", justify="right")
        for name, value in rows:
            table.add_row(name, value)

        Console().print(table)

    except ImportError:
        width = 35
        separator = "-" * (width + 16)
        print(separator)
        print(f"{'Fraud Detection Evaluation Metrics':^{width + 16}}")
        print(separator)
        for name, value in rows:
            print(f"  {name:<{width}} {value:>10}")
        print(separator)
