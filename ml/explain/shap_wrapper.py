from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SHAP explainer wrapper
# ---------------------------------------------------------------------------

class SHAPExplainer:
    """SHAP-based explainability wrapper for LGBMFraudDetector.

    Uses ``shap.TreeExplainer`` which is exact (no sampling) and runs
    in milliseconds for LightGBM models.

    Parameters
    ----------
    model_path:   Directory from which the LGBMFraudDetector is loaded.
    X_background: Background dataset for TreeExplainer (used as the
                  reference distribution for SHAP values).
    """

    def __init__(self, model_path: Path, X_background: pd.DataFrame) -> None:
        from ml.models.lgbm import LGBMFraudDetector

        self._fraud_model = LGBMFraudDetector.load(model_path)
        # Access the underlying lgb.LGBMClassifier
        self._clf = self._fraud_model._clf

        self._explainer = shap.TreeExplainer(
            self._clf,
            data=X_background,
            feature_perturbation="interventional",
        )
        self._feature_names: list[str] = list(X_background.columns)
        logger.info(
            "TreeExplainer initialised with %d background samples and %d features.",
            len(X_background),
            len(self._feature_names),
        )

    # ------------------------------------------------------------------
    def explain(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for all rows in X.

        Returns
        -------
        np.ndarray of shape [n_samples, n_features].
        For binary classifiers, TreeExplainer returns values for class 1
        (fraud) when ``output_names`` is set; otherwise we pick index [1].
        """
        raw = self._explainer.shap_values(X)
        # LightGBM binary → list of two arrays [class0, class1]
        if isinstance(raw, list) and len(raw) == 2:
            return raw[1]
        return np.array(raw)

    # ------------------------------------------------------------------
    def expected_value(self) -> float:
        """Base rate / expected value (log-odds or probability depending on model).

        For binary LightGBM the explainer exposes a list; we return [1].
        """
        ev = self._explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            return float(ev[1])
        return float(ev)

    # ------------------------------------------------------------------
    def plot_summary(
        self,
        X: pd.DataFrame,
        output_path: Path,
        max_display: int = 20,
    ) -> None:
        """Generate and save a SHAP summary (beeswarm) plot as PNG.

        Parameters
        ----------
        X:            Feature DataFrame to compute SHAP values for.
        output_path:  Destination PNG path.
        max_display:  Maximum features shown.
        """
        shap_values = self.explain(X)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        shap.summary_plot(
            shap_values,
            X,
            max_display=max_display,
            show=False,
            feature_names=self._feature_names,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Summary plot saved to %s", output_path)

    # ------------------------------------------------------------------
    def plot_waterfall(
        self,
        shap_values: np.ndarray,
        X_row: pd.Series,
        output_path: Path,
    ) -> None:
        """Generate and save a SHAP waterfall plot for a single transaction.

        Parameters
        ----------
        shap_values: 1-D array of SHAP values for this row [n_features].
        X_row:       Feature Series for the transaction.
        output_path: Destination PNG path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        explanation = shap.Explanation(
            values=shap_values,
            base_values=self.expected_value(),
            data=X_row.values,
            feature_names=list(X_row.index),
        )
        shap.waterfall_plot(explanation, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Waterfall plot saved to %s", output_path)

    # ------------------------------------------------------------------
    def per_transaction_report(self, X: pd.DataFrame) -> pd.DataFrame:
        """Produce a per-transaction explainability report.

        Columns:
        - TransactionID  (int index if not present as column)
        - fraud_prob     (predicted probability)
        - top_5_features (JSON string mapping feature name → SHAP value)
        - shap_sum       (sum of all SHAP values; should ≈ logit(fraud_prob))

        Parameters
        ----------
        X: Feature DataFrame; index is used as TransactionID if the column
           is absent.

        Returns
        -------
        pd.DataFrame with one row per transaction.
        """
        shap_values = self.explain(X)   # [n, d]

        raw_proba = self._fraud_model.predict_proba(X)
        if hasattr(raw_proba, "shape") and raw_proba.ndim == 2:
            fraud_prob = raw_proba[:, 1]
        else:
            fraud_prob = np.array(raw_proba)

        records: list[dict] = []
        for i, idx in enumerate(X.index):
            sv = shap_values[i]                # shape [d]
            top5_idx = np.argsort(np.abs(sv))[::-1][:5]
            top5 = {self._feature_names[j]: round(float(sv[j]), 6) for j in top5_idx}

            records.append(
                {
                    "TransactionID": idx,
                    "fraud_prob": round(float(fraud_prob[i]), 6),
                    "top_5_features": json.dumps(top5),
                    "shap_sum": round(float(sv.sum()), 6),
                }
            )

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="graphpulse-explain-shap",
        description="SHAP explanations for the LightGBM fraud model.",
    )
    parser.add_argument(
        "--model-dir",
        default="artifacts/lgbm",
        help="Directory containing the saved LGBMFraudDetector.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Directory containing processed feature parquet/CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/explanations",
        help="Output directory for the CSV report and PNG plots.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sample data
    feature_files = list(data_dir.glob("*.parquet")) + list(data_dir.glob("*.csv"))
    if not feature_files:
        logger.error("No data files found in %s", data_dir)
        raise FileNotFoundError(f"No parquet/CSV files in {data_dir}")

    data_file = feature_files[0]
    if data_file.suffix == ".parquet":
        X_all = pd.read_parquet(data_file)
    else:
        X_all = pd.read_csv(data_file)

    # Drop label column if present
    for label_col in ("isFraud", "label", "y"):
        if label_col in X_all.columns:
            X_all = X_all.drop(columns=[label_col])

    # Use up to 1000 rows
    X_sample = X_all.iloc[:1000].copy()
    X_background = X_all.iloc[:200].copy()  # smaller background for speed

    logger.info(
        "Loaded %d rows from %s; using %d for explanations.",
        len(X_all),
        data_file.name,
        len(X_sample),
    )

    explainer = SHAPExplainer(model_path=model_dir, X_background=X_background)

    # Per-transaction report
    report_df = explainer.per_transaction_report(X_sample)
    report_path = output_dir / "shap_per_transaction.csv"
    report_df.to_csv(report_path, index=False)
    logger.info("Per-transaction SHAP report saved to %s", report_path)

    # Summary plot
    explainer.plot_summary(X_sample, output_path=output_dir / "shap_summary.png")

    logger.info("SHAP explanation pipeline complete.")


if __name__ == "__main__":
    main()
