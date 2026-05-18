from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TabularConfig:
    data_dir: str = "data/raw/ieee_cis"
    output_dir: str = "data/features"
    train_cutoff: float = 0.8        # fraction of data used for training (time-based)
    target_col: str = "isFraud"
    drop_cols: list | None = None    # auto-detected when None


class IEEECISDataset:
    """IEEE-CIS fraud detection dataset loader and feature engineer."""

    def __init__(self, config: TabularConfig) -> None:
        self.config = config
        self._label_encoders: dict[str, dict[Any, int]] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_raw(self) -> pd.DataFrame:
        """Read train_transaction.csv + train_identity.csv and merge on TransactionID.

        Returns
        -------
        pd.DataFrame
            Merged transaction + identity DataFrame.
        """
        data_dir = Path(self.config.data_dir)
        tx_path = data_dir / "train_transaction.csv"
        id_path = data_dir / "train_identity.csv"

        logger.info("Loading transactions from %s", tx_path)
        df_tx = pd.read_csv(tx_path)

        if id_path.exists():
            logger.info("Loading identity data from %s", id_path)
            df_id = pd.read_csv(id_path)
            df = df_tx.merge(df_id, on="TransactionID", how="left")
        else:
            logger.warning("Identity file not found at %s — skipping merge.", id_path)
            df = df_tx

        logger.info("Loaded %d rows, %d columns.", df.shape[0], df.shape[1])
        return df

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full feature engineering pipeline.

        Steps
        -----
        1. Drop columns with >50% missing values.
        2. Label-encode all object (string) columns; store the mapping.
        3. Fill remaining numeric NaN with -999 (LightGBM handles sentinel natively).
        4. Add engineered features:
           - ``TransactionAmt_log`` = log1p(TransactionAmt)
           - ``hour_of_day``        = (TransactionDT // 3600) % 24
           - ``day_of_week``        = (TransactionDT // 86400) % 7

        Returns
        -------
        pd.DataFrame
            Processed DataFrame with engineered features.
        """
        df = df.copy()

        # --- 1. Drop high-missingness columns ---
        missing_frac = df.isnull().mean()
        high_miss = missing_frac[missing_frac > 0.5].index.tolist()
        drop_cols = self.config.drop_cols or []
        all_drop = list(set(high_miss) | set(drop_cols))
        # Never drop the target
        all_drop = [c for c in all_drop if c != self.config.target_col]
        logger.info("Dropping %d high-missingness columns.", len(all_drop))
        df = df.drop(columns=all_drop, errors="ignore")

        # --- 2. Label-encode object columns ---
        obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in obj_cols:
            if col == self.config.target_col:
                continue
            unique_vals = df[col].dropna().unique().tolist()
            mapping = {v: i for i, v in enumerate(sorted(map(str, unique_vals)))}
            mapping["__nan__"] = -1
            self._label_encoders[col] = mapping
            df[col] = df[col].fillna("__nan__").astype(str).map(
                lambda v, m=mapping: m.get(v, -1)  # noqa: B023
            )

        # --- 3. Fill numeric NaN with -999 ---
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols_no_target = [c for c in num_cols if c != self.config.target_col]
        df[num_cols_no_target] = df[num_cols_no_target].fillna(-999)

        # --- 4. Engineered features ---
        if "TransactionAmt" in df.columns:
            df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"].clip(lower=0))

        if "TransactionDT" in df.columns:
            df["hour_of_day"] = (df["TransactionDT"] // 3600) % 24
            df["day_of_week"] = (df["TransactionDT"] // 86400) % 7

        logger.info("Feature engineering complete. Shape: %s", df.shape)
        return df

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------

    def get_splits(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Time-based train/validation split.

        Sorts by TransactionDT and splits at self.config.train_cutoff fraction.

        Returns
        -------
        tuple
            (X_train, X_val, y_train, y_val)
        """
        target = self.config.target_col
        if "TransactionDT" in df.columns:
            df = df.sort_values("TransactionDT").reset_index(drop=True)
        else:
            logger.warning("TransactionDT not found; using original row order for split.")

        n = len(df)
        split_idx = int(n * self.config.train_cutoff)

        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        feature_cols = [c for c in df.columns if c != target]
        X_train = train_df[feature_cols]
        X_val = val_df[feature_cols]
        y_train = train_df[target]
        y_val = val_df[target]

        logger.info(
            "Train: %d rows | Val: %d rows | Fraud rate train=%.3f val=%.3f",
            len(X_train),
            len(X_val),
            y_train.mean(),
            y_val.mean(),
        )
        return X_train, X_val, y_train, y_val


# ---------------------------------------------------------------------------
# Synthetic dataset (smoke tests)
# ---------------------------------------------------------------------------

class SyntheticFraudDataset:
    """Generate synthetic tabular fraud data for unit tests and smoke tests."""

    @staticmethod
    def generate(
        n_samples: int = 10_000, fraud_rate: float = 0.05
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Create a synthetic DataFrame with 50 numeric features.

        Parameters
        ----------
        n_samples : int
            Total number of rows to generate.
        fraud_rate : float
            Fraction of rows labelled as fraudulent.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            (X, y) where X has 50 feature columns and y is binary {0, 1}.
        """
        rng = np.random.default_rng(seed=42)
        n_fraud = int(n_samples * fraud_rate)
        n_legit = n_samples - n_fraud

        # Fraudulent transactions skewed toward higher amounts / off-hours
        fraud_data = rng.standard_normal((n_fraud, 48)) + rng.uniform(0.5, 2.0, (n_fraud, 48))
        legit_data = rng.standard_normal((n_legit, 48))

        fraud_amounts = rng.exponential(scale=300, size=n_fraud)
        legit_amounts = rng.exponential(scale=80, size=n_legit)

        fraud_hours = rng.integers(0, 6, size=n_fraud)   # late night
        legit_hours = rng.integers(8, 22, size=n_legit)  # business hours

        amounts = np.concatenate([fraud_amounts, legit_amounts])
        hours = np.concatenate([fraud_hours, legit_hours])
        features = np.vstack([fraud_data, legit_data])

        feature_cols = [f"V{i}" for i in range(48)] + ["TransactionAmt", "hour_of_day"]
        X = pd.DataFrame(
            np.column_stack([features, amounts, hours]),
            columns=feature_cols,
        )

        y_arr = np.concatenate([np.ones(n_fraud, dtype=int), np.zeros(n_legit, dtype=int)])

        # Shuffle
        idx = rng.permutation(n_samples)
        X = X.iloc[idx].reset_index(drop=True)
        y = pd.Series(y_arr[idx], name="isFraud")

        return X, y


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry-point for the tabular data pipeline.

    Usage
    -----
    python -m ml.data.tabular_dataset build-features --data-dir data/raw/ieee_cis
    """
    parser = argparse.ArgumentParser(description="GraphPulse tabular feature pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_p = subparsers.add_parser("build-features", help="Build features from raw CSV files.")
    build_p.add_argument("--data-dir", default="data/raw/ieee_cis", help="Path to raw IEEE-CIS data.")
    build_p.add_argument("--output-dir", default="data/features", help="Output directory for feature files.")
    build_p.add_argument("--train-cutoff", type=float, default=0.8, help="Train/val time split fraction.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "build-features":
        config = TabularConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            train_cutoff=args.train_cutoff,
        )
        dataset = IEEECISDataset(config)
        df_raw = dataset.load_raw()
        df_feat = dataset.build_features(df_raw)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "features.parquet"
        df_feat.to_parquet(out_path, index=False)
        logger.info("Saved features to %s", out_path)


if __name__ == "__main__":
    main()
