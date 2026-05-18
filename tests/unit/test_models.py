import pytest
import numpy as np
import pandas as pd
import torch

from ml.models.lgbm import LGBMFraudDetector, LGBMConfig
from ml.models.catboost import CatBoostFraudDetector, CatBoostConfig
from ml.models.graphsage import GraphSAGEClassifier, GraphSAGEConfig
from ml.models.tgn import TGNFraudClassifier, TGNConfig
from ml.data.tabular_dataset import IEEECISDataset, TabularConfig, SyntheticFraudDataset
from ml.online.river_learner import RiverFraudLearner, OnlineLearnerConfig
from ml.online.adwin_drift import BatchDriftMonitor, DriftConfig
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# LGBMConfig
# ---------------------------------------------------------------------------

class TestLGBMConfig:
    def test_defaults(self):
        cfg = LGBMConfig()
        assert cfg.n_estimators == 2000
        assert cfg.scale_pos_weight == 20.0


# ---------------------------------------------------------------------------
# LGBMFraudDetector
# ---------------------------------------------------------------------------

class TestLGBMFraudDetector:
    def _make_data(self, n: int = 200, n_features: int = 10, n_fraud: int = 10):
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((n, n_features)), columns=[f"f{i}" for i in range(n_features)])
        y = pd.Series(np.zeros(n, dtype=int))
        fraud_idx = rng.choice(n, size=n_fraud, replace=False)
        y.iloc[fraud_idx] = 1
        split = int(n * 0.8)
        return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]

    def test_init(self):
        detector = LGBMFraudDetector(LGBMConfig())
        assert detector is not None

    def test_predict_proba_shape(self):
        X_train, y_train, X_val, y_val = self._make_data()
        detector = LGBMFraudDetector(LGBMConfig(n_estimators=10))
        detector.fit(X_train, y_train, X_val, y_val)
        proba = detector.predict_proba(X_val)
        assert proba.shape == (len(X_val), 2)

    def test_predict_binary(self):
        X_train, y_train, X_val, y_val = self._make_data()
        detector = LGBMFraudDetector(LGBMConfig(n_estimators=10))
        detector.fit(X_train, y_train, X_val, y_val)
        preds = detector.predict(X_val)
        assert set(np.unique(preds)).issubset({0, 1})
        assert preds.shape == (len(X_val),)

    def test_feature_importance_shape(self):
        n_features = 10
        X_train, y_train, X_val, y_val = self._make_data(n_features=n_features)
        detector = LGBMFraudDetector(LGBMConfig(n_estimators=10))
        detector.fit(X_train, y_train, X_val, y_val)
        fi = detector.feature_importance()
        assert isinstance(fi, pd.DataFrame)
        assert list(fi.columns) == ["feature", "importance"]
        assert len(fi) == n_features


# ---------------------------------------------------------------------------
# CatBoostFraudDetector
# ---------------------------------------------------------------------------

class TestCatBoostFraudDetector:
    def _make_data(self, n: int = 200, n_features: int = 10):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((n, n_features)), columns=[f"f{i}" for i in range(n_features)])
        y = pd.Series((rng.uniform(size=n) < 0.05).astype(int))
        split = int(n * 0.8)
        return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]

    def test_init(self):
        detector = CatBoostFraudDetector(CatBoostConfig())
        assert detector is not None

    def test_predict_proba_shape(self):
        X_train, y_train, X_val, y_val = self._make_data()
        detector = CatBoostFraudDetector(CatBoostConfig(iterations=5))
        detector.fit(X_train, y_train, X_val, y_val)
        proba = detector.predict_proba(X_val)
        assert proba.shape == (len(X_val), 2)


# ---------------------------------------------------------------------------
# GraphSAGEConfig
# ---------------------------------------------------------------------------

class TestGraphSAGEConfig:
    def test_defaults(self):
        cfg = GraphSAGEConfig()
        assert cfg.in_channels == 128
        assert cfg.n_layers == 3


# ---------------------------------------------------------------------------
# GraphSAGEClassifier
# ---------------------------------------------------------------------------

class TestGraphSAGEClassifier:
    def test_forward_shape(self):
        cfg = GraphSAGEConfig(
            in_channels=16,
            hidden_channels=32,
            out_channels=1,
            n_layers=2,
        )
        model = GraphSAGEClassifier(cfg)
        model.eval()

        data = Data(
            x=torch.randn(50, 16),
            edge_index=torch.randint(0, 50, (2, 100)),
        )

        with torch.no_grad():
            out = model(data)

        assert out.shape == (50, 1)


# ---------------------------------------------------------------------------
# TGNConfig
# ---------------------------------------------------------------------------

class TestTGNConfig:
    def test_defaults(self):
        cfg = TGNConfig()
        assert cfg.num_nodes > 0
        assert cfg.memory_dim > 0
        assert cfg.time_dim > 0
        assert cfg.embedding_dim > 0
        assert cfg.batch_size > 0


# ---------------------------------------------------------------------------
# SyntheticFraudDataset
# ---------------------------------------------------------------------------

class TestSyntheticFraudDataset:
    def test_generate_shape_and_fraud_rate(self):
        n_samples = 500
        fraud_rate = 0.05
        X, y = SyntheticFraudDataset.generate(n_samples=n_samples, fraud_rate=fraud_rate)
        assert len(X) == n_samples
        assert len(y) == n_samples
        actual_rate = y.mean()
        assert abs(actual_rate - fraud_rate) <= 0.03


# ---------------------------------------------------------------------------
# IEEECISDataset feature engineering
# ---------------------------------------------------------------------------

class TestIEEECISDatasetFeatures:
    def _make_df(self, n: int = 300, n_features: int = 20):
        rng = np.random.default_rng(7)
        df = pd.DataFrame(
            rng.standard_normal((n, n_features)),
            columns=[f"V{i}" for i in range(n_features)],
        )
        df["TransactionDT"] = np.arange(n)
        df["TransactionAmt"] = np.abs(rng.standard_normal(n)) * 100 + 10
        df["isFraud"] = (rng.uniform(size=n) < 0.05).astype(int)
        return df

    def test_build_features_drops_high_na(self):
        df = self._make_df()
        df["high_na_col"] = np.where(
            np.random.default_rng(1).uniform(size=len(df)) < 0.60, np.nan, 1.0
        )
        dataset = IEEECISDataset(TabularConfig())
        df_feat = dataset.build_features(df)
        assert "high_na_col" not in df_feat.columns

    def test_build_features_fills_numeric_na(self):
        df = self._make_df()
        mask = np.random.default_rng(2).uniform(size=len(df)) < 0.30
        df.loc[mask, "V0"] = np.nan
        dataset = IEEECISDataset(TabularConfig())
        df_feat = dataset.build_features(df)
        assert df_feat.isnull().sum().sum() == 0

    def test_time_split_preserves_order(self):
        df = self._make_df(n=500)
        dataset = IEEECISDataset(TabularConfig())
        df_feat = dataset.build_features(df)
        X_train, X_val, y_train, y_val = dataset.get_splits(df_feat)
        if "TransactionDT" in X_train.columns and "TransactionDT" in X_val.columns:
            assert X_train["TransactionDT"].max() <= X_val["TransactionDT"].min()


# ---------------------------------------------------------------------------
# OnlineLearner
# ---------------------------------------------------------------------------

class TestOnlineLearner:
    def _make_sample(self, n_features: int = 10):
        rng = np.random.default_rng(99)
        x = {f"f{i}": float(rng.standard_normal()) for i in range(n_features)}
        y = int(rng.uniform() < 0.05)
        return x, y

    def test_learn_one_returns_prob(self):
        learner = RiverFraudLearner(OnlineLearnerConfig())
        x, y = self._make_sample()
        prob = learner.learn_one(x, y)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_n_seen_increments(self):
        learner = RiverFraudLearner(OnlineLearnerConfig())
        rng = np.random.default_rng(55)
        for _ in range(10):
            x = {f"f{i}": float(rng.standard_normal()) for i in range(10)}
            y = int(rng.uniform() < 0.05)
            learner.learn_one(x, y)
        assert learner.n_seen == 10


# ---------------------------------------------------------------------------
# DriftMonitor
# ---------------------------------------------------------------------------

class TestDriftMonitor:
    def test_update_returns_dict(self):
        monitor = BatchDriftMonitor(DriftConfig())
        result = monitor.update(y_score=0.3, y_true=0)
        assert isinstance(result, dict)
        assert "score_drift" in result
        assert "error_drift" in result
