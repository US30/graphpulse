import pytest
import numpy as np
import pandas as pd
import torch
import time

from torch_geometric.data import Data

from ml.models.lgbm import LGBMFraudDetector, LGBMConfig
from ml.models.catboost import CatBoostFraudDetector, CatBoostConfig
from ml.models.graphsage import GraphSAGEClassifier, GraphSAGEConfig
from ml.data.tabular_dataset import IEEECISDataset, TabularConfig, SyntheticFraudDataset
from ml.online.river_learner import RiverFraudLearner, OnlineLearnerConfig
from ml.eval.metrics import compute_metrics

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_tabular_data():
    """1000 rows, 50 features, 5% fraud — returns (X_train, y_train, X_val, y_val)."""
    rng = np.random.default_rng(42)
    n = 1000
    n_features = 50
    X = pd.DataFrame(rng.standard_normal((n, n_features)), columns=[f"V{i}" for i in range(n_features)])
    y = pd.Series((rng.uniform(size=n) < 0.05).astype(int))
    split = int(n * 0.8)
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


@pytest.fixture
def synthetic_graph_data():
    """200 nodes, 400 random edges, 5% fraud labels — PyG Data object."""
    torch.manual_seed(0)
    n_nodes = 200
    n_edges = 400
    x = torch.randn(n_nodes, 16)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    y = torch.bernoulli(torch.full((n_nodes,), 0.05)).long()
    t = torch.arange(n_edges, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.t = t
    return data


@pytest.fixture
def synthetic_raw_df():
    """Raw-style DataFrame mimicking IEEE-CIS columns for feature builder smoke tests."""
    rng = np.random.default_rng(11)
    n = 500
    df = pd.DataFrame(
        rng.standard_normal((n, 50)),
        columns=[f"V{i}" for i in range(50)],
    )
    df["TransactionDT"] = np.arange(n)
    df["TransactionAmt"] = np.abs(rng.standard_normal(n)) * 100 + 10
    df["ProductCD"] = rng.choice(["W", "H", "C", "S", "R"], size=n)
    df["card1"] = rng.integers(1000, 9999, size=n)
    df["isFraud"] = (rng.uniform(size=n) < 0.05).astype(int)
    return df


# ---------------------------------------------------------------------------
# LGBM Smoke
# ---------------------------------------------------------------------------

class TestLGBMSmoke:
    def test_smoke_fit_predict(self, synthetic_tabular_data):
        X_train, y_train, X_val, y_val = synthetic_tabular_data

        cfg = LGBMConfig(n_estimators=10, early_stopping_rounds=5)
        detector = LGBMFraudDetector(cfg)

        t0 = time.time()
        detector.fit(X_train, y_train, X_val, y_val)
        elapsed = time.time() - t0

        proba = detector.predict_proba(X_val)
        assert proba.shape == (len(X_val), 2)

        m = compute_metrics(y_val.values, proba[:, 1])
        assert m.pr_auc > 0.0
        assert elapsed < 30, f"Training took {elapsed:.1f}s — exceeded 30s budget"


# ---------------------------------------------------------------------------
# CatBoost Smoke
# ---------------------------------------------------------------------------

class TestCatBoostSmoke:
    def test_smoke_fit_predict(self, synthetic_tabular_data):
        X_train, y_train, X_val, y_val = synthetic_tabular_data

        cfg = CatBoostConfig(iterations=10)
        detector = CatBoostFraudDetector(cfg)

        t0 = time.time()
        detector.fit(X_train, y_train, X_val, y_val)
        elapsed = time.time() - t0

        proba = detector.predict_proba(X_val)
        assert proba.shape == (len(X_val), 2)

        m = compute_metrics(y_val.values, proba[:, 1])
        assert m.pr_auc > 0.0
        assert elapsed < 30, f"Training took {elapsed:.1f}s — exceeded 30s budget"


# ---------------------------------------------------------------------------
# GraphSAGE Smoke
# ---------------------------------------------------------------------------

class TestGraphSAGESmoke:
    def test_smoke_forward(self, synthetic_graph_data):
        data = synthetic_graph_data

        cfg = GraphSAGEConfig(
            in_channels=16,
            hidden_channels=32,
            out_channels=1,
            n_layers=2,
        )
        model = GraphSAGEClassifier(cfg)
        model.eval()

        with torch.no_grad():
            out = model(data)

        assert out.shape == (200, 1)


# ---------------------------------------------------------------------------
# River Online Learner Smoke
# ---------------------------------------------------------------------------

class TestRiverLearnerSmoke:
    def test_stream_50_samples(self):
        learner = RiverFraudLearner(OnlineLearnerConfig())
        rng = np.random.default_rng(77)

        for _ in range(50):
            x = {f"f{i}": float(rng.standard_normal()) for i in range(15)}
            y = int(rng.uniform() < 0.05)
            learner.learn_one(x, y)

        assert learner.n_seen == 50


# ---------------------------------------------------------------------------
# Feature Builder Smoke
# ---------------------------------------------------------------------------

class TestFeatureBuilderSmoke:
    def test_build_features_smoke(self, synthetic_raw_df):
        dataset = IEEECISDataset(TabularConfig())
        df_feat = dataset.build_features(synthetic_raw_df)
        X_train, X_val, y_train, y_val = dataset.get_splits(df_feat)
        assert X_train is not None
        assert len(X_train) + len(X_val) == len(synthetic_raw_df)
