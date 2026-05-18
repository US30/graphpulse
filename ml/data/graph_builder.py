from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, HeteroData

logger = logging.getLogger(__name__)


@dataclass
class GraphConfig:
    data_dir: str = "data/features"
    output_dir: str = "data/graph"
    node_features: list | None = None   # feature columns used as node features (None = auto)
    edge_features: list | None = None   # transaction features used as edge attributes (None = auto)
    max_nodes: int = 500_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise a 2-D array column-wise. Safe against zero-variance columns."""
    min_v = arr.min(axis=0, keepdims=True)
    max_v = arr.max(axis=0, keepdims=True)
    denom = np.where(max_v - min_v == 0, 1.0, max_v - min_v)
    return (arr - min_v) / denom


_DEFAULT_EDGE_FEATURES = ["TransactionAmt_log", "hour_of_day", "day_of_week"]
_DEFAULT_NODE_NUMERIC_EXCLUDE = {"isFraud", "TransactionID", "TransactionDT"}


class TransactionGraphBuilder:
    """Build PyG homogeneous and heterogeneous graphs from IEEE-CIS feature DataFrames."""

    def __init__(self, config: GraphConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _card_id(self, df: pd.DataFrame) -> pd.Series:
        """Composite card identifier: card1 + '_' + card2."""
        c1 = df["card1"].astype(str) if "card1" in df.columns else pd.Series(["unk"] * len(df))
        c2 = df["card2"].astype(str) if "card2" in df.columns else pd.Series(["0"] * len(df))
        return c1 + "_" + c2

    def _addr_id(self, df: pd.DataFrame) -> pd.Series:
        """Composite address identifier: addr1 + '_' + addr2."""
        a1 = df["addr1"].astype(str) if "addr1" in df.columns else pd.Series(["unk"] * len(df))
        a2 = df["addr2"].astype(str) if "addr2" in df.columns else pd.Series(["0"] * len(df))
        return a1 + "_" + a2

    def _edge_feat_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Build edge feature matrix from available edge feature columns."""
        edge_cols = self.config.edge_features or _DEFAULT_EDGE_FEATURES
        available = [c for c in edge_cols if c in df.columns]
        if not available:
            return np.zeros((len(df), 1), dtype=np.float32)
        arr = df[available].fillna(0.0).to_numpy(dtype=np.float32)
        return _norm(arr)

    def _tx_node_features(self, df: pd.DataFrame) -> np.ndarray:
        """Select numeric columns as transaction node features."""
        exclude = _DEFAULT_NODE_NUMERIC_EXCLUDE | {"card1", "card2", "addr1", "addr2"}
        if self.config.node_features:
            cols = [c for c in self.config.node_features if c in df.columns]
        else:
            cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude
            ][:64]  # cap at 64 dims for memory sanity
        if not cols:
            return np.zeros((len(df), 1), dtype=np.float32)
        arr = df[cols].fillna(-999.0).to_numpy(dtype=np.float32)
        arr = np.where(arr == -999.0, 0.0, arr)  # zero-fill sentinels for embedding
        return _norm(arr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_homogeneous(self, df: pd.DataFrame) -> Data:
        """Build a homogeneous PyG Data object from the feature DataFrame.

        Node layout (contiguous blocks)
        --------------------------------
        [0 .. N_tx-1]              — transactions
        [N_tx .. N_tx+N_card-1]   — card entities
        [N_tx+N_card .. total-1]  — address entities

        Edges (directed, src→dst)
        -------------------------
        transaction_i → card_j     (uses_card)
        transaction_i → address_k  (ships_to)

        Returns
        -------
        Data
            PyG Data with fields: x, edge_index, edge_attr, t, y.
        """
        df = df.reset_index(drop=True)
        n_tx = len(df)

        # --- Node indices ---
        card_ids = self._card_id(df)
        addr_ids = self._addr_id(df)

        unique_cards = card_ids.unique().tolist()
        unique_addrs = addr_ids.unique().tolist()

        card_map = {c: i + n_tx for i, c in enumerate(unique_cards)}
        addr_map = {a: i + n_tx + len(unique_cards) for i, a in enumerate(unique_addrs)}

        total_nodes = n_tx + len(unique_cards) + len(unique_addrs)
        logger.info(
            "Graph: %d tx nodes, %d card nodes, %d addr nodes → %d total",
            n_tx, len(unique_cards), len(unique_addrs), total_nodes,
        )

        # --- Edge indices ---
        tx_idx = torch.arange(n_tx, dtype=torch.long)
        card_idx = torch.tensor([card_map[c] for c in card_ids], dtype=torch.long)
        addr_idx = torch.tensor([addr_map[a] for a in addr_ids], dtype=torch.long)

        # Both edge types share the same transaction source
        src = torch.cat([tx_idx, tx_idx])
        dst = torch.cat([card_idx, addr_idx])
        edge_index = torch.stack([src, dst], dim=0)  # [2, 2*N_tx]

        # --- Node features ---
        tx_feat = self._tx_node_features(df)            # [N_tx, feat_dim]
        feat_dim = tx_feat.shape[1]

        # Card and address nodes get zero features
        card_feat = np.zeros((len(unique_cards), feat_dim), dtype=np.float32)
        addr_feat = np.zeros((len(unique_addrs), feat_dim), dtype=np.float32)
        x = torch.tensor(
            np.vstack([tx_feat, card_feat, addr_feat]), dtype=torch.float
        )  # [total_nodes, feat_dim]

        # --- Edge features ---
        edge_feat_np = self._edge_feat_matrix(df)     # [N_tx, edge_feat_dim]
        edge_feat_double = np.vstack([edge_feat_np, edge_feat_np])  # repeat for both edge types
        edge_attr = torch.tensor(edge_feat_double, dtype=torch.float)

        # --- Timestamps ---
        t_col = df["TransactionDT"].fillna(0.0).to_numpy(dtype=np.float32) if "TransactionDT" in df.columns else np.zeros(n_tx, dtype=np.float32)
        t_all = np.concatenate([t_col, t_col])
        t = torch.tensor(t_all, dtype=torch.float)

        # --- Labels ---
        target_col = "isFraud"
        if target_col in df.columns:
            y_tx = df[target_col].fillna(0).to_numpy(dtype=np.long)
        else:
            y_tx = np.zeros(n_tx, dtype=np.long)
        y = torch.tensor(
            np.concatenate([y_tx, np.full(len(unique_cards) + len(unique_addrs), -1, dtype=np.long)]),
            dtype=torch.long,
        )

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, t=t, y=y)
        data.num_nodes = total_nodes
        return data

    def build_heterogeneous(self, df: pd.DataFrame) -> HeteroData:
        """Build a heterogeneous PyG HeteroData object.

        Node types: ``transaction``, ``card``, ``address``.
        Edge types: ``(transaction, uses, card)``, ``(transaction, ships_to, address)``.

        Returns
        -------
        HeteroData
        """
        df = df.reset_index(drop=True)
        n_tx = len(df)

        card_ids = self._card_id(df)
        addr_ids = self._addr_id(df)
        unique_cards = card_ids.unique().tolist()
        unique_addrs = addr_ids.unique().tolist()

        card_map = {c: i for i, c in enumerate(unique_cards)}
        addr_map = {a: i for i, a in enumerate(unique_addrs)}

        # --- Node features ---
        tx_feat = self._tx_node_features(df)
        feat_dim = tx_feat.shape[1]
        card_feat = np.zeros((len(unique_cards), feat_dim), dtype=np.float32)
        addr_feat = np.zeros((len(unique_addrs), feat_dim), dtype=np.float32)

        # --- Edge connectivity ---
        tx_idx = torch.arange(n_tx, dtype=torch.long)
        card_idx = torch.tensor([card_map[c] for c in card_ids], dtype=torch.long)
        addr_idx = torch.tensor([addr_map[a] for a in addr_ids], dtype=torch.long)

        edge_feat_np = self._edge_feat_matrix(df)
        edge_attr = torch.tensor(edge_feat_np, dtype=torch.float)

        t_col = df["TransactionDT"].fillna(0.0).to_numpy(dtype=np.float32) if "TransactionDT" in df.columns else np.zeros(n_tx, dtype=np.float32)
        t = torch.tensor(t_col, dtype=torch.float)

        target_col = "isFraud"
        if target_col in df.columns:
            y = torch.tensor(df[target_col].fillna(0).to_numpy(dtype=np.long), dtype=torch.long)
        else:
            y = torch.zeros(n_tx, dtype=torch.long)

        data = HeteroData()

        # Node features
        data["transaction"].x = torch.tensor(tx_feat, dtype=torch.float)
        data["transaction"].y = y
        data["card"].x = torch.tensor(card_feat, dtype=torch.float)
        data["address"].x = torch.tensor(addr_feat, dtype=torch.float)

        # Edges: transaction → card
        data["transaction", "uses", "card"].edge_index = torch.stack([tx_idx, card_idx], dim=0)
        data["transaction", "uses", "card"].edge_attr = edge_attr
        data["transaction", "uses", "card"].t = t

        # Edges: transaction → address
        data["transaction", "ships_to", "address"].edge_index = torch.stack([tx_idx, addr_idx], dim=0)
        data["transaction", "ships_to", "address"].edge_attr = edge_attr
        data["transaction", "ships_to", "address"].t = t

        logger.info(
            "HeteroData: %d tx | %d cards | %d addrs",
            n_tx, len(unique_cards), len(unique_addrs),
        )
        return data

    def save(self, graph: Data | HeteroData, path: Path) -> None:
        """Persist a PyG graph to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, path)
        logger.info("Graph saved to %s", path)

    def load(self, path: Path) -> Data | HeteroData:
        """Load a PyG graph from disk."""
        path = Path(path)
        graph = torch.load(path, weights_only=False)
        logger.info("Graph loaded from %s", path)
        return graph


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry-point for graph construction.

    Usage
    -----
    python -m ml.data.graph_builder build --features data/features/features.parquet
    """
    parser = argparse.ArgumentParser(description="GraphPulse transaction graph builder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_p = subparsers.add_parser("build", help="Build PyG graph from features parquet.")
    build_p.add_argument("--features", default="data/features/features.parquet")
    build_p.add_argument("--output-dir", default="data/graph")
    build_p.add_argument("--hetero", action="store_true", help="Build HeteroData graph.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "build":
        config = GraphConfig(output_dir=args.output_dir)
        builder = TransactionGraphBuilder(config)

        feat_path = Path(args.features)
        logger.info("Loading features from %s", feat_path)
        df = pd.read_parquet(feat_path)

        if args.hetero:
            graph = builder.build_heterogeneous(df)
            out_path = Path(args.output_dir) / "hetero_graph.pt"
        else:
            graph = builder.build_homogeneous(df)
            out_path = Path(args.output_dir) / "homo_graph.pt"

        builder.save(graph, out_path)


if __name__ == "__main__":
    main()
