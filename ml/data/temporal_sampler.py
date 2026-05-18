from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torch_geometric.data import Data


@dataclass
class SamplerConfig:
    batch_size: int = 200
    neg_sample_ratio: float = 1.0   # negative edges per positive edge
    seed: int = 42


class TemporalEdgeSampler:
    """Chronological mini-batch sampler for Temporal Graph Network (TGN) training.

    Yields positive transaction edges in time-sorted order alongside randomly
    sampled negative (non-existing) destination nodes.

    Parameters
    ----------
    data : Data
        PyG Data object with fields:
        - ``edge_index`` [2, E] — source and destination node indices.
        - ``t`` [E] — edge timestamps (floats, e.g. TransactionDT).
        - ``edge_attr`` [E, F] — edge feature messages.
        - ``y`` [N] — node labels (-1 for non-transaction nodes).
    config : SamplerConfig
        Sampler hyper-parameters.
    """

    def __init__(self, data: Data, config: SamplerConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed=config.seed)

        # Extract edges and sort chronologically
        src_all = data.edge_index[0].numpy()
        dst_all = data.edge_index[1].numpy()
        t_all = data.t.numpy() if data.t is not None else np.zeros(src_all.shape[0])
        msg_all = data.edge_attr.numpy() if data.edge_attr is not None else np.zeros((src_all.shape[0], 1))

        order = np.argsort(t_all, kind="stable")
        self.src = src_all[order]           # [E]
        self.dst = dst_all[order]           # [E]
        self.t = t_all[order]              # [E]
        self.msg = msg_all[order]          # [E, F]

        self.num_edges = len(self.src)
        self.num_nodes = int(data.num_nodes) if data.num_nodes is not None else int(max(src_all.max(), dst_all.max()) + 1)
        self.batch_size = config.batch_size
        self.neg_ratio = config.neg_sample_ratio

        # Build adjacency set for negative sampling (src → set of dst)
        self._adj: dict[int, set[int]] = {}
        for s, d in zip(self.src, self.dst):
            self._adj.setdefault(int(s), set()).add(int(d))

    # ------------------------------------------------------------------
    # Negative sampling
    # ------------------------------------------------------------------

    def _sample_negatives(self, src_batch: np.ndarray) -> np.ndarray:
        """Sample random negative destination nodes not connected to each source.

        For each source node s_i, draw ``neg_sample_ratio`` random dst nodes
        that do NOT appear as an existing neighbour of s_i.

        Returns
        -------
        np.ndarray
            Shape [B * neg_k] of negative destination indices, where
            neg_k = max(1, int(self.neg_ratio)).
        """
        neg_k = max(1, int(self.neg_ratio))
        neg_dsts = []
        for s in src_batch:
            existing = self._adj.get(int(s), set())
            candidates = []
            attempts = 0
            while len(candidates) < neg_k and attempts < neg_k * 10:
                rand_dst = self.rng.integers(0, self.num_nodes)
                if rand_dst not in existing:
                    candidates.append(rand_dst)
                attempts += 1
            # Pad with random nodes if not enough unique negatives found
            while len(candidates) < neg_k:
                candidates.append(self.rng.integers(0, self.num_nodes))
            neg_dsts.extend(candidates)
        return np.array(neg_dsts, dtype=np.int64)

    # ------------------------------------------------------------------
    # Iterator protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of batches (ceil division)."""
        return (self.num_edges + self.batch_size - 1) // self.batch_size

    def __iter__(
        self,
    ) -> Iterator[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """Yield mini-batches in chronological order.

        Yields
        ------
        tuple of Tensors
            ``(src, dst, t, msg, neg_dst)`` where:
            - ``src``     : [B] source node indices (long).
            - ``dst``     : [B] destination node indices (long).
            - ``t``       : [B] edge timestamps (float).
            - ``msg``     : [B, F] edge feature messages (float).
            - ``neg_dst`` : [B * neg_k] negative destination indices (long).
        """
        for start in range(0, self.num_edges, self.batch_size):
            end = min(start + self.batch_size, self.num_edges)

            src_b = self.src[start:end]
            dst_b = self.dst[start:end]
            t_b = self.t[start:end]
            msg_b = self.msg[start:end]
            neg_dst_b = self._sample_negatives(src_b)

            yield (
                torch.tensor(src_b, dtype=torch.long),
                torch.tensor(dst_b, dtype=torch.long),
                torch.tensor(t_b, dtype=torch.float),
                torch.tensor(msg_b, dtype=torch.float),
                torch.tensor(neg_dst_b, dtype=torch.long),
            )
