from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)


@dataclass
class TGNConfig:
    node_feat_dim: int = 64        # raw node feature dimension
    edge_feat_dim: int = 32        # raw edge (transaction) feature dimension
    memory_dim: int = 100          # TGN memory state size
    time_dim: int = 100            # time embedding dimension
    embedding_dim: int = 100       # output node embedding dim
    n_heads: int = 2               # TransformerConv heads
    n_layers: int = 1              # GNN message-passing layers
    dropout: float = 0.1
    num_nodes: int = 500_000       # upper bound on node count (padded)
    batch_size: int = 200
    neighbor_size: int = 10        # number of temporal neighbors to sample


class TGNFraudClassifier(nn.Module):
    """Temporal Graph Network (TGN) classifier for real-time fraud detection.

    Combines TGN memory, TransformerConv message passing, and a link-level
    classifier head that scores each (src, dst) transaction pair.
    """

    def __init__(self, config: TGNConfig) -> None:
        super().__init__()
        self.config = config

        # TGN persistent memory module
        self.memory = TGNMemory(
            config.num_nodes,
            config.edge_feat_dim,
            config.memory_dim,
            config.time_dim,
            message_module=IdentityMessage(
                config.edge_feat_dim,
                config.memory_dim,
                config.time_dim,
            ),
            aggregator_module=LastAggregator(),
        )

        # GNN aggregation over temporal neighborhood
        # TransformerConv input: memory_dim + node_feat_dim → embedding_dim
        self.gnn = TransformerConv(
            in_channels=config.memory_dim + config.node_feat_dim,
            out_channels=config.embedding_dim // config.n_heads,
            heads=config.n_heads,
            dropout=config.dropout,
        )

        # Link-level fraud classifier head (operates on concatenated src+dst embeddings)
        self.link_pred = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1),
        )

        # Temporal neighborhood loader
        self.neighbor_loader = LastNeighborLoader(
            config.num_nodes,
            size=config.neighbor_size,
        )

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        t: torch.Tensor,
        msg: torch.Tensor,
        edge_index: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute fraud logits for a batch of (src, dst) transaction pairs.

        Parameters
        ----------
        src : torch.Tensor
            Source node indices of shape [B].
        dst : torch.Tensor
            Destination node indices of shape [B].
        t : torch.Tensor
            Transaction timestamps of shape [B].
        msg : torch.Tensor
            Edge/transaction feature messages of shape [B, edge_feat_dim].
        edge_index : torch.Tensor
            Temporal neighborhood edge connectivity of shape [2, E].
        x : torch.Tensor
            Raw node features for all nodes in the subgraph [N, node_feat_dim].

        Returns
        -------
        torch.Tensor
            Fraud logits of shape [B, 1] for each (src, dst) pair.
        """
        # Retrieve memory states for all unique nodes
        n_id = torch.cat([src, dst]).unique()
        z_mem, last_update = self.memory(n_id)

        # Concatenate memory state with raw node features
        # x[n_id] selects features for the relevant nodes
        x_sub = x[n_id]
        z_in = torch.cat([z_mem, x_sub], dim=-1)  # [N_sub, memory_dim + node_feat_dim]

        # GNN message passing over temporal neighborhood
        z_out = self.gnn(z_in, edge_index)  # [N_sub, embedding_dim]

        # Build lookup from global node id → position in n_id for indexing
        id_to_pos = {nid.item(): pos for pos, nid in enumerate(n_id)}
        src_pos = torch.tensor([id_to_pos[s.item()] for s in src], device=src.device)
        dst_pos = torch.tensor([id_to_pos[d.item()] for d in dst], device=dst.device)

        z_src = z_out[src_pos]  # [B, embedding_dim]
        z_dst = z_out[dst_pos]  # [B, embedding_dim]

        # Concatenate and score
        z_pair = torch.cat([z_src, z_dst], dim=-1)  # [B, embedding_dim * 2]
        logits = self.link_pred(z_pair)              # [B, 1]
        return logits

    def reset_memory(self) -> None:
        """Reset TGN memory to zero state (call between independent graph segments)."""
        self.memory.reset_state()

    def detach_memory(self) -> None:
        """Detach memory from the computation graph for truncated BPTT."""
        self.memory.detach_()

    def update_memory(
        self,
        n_id: torch.Tensor,
        msg: torch.Tensor,
        t: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> None:
        """Update TGN memory after processing a batch.

        Parameters
        ----------
        n_id : torch.Tensor
            Node indices involved in the batch.
        msg : torch.Tensor
            Edge messages of shape [E, edge_feat_dim].
        t : torch.Tensor
            Timestamps of shape [E].
        edge_index : torch.Tensor
            Edge connectivity of shape [2, E].
        """
        self.memory.update_state(
            src=edge_index[0],
            dst=edge_index[1],
            t=t,
            raw_msg=msg,
        )
