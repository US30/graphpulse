from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import HeteroData


@dataclass
class HGTConfig:
    hidden_channels: int = 128
    out_channels: int = 1
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.2
    node_types: list = field(
        default_factory=lambda: ["transaction", "card", "address"]
    )
    edge_types: list = field(
        default_factory=lambda: [
            ("transaction", "uses", "card"),
            ("transaction", "ships_to", "address"),
        ]
    )


class HGTClassifier(nn.Module):
    """Heterogeneous Graph Transformer (HGT) classifier for fraud detection.

    Supports heterogeneous node and edge types (transaction, card, address).
    """

    def __init__(self, config: HGTConfig, metadata: tuple) -> None:
        """
        Parameters
        ----------
        config : HGTConfig
            Model hyperparameters.
        metadata : tuple
            PyG metadata tuple (node_types, edge_types) from the HeteroData graph.
            Pass ``hetero_data.metadata()`` here at construction time.
        """
        super().__init__()
        self.config = config

        # Per-node-type linear projection from raw features → hidden_channels
        # Linear(-1, ...) defers input dimension inference to first forward pass.
        self.lin_dict = nn.ModuleDict(
            {nt: Linear(-1, config.hidden_channels) for nt in config.node_types}
        )

        # Stack of HGTConv layers
        self.convs = nn.ModuleList(
            [
                HGTConv(
                    config.hidden_channels,
                    config.hidden_channels,
                    metadata,
                    config.n_heads,
                    group="sum",
                )
                for _ in range(config.n_layers)
            ]
        )

        self.dropout = nn.Dropout(p=config.dropout)

        # Classification head applied to "transaction" node embeddings
        self.head = nn.Linear(config.hidden_channels, config.out_channels)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run HGT forward pass over the heterogeneous graph.

        Parameters
        ----------
        x_dict : dict[str, Tensor]
            Mapping from node type name → feature matrix [N_type, feat_dim].
        edge_index_dict : dict[tuple, Tensor]
            Mapping from edge type triple → edge_index [2, E_type].

        Returns
        -------
        dict[str, Tensor]
            Contains key ``"transaction"`` with fraud logits of shape
            [N_transaction, out_channels] (pre-sigmoid).
        """
        # Project each node type to hidden_channels
        h_dict: dict[str, torch.Tensor] = {}
        for node_type, lin in self.lin_dict.items():
            if node_type in x_dict:
                h_dict[node_type] = F.relu(lin(x_dict[node_type]))

        # Apply stacked HGTConv layers
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {k: F.relu(self.dropout(v)) for k, v in h_dict.items()}

        # Compute fraud logits for transaction nodes
        transaction_emb = h_dict["transaction"]            # [N_tx, hidden_channels]
        logits = self.head(transaction_emb)                # [N_tx, out_channels]

        return {"transaction": logits}

    def predict_proba_transactions(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
    ) -> torch.Tensor:
        """Return sigmoid fraud probabilities for transaction nodes.

        Returns
        -------
        torch.Tensor
            Shape [N_transaction, out_channels] in [0, 1].
        """
        out = self.forward(x_dict, edge_index_dict)
        return torch.sigmoid(out["transaction"])
