from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data


@dataclass
class GraphSAGEConfig:
    in_channels: int = 128
    hidden_channels: int = 256
    out_channels: int = 1          # binary fraud score (pre-sigmoid)
    n_layers: int = 3
    dropout: float = 0.3
    aggregator: str = "mean"       # mean | max | lstm


class GraphSAGEEncoder(nn.Module):
    """Multi-layer GraphSAGE encoder that produces node embeddings."""

    def __init__(self, config: GraphSAGEConfig) -> None:
        super().__init__()
        self.config = config

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        in_ch = config.in_channels
        for i in range(config.n_layers):
            out_ch = config.hidden_channels
            aggr = config.aggregator
            self.convs.append(SAGEConv(in_ch, out_ch, aggr=aggr))
            self.bns.append(nn.BatchNorm1d(out_ch))
            in_ch = out_ch

        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute node embeddings via stacked SAGEConv layers.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [N, in_channels].
        edge_index : torch.Tensor
            Edge connectivity of shape [2, E].

        Returns
        -------
        torch.Tensor
            Node embeddings of shape [N, hidden_channels].
        """
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = self.dropout(x)
        return x

    def reset_parameters(self) -> None:
        """Re-initialize all learnable parameters."""
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


class GraphSAGEClassifier(nn.Module):
    """GraphSAGE encoder + linear classification head for node-level fraud detection."""

    def __init__(self, config: GraphSAGEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = GraphSAGEEncoder(config)
        self.head = nn.Linear(config.hidden_channels, config.out_channels)

    def forward(self, data: Data) -> torch.Tensor:
        """Run forward pass over a PyG Data object.

        Returns
        -------
        torch.Tensor
            Node-level fraud logits of shape [N, out_channels] (pre-sigmoid).
        """
        x = self.encoder(data.x, data.edge_index)
        logits = self.head(x)
        return logits

    def predict_proba(self, data: Data) -> torch.Tensor:
        """Return fraud probabilities in [0, 1] of shape [N, out_channels]."""
        logits = self.forward(data)
        return torch.sigmoid(logits)
