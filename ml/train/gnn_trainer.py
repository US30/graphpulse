from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
import lightning as L
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class GNNLightningModule(L.LightningModule):
    """Generic Lightning wrapper for GNN fraud classifiers.

    Expects the wrapped model to return raw logits (shape [N] or [N, 1])
    for node-level binary classification.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-3,
        pos_weight: float = 20.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.pos_weight: torch.Tensor  # registered as buffer below

        # Use register_buffer so it moves with the module to the right device
        self.register_buffer("pos_weight_buf", torch.tensor(pos_weight))

        # torchmetrics — reset each epoch automatically
        self.train_auprc = BinaryAveragePrecision()
        self.val_auprc = BinaryAveragePrecision()
        self.val_auroc = BinaryAUROC()

        self.save_hyperparameters(ignore=["model"])

    # ------------------------------------------------------------------
    def _forward_batch(self, batch):
        """Run model forward pass; return (logits_flat, labels_flat)."""
        # batch.y may contain labels for all nodes; we train on seed nodes
        # (NeighborLoader sets batch.batch_size = number of seed nodes)
        logits = self.model(batch.x, batch.edge_index)
        if logits.dim() == 2:
            logits = logits.squeeze(-1)

        n_seed = getattr(batch, "batch_size", logits.size(0))
        logits = logits[:n_seed]
        labels = batch.y[:n_seed].float()
        return logits, labels

    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        logits, labels = self._forward_batch(batch)
        pos_w = self.pos_weight_buf.to(logits.device)
        loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_w
        )

        probs = torch.sigmoid(logits).detach()
        labels_int = labels.int()
        self.train_auprc.update(probs, labels_int)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_auprc",
            self.train_auprc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx: int) -> None:
        logits, labels = self._forward_batch(batch)
        pos_w = self.pos_weight_buf.to(logits.device)
        loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_w
        )

        probs = torch.sigmoid(logits).detach()
        labels_int = labels.int()
        self.val_auprc.update(probs, labels_int)
        self.val_auroc.update(probs, labels_int)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_auprc", self.val_auprc, on_epoch=True, prog_bar=True)
        self.log("val_auroc", self.val_auroc, on_epoch=True, prog_bar=True)

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer is not None else 50,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


# ---------------------------------------------------------------------------
# High-level trainer facade
# ---------------------------------------------------------------------------

class GNNTrainer:
    """Orchestrates NeighborLoader creation and Lightning training for GNNs."""

    def __init__(self, model: torch.nn.Module, config: Any) -> None:
        self.model = model
        self.config = config

    # ------------------------------------------------------------------
    @staticmethod
    def _time_based_masks(graph_data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """Split node indices 80/20 by edge timestamp.

        For each node we assign a 'last seen time' = max of its incident
        edge timestamps.  Nodes in the earliest 80% form the train set.
        Nodes with no edges default to the train set.
        """
        n_nodes = graph_data.num_nodes
        device = graph_data.edge_index.device

        # Default: all nodes in train
        node_time = torch.zeros(n_nodes, device=device)

        if hasattr(graph_data, "edge_time") and graph_data.edge_time is not None:
            edge_time = graph_data.edge_time.to(device)
            src, dst = graph_data.edge_index
            # Scatter max over source and destination
            for node_idx in [src, dst]:
                node_time.scatter_reduce_(
                    0,
                    node_idx,
                    edge_time[: node_idx.size(0)],
                    reduce="amax",
                    include_self=True,
                )

        cutoff = torch.quantile(node_time.float(), 0.80)
        train_mask = node_time <= cutoff
        val_mask = ~train_mask

        # Ensure at least some validation nodes
        if val_mask.sum() == 0:
            logger.warning(
                "No val nodes after time split — using random 20%% split."
            )
            perm = torch.randperm(n_nodes)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask[perm[: max(1, n_nodes // 5)]] = True
            train_mask = ~val_mask

        return train_mask, val_mask

    # ------------------------------------------------------------------
    def train(self, graph_data: Data) -> dict[str, float]:
        """Train the GNN model and return validation metrics.

        Returns
        -------
        dict with keys "val_auprc" and "val_auroc".
        """
        cfg = self.config

        # Resolve hyper-parameters with sensible defaults
        max_epochs: int = getattr(cfg, "max_epochs", 50)
        lr: float = getattr(cfg, "lr", 1e-3)
        pos_weight: float = getattr(cfg, "pos_weight", 20.0)
        num_neighbors: list[int] = getattr(cfg, "num_neighbors", [25, 10])
        batch_size: int = getattr(cfg, "batch_size", 512)

        train_mask, val_mask = self._time_based_masks(graph_data)

        train_idx = train_mask.nonzero(as_tuple=False).squeeze()
        val_idx = val_mask.nonzero(as_tuple=False).squeeze()

        logger.info(
            "Graph split — train nodes: %d  val nodes: %d",
            train_idx.size(0),
            val_idx.size(0),
        )

        train_loader = NeighborLoader(
            graph_data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=train_idx,
            shuffle=True,
        )
        val_loader = NeighborLoader(
            graph_data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=val_idx,
            shuffle=False,
        )

        module = GNNLightningModule(
            model=self.model,
            lr=lr,
            pos_weight=pos_weight,
        )

        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        logger.info("Using accelerator: %s", accelerator)

        trainer = L.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=1,
            log_every_n_steps=10,
            enable_model_summary=True,
            enable_progress_bar=True,
        )

        trainer.fit(module, train_loader, val_loader)

        # Extract final epoch validation metrics from callback_metrics
        cb = trainer.callback_metrics
        val_auprc = float(cb.get("val_auprc", 0.0))
        val_auroc = float(cb.get("val_auroc", 0.0))

        logger.info(
            "GNN training finished — val_auprc=%.4f  val_auroc=%.4f",
            val_auprc,
            val_auroc,
        )

        return {"val_auprc": val_auprc, "val_auroc": val_auroc}
