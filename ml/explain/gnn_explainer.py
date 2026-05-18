from __future__ import annotations

import logging

import torch
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TGN explainer wrapper
# ---------------------------------------------------------------------------

class TGNExplainerWrapper:
    """Node-level GNNExplainer for TGN (and other GNN) fraud classifiers.

    Wraps ``torch_geometric.explain.Explainer`` with the ``GNNExplainer``
    algorithm to produce subgraph rationales for individual suspicious nodes.

    Parameters
    ----------
    model: A trained GNN model (TGNFraudClassifier or any compatible module)
           that accepts ``(x, edge_index)`` and returns per-node probabilities.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="binary_classification",
                task_level="node",
                return_type="probs",
            ),
        )
        logger.info(
            "TGNExplainerWrapper initialised with GNNExplainer(epochs=200)."
        )

    # ------------------------------------------------------------------
    def explain_node(self, node_idx: int, data: Data) -> dict:
        """Explain the model's prediction for a single node.

        Parameters
        ----------
        node_idx: Index of the node to explain.
        data:     The full (or local) PyG Data object containing ``x`` and
                  ``edge_index`` (and optionally ``edge_attr``).

        Returns
        -------
        dict with:
        - node_idx (int)
        - node_feat_importance (list[float]): per-feature attribution, shape [d].
        - top_subgraph_edges (list[tuple[int,int]]): edges with highest masks.
        - explanation_score (float): scalar summarising explanation quality
          (mean absolute node feature importance).
        """
        kwargs: dict = dict(
            x=data.x,
            edge_index=data.edge_index,
            index=node_idx,
        )
        if data.edge_attr is not None:
            kwargs["edge_attr"] = data.edge_attr

        try:
            explanation = self.explainer(**kwargs)
        except Exception as exc:
            logger.error(
                "GNNExplainer failed for node %d: %s", node_idx, exc
            )
            raise

        # Node feature importance
        if explanation.node_mask is not None:
            # node_mask shape: [n_nodes, n_features]  or [n_features]
            nm = explanation.node_mask
            if nm.dim() == 2:
                feat_imp = nm[node_idx].cpu().tolist()
            else:
                feat_imp = nm.cpu().tolist()
        else:
            feat_imp = []

        # Top-k subgraph edges by edge mask value
        top_edges: list[tuple[int, int]] = []
        if explanation.edge_mask is not None:
            em = explanation.edge_mask.cpu()  # [n_edges]
            top_k = min(10, em.size(0))
            top_edge_idx = torch.argsort(em, descending=True)[:top_k]
            src = data.edge_index[0].cpu()
            dst = data.edge_index[1].cpu()
            top_edges = [
                (int(src[i]), int(dst[i])) for i in top_edge_idx.tolist()
            ]

        explanation_score = float(
            torch.tensor(feat_imp).abs().mean().item()
        ) if feat_imp else 0.0

        return {
            "node_idx": node_idx,
            "node_feat_importance": feat_imp,
            "top_subgraph_edges": top_edges,
            "explanation_score": explanation_score,
        }

    # ------------------------------------------------------------------
    def explain_batch(
        self,
        node_indices: list[int],
        data: Data,
    ) -> list[dict]:
        """Explain model predictions for a list of node indices.

        Parameters
        ----------
        node_indices: List of node indices to explain.
        data:         PyG Data object shared across all explanations.

        Returns
        -------
        List of explanation dicts (same format as ``explain_node``),
        in the same order as ``node_indices``.  Nodes that fail to
        explain are included with an ``"error"`` key instead of raising.
        """
        results: list[dict] = []
        total = len(node_indices)
        for i, nid in enumerate(node_indices):
            try:
                result = self.explain_node(nid, data)
                results.append(result)
                if (i + 1) % max(1, total // 10) == 0:
                    logger.info(
                        "Explained %d / %d nodes.", i + 1, total
                    )
            except Exception as exc:
                logger.warning(
                    "Skipping node %d due to error: %s", nid, exc
                )
                results.append(
                    {
                        "node_idx": nid,
                        "node_feat_importance": [],
                        "top_subgraph_edges": [],
                        "explanation_score": 0.0,
                        "error": str(exc),
                    }
                )
        return results
