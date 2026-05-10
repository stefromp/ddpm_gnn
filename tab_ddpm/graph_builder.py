"""
Graph construction utilities for GraphAwareDenoiser.

Two modes:
  static  — edges derived once from training-data statistics (Pearson
             correlation, Cramér's V, normalised mutual information).
  dynamic — soft adjacency learned end-to-end from node embeddings via
             scaled dot-product attention (DynamicAdjacency module).
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Static adjacency
# ---------------------------------------------------------------------------

def build_static_adjacency(
    X_num: Optional[np.ndarray],
    X_cat: Optional[np.ndarray],
    d_num: int,
    cat_sizes: List[int],
    threshold: float = 0.1,
) -> torch.Tensor:
    """Return a (N, N) float32 adjacency matrix derived from training data.

    N = d_num + len(cat_sizes).  One node per numerical feature and one node
    per categorical feature (regardless of cardinality).

    Edge rules (undirected, based on absolute-value scores):
      num–num : |Pearson correlation| > threshold
      cat–cat : Cramér's V            > threshold
      num–cat : normalised MI         > threshold

    Self-loops are always included.

    Args:
        X_num:     (n_samples, d_num) float array, or None when d_num == 0.
        X_cat:     (n_samples, n_cat) integer array, or None when no cats.
        d_num:     number of numerical features.
        cat_sizes: cardinalities of each categorical feature (no zeros).
        threshold: minimum score required to add an edge.

    Returns:
        adj: float32 tensor of shape (N, N) with values in {0.0, 1.0}.
    """
    n_cat = len(cat_sizes)
    N = d_num + n_cat
    adj = np.zeros((N, N), dtype=np.float32)
    np.fill_diagonal(adj, 1.0)

    if N == 0:
        return torch.from_numpy(adj)

    # --- num–num : Pearson correlation ---
    if X_num is not None and d_num >= 2:
        corr = np.corrcoef(X_num.T)          # (d_num, d_num), may contain NaN
        corr = np.nan_to_num(corr, nan=0.0)
        for i in range(d_num):
            for j in range(i + 1, d_num):
                if abs(corr[i, j]) > threshold:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

    # --- cat–cat : Cramér's V ---
    if X_cat is not None and n_cat >= 2:
        from scipy.stats import chi2_contingency  # lazy import

        for i in range(n_cat):
            for j in range(i + 1, n_cat):
                table = _contingency_table(X_cat[:, i], X_cat[:, j])
                chi2, _, _, _ = chi2_contingency(table, correction=False)
                n_obs = table.sum()
                r, k = table.shape
                denom = n_obs * (min(r, k) - 1)
                cramer_v = float(np.sqrt(chi2 / (denom + 1e-12)))
                cramer_v = min(cramer_v, 1.0)
                if cramer_v > threshold:
                    ni, nj = d_num + i, d_num + j
                    adj[ni, nj] = 1.0
                    adj[nj, ni] = 1.0

    # --- num–cat : normalised mutual information ---
    if X_num is not None and X_cat is not None and d_num > 0 and n_cat > 0:
        from sklearn.feature_selection import mutual_info_classif  # lazy import

        for j in range(n_cat):
            mi_scores = mutual_info_classif(
                X_num, X_cat[:, j], discrete_features=False, random_state=0
            )
            _, counts = np.unique(X_cat[:, j], return_counts=True)
            probs = counts / counts.sum()
            h_cat = float(-np.sum(probs * np.log(probs + 1e-12)))
            for i in range(d_num):
                nmi = float(mi_scores[i]) / (h_cat + 1e-12)
                nmi = min(nmi, 1.0)
                if nmi > threshold:
                    nj = d_num + j
                    adj[i, nj] = 1.0
                    adj[nj, i] = 1.0

    return torch.from_numpy(adj)


def _contingency_table(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Build an (r, k) integer contingency table for two integer arrays."""
    a_vals, a_inv = np.unique(a, return_inverse=True)
    b_vals, b_inv = np.unique(b, return_inverse=True)
    table = np.zeros((len(a_vals), len(b_vals)), dtype=np.int64)
    np.add.at(table, (a_inv, b_inv), 1)
    return table


# ---------------------------------------------------------------------------
# Dynamic adjacency
# ---------------------------------------------------------------------------

class DynamicAdjacency(nn.Module):
    """Learnable soft adjacency derived from node embeddings each forward pass.

    Uses scaled dot-product attention between node embeddings to produce a
    (B, N, N) adjacency in (0, 1).  Optionally sparsifies to the top-k
    neighbours per node (off-diagonal), keeping the result differentiable via
    a straight-through mask.  Self-loops are always set to 1.

    Args:
        n_nodes: number of graph nodes N.
        d_model: node embedding dimension.
        top_k:   if > 0, keep only the top-k off-diagonal entries per row;
                 0 means dense soft adjacency (full sigmoid).
    """

    def __init__(self, n_nodes: int, d_model: int, top_k: int = 0) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.top_k = top_k
        self.scale = math.sqrt(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        """Compute a (B, N, N) soft adjacency from node embeddings.

        Args:
            node_emb: (B, N, d_model) node embedding tensor.

        Returns:
            adj: (B, N, N) float tensor with values in [0, 1].
        """
        B, N, _ = node_emb.shape
        Q = self.q_proj(node_emb)                              # (B, N, D)
        K = self.k_proj(node_emb)                              # (B, N, D)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, N, N)
        adj = torch.sigmoid(scores)

        if self.top_k > 0 and self.top_k < N - 1:
            eye = torch.eye(N, device=adj.device, dtype=adj.dtype).unsqueeze(0)
            adj_no_self = adj * (1.0 - eye)
            _, topk_idx = torch.topk(adj_no_self, k=self.top_k, dim=-1)
            mask = torch.zeros_like(adj_no_self).scatter_(-1, topk_idx, 1.0)
            adj = adj * mask

        # Enforce self-loops
        eye = torch.eye(N, device=adj.device, dtype=adj.dtype).unsqueeze(0)
        adj = torch.clamp(adj + eye, max=1.0)
        return adj
