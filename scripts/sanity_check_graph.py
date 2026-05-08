"""
Sanity check for the graph-aware denoiser components.

Loads the 'cardio' dataset (small, mixed types), builds a static adjacency,
instantiates GraphAwareDenoiser in both static and dynamic modes, and verifies:
  1. Static adjacency shape and density are printed.
  2. Forward pass output shape equals input shape.
  3. Backward pass assigns gradients to all parameters.

Run from the repository root:
    python scripts/sanity_check_graph.py
"""

import sys
import os

# Allow imports from scripts/ and the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from tab_ddpm.graph_builder import build_static_adjacency
from tab_ddpm.graph_denoiser import GraphAwareDenoiser

# ---------------------------------------------------------------------------
# 1. Load dataset (raw numpy arrays — no transformation needed for the check)
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'cardio')

X_num = np.load(os.path.join(DATA_PATH, 'X_num_train.npy')).astype(np.float32)
X_cat = np.load(os.path.join(DATA_PATH, 'X_cat_train.npy')).astype(np.int64)

d_num = X_num.shape[1]          # 5 numerical features
n_cat = X_cat.shape[1]          # 6 categorical features
cat_sizes = [int(X_cat[:, i].max()) + 1 for i in range(n_cat)]

print("=" * 60)
print(f"Dataset : cardio  (train rows = {X_num.shape[0]})")
print(f"d_num   : {d_num}")
print(f"n_cat   : {n_cat}  sizes={cat_sizes}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 2. Build static adjacency
# ---------------------------------------------------------------------------
adj = build_static_adjacency(
    X_num=X_num,
    X_cat=X_cat,
    d_num=d_num,
    cat_sizes=cat_sizes,
    threshold=0.1,
)

N = d_num + n_cat
assert adj.shape == (N, N), f"Expected ({N},{N}), got {adj.shape}"
assert (adj.diagonal() == 1.0).all(), "Self-loops missing in static adjacency"

n_edges_off_diag = int(adj.sum().item()) - N
density = n_edges_off_diag / max(N * (N - 1), 1)
print(f"\n[static adj] shape={tuple(adj.shape)}  "
      f"off-diag edges={n_edges_off_diag}  density={density:.3f}")
print("Static adjacency check PASSED\n")

# ---------------------------------------------------------------------------
# 3. Build a synthetic batch for forward / backward checks
# ---------------------------------------------------------------------------
B = 8
d_in = d_num + sum(cat_sizes)
# Random numerical part
x_num_part = torch.randn(B, d_num)
# Random one-hot categorical parts
cat_parts = []
for K in cat_sizes:
    idx = torch.randint(0, K, (B,))
    ohe = torch.zeros(B, K)
    ohe.scatter_(1, idx.unsqueeze(1), 1.0)
    cat_parts.append(ohe)
x = torch.cat([x_num_part] + cat_parts, dim=1)          # (B, d_in)
timesteps = torch.randint(0, 1000, (B,))
y = torch.randint(0, 2, (B,))

assert x.shape == (B, d_in), f"Unexpected batch shape: {x.shape}"

# ---------------------------------------------------------------------------
# 4a. Static mode — forward + backward
# ---------------------------------------------------------------------------
print("[static mode]")
model_static = GraphAwareDenoiser(
    d_in=d_in,
    num_classes=2,
    is_y_cond=True,
    d_num=d_num,
    cat_sizes=cat_sizes,
    d_model=64,
    n_layers=2,
    n_heads=4,
    graph_mode='static',
    adjacency=adj,
    top_k=0,
)

out_static = model_static(x, timesteps, y)
assert out_static.shape == (B, d_in), (
    f"Static forward: expected ({B},{d_in}), got {out_static.shape}"
)
print(f"  forward  OK — output shape {tuple(out_static.shape)}")

loss_static = out_static.sum()
loss_static.backward()
no_grad = [n for n, p in model_static.named_parameters() if p.grad is None]
assert len(no_grad) == 0, f"Params with no grad (static): {no_grad}"
print(f"  backward OK — all {sum(1 for _ in model_static.parameters())} params received gradients")
print("Static mode check PASSED\n")

# ---------------------------------------------------------------------------
# 4b. Dynamic mode — forward + backward
# ---------------------------------------------------------------------------
print("[dynamic mode]")
model_dynamic = GraphAwareDenoiser(
    d_in=d_in,
    num_classes=2,
    is_y_cond=True,
    d_num=d_num,
    cat_sizes=cat_sizes,
    d_model=64,
    n_layers=2,
    n_heads=4,
    graph_mode='dynamic',
    adjacency=None,
    top_k=0,
)

out_dynamic = model_dynamic(x, timesteps, y)
assert out_dynamic.shape == (B, d_in), (
    f"Dynamic forward: expected ({B},{d_in}), got {out_dynamic.shape}"
)
print(f"  forward  OK — output shape {tuple(out_dynamic.shape)}")

loss_dynamic = out_dynamic.sum()
loss_dynamic.backward()
no_grad = [n for n, p in model_dynamic.named_parameters() if p.grad is None]
assert len(no_grad) == 0, f"Params with no grad (dynamic): {no_grad}"
print(f"  backward OK — all {sum(1 for _ in model_dynamic.parameters())} params received gradients")
print("Dynamic mode check PASSED\n")

# ---------------------------------------------------------------------------
# 5. Edge case: no categorical features (all numerical)
# ---------------------------------------------------------------------------
print("[no-categorical edge case]")
d_num_only = 8
x_num_only = torch.randn(B, d_num_only)
adj_num_only = build_static_adjacency(
    X_num=np.random.randn(100, d_num_only).astype(np.float32),
    X_cat=None,
    d_num=d_num_only,
    cat_sizes=[],
    threshold=0.0,   # low threshold ensures some edges
)
model_num_only = GraphAwareDenoiser(
    d_in=d_num_only,
    num_classes=0,
    is_y_cond=False,
    d_num=d_num_only,
    cat_sizes=[],
    d_model=32,
    n_layers=1,
    n_heads=4,
    graph_mode='static',
    adjacency=adj_num_only,
)
out_num_only = model_num_only(x_num_only, timesteps)
assert out_num_only.shape == (B, d_num_only)
print(f"  forward OK — output shape {tuple(out_num_only.shape)}")
print("No-categorical edge case PASSED\n")

print("=" * 60)
print("All sanity checks PASSED.")
print("=" * 60)
