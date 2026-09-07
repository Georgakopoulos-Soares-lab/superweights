"""
Shared math for E7 -- exact bilinear-amplifier operator dimensionality.

    U_k = sum_i  d_i * outer(g_i, u_i),   d_i = W_down[k,i],  g_i = W_gate[i,:],  u_i = W_up[i,:]

U_k is the [d_model, d_model] matrix already implicitly defined by ||U_k||_F in
uk_frobenius.py and by E5/E6's exact quadratic form -- this module makes the operator itself,
not just its norm, and reports its singular spectrum. Orientation matches the existing
brute-force construction used in E5's and E6's own test suites
(`Uk += d[k,i] * torch.outer(g[i], u[i])`) exactly; a global transpose would not change any
singular value.

No cross terms are dropped here (unlike the diagonal c_{k,i} decomposition): U_k is
materialized in closed form via a single [d_model, d_ffn] x [d_ffn, d_model] matmul,

    U_k = (d[:, None] * W_gate)^T @ W_up

which is mathematically identical to the sum above (linearity), and is the same cost class
already validated fast enough in E5/E6 (O(d_model^2 * d_ffn)), avoiding a slow per-coordinate
Python loop for real model sizes.

Primary metrics (Phase 2, fixed set of two, no others):
    q1       = sigma_1^2 / sum_j sigma_j^2         -- top singular-energy share
    PR_spec  = (sum_j sigma_j^2)^2 / sum_j sigma_j^4  -- spectral participation ratio
Secondary, algebraically redundant with q1, reported but never primary:
    stable_rank = 1 / q1
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@torch.no_grad()
def materialize_Uk(W_gate: torch.Tensor, W_up: torch.Tensor, w_down_row: torch.Tensor,
                    dtype: torch.dtype = torch.float64, device: str | None = None) -> torch.Tensor:
    """U_k = (d * W_gate)^T @ W_up, exactly. [d_model, d_model], float64 by default."""
    dev = device or (W_gate.device if W_gate.is_cuda else "cpu")
    G = W_gate.to(dtype).to(dev)
    U = W_up.to(dtype).to(dev)
    d = w_down_row.to(dtype).to(dev)
    return (d[:, None] * G).T @ U


@torch.no_grad()
def singular_values(Uk: torch.Tensor) -> torch.Tensor:
    """Descending singular values of U_k, float64, on CPU (svdvals only -- no singular
    vectors are needed for q1/PR_spec, cheaper than a full SVD)."""
    return torch.linalg.svdvals(Uk).cpu()


@dataclass
class SpectralMetrics:
    q1: float
    pr_spec: float
    stable_rank: float
    sigma1: float
    frob_norm: float          # sqrt(sum sigma_j^2) -- must equal ||U_k||_F exactly
    n_singular_values: int


def spectral_metrics(sigmas: torch.Tensor) -> SpectralMetrics:
    sig2 = (sigmas.double()) ** 2
    total = float(sig2.sum())
    q1 = float(sig2[0]) / total
    pr_spec = (total ** 2) / float((sig2 ** 2).sum())
    return SpectralMetrics(
        q1=q1, pr_spec=pr_spec, stable_rank=1.0 / q1, sigma1=float(sigmas[0]),
        frob_norm=total ** 0.5, n_singular_values=int(sigmas.numel()),
    )


@torch.no_grad()
def row_spectral_metrics(W_gate: torch.Tensor, W_up: torch.Tensor, w_down_row: torch.Tensor,
                          device: str | None = None) -> tuple[SpectralMetrics, torch.Tensor]:
    """Convenience wrapper: materialize U_k, take its spectrum, compute q1/PR_spec.
    Returns (metrics, full descending singular-value vector) -- the vector is kept for
    supplementary reporting (e.g. a scree-style top-k share table), never as a new primary
    metric."""
    Uk = materialize_Uk(W_gate, W_up, w_down_row, device=device)
    sigmas = singular_values(Uk)
    del Uk
    return spectral_metrics(sigmas), sigmas
