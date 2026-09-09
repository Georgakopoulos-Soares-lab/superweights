"""
Shared math for E6 Stage A -- exact cross-pathway interaction geometry of a gated-FFN
high-gain row.

Pair-index convention (fixed, stated once, used consistently everywhere in this module):
given the elementwise-Gram matrix

    K[i,j] = (W_gate[i,:] . W_gate[j,:]) * (W_up[i,:] . W_up[j,:])          [d_ffn, d_ffn]

and, for a fixed target row k, d_i = W_down[k, i], define

    X[i,j] = d_i * d_j * K[i,j]                                            [d_ffn, d_ffn]

X is symmetric (K is symmetric because dot products are). Sums below range over ALL ORDERED
index pairs (i, j) with i in 0..d_ffn-1, j in 0..d_ffn-1 -- i.e. every unordered pair {i,j},
i != j, is counted TWICE (once as (i,j), once as (j,i)), exactly matching
sum_{i,j} X[i,j] = d^T K d = S_exact^2. This is deliberate: it keeps the identity
S_exact^2 = S_diag^2 + (cross sum) exact with no extra factor-of-2 bookkeeping anywhere else
in this module or its tests.

Reuses experiments/frozen/E5_dimensionality/dimensionality_lib.py's already-validated
exact_uk_all_rows() for the layer-wide reference distribution (Task 2F), and
experiments/src/uk_frobenius.py's uk_frobenius() for the diagonal form -- per instruction,
E5's math is reused, not reimplemented, and neither module is modified.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "E5_dimensionality"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dimensionality_lib import exact_uk_all_rows  # noqa: E402  (E5, read-only reuse)
from uk_frobenius import uk_frobenius  # noqa: E402


# --------------------------------------------------------------------------------------
# Pair matrix and per-row cross geometry (Task 2 A-E)
# --------------------------------------------------------------------------------------

@torch.no_grad()
def pair_matrix(W_gate: torch.Tensor, W_up: torch.Tensor,
                 dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """K[i,j] = (g_i . g_j)(u_i . u_j). Symmetric, [d_ffn, d_ffn]. K[i,i] = A_i (E5's factor_A)."""
    G = W_gate.to(dtype)
    U = W_up.to(dtype)
    return (G @ G.T) * (U @ U.T)


@dataclass
class RowCrossGeometry:
    s_exact: float
    s_diag: float
    f_cross: float                  # signed, NOT clipped to [0,1]
    x_pos: float                    # sum over ordered i!=j of max(X_ij, 0)
    x_neg: float                    # sum over ordered i!=j of min(X_ij, 0), <= 0
    x_pos_norm: float                # x_pos / S_exact^2
    x_neg_norm: float                # x_neg / S_exact^2 (<= 0)
    gross_cross_norm: float          # x_pos_norm - x_neg_norm = (|x_pos| + |x_neg|) / S_exact^2
    kappa: float | str               # coherence statistic, or "undefined (no cross mass)"
    d_ffn: int


@torch.no_grad()
def row_cross_geometry(K: torch.Tensor, w_down_row: torch.Tensor,
                        dtype: torch.dtype = torch.float64) -> RowCrossGeometry:
    """Full cross-pathway decomposition for one target row. K is [d_ffn, d_ffn] (pair_matrix
    output); w_down_row is W_down[k, :], shape [d_ffn]. Materializes one [d_ffn, d_ffn] dense
    matrix -- memory-safe for every candidate row in this experiment's panel (largest is
    OLMo-7B at d_ffn=11008, ~968MB in float64, well within the confirmed-idle A100-40GB or
    the node's 60GB free system RAM)."""
    d = w_down_row.to(dtype)
    X = torch.outer(d, d) * K                      # [d_ffn, d_ffn]
    diag = torch.diagonal(X)
    s_exact_sq = float(X.sum())
    s_diag_sq = float(diag.sum())
    f_cross = (s_exact_sq - s_diag_sq) / s_exact_sq

    off = X.clone()
    off.fill_diagonal_(0.0)
    x_pos = float(off.clamp_min(0).sum())
    x_neg = float(off.clamp_max(0).sum())

    # identity check (not a test -- a runtime guard against a silent convention error)
    cross_sum = s_exact_sq - s_diag_sq
    assert abs((x_pos + x_neg) - cross_sum) <= 1e-6 * max(abs(cross_sum), 1.0), \
        f"x_pos + x_neg ({x_pos + x_neg}) != S_exact^2 - S_diag^2 ({cross_sum})"

    x_pos_norm = x_pos / s_exact_sq
    x_neg_norm = x_neg / s_exact_sq
    gross = x_pos_norm - x_neg_norm

    denom = x_pos + (-x_neg)
    kappa: float | str
    if denom > 0:
        kappa = (x_pos + x_neg) / denom
    else:
        kappa = "undefined (no cross mass)"

    return RowCrossGeometry(
        s_exact=float(max(s_exact_sq, 0.0) ** 0.5), s_diag=float(max(s_diag_sq, 0.0) ** 0.5),
        f_cross=f_cross, x_pos=x_pos, x_neg=x_neg,
        x_pos_norm=x_pos_norm, x_neg_norm=x_neg_norm, gross_cross_norm=gross,
        kappa=kappa, d_ffn=int(d.numel()),
    )


# --------------------------------------------------------------------------------------
# Layer-wide f_cross reference distribution (Task 2F) -- reuses E5's exact_uk_all_rows
# --------------------------------------------------------------------------------------

@dataclass
class LayerFCrossDistribution:
    f_cross: np.ndarray       # [d_model]
    s_exact: np.ndarray       # [d_model]
    s_diag: np.ndarray        # [d_model]
    median: float
    iqr: float
    d_model: int


@torch.no_grad()
def layer_fcross_distribution(W_gate: torch.Tensor, W_up: torch.Tensor, W_down: torch.Tensor,
                               device: str = "cuda") -> LayerFCrossDistribution:
    """f_cross for every output row of one layer. Cost class identical to E5's own
    exact_uk_all_rows call (already validated fast enough for all six E5 models, largest
    d_ffn=14336, in single-digit seconds on this node's A100)."""
    s_exact = exact_uk_all_rows(W_gate, W_up, W_down, device=device).numpy()
    s_diag = uk_frobenius(W_gate, W_up, W_down).numpy()
    f_cross = (s_exact ** 2 - s_diag ** 2) / (s_exact ** 2)
    med = float(np.median(f_cross))
    q75, q25 = np.percentile(f_cross, [75, 25])
    return LayerFCrossDistribution(f_cross=f_cross, s_exact=s_exact, s_diag=s_diag,
                                    median=med, iqr=float(q75 - q25), d_model=len(f_cross))


def candidate_layer_position(candidate_f_cross: float, dist: LayerFCrossDistribution) -> dict:
    """Percentile + robust z-like position of one candidate row's f_cross within its own
    layer's full distribution. Same robust-MAD convention as E5's AE_* statistics, for
    methodological consistency across the two experiments (not a new invented metric)."""
    pct = 100.0 * float((dist.f_cross <= candidate_f_cross).mean())
    mad = float(np.median(np.abs(dist.f_cross - dist.median)))
    z = (candidate_f_cross - dist.median) / (1.4826 * mad) if mad > 0 else "undefined (MAD=0)"
    return dict(percentile=pct, robust_z=z, layer_median=dist.median, layer_iqr=dist.iqr,
                d_model=dist.d_model)
