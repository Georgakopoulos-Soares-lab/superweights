"""
Shared math for E5 Gate 0 -- weight-only mechanism of NLP-vs-genomic FFN scalarization.

Pure functions only: no I/O, no model loading, no hard-coded checkpoint/row values. Every
formula here is fixed by `docs/prereg/PREREG_dimensionality_gate0.md` and must not change
after that file is locked. If a formula needs to change, the prereg is re-locked (a new,
separately-disclosed entry), not silently edited alongside this module.

Conventions match `manuscript/src/uk_frobenius.py` throughout:
    W_gate, W_up : [d_ffn, d_model]
    W_down       : [d_model, d_ffn]
c_{k,i} here is called C_i (D_i * A_i) for a fixed target row k, consistent with
uk_frobenius.uk_contributions()[k, :].
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, asdict

import numpy as np
import torch


# --------------------------------------------------------------------------------------
# D / A / C factor vectors (Gate 0C)
# --------------------------------------------------------------------------------------

@torch.no_grad()
def factor_D(w_down_row: torch.Tensor, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """D_i = W_down[k, i]^2, for one target row. Input: [d_ffn]. Output: [d_ffn]."""
    w = w_down_row.to(dtype)
    return w * w


@torch.no_grad()
def factor_A(W_gate: torch.Tensor, W_up: torch.Tensor,
             dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """A_i = ||W_gate[i,:]||^2 * ||W_up[i,:]||^2. Independent of k. Output: [d_ffn]."""
    g = W_gate.to(dtype)
    u = W_up.to(dtype)
    return (g * g).sum(dim=1) * (u * u).sum(dim=1)


@torch.no_grad()
def factor_C(D: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """C_i = D_i * A_i. Identical to uk_frobenius.uk_contributions()[k, :]."""
    return D * A


# --------------------------------------------------------------------------------------
# Concentration profile of an arbitrary nonnegative vector (Gate 0C)
# --------------------------------------------------------------------------------------

@dataclass
class VectorProfile:
    top1_index: int
    top1_share: float
    top5_share: float
    top10_share: float
    participation_ratio: float
    d_ffn: int


@torch.no_grad()
def vector_profile(vec: torch.Tensor) -> VectorProfile:
    """Same formulas as uk_frobenius.row_granularity, applied to any nonnegative vector
    (D, A, or C), not just a row extracted from a full [d_model, d_ffn] contributions
    matrix. total must be > 0 -- the target rows in the primary panel are all previously
    confirmed nonzero high-gain rows, so this is not expected to fire."""
    total = vec.sum()
    if total <= 0:
        raise ValueError("vector has zero or negative total -- cannot profile")
    order = torch.argsort(vec, descending=True)
    top10 = min(10, vec.numel())
    return VectorProfile(
        top1_index=int(order[0]),
        top1_share=float(vec[order[0]] / total),
        top5_share=float(vec[order[:5]].sum() / total),
        top10_share=float(vec[order[:top10]].sum() / total),
        participation_ratio=float(total.pow(2) / (vec * vec).sum()),
        d_ffn=int(vec.numel()),
    )


@torch.no_grad()
def rank_of(vec: torch.Tensor, index: int) -> int:
    """1-based rank of `index` within `vec` (descending). Matches uk_frobenius.scalar_rank."""
    return int((vec > vec[index]).sum()) + 1


# --------------------------------------------------------------------------------------
# Exact (cross-term) quadratic form, all rows at once (Gate 0B)
# --------------------------------------------------------------------------------------

@torch.no_grad()
def exact_uk_all_rows(W_gate: torch.Tensor, W_up: torch.Tensor, W_down: torch.Tensor,
                       device: str = "cuda") -> torch.Tensor:
    """Exact ||U_k||_F for every output row k, via the elementwise-Gram identity:

        K = (W_gate W_gate^T) elementwise* (W_up W_up^T)      [d_ffn, d_ffn], K[i,i] = A_i
        ||U_k||_F^2 = W_down[k,:] @ K @ W_down[k,:]^T

    Same identity used for GENERator PROK in
    experiments/E4_granularity/resolve_n009_prok_layer.py; reimplemented here standalone so
    E5 does not depend on E4's internals. float64 throughout. Returns a [d_model] CPU tensor.
    """
    dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    G = W_gate.double().to(dev)
    U = W_up.double().to(dev)
    D = W_down.double().to(dev)
    K = (G @ G.T) * (U @ U.T)
    out = ((D @ K) * D).sum(dim=1).clamp_min(0).sqrt()
    del K, G, U, D
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out.cpu()


def cross_term_fraction(s_exact: float, s_diag: float) -> float:
    """f_cross = (S_exact^2 - S_diag^2) / S_exact^2, as defined in the Gate-0 prereg."""
    return (s_exact ** 2 - s_diag ** 2) / (s_exact ** 2)


# --------------------------------------------------------------------------------------
# Multiplicative-alignment permutation null (Gate 0D)
# --------------------------------------------------------------------------------------

@dataclass
class PermutationNull:
    max_c: np.ndarray
    top1_share: np.ndarray
    participation_ratio: np.ndarray
    n_perm: int


def permutation_null(D: torch.Tensor, A: torch.Tensor, n_perm: int, seed_seq) -> PermutationNull:
    """Fix D and A; for n_perm random permutations of A's index, recompute C_pi = D * A[pi]
    and record max, top1_share, PR. `seed_seq` is a numpy SeedSequence (see prereg §0D for
    the fixed panel-order spawning scheme) -- passing an int here is a deliberate error to
    catch accidental unseeded/reseeded calls."""
    if not isinstance(seed_seq, np.random.SeedSequence):
        raise TypeError("seed_seq must be a numpy SeedSequence (see prereg §0D)")
    rng = np.random.default_rng(seed_seq)
    d_np = D.numpy().astype(np.float64)
    a_np = A.numpy().astype(np.float64)
    n = d_np.shape[0]

    max_c = np.empty(n_perm, dtype=np.float64)
    top1 = np.empty(n_perm, dtype=np.float64)
    pr = np.empty(n_perm, dtype=np.float64)

    for p in range(n_perm):
        perm = rng.permutation(n)
        c_pi = d_np * a_np[perm]
        total = c_pi.sum()
        m = c_pi.max()
        max_c[p] = m
        top1[p] = m / total
        pr[p] = (total * total) / (c_pi * c_pi).sum()

    return PermutationNull(max_c=max_c, top1_share=top1, participation_ratio=pr, n_perm=n_perm)


def _mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


@dataclass
class AlignmentExcess:
    observed: float
    null_median: float
    null_mad: float
    ae: float | str          # float, or "undefined (MAD=0)"
    percentile: float        # oriented so higher = more concentrated / more surprising


def alignment_excess_max(observed_max: float, null: PermutationNull) -> AlignmentExcess:
    med = float(np.median(null.max_c))
    mad = _mad(null.max_c)
    ae = (observed_max - med) / (1.4826 * mad) if mad > 0 else "undefined (MAD=0)"
    pct = 100.0 * float((null.max_c <= observed_max).mean())
    return AlignmentExcess(observed_max, med, mad, ae, pct)


def alignment_excess_top1share(observed_share: float, null: PermutationNull) -> AlignmentExcess:
    med = float(np.median(null.top1_share))
    mad = _mad(null.top1_share)
    ae = (observed_share - med) / (1.4826 * mad) if mad > 0 else "undefined (MAD=0)"
    pct = 100.0 * float((null.top1_share <= observed_share).mean())
    return AlignmentExcess(observed_share, med, mad, ae, pct)


def alignment_excess_pr(observed_pr: float, null: PermutationNull) -> AlignmentExcess:
    """Oriented so positive AE / high percentile = MORE concentrated than null (low PR)."""
    med = float(np.median(null.participation_ratio))
    mad = _mad(null.participation_ratio)
    ae = (med - observed_pr) / (1.4826 * mad) if mad > 0 else "undefined (MAD=0)"
    pct = 100.0 * float((null.participation_ratio >= observed_pr).mean())
    return AlignmentExcess(observed_pr, med, mad, ae, pct)


# --------------------------------------------------------------------------------------
# Exact (enumerated, non-asymptotic) rank-sum test for tiny samples (Gate-0 decision rule)
# --------------------------------------------------------------------------------------

def exact_ranksum_pvalue(group_a: list[float], group_b: list[float]) -> dict:
    """One-sided exact Mann-Whitney/Wilcoxon rank-sum test: H1 = group_a values tend to
    exceed group_b values. Exact by full enumeration of all C(n, n_a) ways to split the
    combined rank vector under exchangeability -- appropriate (and honest) for the tiny
    n=3+3 sample this experiment uses; no normal/asymptotic approximation anywhere.

    Average-rank tie handling (standard). Returns the observed rank-sum for group_a, the
    number of (equally likely) arrangements at least as extreme, the total arrangement
    count, and the exact one-sided p-value.
    """
    combined = list(group_a) + list(group_b)
    n_a, n = len(group_a), len(combined)
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1)
    # average-rank tie correction
    uniq, inv, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]

    observed_rank_sum_a = float(ranks[:n_a].sum())

    total_arrangements = 0
    at_least_as_extreme = 0
    for idx_a in itertools.combinations(range(n), n_a):
        total_arrangements += 1
        rs = sum(ranks[i] for i in idx_a)
        if rs >= observed_rank_sum_a - 1e-9:
            at_least_as_extreme += 1

    p_value = at_least_as_extreme / total_arrangements
    max_possible_rank_sum = sum(sorted(ranks)[-n_a:])
    return {
        "observed_rank_sum_a": observed_rank_sum_a,
        "max_possible_rank_sum_a": float(max_possible_rank_sum),
        "is_complete_separation": bool(observed_rank_sum_a >= max_possible_rank_sum - 1e-9),
        "at_least_as_extreme_count": at_least_as_extreme,
        "total_arrangements": total_arrangements,
        "p_value_one_sided": p_value,
    }


def complete_separation(nlp_values: list[float], genomic_values: list[float]) -> bool:
    """Gate-0 primary decision predicate: every NLP value strictly exceeds every genomic
    value. Undefined ('undefined (MAD=0)') entries are treated as failing the comparison
    (they cannot be shown to exceed anything) and are reported separately by the caller."""
    for v in nlp_values:
        if not isinstance(v, (int, float)):
            return False
    for v in genomic_values:
        if not isinstance(v, (int, float)):
            return False
    return min(nlp_values) > max(genomic_values)
