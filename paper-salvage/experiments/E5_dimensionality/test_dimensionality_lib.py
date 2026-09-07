"""
Synthetic-tensor tests for dimensionality_lib.py -- run and green BEFORE the Gate-0 prereg is
locked (per PREREG_dimensionality_gate0.md: no real target-row weight may be passed through
this code before the lock; synthetic/planted tensors are explicitly allowed pre-lock).

    python experiments/E5_dimensionality/test_dimensionality_lib.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dimensionality_lib import (  # noqa: E402
    factor_D, factor_A, factor_C, vector_profile, rank_of,
    exact_uk_all_rows, cross_term_fraction,
    permutation_null, alignment_excess_max, alignment_excess_top1share, alignment_excess_pr,
    exact_ranksum_pvalue, complete_separation,
)
from uk_frobenius import uk_contributions, uk_frobenius, row_granularity, scalar_rank  # noqa: E402


def test_factors_match_uk_frobenius():
    torch.manual_seed(0)
    d_model, d_ffn = 32, 128
    K, I = 5, 11
    g = torch.randn(d_ffn, d_model)
    u = torch.randn(d_ffn, d_model)
    d = torch.randn(d_model, d_ffn) * 0.01
    d[K, I] = 40.0

    D = factor_D(d[K])
    A = factor_A(g, u)
    C = factor_C(D, A)

    ref_c = uk_contributions(g, u, d)[K]  # existing predictor's c_{k,i} for row K
    assert torch.allclose(C, ref_c.double(), rtol=1e-9), "C_i must equal uk_contributions row K exactly"

    ref_uk = uk_frobenius(g, u, d)[K]
    assert abs(float(C.sum().sqrt()) - float(ref_uk)) / float(ref_uk) < 1e-9, \
        "sqrt(sum C_i) must equal uk_frobenius()[K]"
    print("  ok: factor_D/A/C reproduce uk_frobenius.uk_contributions exactly")


def test_vector_profile_matches_row_granularity():
    torch.manual_seed(1)
    d_model, d_ffn = 20, 200
    K, I = 3, 17
    g = torch.randn(d_ffn, d_model)
    u = torch.randn(d_ffn, d_model)
    d = torch.randn(d_model, d_ffn) * 0.01
    d[K, I] = 60.0

    contrib = uk_contributions(g, u, d)
    gran = row_granularity(contrib, K)

    D = factor_D(d[K])
    A = factor_A(g, u)
    C = factor_C(D, A)
    prof = vector_profile(C)

    assert prof.top1_index == gran.top1_index
    assert abs(prof.top1_share - gran.top1_share) < 1e-9
    assert abs(prof.top5_share - gran.top5_share) < 1e-9
    assert abs(prof.participation_ratio - gran.participation_ratio) < 1e-9
    assert rank_of(C, I) == scalar_rank(contrib, K, I) == 1
    print("  ok: vector_profile(C) matches uk_frobenius.row_granularity exactly")

    # distributed control
    d2 = torch.randn(d_model, d_ffn) * 0.01
    D2 = factor_D(d2[0])
    C2 = factor_C(D2, A)
    prof2 = vector_profile(C2)
    assert prof2.participation_ratio > 0.1 * d_ffn, "distributed control should have high PR"
    print(f"  ok: distributed control PR={prof2.participation_ratio:.1f} of {d_ffn}")


def test_exact_matches_bruteforce_and_diagonal_when_orthogonal():
    torch.manual_seed(2)
    d_model, d_ffn = 12, 40
    K = 4
    g = torch.randn(d_ffn, d_model)
    u = torch.randn(d_ffn, d_model)
    d = torch.randn(d_model, d_ffn) * 0.3

    # brute-force ||U_k||_F^2 for row K: U_k = sum_i d[k,i] * outer(g_i, u_i)
    Uk = torch.zeros(d_model, d_model, dtype=torch.float64)
    for i in range(d_ffn):
        Uk += d[K, i].double() * torch.outer(g[i].double(), u[i].double())
    bruteforce_sq = float((Uk * Uk).sum())

    exact_all = exact_uk_all_rows(g, u, d, device="cpu")
    exact_sq = float(exact_all[K] ** 2)
    assert abs(exact_sq - bruteforce_sq) / bruteforce_sq < 1e-8, \
        "exact_uk_all_rows must match the brute-force materialized U_k"
    print(f"  ok: exact_uk_all_rows matches brute-force U_k (rel err "
          f"{abs(exact_sq - bruteforce_sq) / bruteforce_sq:.2e})")

    D = factor_D(d[K])
    A = factor_A(g, u)
    diag_sq = float(factor_C(D, A).sum())
    f_cross = cross_term_fraction(float(exact_all[K]), math.sqrt(diag_sq))
    # random g/u are generically non-orthogonal at this size -- cross terms should be
    # present and f_cross should NOT be pathologically >1 or <0-with-huge-magnitude.
    assert -1.0 < f_cross < 1.0, f"f_cross out of a sane range: {f_cross}"
    print(f"  ok: f_cross = {f_cross:.4f} on generic random weights (sanity range)")


def test_exact_equals_diagonal_when_rows_orthogonal_by_construction():
    """If gate/up rows are constructed pairwise-orthogonal (g_i . g_j = 0 for i != j, same
    for u), K is exactly diagonal and f_cross must be ~0 to float64 precision."""
    d_model, d_ffn = 16, 16  # orthogonal basis needs d_ffn <= d_model
    K = 2
    g = torch.eye(d_ffn, d_model, dtype=torch.float64) * 3.0
    u = torch.eye(d_ffn, d_model, dtype=torch.float64) * 2.0
    d = torch.randn(d_model, d_ffn, dtype=torch.float64) * 0.5

    exact_all = exact_uk_all_rows(g, u, d, device="cpu")
    D = factor_D(d[K])
    A = factor_A(g, u)
    diag_sq = float(factor_C(D, A).sum())
    f_cross = cross_term_fraction(float(exact_all[K]), math.sqrt(diag_sq))
    assert abs(f_cross) < 1e-9, f"orthogonal construction should give f_cross ~ 0, got {f_cross}"
    print(f"  ok: orthogonal-by-construction gives f_cross={f_cross:.2e} (~0, as required)")


def test_permutation_null_detects_planted_alignment():
    d_ffn = 300
    rng_seed = np.random.SeedSequence(999)
    rng = np.random.default_rng(rng_seed)
    D = torch.tensor(rng.exponential(1.0, size=d_ffn))
    A = torch.tensor(rng.exponential(1.0, size=d_ffn))
    # plant strong alignment at index 0: both D and A get a large value there
    D[0] = 50.0
    A[0] = 50.0
    C = factor_C(D, A)
    prof = vector_profile(C)

    null = permutation_null(D, A, n_perm=2000, seed_seq=np.random.SeedSequence(42))
    ae = alignment_excess_top1share(prof.top1_share, null)
    assert isinstance(ae.ae, float) and ae.ae > 3.0, \
        f"planted alignment should give a large positive AE, got {ae.ae}"
    assert ae.percentile > 99.0, f"planted alignment should be far in the null's tail, got {ae.percentile}"
    print(f"  ok: planted alignment detected -- AE_top1share={ae.ae:.2f}, percentile={ae.percentile:.2f}")


def test_permutation_null_null_case_lands_near_median():
    """If D and A are independent random vectors with NO planted alignment, the observed
    C should look like a typical permutation draw -- percentile near 50, not in a tail."""
    d_ffn = 500
    rng = np.random.default_rng(np.random.SeedSequence(7))
    D = torch.tensor(rng.exponential(1.0, size=d_ffn))
    A = torch.tensor(rng.exponential(1.0, size=d_ffn))
    C = factor_C(D, A)
    prof = vector_profile(C)

    null = permutation_null(D, A, n_perm=3000, seed_seq=np.random.SeedSequence(123))
    ae = alignment_excess_top1share(prof.top1_share, null)
    assert isinstance(ae.ae, float)
    assert 5.0 < ae.percentile < 95.0, \
        f"unaligned D/A should not land in a permutation-null tail, got percentile={ae.percentile}"
    print(f"  ok: no-alignment control lands mid-null -- percentile={ae.percentile:.2f}")


def test_permutation_null_rejects_int_seed():
    D = torch.rand(10) + 0.1
    A = torch.rand(10) + 0.1
    try:
        permutation_null(D, A, n_perm=10, seed_seq=42)
        raise AssertionError("should have raised TypeError for a bare int seed")
    except TypeError:
        print("  ok: bare int seed rejected (must pass a numpy SeedSequence)")


def test_pr_orientation():
    """alignment_excess_pr must be oriented so a LOWER observed PR than the null gives a
    POSITIVE AE (more concentrated than null)."""
    class FakeNull:
        participation_ratio = np.array([10.0, 12.0, 11.0, 9.0, 13.0, 11.5, 10.5])
    ae_low = alignment_excess_pr(2.0, FakeNull())   # much more concentrated than null
    ae_high = alignment_excess_pr(20.0, FakeNull())  # much less concentrated than null
    assert isinstance(ae_low.ae, float) and ae_low.ae > 0
    assert isinstance(ae_high.ae, float) and ae_high.ae < 0
    assert ae_low.ae > ae_high.ae
    print(f"  ok: PR orientation correct (low-PR AE={ae_low.ae:.2f} > high-PR AE={ae_high.ae:.2f})")


def test_exact_ranksum_complete_separation_is_p_005():
    nlp = [10.0, 11.0, 12.0]
    genomic = [1.0, 2.0, 3.0]
    result = exact_ranksum_pvalue(nlp, genomic)
    assert result["is_complete_separation"] is True
    assert result["total_arrangements"] == 20
    assert result["at_least_as_extreme_count"] == 1
    assert abs(result["p_value_one_sided"] - 0.05) < 1e-12
    print(f"  ok: complete separation gives exact p={result['p_value_one_sided']:.4f} (1/20)")


def test_exact_ranksum_overlap_is_not_complete_separation():
    nlp = [10.0, 2.0, 12.0]        # one NLP value (2.0) below all genomic values
    genomic = [1.0, 5.0, 3.0]
    result = exact_ranksum_pvalue(nlp, genomic)
    assert result["is_complete_separation"] is False
    assert result["p_value_one_sided"] > 0.05
    print(f"  ok: overlapping groups correctly NOT flagged as complete separation "
          f"(p={result['p_value_one_sided']:.4f})")


def test_exact_ranksum_handles_ties():
    nlp = [5.0, 5.0, 9.0]
    genomic = [5.0, 1.0, 2.0]
    result = exact_ranksum_pvalue(nlp, genomic)
    assert 0.0 < result["p_value_one_sided"] <= 1.0
    print(f"  ok: tie handling does not crash (p={result['p_value_one_sided']:.4f})")


def test_complete_separation_predicate():
    assert complete_separation([5.0, 6.0, 7.0], [1.0, 2.0, 3.0]) is True
    assert complete_separation([5.0, 1.0, 7.0], [1.0, 2.0, 3.0]) is False
    assert complete_separation([5.0, 6.0, "undefined (MAD=0)"], [1.0, 2.0, 3.0]) is False
    print("  ok: complete_separation predicate correct, including the MAD=0 guard")


def main() -> None:
    tests = [
        test_factors_match_uk_frobenius,
        test_vector_profile_matches_row_granularity,
        test_exact_matches_bruteforce_and_diagonal_when_orthogonal,
        test_exact_equals_diagonal_when_rows_orthogonal_by_construction,
        test_permutation_null_detects_planted_alignment,
        test_permutation_null_null_case_lands_near_median,
        test_permutation_null_rejects_int_seed,
        test_pr_orientation,
        test_exact_ranksum_complete_separation_is_p_005,
        test_exact_ranksum_overlap_is_not_complete_separation,
        test_exact_ranksum_handles_ties,
        test_complete_separation_predicate,
    ]
    for t in tests:
        print(f"{t.__name__} ...")
        t()
    print(f"\nall {len(tests)} tests passed (synthetic tensors only -- no real checkpoint weight "
          f"was loaded, consistent with the pre-lock rule)")


if __name__ == "__main__":
    main()
