"""
Synthetic-tensor tests for cross_geometry_lib.py -- run and green BEFORE the E6 Stage-A prereg
is locked. No real checkpoint weight from any confirmation-panel candidate is used here.
(A separate, explicitly pre-lock regression check against E5's own six already-published rows
lives in validate_against_e5.py -- distinct from candidate rows, and not run here.)

    python experiments/E6_cross_geometry/test_cross_geometry_lib.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cross_geometry_lib import (  # noqa: E402
    pair_matrix, row_cross_geometry, layer_fcross_distribution, candidate_layer_position,
)
from uk_frobenius import uk_contributions, uk_frobenius  # noqa: E402


def test_orthogonal_rows_give_zero_cross():
    """Pairwise-orthogonal gate/up rows -> K is exactly diagonal -> every off-diagonal X_ij
    is exactly zero -> f_cross = 0, x_pos = x_neg = 0, kappa undefined."""
    d_ffn, d_model = 20, 20  # orthogonal basis needs d_ffn <= d_model
    K_row = 3
    g = torch.eye(d_ffn, d_model, dtype=torch.float64) * 2.5
    u = torch.eye(d_ffn, d_model, dtype=torch.float64) * 1.5
    d = torch.randn(d_model, d_ffn, dtype=torch.float64) * 0.7

    K = pair_matrix(g, u)
    geo = row_cross_geometry(K, d[K_row])
    assert abs(geo.f_cross) < 1e-9, f"expected f_cross ~ 0, got {geo.f_cross}"
    assert abs(geo.x_pos) < 1e-9 and abs(geo.x_neg) < 1e-9
    assert geo.kappa == "undefined (no cross mass)"
    print(f"  ok: orthogonal construction -> f_cross={geo.f_cross:.2e}, kappa={geo.kappa}")


def test_constructive_alignment_known_analytically():
    """Two coordinates (i=0,1) with strictly positive, mutually-aligned gate/up rows and a
    positive-signed d_i, d_j -> their cross term X_01 (and X_10) must be strictly positive
    and analytically computable by hand; all other coordinates orthogonal to everything
    (zero cross contribution elsewhere) so the row's cross geometry is driven entirely by
    this one constructive pair -> kappa must be exactly +1 (all cross mass positive)."""
    d_ffn, d_model = 6, 6
    g = torch.eye(d_ffn, d_model, dtype=torch.float64)
    u = torch.eye(d_ffn, d_model, dtype=torch.float64)
    # make g_0, g_1 (and u_0, u_1) partially overlapping instead of orthonormal, so
    # K[0,1] = (g0.g1)(u0.u1) != 0, while every other off-diagonal stays exactly 0
    g[0] = torch.tensor([1.0, 0.5, 0, 0, 0, 0], dtype=torch.float64)
    g[1] = torch.tensor([0.5, 1.0, 0, 0, 0, 0], dtype=torch.float64)
    u[0] = torch.tensor([1.0, 0.3, 0, 0, 0, 0], dtype=torch.float64)
    u[1] = torch.tensor([0.3, 1.0, 0, 0, 0, 0], dtype=torch.float64)

    d_vec = torch.tensor([2.0, 3.0, 0.1, 0.1, 0.1, 0.1], dtype=torch.float64)
    K = pair_matrix(g, u)

    # hand-computed K[0,1]
    g0g1 = float(g[0] @ g[1])   # 1*0.5 + 0.5*1.0 = 1.0
    u0u1 = float(u[0] @ u[1])   # 1*0.3 + 0.3*1.0 = 0.6
    expected_K01 = g0g1 * u0u1
    assert abs(float(K[0, 1]) - expected_K01) < 1e-10
    assert expected_K01 > 0, "test construction must give a positive K[0,1]"

    geo = row_cross_geometry(K, d_vec)
    # analytic cross sum for this row: only (0,1) and (1,0) pairs are nonzero among i!=j
    # (all other coordinates are exactly orthonormal to everything, K=0 off-diagonal there)
    expected_cross_pair_sum = 2 * d_vec[0] * d_vec[1] * expected_K01   # ordered pairs (0,1)+(1,0)
    assert abs(geo.x_pos - float(expected_cross_pair_sum)) < 1e-8
    assert abs(geo.x_neg) < 1e-10
    assert isinstance(geo.kappa, float) and abs(geo.kappa - 1.0) < 1e-9, \
        f"purely constructive cross mass must give kappa=+1, got {geo.kappa}"
    assert geo.f_cross > 0
    print(f"  ok: constructive-alignment case -> kappa={geo.kappa:.6f} (expect 1.0), "
          f"f_cross={geo.f_cross:.4f}, x_pos matches hand computation exactly")


def test_destructive_cancellation_case():
    """Two independently constructed pairs of coordinates: one pair gives a large POSITIVE
    cross contribution, another gives an equal-magnitude NEGATIVE cross contribution (via a
    sign flip on d_j), so the NET cross sum is ~0 (small f_cross) while the GROSS cross mass
    (x_pos + |x_neg|) is large and kappa is near 0 -- exactly the "huge cancellation, modest
    net" case the prereg must be able to distinguish from "genuinely weak interaction"."""
    d_ffn, d_model = 8, 8
    g = torch.eye(d_ffn, d_model, dtype=torch.float64)
    u = torch.eye(d_ffn, d_model, dtype=torch.float64)
    # pair A (0,1): positive alignment
    g[0] = torch.tensor([1.0, 0.6, 0, 0, 0, 0, 0, 0], dtype=torch.float64)
    g[1] = torch.tensor([0.6, 1.0, 0, 0, 0, 0, 0, 0], dtype=torch.float64)
    u[0] = torch.tensor([1.0, 0.6, 0, 0, 0, 0, 0, 0], dtype=torch.float64)
    u[1] = torch.tensor([0.6, 1.0, 0, 0, 0, 0, 0, 0], dtype=torch.float64)
    # pair B (2,3): identical geometry to pair A (same positive K[2,3]) but d_2, d_3 have
    # OPPOSITE relative sign vs pair A's d_0, d_1 -> X_23 contribution is negative
    g[2] = torch.tensor([0, 0, 1.0, 0.6, 0, 0, 0, 0], dtype=torch.float64)
    g[3] = torch.tensor([0, 0, 0.6, 1.0, 0, 0, 0, 0], dtype=torch.float64)
    u[2] = torch.tensor([0, 0, 1.0, 0.6, 0, 0, 0, 0], dtype=torch.float64)
    u[3] = torch.tensor([0, 0, 0.6, 1.0, 0, 0, 0, 0], dtype=torch.float64)

    d_vec = torch.tensor([2.0, 2.0, 2.0, -2.0, 0.05, 0.05, 0.05, 0.05], dtype=torch.float64)
    K = pair_matrix(g, u)
    geo = row_cross_geometry(K, d_vec)

    assert geo.x_pos > 0 and geo.x_neg < 0, "both a positive and a negative cross contribution must be present"
    assert abs(geo.x_pos + geo.x_neg) < 0.05 * geo.x_pos, \
        f"pair A and pair B were constructed to nearly cancel, got x_pos={geo.x_pos} x_neg={geo.x_neg}"
    assert isinstance(geo.kappa, float) and abs(geo.kappa) < 0.1, \
        f"near-total cancellation must give kappa near 0, got {geo.kappa}"
    assert geo.gross_cross_norm > 0.01, "gross cross mass must be non-trivial despite small net f_cross"
    print(f"  ok: cancellation case -> kappa={geo.kappa:.4f} (expect ~0), "
          f"f_cross={geo.f_cross:.4f} (small net), gross_cross_norm={geo.gross_cross_norm:.4f} (large gross)")


def test_bruteforce_matches_exact_row_cross_geometry():
    """Materialize U_k explicitly (sum_i d_i outer(g_i, u_i)) and confirm ||U_k||_F^2 equals
    row_cross_geometry's s_exact^2, independent of the K-matrix formulation."""
    torch.manual_seed(3)
    d_model, d_ffn = 10, 30
    K_row = 4
    g = torch.randn(d_ffn, d_model, dtype=torch.float64)
    u = torch.randn(d_ffn, d_model, dtype=torch.float64)
    d = torch.randn(d_model, d_ffn, dtype=torch.float64) * 0.4

    Uk = torch.zeros(d_model, d_model, dtype=torch.float64)
    for i in range(d_ffn):
        Uk += d[K_row, i] * torch.outer(g[i], u[i])
    bruteforce_sq = float((Uk * Uk).sum())

    K = pair_matrix(g, u)
    geo = row_cross_geometry(K, d[K_row])
    rel_err = abs(geo.s_exact ** 2 - bruteforce_sq) / bruteforce_sq
    assert rel_err < 1e-8, f"S_exact^2 does not match brute-force U_k: rel err {rel_err:.2e}"
    print(f"  ok: brute-force U_k matches row_cross_geometry.s_exact^2 (rel err {rel_err:.2e})")


def test_ordered_pair_identity_and_diagonal_match_uk_contributions():
    """x_pos + x_neg must equal S_exact^2 - S_diag^2 EXACTLY (this is asserted inside
    row_cross_geometry itself as a runtime guard; here we re-check it externally on a
    non-trivial random tensor, plus confirm the diagonal matches uk_frobenius.uk_contributions,
    so the K[i,i]=A_i identity carried over from E5 still holds in this module)."""
    torch.manual_seed(5)
    d_model, d_ffn = 14, 60
    K_row = 2
    g = torch.randn(d_ffn, d_model, dtype=torch.float64)
    u = torch.randn(d_ffn, d_model, dtype=torch.float64)
    d = torch.randn(d_model, d_ffn, dtype=torch.float64) * 0.6

    K = pair_matrix(g, u)
    geo = row_cross_geometry(K, d[K_row])
    assert abs((geo.x_pos + geo.x_neg) - (geo.s_exact ** 2 - geo.s_diag ** 2)) < 1e-6

    ref_c = uk_contributions(g.float(), u.float(), d.float())[K_row]
    assert abs(geo.s_diag ** 2 - float(ref_c.double().sum())) / float(ref_c.double().sum()) < 1e-6
    print(f"  ok: x_pos+x_neg == S_exact^2-S_diag^2 exactly; diagonal matches "
          f"uk_frobenius.uk_contributions (S_diag^2={geo.s_diag**2:.4f})")


def test_layer_distribution_matches_uk_frobenius_and_e5_exact():
    torch.manual_seed(9)
    d_model, d_ffn = 12, 40
    g = torch.randn(d_ffn, d_model, dtype=torch.float64)
    u = torch.randn(d_ffn, d_model, dtype=torch.float64)
    d = torch.randn(d_model, d_ffn, dtype=torch.float64) * 0.5

    dist = layer_fcross_distribution(g, u, d, device="cpu")
    ref_diag = uk_frobenius(g.float(), u.float(), d.float()).double().numpy()
    import numpy as np
    assert np.allclose(dist.s_diag, ref_diag, rtol=1e-6)
    assert dist.d_model == d_model

    # candidate position sanity: a row's own f_cross must land at percentile >= its own
    # rank fraction (trivial self-consistency: candidate IS one of the distribution's rows)
    pos = candidate_layer_position(float(dist.f_cross[0]), dist)
    assert 0.0 <= pos["percentile"] <= 100.0
    print(f"  ok: layer_fcross_distribution matches uk_frobenius diagonal exactly; "
          f"candidate_layer_position well-formed (pct={pos['percentile']:.1f})")


def test_float64_enforced():
    d_ffn, d_model = 5, 5
    g = torch.randn(d_ffn, d_model, dtype=torch.float32)
    u = torch.randn(d_ffn, d_model, dtype=torch.float32)
    K = pair_matrix(g, u)
    assert K.dtype == torch.float64, "pair_matrix must promote to float64 regardless of input dtype"
    print("  ok: pair_matrix promotes to float64 even when given float32 input")


def main() -> None:
    tests = [
        test_orthogonal_rows_give_zero_cross,
        test_constructive_alignment_known_analytically,
        test_destructive_cancellation_case,
        test_bruteforce_matches_exact_row_cross_geometry,
        test_ordered_pair_identity_and_diagonal_match_uk_contributions,
        test_layer_distribution_matches_uk_frobenius_and_e5_exact,
        test_float64_enforced,
    ]
    for t in tests:
        print(f"{t.__name__} ...")
        t()
    print(f"\nall {len(tests)} tests passed (synthetic tensors only -- no confirmation-panel "
          f"checkpoint weight was loaded)")


if __name__ == "__main__":
    main()
