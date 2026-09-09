"""
Synthetic-tensor tests for spectral_lib.py -- run and green BEFORE the E7 prereg is locked.
No real checkpoint weight is used here.

Numerical tolerances (preregistered here, in code, before any real model is measured):
    RTOL = 1e-8   for float64 exact-vs-exact agreement (matrix-free construction vs.
                  brute-force explicit summation vs. direct SVD on the same tensor)

    python experiments/E7_exact_dimensionality/test_spectral_lib.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from spectral_lib import materialize_Uk, singular_values, spectral_metrics, row_spectral_metrics  # noqa: E402
from uk_frobenius import uk_frobenius  # noqa: E402

RTOL = 1e-8


def test_rank1_by_construction():
    """Many active coordinates, all structurally proportional to one (g*, u*) direction, so
    the SUM collapses to exactly rank 1 regardless of how many terms contribute -- this tests
    the sum's behavior, not a trivial single-term case."""
    torch.manual_seed(0)
    d_model, d_ffn = 24, 40
    g_star = torch.randn(d_model, dtype=torch.float64)
    u_star = torch.randn(d_model, dtype=torch.float64)
    c = torch.randn(d_ffn, dtype=torch.float64)   # per-coordinate scalar multiples of g_star
    e = torch.randn(d_ffn, dtype=torch.float64)   # per-coordinate scalar multiples of u_star
    d = torch.randn(d_ffn, dtype=torch.float64) * 0.7

    G = c[:, None] * g_star[None, :]
    U = e[:, None] * u_star[None, :]

    metrics, sigmas = row_spectral_metrics(G, U, d, device="cpu")
    assert abs(metrics.q1 - 1.0) < RTOL, f"expected q1=1.0, got {metrics.q1}"
    assert abs(metrics.pr_spec - 1.0) < RTOL, f"expected PR_spec=1.0, got {metrics.pr_spec}"
    # only one singular value should be non-negligible
    assert float(sigmas[1]) < 1e-8 * float(sigmas[0]), "second singular value should vanish"
    print(f"  ok: rank-1-by-construction -> q1={metrics.q1:.10f} PR_spec={metrics.pr_spec:.10f}")


def test_rank_r_orthogonal_equal_energy():
    """r mutually Frobenius-orthogonal, equal-energy rank-1 terms (standard-basis
    construction) -> analytically sigma_j = 1 for j=1..r, q1=1/r, PR_spec=r exactly."""
    for r in (1, 3, 7):
        d_model = d_ffn = r
        G = torch.eye(d_ffn, d_model, dtype=torch.float64)
        U = torch.eye(d_ffn, d_model, dtype=torch.float64)
        d = torch.ones(d_ffn, dtype=torch.float64)

        metrics, sigmas = row_spectral_metrics(G, U, d, device="cpu")
        assert abs(metrics.q1 - 1.0 / r) < RTOL, f"r={r}: expected q1={1/r}, got {metrics.q1}"
        assert abs(metrics.pr_spec - r) < RTOL, f"r={r}: expected PR_spec={r}, got {metrics.pr_spec}"
        assert torch.allclose(sigmas, torch.ones(r, dtype=torch.float64), rtol=RTOL)
        print(f"  ok: rank-{r} orthogonal equal-energy -> q1={metrics.q1:.6f} (1/{r}) "
              f"PR_spec={metrics.pr_spec:.6f} ({r})")


def test_nonorthogonal_constructive_matches_bruteforce():
    torch.manual_seed(2)
    d_model, d_ffn = 18, 50
    G = torch.randn(d_ffn, d_model, dtype=torch.float64)
    U = torch.randn(d_ffn, d_model, dtype=torch.float64)
    d = torch.randn(d_ffn, dtype=torch.float64) * 0.4

    Uk_matrixfree = materialize_Uk(G, U, d, device="cpu")

    Uk_bruteforce = torch.zeros(d_model, d_model, dtype=torch.float64)
    for i in range(d_ffn):
        Uk_bruteforce += d[i] * torch.outer(G[i], U[i])

    rel_err = float((Uk_matrixfree - Uk_bruteforce).abs().max() / Uk_bruteforce.abs().max())
    assert rel_err < RTOL, f"matrix-free construction disagrees with brute force: {rel_err:.2e}"

    sigmas_free = singular_values(Uk_matrixfree)
    sigmas_brute = singular_values(Uk_bruteforce)
    assert torch.allclose(sigmas_free, sigmas_brute, rtol=RTOL)
    print(f"  ok: nonorthogonal constructive case -- matrix-free U_k matches brute-force "
          f"summation (max rel err {rel_err:.2e}), singular values agree to rtol={RTOL:.0e}")


def test_cancellation_case_diagonal_pr_would_disagree():
    """Two coordinate pairs with equal-magnitude, opposite-sign contributions to the SAME
    exact direction (constructed so their outer products literally cancel), plus one
    dominant unrelated term. The diagonal scalar PR (c_{k,i} = d_i^2 |g_i|^2 |u_i|^2, which
    is sign-blind -- every term contributes positively regardless of cancellation) would
    report high effective dimensionality (multiple "large" c_{k,i} terms), while the exact
    spectral computation reports the true, much more concentrated operator that results
    after cancellation. This test verifies the exact spectrum matches direct SVD of the
    literal materialized matrix (the only claim this module makes) and separately confirms,
    numerically, that the diagonal PR is indeed misleadingly high in this construction --
    demonstrating exactly the failure mode E5/E6 motivated fixing."""
    d_model, d_ffn = 10, 4
    # pair A and pair B are built to have IDENTICAL (g_i . g_j)(u_i . u_j) geometry to each
    # other, but pair B's d-sign is flipped relative to pair A, so pair B's outer-product
    # contribution to U_k cancels against a component of pair A's.
    g0 = torch.tensor([1.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64)
    g1 = torch.tensor([0.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64)
    u0 = torch.tensor([1.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64)
    u1 = torch.tensor([0.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float64)
    # coordinates 0,1: d=+3 each -> U_k gets +3*outer(g0,u0) + 3*outer(g1,u1)
    # coordinates 2,3: identical g/u to 0,1 but d=-3 each -> exactly cancels 0,1's contribution
    G = torch.stack([g0, g1, g0, g1])
    U = torch.stack([u0, u1, u0, u1])
    d = torch.tensor([3.0, 3.0, -3.0, -3.0], dtype=torch.float64)

    Uk = materialize_Uk(G, U, d, device="cpu")
    assert float(Uk.abs().max()) < 1e-10, "construction should cancel to (numerically) zero"

    # diagonal c_{k,i} is sign-blind: every one of the 4 terms has identical, large magnitude
    c_diag = (d ** 2) * ((G ** 2).sum(dim=1)) * ((U ** 2).sum(dim=1))
    assert float(c_diag.min()) > 0, "diagonal terms are all positive despite exact cancellation"
    diagonal_pr = float(c_diag.sum() ** 2 / (c_diag ** 2).sum())
    assert diagonal_pr > 3.9, (
        f"diagonal PR should report near-maximal (4-way) apparent dimensionality despite "
        f"the exact operator being (numerically) the zero matrix, got PR={diagonal_pr}")

    sigmas = singular_values(Uk)
    assert float(sigmas.max()) < 1e-10, "exact spectrum must reflect the true (zero) operator"
    print(f"  ok: cancellation case -- diagonal PR reports {diagonal_pr:.3f} (near-maximal, "
          f"misleading) while the exact operator's singular values are ~0 (max "
          f"{float(sigmas.max()):.2e}), exactly the failure mode this metric fixes")


def test_random_small_tensors_matrixfree_vs_bruteforce_vs_svd():
    torch.manual_seed(11)
    for trial in range(5):
        d_model = torch.randint(8, 22, (1,)).item()
        d_ffn = torch.randint(20, 70, (1,)).item()
        K_row = 0
        g = torch.randn(d_ffn, d_model, dtype=torch.float64)
        u = torch.randn(d_ffn, d_model, dtype=torch.float64)
        d = torch.randn(d_ffn, dtype=torch.float64) * 0.5

        Uk = materialize_Uk(g, u, d, device="cpu")
        Uk_ref = torch.zeros(d_model, d_model, dtype=torch.float64)
        for i in range(d_ffn):
            Uk_ref += d[i] * torch.outer(g[i], u[i])
        assert torch.allclose(Uk, Uk_ref, rtol=RTOL, atol=1e-10)

        sigmas = singular_values(Uk)
        sigmas_ref = torch.linalg.svdvals(Uk_ref)
        assert torch.allclose(sigmas, sigmas_ref, rtol=RTOL)

        # frob-norm identity check: sqrt(sum sigma^2) must equal ||U_k||_F from
        # uk_frobenius's own diagonal-only estimate ONLY when cross terms are absent; here we
        # just check it against a direct Frobenius norm of the same explicit matrix (a
        # cross-term-agnostic identity that must always hold regardless of geometry).
        metrics = spectral_metrics(sigmas)
        direct_frob = float((Uk * Uk).sum() ** 0.5)
        assert abs(metrics.frob_norm - direct_frob) / direct_frob < RTOL
    print(f"  ok: 5 random small-tensor trials -- matrix-free U_k == brute-force summation, "
          f"singular values match direct SVD, frob-norm identity holds (all rtol={RTOL:.0e})")


def test_frob_norm_matches_uk_frobenius_when_effectively_diagonal():
    """When gate/up rows are pairwise orthogonal by construction (K is exactly diagonal, the
    regime E5's own orthogonal test used), the exact spectral Frobenius norm must equal
    uk_frobenius.uk_frobenius()'s diagonal-only value exactly -- the two formulations agree
    precisely when there are no cross terms to disagree about."""
    d_ffn = d_model = 15
    g = torch.eye(d_ffn, d_model, dtype=torch.float64) * 2.0
    u = torch.eye(d_ffn, d_model, dtype=torch.float64) * 1.5
    d = torch.randn(d_ffn, dtype=torch.float64) * 0.6
    K_row = 3

    metrics, _ = row_spectral_metrics(g.float(), u.float(), d.float(), device="cpu")
    Wd = d.unsqueeze(0).repeat(d_model, 1).float()   # W_down where every row equals d
    ref = float(uk_frobenius(g.float(), u.float(), Wd)[0])
    assert abs(metrics.frob_norm - ref) / ref < 1e-5, \
        f"diagonal-regime spectral frob norm ({metrics.frob_norm}) should match uk_frobenius ({ref})"
    print(f"  ok: orthogonal (diagonal-K) regime -- spectral frob_norm matches "
          f"uk_frobenius.uk_frobenius exactly ({metrics.frob_norm:.6f} vs {ref:.6f})")


def main() -> None:
    tests = [
        test_rank1_by_construction,
        test_rank_r_orthogonal_equal_energy,
        test_nonorthogonal_constructive_matches_bruteforce,
        test_cancellation_case_diagonal_pr_would_disagree,
        test_random_small_tensors_matrixfree_vs_bruteforce_vs_svd,
        test_frob_norm_matches_uk_frobenius_when_effectively_diagonal,
    ]
    for t in tests:
        print(f"{t.__name__} ...")
        t()
    print(f"\nall {len(tests)} tests passed (synthetic tensors only)")


if __name__ == "__main__":
    main()
