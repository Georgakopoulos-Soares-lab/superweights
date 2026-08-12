"""Self-test: run before trusting any audit output.

    python src/test_uk_frobenius.py

Checks the vectorised form against the brute-force definition, that a planted scalar
super-weight is recovered at rank 1 in both row and scalar, and that the granularity
metric separates a scalar-dominated row from a distributed one.
"""

import torch

from uk_frobenius import (
    uk_contributions,
    uk_frobenius,
    row_granularity,
    scalar_rank,
    layer_report,
)


def main() -> None:
    torch.manual_seed(0)
    d_model, d_ffn = 64, 256
    K, I = 7, 13

    g = torch.randn(d_ffn, d_model)
    u = torch.randn(d_ffn, d_model)
    d = torch.randn(d_model, d_ffn) * 0.01
    d[K, I] = 50.0  # planted scalar super-weight

    c = uk_contributions(g, u, d)
    uk = uk_frobenius(g, u, d)

    # 1. vectorised == definition
    ref = sum(
        float(d[K, i]) ** 2 * float(g[i] @ g[i]) * float(u[i] @ u[i])
        for i in range(d_ffn)
    ) ** 0.5
    assert abs(float(uk[K]) - ref) / ref < 1e-9, "vectorised form disagrees with definition"

    # 2. row recovery
    assert int(uk.argmax()) == K, "planted row not ranked first"
    rep, _ = layer_report(g, u, d, layer=0, query_rows=[K])
    assert rep.query_ranks[K]["rank"] == 1

    # 3. scalar recovery
    gr = row_granularity(c, K)
    assert gr.top1_index == I, "planted scalar not the dominant contributor"
    assert scalar_rank(c, K, I) == 1
    assert gr.top1_share > 0.99 and gr.regime == "scalar-dominated"

    # 4. distributed control
    d2 = torch.randn(d_model, d_ffn) * 0.01
    gr2 = row_granularity(uk_contributions(g, u, d2), 0)
    assert gr2.regime == "distributed"
    assert gr2.participation_ratio > 0.1 * d_ffn

    print("all checks passed")
    print(f"  planted row      rank 1, max/median {rep.max_over_median:.1f}")
    print(f"  planted scalar   rank 1, top1_share {gr.top1_share:.4f}, PR {gr.participation_ratio:.2f}")
    print(f"  control row      PR {gr2.participation_ratio:.1f} of {d_ffn}, regime {gr2.regime}")


if __name__ == "__main__":
    main()
