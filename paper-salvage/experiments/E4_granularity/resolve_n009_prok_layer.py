"""
N-009 / C-001 — provenance resolution for the GENERator PROK structural rank.

Provenance chain traced backward from the claim, NOT inferred from SW_TARGETS:

  CLAIMS_LEDGER C-001  ("rank 1/3,072 at the step-up layer, 12x / 18x ratio")
    -> results/sw_mechanistic_generator_prokaryote.json
         sw_rows_by_layer = {"2": [1927]}
         frob_norm_uk_by_layer["2"][1927] = 2648.4773, rank 1/3072, max/median 17.90
    -> scripts/analysis/run_sw_mechanistic.py :: experiment2_frob_uk
    -> configs/generator_prokaryote.yaml :: GenerTeam/GENERator-v2-prokaryote-3b-base

So the claim's layer is **2** — the same layer E4 used. The layer-mismatch hypothesis in
N-009 is wrong, and the discrepancy lies elsewhere.

This script recomputes BOTH quantities on the SAME weights at that exact layer:

  A. exact      ||U_k||_F = || W_gate^T diag(W_down[k,:]) W_up ||_F
                (what run_sw_mechanistic.py computed, cross terms included)
  B. decomposed sqrt( sum_i c_{k,i} ),  c_{k,i} = W_down[k,i]^2 ||W_gate[i,:]||^2 ||W_up[i,:]||^2
                (what src/uk_frobenius.py computes -- the rank-1 outer-product bound)

A and B coincide only when the rank-1 terms are mutually Frobenius-orthogonal.

Reports both, plus the c_{k,i} granularity decomposition with cumulative shares.
Measurement only -- no interpretation of why the two differ.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SALVAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SALVAGE / "src"))

from uk_frobenius import row_granularity, uk_contributions, uk_frobenius  # noqa: E402
from scripts.interpretability.run_sw_broadcast_impulse import _get_blocks, _load_model  # noqa: E402

LAYER = 2          # from provenance, not from SW_TARGETS
ROW = 1927
STORED_VALUE = 2648.4773       # frob_norm_uk_by_layer["2"][1927]
STORED_MEDIAN = 147.97756958007812
STORED_MAX_OVER_MED = 17.90


@torch.no_grad()
def exact_uk_frobenius(Wg: torch.Tensor, Wu: torch.Tensor, Wd: torch.Tensor,
                       device: str = "cuda") -> torch.Tensor:
    """Exact ||U_k||_F, the same quantity run_sw_mechanistic.py::experiment2_frob_uk builds.

    Closed form, so the (d_model x d_model) matrix per k never has to be materialised:

        U_k = W_gate^T diag(d_k) W_up,   d_k = W_down[k,:]
        ||U_k||_F^2 = tr(U^T diag(d_k) G G^T diag(d_k) U)
                    = sum_{i,j} d_i d_j (g_i . g_j)(u_i . u_j)
                    = d_k^T ( G G^T  (elementwise*)  U U^T ) d_k

    So with K = (G G^T) * (U U^T), all k at once is rowwise (D @ K * D).sum(1).

    Note the exact relationship to the decomposition in src/uk_frobenius.py: the diagonal
    K_ii = ||g_i||^2 ||u_i||^2, so c_{k,i} = D[k,i]^2 * K_ii is the DIAGONAL of this form.
    The decomposition drops every i != j cross term.
    """
    G = Wg.float().to(device)
    U = Wu.float().to(device)
    D = Wd.float().to(device)
    K = (G @ G.T) * (U @ U.T)                 # (d_ffn, d_ffn)
    out = ((D @ K) * D).sum(dim=1).clamp_min(0).sqrt()
    del K, G, U, D
    torch.cuda.empty_cache()
    return out.double().cpu()


def rank_of(vec: torch.Tensor, row: int) -> int:
    return int((vec > vec[row]).sum()) + 1


def main() -> int:
    print(f"GENERator PROK — provenance-resolved layer {LAYER}, row {ROW}")
    model, _ = _load_model("generator_prokaryote", "cuda")
    mlp = _get_blocks(model, "llama")[LAYER].mlp
    Wg = mlp.gate_proj.weight.detach().cpu()
    Wu = mlp.up_proj.weight.detach().cpu()
    Wd = mlp.down_proj.weight.detach().cpu()
    print(f"  shapes: gate {tuple(Wg.shape)} up {tuple(Wu.shape)} down {tuple(Wd.shape)}")
    d_ffn, d_model = Wg.shape
    assert Wd.shape == (d_model, d_ffn), "shape guard failed"
    print(f"  shape check OK (d_model={d_model}, d_ffn={d_ffn})")
    del model
    torch.cuda.empty_cache()

    res: dict = {"layer": LAYER, "row": ROW,
                 "stored": {"value": STORED_VALUE, "median": STORED_MEDIAN,
                            "rank": 1, "max_over_median": STORED_MAX_OVER_MED,
                            "source": "results/sw_mechanistic_generator_prokaryote.json"}}

    # ---- A. exact ----------------------------------------------------------
    ex = exact_uk_frobenius(Wg, Wu, Wd, device='cuda')
    ex_rank = rank_of(ex, ROW)
    ex_med = float(ex.median())
    res["exact"] = {"value": float(ex[ROW]), "rank": ex_rank, "d_model": d_model,
                    "percentile": 100.0 * (1.0 - (ex_rank - 1) / d_model),
                    "median": ex_med, "max": float(ex.max()),
                    "max_over_median": float(ex.max() / ex_med)}
    print(f"\n  A. EXACT ||U_k||_F   (run_sw_mechanistic formula)")
    print(f"     value {float(ex[ROW]):.4f}   rank {ex_rank}/{d_model}   "
          f"median {ex_med:.4f}   max/med {float(ex.max()/ex_med):.2f}")
    print(f"     stored was value {STORED_VALUE:.4f}, rank 1, max/med {STORED_MAX_OVER_MED:.2f}"
          f"   -> reproduces: {abs(float(ex[ROW]) - STORED_VALUE) / STORED_VALUE < 1e-3}")

    # ---- B. decomposition --------------------------------------------------
    dec = uk_frobenius(Wg, Wu, Wd)
    dec_rank = rank_of(dec, ROW)
    dec_med = float(dec.median())
    res["decomposed"] = {"value": float(dec[ROW]), "rank": dec_rank,
                         "percentile": 100.0 * (1.0 - (dec_rank - 1) / d_model),
                         "median": dec_med, "max": float(dec.max()),
                         "max_over_median": float(dec.max() / dec_med)}
    print(f"\n  B. DECOMPOSED sqrt(sum_i c_ki)   (src/uk_frobenius.py)")
    print(f"     value {float(dec[ROW]):.4f}   rank {dec_rank}/{d_model}   "
          f"median {dec_med:.4f}   max/med {float(dec.max()/dec_med):.2f}")

    # ---- c_{k,i} granularity at this layer ---------------------------------
    contrib = uk_contributions(Wg, Wu, Wd)
    gran = row_granularity(contrib, ROW)
    c = contrib[ROW]
    order = torch.argsort(c, descending=True)
    total = float(c.sum())
    cum = {}
    for n in (1, 5, 10, 50, 100):
        cum[f"top{n}"] = float(c[order[:n]].sum() / total)
    res["granularity"] = {"top1_index": gran.top1_index, "top1_share": gran.top1_share,
                          "participation_ratio": gran.participation_ratio,
                          "d_ffn": gran.d_ffn, "regime": gran.regime,
                          "cumulative_share": cum}
    print(f"\n  c_ki decomposition at layer {LAYER}, row {ROW}")
    print(f"     top1_index {gran.top1_index}   top1_share {gran.top1_share:.6f}   "
          f"PR {gran.participation_ratio:.2f} of {gran.d_ffn}")
    print("     cumulative: " + "  ".join(f"top{n} {cum[f'top{n}']:.4f}"
                                          for n in (1, 5, 10, 50, 100)))

    out = ROOT / "results" / "n009_prok_layer_resolution.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\n  saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
