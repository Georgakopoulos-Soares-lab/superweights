"""
Exact ||U_k||_F for EVERY output row of a layer, without any SVD.

Identity (exact, no cross terms dropped -- unlike uk_frobenius.py's diagonal c_{k,i} form):

    U_k = (d .* W_gate)^T @ W_up,   d = W_down[k, :]
    ||U_k||_F^2 = sum_{a,b} ( sum_i d_i G[i,a] U[i,b] )^2
                = sum_{i,j} d_i d_j (G G^T)[i,j] (U U^T)[i,j]
                = d^T [ (G G^T) .* (U U^T) ] d

So with K = (G G^T) .* (U U^T)  [d_ffn, d_ffn], computed ONCE per layer:

    ||U_k||_F^2 for all k = rowwise_sum( (W_down @ K) .* W_down )

This is mathematically identical to sqrt(sum_j sigma_j^2) from
spectral_lib.row_spectral_metrics (validated below against that function on real rows),
but costs two matmuls per layer instead of d_model full SVDs.

Used by E10's protocol correction to rank published high-gain candidate rows by exact
operator MAGNITUDE (and by magnitude relative to their own layer's median, the
uk_frobenius.layer_report convention), rather than by q1 -- which measures internal
spectral concentration, not gain.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

SALVAGE = Path(__file__).resolve().parents[2]
ROOT = SALVAGE  # repo root; harness paths are written relative to it
sys.path.insert(0, str(SALVAGE / "src"))
sys.path.insert(0, str(SALVAGE / "experiments" / "E7_exact_dimensionality"))

from spectral_lib import row_spectral_metrics  # noqa: E402  (validation only)

# Published high-gain candidate rows per model (Yu et al. 2024 Table 2 / this project's
# own frozen prospective detection). NOT a new row search -- these already existed pre-E10.
PANEL = {
    "olmo": dict(
        repo="allenai/OLMo-7B-0724-hf",
        rows={1: [269], 2: [269], 7: [269], 24: [269]},
        canonical=(1, 269),   # E1/E5/E7's designated primary
    ),
    "phi3": dict(
        repo="microsoft/Phi-3-mini-4k-instruct",
        rows={2: [525, 1693, 1113], 4: [525, 1113, 1693]},
        canonical=None,       # E7 treats all six as one model-level unit
        packed=True,          # Phi3MLP packs gate+up into gate_up_proj
    ),
}


def fetch_layer_tensors(repo: str, layer: int, packed: bool = False):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    if packed:
        names = {"gate_up": f"model.layers.{layer}.mlp.gate_up_proj.weight",
                 "down": f"model.layers.{layer}.mlp.down_proj.weight"}
    else:
        names = {"gate": f"model.layers.{layer}.mlp.gate_proj.weight",
                 "up": f"model.layers.{layer}.mlp.up_proj.weight",
                 "down": f"model.layers.{layer}.mlp.down_proj.weight"}
    idx_path = hf_hub_download(repo, "model.safetensors.index.json")
    weight_map = json.loads(Path(idx_path).read_text())["weight_map"]
    missing = [n for n in names.values() if n not in weight_map]
    if missing:
        raise RuntimeError(f"tensor names not in index: {missing}")
    loaded = {}
    for shard in sorted({weight_map[n] for n in names.values()}):
        data = load_file(hf_hub_download(repo, shard))
        for key, tname in names.items():
            if tname in data:
                loaded[key] = data[tname]
        del data
    if packed:
        gu = loaded["gate_up"]
        d_ffn = gu.shape[0] // 2
        # Phi3MLP: chunk(2, dim=-1) on the OUTPUT -> first half gate, second half up,
        # i.e. rows [0:d_ffn] = gate, [d_ffn:] = up (verified in E7 MODEL_PANEL.md).
        return {"gate": gu[:d_ffn], "up": gu[d_ffn:], "down": loaded["down"]}
    return loaded


@torch.no_grad()
def exact_uk_norms_all_rows(W_gate, W_up, W_down, device="cuda", dtype=torch.float64):
    """||U_k||_F for every k, exactly, via the Gram identity. Returns [d_model] on CPU."""
    G = W_gate.to(dtype).to(device)
    U = W_up.to(dtype).to(device)
    D = W_down.to(dtype).to(device)
    K = (G @ G.T) * (U @ U.T)           # [d_ffn, d_ffn], exact, includes all cross terms
    del G, U
    torch.cuda.empty_cache()
    sq = ((D @ K) * D).sum(dim=1)       # [d_model]
    del K
    torch.cuda.empty_cache()
    return sq.clamp_min(0).sqrt().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(PANEL))
    args = ap.parse_args()
    spec = PANEL[args.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    packed = spec.get("packed", False)

    out_layers = {}
    validated = False
    for layer, target_rows in spec["rows"].items():
        print(f"\nlayer {layer}: fetching tensors ...")
        t = fetch_layer_tensors(spec["repo"], layer, packed=packed)
        Wg, Wu, Wd = t["gate"], t["up"], t["down"]
        d_ffn, d_model = Wg.shape
        assert Wd.shape == (d_model, d_ffn), f"shape mismatch: {Wd.shape} vs {(d_model, d_ffn)}"
        print(f"  shapes gate{tuple(Wg.shape)} up{tuple(Wu.shape)} down{tuple(Wd.shape)}")

        norms = exact_uk_norms_all_rows(Wg, Wu, Wd, device=device)

        # Validate the Gram identity against spectral_lib's SVD-based frob_norm ONCE,
        # on the first target row -- they must agree to floating-point tolerance.
        if not validated:
            r0 = target_rows[0]
            m, _ = row_spectral_metrics(Wg, Wu, Wd[r0], device=device)
            gram_val, svd_val = float(norms[r0]), m.frob_norm
            rel = abs(gram_val - svd_val) / max(svd_val, 1e-12)
            print(f"  VALIDATION L{layer}/r{r0}: gram={gram_val:.10g} svd={svd_val:.10g} "
                  f"rel_err={rel:.3e}")
            assert rel < 1e-6, f"Gram identity disagrees with SVD: {gram_val} vs {svd_val}"
            validated = True

        med = float(norms.median())
        order = torch.argsort(norms, descending=True)
        rank_of = {int(r): int((order == r).nonzero()[0, 0]) for r in target_rows}
        out_layers[layer] = dict(
            layer=layer, d_model=d_model, d_ffn=d_ffn,
            layer_median_uk=med, layer_max_uk=float(norms.max()),
            layer_max_over_median=float(norms.max()) / med if med > 0 else float("inf"),
            targets=[dict(row=int(r), uk_norm=float(norms[r]),
                          uk_over_layer_median=float(norms[r]) / med if med > 0 else float("inf"),
                          within_layer_rank=rank_of[int(r)])
                     for r in target_rows],
            top10_rows_this_layer=[dict(row=int(order[i]), uk_norm=float(norms[order[i]]))
                                    for i in range(10)],
        )
        for tg in out_layers[layer]["targets"]:
            print(f"  L{layer}/r{tg['row']}: ||U_k||_F={tg['uk_norm']:.6g}  "
                  f"{tg['uk_over_layer_median']:.1f}x layer median  "
                  f"within-layer rank #{tg['within_layer_rank']} of {d_model}")
        del Wg, Wu, Wd, t

    result = dict(model=args.model, repo=spec["repo"], canonical=spec.get("canonical"),
                  method="exact ||U_k||_F via Gram identity (no SVD), float64, "
                         "validated against spectral_lib.row_spectral_metrics",
                  layers=out_layers)
    p = ROOT / "results" / f"e10_exact_uknorm_{args.model}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
