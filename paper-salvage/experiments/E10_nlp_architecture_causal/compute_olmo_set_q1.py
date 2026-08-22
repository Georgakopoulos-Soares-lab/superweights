"""
experiments/E10_nlp_architecture_causal/compute_olmo_set_q1.py

E10 Step D1: OLMo-7B has a pre-existing SET of candidate rows (Yu et al. Table 2 lists output
row 269 recurring at layers 1, 2, 7, 24, each with a different scalar i). Only layer 1's exact
q1 was ever computed (E7). This script applies the IDENTICAL already-existing exact metric
(spectral_lib.row_spectral_metrics, the same E7 machinery) to the other 3 published layers, so
E10's Step D1 top-K freeze can rank the whole pre-existing set rather than just the one row.

This is not a new structural metric and not a new row search: the 4 (layer, row=269)
coordinates are all already published in Yu et al. (2024) Table 2 -- nothing here searches for
new candidates.

Weight-only; only the shards holding each target layer's gate/up/down tensors are fetched
(same fetch pattern as E1_nlp_validation/run_e1_retrospective.py::fetch_layer_tensors).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

SALVAGE = Path(__file__).resolve().parents[2]
ROOT = SALVAGE.parent
sys.path.insert(0, str(SALVAGE / "src"))
sys.path.insert(0, str(SALVAGE / "experiments" / "E7_exact_dimensionality"))

from spectral_lib import row_spectral_metrics  # noqa: E402

REPO = "allenai/OLMo-7B-0724-hf"
ROW = 269
# (layer, published scalar i) -- from Yu et al. Table 2, already cited in
# E6_cross_geometry/CONFIRMATION_PANEL.md's "NLP candidates" table.
LAYERS = {1: 7467, 2: 8275, 7: 453, 24: 2300}


def fetch_layer_tensors(repo: str, layer: int) -> dict[str, torch.Tensor]:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    names = {
        "gate": f"model.layers.{layer}.mlp.gate_proj.weight",
        "up": f"model.layers.{layer}.mlp.up_proj.weight",
        "down": f"model.layers.{layer}.mlp.down_proj.weight",
    }
    idx_path = hf_hub_download(repo, "model.safetensors.index.json")
    weight_map = json.loads(Path(idx_path).read_text())["weight_map"]
    missing = [n for n in names.values() if n not in weight_map]
    if missing:
        raise RuntimeError(f"tensor names not in index: {missing}")
    shards = sorted({weight_map[n] for n in names.values()})
    loaded: dict[str, torch.Tensor] = {}
    for shard in shards:
        p = hf_hub_download(repo, shard)
        data = load_file(p)
        for key, tname in names.items():
            if tname in data:
                loaded[key] = data[tname]
        del data
    if len(loaded) != 3:
        raise RuntimeError(f"only found {sorted(loaded)}")
    return loaded


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for layer, i_pub in LAYERS.items():
        print(f"layer {layer}: fetching gate/up/down ...")
        t = fetch_layer_tensors(REPO, layer)
        Wg, Wu, Wd = t["gate"], t["up"], t["down"]
        d_ffn, d_model = Wg.shape
        assert Wd.shape == (d_model, d_ffn), f"shape mismatch at layer {layer}: {Wd.shape}"
        metrics, _ = row_spectral_metrics(Wg, Wu, Wd[ROW], device=device)
        print(f"  L{layer}/r{ROW} (published i={i_pub}): q1={metrics.q1:.6f} "
              f"pr_spec={metrics.pr_spec:.4f} frob={metrics.frob_norm:.6g}")
        results.append(dict(layer=layer, row=ROW, published_i=i_pub, q1=metrics.q1,
                             pr_spec=metrics.pr_spec, stable_rank=metrics.stable_rank,
                             frob_norm=metrics.frob_norm,
                             n_singular_values=metrics.n_singular_values))

    out = dict(model="OLMo-7B", repo=REPO, row=ROW, rows=results)
    out_path = ROOT / "results" / "e10_olmo_set_q1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
