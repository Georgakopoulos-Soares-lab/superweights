"""
E1 — NLP cold-weight validation, RETROSPECTIVE ARM ONLY.

Tests the closed-form predictor ‖U_k‖_F against the published super-weight coordinates of
Yu et al. (2024), arXiv:2411.07191, Table 2 "Super Weight Directory". Weights only — no
forward pass, no GPU.

  Level 1 — row recovery:    rank of the published output row k among all d_model rows,
                             plus max/median ratio for the layer.
  Level 2 — scalar recovery: within row k, top1_index / top1_share / participation_ratio,
                             and the rank of the published hidden index i.

Reporting rule (PHASE_1_BLOCKING §E1): "recovers the published super-weight output rows" if
only Level 1 holds. Scalar recovery may be claimed only if Level 2 also holds.

The prospective arm is NOT run here — it needs a model choice and a lock.

Coordinates are read from the local copy of the paper (docs/superweight_paper.txt, Table 2),
not from memory. Layer indices are HuggingFace-ready per the paper's own note:
"for Llama-7B ... access the super weight using layers[2].mlp.down_proj.weight[3968, 7003]".

Only the safetensors shards containing the three target tensors are downloaded, so this
costs a few GB rather than a full checkpoint per model.

Usage:
  python run_e1_retrospective.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SALVAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SALVAGE / "src"))

from uk_frobenius import layer_report, row_granularity, scalar_rank  # noqa: E402

# Yu et al. (2024) Table 2. (repo_id, layer, row k, index i)
# Where the paper lists several coordinates for a model, the FIRST (lowest-layer) entry is
# used as the primary, matching how the paper itself illustrates the directory.
PUBLISHED = [
    {"name": "Llama-7B",   "repo": "huggyllama/llama-7b",       "layer": 2, "k": 3968, "i": 7003},
    {"name": "Mistral-7B", "repo": "mistralai/Mistral-7B-v0.1", "layer": 1, "k": 2070, "i": 7310},
    {"name": "OLMo-7B",    "repo": "allenai/OLMo-7B-0724-hf",   "layer": 1, "k": 269,  "i": 7467},
]


def fetch_layer_tensors(repo: str, layer: int) -> dict[str, torch.Tensor] | None:
    """Download only the shard(s) holding this layer's gate/up/down projections."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    names = {
        "gate": f"model.layers.{layer}.mlp.gate_proj.weight",
        "up":   f"model.layers.{layer}.mlp.up_proj.weight",
        "down": f"model.layers.{layer}.mlp.down_proj.weight",
    }
    try:
        idx_path = hf_hub_download(repo, "model.safetensors.index.json")
        weight_map = json.loads(Path(idx_path).read_text())["weight_map"]
        missing = [n for n in names.values() if n not in weight_map]
        if missing:
            print(f"    tensor names not in index: {missing[:3]}")
            return None
        shards = sorted({weight_map[n] for n in names.values()})
        print(f"    shards needed: {shards}")
        loaded: dict[str, torch.Tensor] = {}
        for shard in shards:
            p = hf_hub_download(repo, shard)
            data = load_file(p)
            for key, tname in names.items():
                if tname in data:
                    loaded[key] = data[tname]
            del data
        if len(loaded) != 3:
            print(f"    only found {sorted(loaded)}")
            return None
        return loaded
    except Exception as exc:
        print(f"    FETCH FAILED: {type(exc).__name__}: {exc}")
        return None


def main() -> int:
    out_rows = []
    for spec in PUBLISHED:
        name, repo, layer, k, i = (spec["name"], spec["repo"], spec["layer"],
                                   spec["k"], spec["i"])
        print(f"\n=== {name} ({repo})  layer {layer}, published [k={k}, i={i}] ===")
        tensors = fetch_layer_tensors(repo, layer)
        if tensors is None:
            out_rows.append({**spec, "status": "UNAVAILABLE"})
            print("    -> skipped")
            continue

        W_gate, W_up, W_down = tensors["gate"], tensors["up"], tensors["down"]
        print(f"    shapes: gate {tuple(W_gate.shape)}  up {tuple(W_up.shape)}  "
              f"down {tuple(W_down.shape)}")

        # Shape guard: canonical convention is gate/up [d_ffn, d_model], down [d_model, d_ffn].
        d_ffn, d_model = W_gate.shape
        if W_down.shape != (d_model, d_ffn):
            print(f"    SHAPE MISMATCH: down is {tuple(W_down.shape)}, expected "
                  f"{(d_model, d_ffn)} — skipping rather than guessing a transpose")
            out_rows.append({**spec, "status": "SHAPE_MISMATCH",
                             "shapes": {kk: list(v.shape) for kk, v in tensors.items()}})
            continue
        if not (0 <= k < d_model and 0 <= i < d_ffn):
            print(f"    COORDINATE OUT OF RANGE for d_model={d_model}, d_ffn={d_ffn}")
            out_rows.append({**spec, "status": "COORD_OUT_OF_RANGE"})
            continue

        rep, contrib = layer_report(W_gate, W_up, W_down, layer=layer, query_rows=[k])
        gran = row_granularity(contrib, k)
        i_rank = scalar_rank(contrib, k, i)
        qr = rep.query_ranks[k]

        print(f"    L1  row rank        {qr['rank']} / {rep.d_model}   "
              f"(pct {qr['percentile']:.3f})")
        print(f"    L1  max/median      {rep.max_over_median:.1f}")
        print(f"    L2  top1_index      {gran.top1_index}   published i = {i}   "
              f"{'MATCH' if gran.top1_index == i else 'no match'}")
        print(f"    L2  rank of i       {i_rank} / {rep.d_ffn}")
        print(f"    L2  top1_share      {gran.top1_share:.4f}   "
              f"PR {gran.participation_ratio:.2f}   regime {gran.regime}")

        out_rows.append({
            **spec, "status": "ok", "d_model": rep.d_model, "d_ffn": rep.d_ffn,
            "level1": {"row_rank": qr["rank"], "percentile": qr["percentile"],
                       "uk_norm": qr["value"], "layer_median": rep.median,
                       "layer_max": rep.max, "max_over_median": rep.max_over_median,
                       "top_rows": rep.top_rows[:5]},
            "level2": {"top1_index": gran.top1_index, "published_i": i,
                       "top1_index_matches": gran.top1_index == i,
                       "rank_of_published_i": i_rank,
                       "top1_share": gran.top1_share, "top5_share": gran.top5_share,
                       "participation_ratio": gran.participation_ratio,
                       "regime": gran.regime},
        })
        del W_gate, W_up, W_down, tensors, contrib

    out = ROOT / "results" / "e1_nlp_retrospective.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source": "Yu et al. 2024 arXiv:2411.07191 Table 2, via docs/superweight_paper.txt",
        "arm": "retrospective only; prospective arm NOT run (needs model choice + lock)",
        "models": out_rows}, indent=2))
    print(f"\n  saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
