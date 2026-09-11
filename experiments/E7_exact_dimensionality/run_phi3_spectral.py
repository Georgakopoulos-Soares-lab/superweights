"""
experiments/E7_exact_dimensionality/run_phi3_spectral.py

Confirmatory spectral computation for Phi-3-mini-4k-instruct's six published (Yu et al.
Table 2) coordinates. Weight-only: fetches only the safetensors shard(s) holding layers 2
and 4's mlp.gate_up_proj / mlp.down_proj, no forward pass.

Phi3MLP packs gate+up in a single gate_up_proj [2*d_ffn, d_model] tensor, chunk(2) ->
(gate, up) -- confirmed from transformers/models/phi3/modeling_phi3.py (see MODEL_PANEL.md).
This adapter is E7-local (not added to uk_frobenius.py), per instruction to prefer
self-contained E7 code.

Usage:
  python run_phi3_spectral.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

SALVAGE = Path(__file__).resolve().parents[2]
ROOT = SALVAGE  # repo root; harness paths are written relative to it
sys.path.insert(0, str(SALVAGE / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from spectral_lib import row_spectral_metrics  # noqa: E402

PREREG = SALVAGE / "docs" / "prereg" / "PREREG_exact_operator_dimensionality.md"
REPO = "microsoft/Phi-3-mini-4k-instruct"

# Yu et al. (2024) Table 2, Phi-3-mini-4k-instruct: 6 coordinates.
PUBLISHED = [
    dict(layer=2, row=525, i=808),
    dict(layer=2, row=1693, i=808),
    dict(layer=2, row=1113, i=808),
    dict(layer=4, row=525, i=2723),
    dict(layer=4, row=1113, i=2723),
    dict(layer=4, row=1693, i=2723),
]


def verify_lock() -> None:
    r = subprocess.run(
        [sys.executable, str(SALVAGE / "src" / "prereg_lock.py"), "verify", str(PREREG)],
        capture_output=True, text=True,
    )
    print(r.stdout.strip())
    if r.returncode != 0 or "OK" not in r.stdout:
        print(r.stderr, file=sys.stderr)
        raise SystemExit("prereg_lock verify FAILED.")


def fetch_layer_tensors(layer: int):
    """Download only the shard(s) holding this layer's gate_up_proj / down_proj weights."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    names = {
        "gate_up": f"model.layers.{layer}.mlp.gate_up_proj.weight",
        "down": f"model.layers.{layer}.mlp.down_proj.weight",
    }
    idx_path = hf_hub_download(REPO, "model.safetensors.index.json")
    weight_map = json.loads(Path(idx_path).read_text())["weight_map"]
    missing = [n for n in names.values() if n not in weight_map]
    if missing:
        raise RuntimeError(f"tensor names not in index: {missing}")
    shards = sorted({weight_map[n] for n in names.values()})
    print(f"  layer {layer}: shards needed: {shards}")
    loaded = {}
    for shard in shards:
        p = hf_hub_download(REPO, shard)
        data = load_file(p)
        for key, tname in names.items():
            if tname in data:
                loaded[key] = data[tname]
        del data
    return loaded["gate_up"], loaded["down"]


def split_gate_up(gate_up: torch.Tensor):
    d_ffn2, d_model = gate_up.shape
    d_ffn = d_ffn2 // 2
    gate, up = gate_up[:d_ffn], gate_up[d_ffn:]
    return gate, up


def main() -> None:
    print("Verifying E7 prereg lock before touching Phi-3 weights ...")
    verify_lock()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    by_layer = {}
    results = []
    for spec in PUBLISHED:
        layer = spec["layer"]
        if layer not in by_layer:
            gate_up, down = fetch_layer_tensors(layer)
            gate, up = split_gate_up(gate_up)
            d_ffn, d_model = gate.shape
            print(f"  layer {layer}: gate{tuple(gate.shape)} up{tuple(up.shape)} "
                  f"down{tuple(down.shape)} (d_ffn={d_ffn}, d_model={d_model})")
            by_layer[layer] = (gate, up, down)
        gate, up, down = by_layer[layer]

        row = spec["row"]
        metrics, sigmas = row_spectral_metrics(gate, up, down[row], device=device)
        print(f"  L{layer} row {row} (published i={spec['i']}): q1={metrics.q1:.6f} "
              f"PR_spec={metrics.pr_spec:.4f} frob={metrics.frob_norm:.6g} "
              f"n_sv={metrics.n_singular_values}")
        results.append(dict(
            layer=layer, row=row, published_i=spec["i"],
            q1=metrics.q1, pr_spec=metrics.pr_spec, stable_rank=metrics.stable_rank,
            frob_norm=metrics.frob_norm, n_singular_values=metrics.n_singular_values,
        ))

    q1_values = [r["q1"] for r in results]
    pr_values = [r["pr_spec"] for r in results]
    q1_values_sorted = sorted(q1_values)
    pr_values_sorted = sorted(pr_values)
    n = len(q1_values_sorted)
    model_q1_median = (q1_values_sorted[n // 2] if n % 2 == 1 else
                        0.5 * (q1_values_sorted[n // 2 - 1] + q1_values_sorted[n // 2]))
    model_pr_median = (pr_values_sorted[n // 2] if n % 2 == 1 else
                        0.5 * (pr_values_sorted[n // 2 - 1] + pr_values_sorted[n // 2]))

    out = dict(model="Phi-3-mini-4k-instruct", group="nlp", repo=REPO,
               rows=results, model_level=dict(q1_median=model_q1_median, pr_spec_median=model_pr_median))
    print(f"\nPhi-3 model-level (median over 6 published rows): "
          f"q1={model_q1_median:.6f} PR_spec={model_pr_median:.4f}")

    out_path = ROOT / "results" / "e7_phi3_spectral.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
