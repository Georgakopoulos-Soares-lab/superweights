"""
scripts/analysis/run_dnabert2_uk_audit.py
-----------------------------------------
Mechanistic ||U_k||_F audit for DNABERT-2, mirroring run_sw_mechanistic.py
(GENERator).

DNABERT-2 uses BertGatedLinearUnitMLP per layer:

    hidden = gated_layers(x)                       # (B, 2 * d_ffn)
    gate, up = hidden[..., :d_ffn], hidden[..., d_ffn:]
    h = GELU(gate) * up
    out = wo(h)

So the GLU structure is structurally identical to GENERator; only the parameter
layout differs:

    gated_layers.weight : (2*d_ffn, d_model)   stacked [gate; up]
      W_gate = gated_layers.weight[:d_ffn, :]
      W_up   = gated_layers.weight[d_ffn:, :]
    wo.weight            : (d_model, d_ffn)    == W_down

We scan **every encoder layer 0..11** (rather than a single SW layer) because
the DNABERT-2 SW ensemble spans multiple layers (3,5,6,7,9 around row 603).
For each layer we compute the rank-1 Frobenius bound

    ||U_k||_F = sqrt( sum_i  W_down[k,i]^2 * ||W_gate[i,:]||^2 * ||W_up[i,:]||^2 )

and report the rank of each known SW row in that layer's distribution.

Output
------
results/sw_mechanistic_dnabert2.json
{
  "model": "dnabert2",
  "d_model": 768, "d_ffn": 3072,
  "sw_rows_by_layer": {layer: [rows...]},     # from super_weight_index.json
  "frob_norm_uk_by_layer": {layer: [float, ...]},   # length d_model
  "sw_row_ranks": {layer: {row: rank_1based}},
  "sw_row_frob": {layer: {row: float}},
  "median_frob_by_layer": {layer: float},
}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent


# ─────────────────────────────────────────────────────────────────────────────
def load_model(config: dict):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        config["model_id"],
        trust_remote_code=config.get("hf_trust_remote_code", True),
        torch_dtype=torch.float32,
    )
    model.eval()
    return model


def extract_mlp_weights(model, layer_idx: int):
    """Return (W_gate, W_up, W_down) as float32 numpy arrays for layer `layer_idx`.

    W_gate : (d_ffn, d_model)
    W_up   : (d_ffn, d_model)
    W_down : (d_model, d_ffn)
    """
    layer = model.encoder.layer[layer_idx]
    mlp = layer.mlp
    gated = mlp.gated_layers.weight.detach().float().cpu().numpy()  # (2*d_ffn, d_model)
    two_dffn, d_model = gated.shape
    d_ffn = two_dffn // 2
    Wg = gated[:d_ffn, :]          # gate (first half — matches modeling source)
    Wu = gated[d_ffn:, :]          # up
    Wd = mlp.wo.weight.detach().float().cpu().numpy()  # (d_model, d_ffn)
    assert Wd.shape == (d_model, d_ffn), f"unexpected wo shape {Wd.shape}"
    return Wg, Wu, Wd


def frob_norm_uk(Wg: np.ndarray, Wu: np.ndarray, Wd: np.ndarray) -> np.ndarray:
    """Rank-1 Frobenius bound (identical formula to run_sw_mechanistic.py).

        ||U_k||_F = sqrt( sum_i  Wd[k,i]^2 * ||Wg[i,:]||^2 * ||Wu[i,:]||^2 )
    """
    norm_g2 = (Wg ** 2).sum(axis=1)                    # (d_ffn,)
    norm_u2 = (Wu ** 2).sum(axis=1)                    # (d_ffn,)
    weight  = norm_g2 * norm_u2                        # (d_ffn,)
    frob    = np.sqrt((Wd ** 2 * weight[None, :]).sum(axis=1))   # (d_model,)
    return frob


# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     default="dnabert2")
    p.add_argument("--sw_index",  default="results/super_weight_index.json")
    p.add_argument("--out",       default="results/sw_mechanistic_dnabert2.json")
    p.add_argument("--config",    default="configs/dnabert2.yaml")
    args = p.parse_args()

    config = yaml.safe_load((ROOT / args.config).read_text())
    n_layers = int(config["num_layers"])

    sw_idx = json.loads((ROOT / args.sw_index).read_text())
    sw_entries = sw_idx.get(args.model, {})
    if isinstance(sw_entries, dict):
        sw_entries = sw_entries.get("results", [])

    sw_rows_by_layer: dict[int, list[int]] = {}
    for e in sw_entries:
        sw_rows_by_layer.setdefault(int(e["layer"]), []).append(int(e["row"]))

    print(f"[uk_audit] model={args.model}  layers=0..{n_layers - 1}")
    print(f"[uk_audit] SW rows by layer:")
    for l, rs in sorted(sw_rows_by_layer.items()):
        print(f"    layer {l}: rows {rs}")

    print("[uk_audit] Loading model …")
    model = load_model(config)

    out = {
        "model":           args.model,
        "sw_rows_by_layer": {str(k): v for k, v in sw_rows_by_layer.items()},
        "frob_norm_uk_by_layer": {},
        "sw_row_ranks":    {},
        "sw_row_frob":     {},
        "median_frob_by_layer": {},
    }

    for li in range(n_layers):
        Wg, Wu, Wd = extract_mlp_weights(model, li)
        d_ffn, d_model = Wg.shape
        if li == 0:
            print(f"[uk_audit] d_ffn={d_ffn}  d_model={d_model}")
            out["d_ffn"]   = int(d_ffn)
            out["d_model"] = int(d_model)

        frob = frob_norm_uk(Wg, Wu, Wd)
        order = np.argsort(-frob)               # descending
        rank_of = {int(r): int(np.where(order == r)[0][0] + 1)
                   for r in range(d_model)}

        out["frob_norm_uk_by_layer"][str(li)] = frob.tolist()
        out["median_frob_by_layer"][str(li)] = float(np.median(frob))

        sw_here = sw_rows_by_layer.get(li, [])
        if sw_here:
            ranks = {int(r): rank_of[r] for r in sw_here}
            fvals = {int(r): float(frob[r]) for r in sw_here}
            out["sw_row_ranks"][str(li)] = ranks
            out["sw_row_frob"][str(li)]  = fvals
            for r in sw_here:
                print(f"  layer {li:2d}  SW row {r:4d}  ||U_k||_F = {frob[r]:8.3f}  "
                      f"rank {rank_of[r]:4d}/{d_model}  median = {np.median(frob):.3f}")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[uk_audit] wrote {out_path}")


if __name__ == "__main__":
    main()
