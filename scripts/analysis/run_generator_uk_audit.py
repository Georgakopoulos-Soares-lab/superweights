"""
scripts/analysis/run_generator_uk_audit.py
------------------------------------------
Mechanistic ||U_k||_F audit for GENERator (EUK and PROK variants),
generating Fig 2 panels C/D data and adding GENERator markers to panel E.

GENERator uses LlamaForCausalLM with SwiGLU MLP per layer:

    gate_out = gate_proj(x)        # (d_ffn,)
    up_out   = up_proj(x)          # (d_ffn,)
    h        = SiLU(gate_out) * up_out
    out      = down_proj(h)        # (d_model,)

So:
    W_gate = gate_proj.weight      # (d_ffn, d_model)
    W_up   = up_proj.weight        # (d_ffn, d_model)
    W_down = down_proj.weight      # (d_model, d_ffn)

    ||U_k||_F = sqrt( sum_i  W_down[k,i]^2 * ||W_gate[i,:]||^2 * ||W_up[i,:]||^2 )

SW rows (from results/super_weight_index.json):
  EUK:  layer 4, rows [2371, 1522]
  PROK: layer 2, row  [1927]

Output:
  results/sw_mechanistic_generator.json
  results/sw_mechanistic_generator_prokaryote.json

Usage:
  python scripts/analysis/run_generator_uk_audit.py --variant euk
  python scripts/analysis/run_generator_uk_audit.py --variant prok
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent

VARIANT_CFG = {
    "euk":  ("generator",            "results/sw_mechanistic_generator.json"),
    "prok": ("generator_prokaryote", "results/sw_mechanistic_generator_prokaryote.json"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="GENERator ||U_k||_F audit")
    parser.add_argument("--variant", choices=["euk", "prok"], required=True,
                        help="'euk' → GENERator-EUK, 'prok' → GENERator-PROK")
    args = parser.parse_args()

    model_key, out_rel = VARIANT_CFG[args.variant]
    out_path = ROOT / out_rel

    cfg = yaml.safe_load((ROOT / f"configs/{model_key}.yaml").read_text())
    n_layers: int = cfg["num_layers"]

    sw_idx = json.loads((ROOT / "results/super_weight_index.json").read_text())
    sw_entries = sw_idx.get(model_key, {}).get("results", [])
    sw_rows_by_layer: dict[int, list[int]] = {}
    for e in sw_entries:
        sw_rows_by_layer.setdefault(int(e["layer"]), []).append(int(e["row"]))
    print(f"[uk_audit] variant={args.variant}  SW rows by layer: {sw_rows_by_layer}")

    print(f"[uk_audit] loading {cfg['model_id']} on CPU …")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        trust_remote_code=cfg.get("hf_trust_remote_code", True),
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()
    print(f"[uk_audit] loaded — scanning {n_layers} layers")

    out: dict = {
        "model":                     model_key,
        "sw_rows_by_layer":          {str(k): v for k, v in sw_rows_by_layer.items()},
        "frob_norm_uk_by_layer":     {},
        "median_frob_by_layer":      {},
        "max_frob_by_layer":         {},
        "concentration_ratio_by_layer": {},
        "sw_row_ranks":              {},
        "sw_row_frob":               {},
        "top1_frob_by_layer":        {},
    }

    for li in range(n_layers):
        blk = model.model.layers[li]
        Wg = blk.mlp.gate_proj.weight.detach().float().cpu().numpy()   # (d_ffn, d_model)
        Wu = blk.mlp.up_proj.weight.detach().float().cpu().numpy()     # (d_ffn, d_model)
        Wd = blk.mlp.down_proj.weight.detach().float().cpu().numpy()   # (d_model, d_ffn)

        d_ffn, d_model = Wg.shape
        assert Wd.shape == (d_model, d_ffn), f"shape mismatch layer {li}"

        if li == 0:
            print(f"[uk_audit] d_model={d_model}  d_ffn={d_ffn}")
            out["d_model"] = int(d_model)
            out["d_ffn"]   = int(d_ffn)

        # ||U_k||_F = sqrt( sum_i W_down[k,i]^2 * ||W_gate[i,:]||^2 * ||W_up[i,:]||^2 )
        norm_g2 = (Wg ** 2).sum(axis=1)              # (d_ffn,)
        norm_u2 = (Wu ** 2).sum(axis=1)              # (d_ffn,)
        weight  = norm_g2 * norm_u2
        frob    = np.sqrt((Wd ** 2 * weight[None, :]).sum(axis=1))   # (d_model,)

        order  = np.argsort(-frob)
        rank_of = {int(r): int(np.where(order == r)[0][0] + 1)
                   for r in range(d_model)}

        median = float(np.median(frob))
        mx     = float(frob.max())
        out["frob_norm_uk_by_layer"][str(li)] = frob.tolist()
        out["median_frob_by_layer"][str(li)]  = median
        out["max_frob_by_layer"][str(li)]     = mx
        out["concentration_ratio_by_layer"][str(li)] = (
            mx / median if median > 0 else float("nan")
        )
        out["top1_frob_by_layer"][str(li)] = int(order[0])

        sw_here = sw_rows_by_layer.get(li, [])
        if sw_here:
            ranks = {int(r): rank_of[r] for r in sw_here}
            fvals = {int(r): float(frob[r]) for r in sw_here}
            out["sw_row_ranks"][str(li)] = ranks
            out["sw_row_frob"][str(li)]  = fvals
            for r in sw_here:
                pct = 100.0 * (1.0 - (ranks[r] - 1) / d_model)
                print(
                    f"  layer {li:2d}  SW row {r:5d}  "
                    f"||U_k||_F = {frob[r]:14.3f}  "
                    f"rank {ranks[r]:5d}/{d_model}  "
                    f"median={median:.3f}  max={mx:.3f}  "
                    f"percentile={pct:.2f}"
                )
        else:
            print(
                f"  layer {li:2d}  median={median:.3f}  "
                f"max={mx:.3f}  top1_row={order[0]}"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[uk_audit] wrote {out_path}")


if __name__ == "__main__":
    main()
