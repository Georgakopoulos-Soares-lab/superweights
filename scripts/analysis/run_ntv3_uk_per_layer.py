"""
scripts/analysis/run_ntv3_uk_per_layer.py
-----------------------------------------
NTv3 ||U_k||_F per-layer audit. Computes the predictor at every
NTv3 transformer block and reports both the SW row's rank (if any
SW is registered at that layer) and the top-1 row of the layer.

Used to support the cross-architecture claim in Table 1 supplementary.

Writes results/ntv3_uk_per_layer.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
_NTV3_CODE_REV = "0ecff3637f0d3ba5b686d1095083218157c2ca34"


def main():
    from transformers import AutoModelForMaskedLM

    print("[ntv3_uk_per_layer] loading NTv3 …")
    model = AutoModelForMaskedLM.from_pretrained(
        "InstaDeepAI/NTv3_650M_pre",
        trust_remote_code=True,
        code_revision=_NTV3_CODE_REV,
        torch_dtype=torch.float32,
    )
    model.eval()

    sw_idx = json.loads((ROOT / "results/super_weight_index.json").read_text())
    sw_entries = sw_idx.get("ntv3", {})
    if isinstance(sw_entries, dict):
        sw_entries = sw_entries.get("results", [])
    sw_rows_by_layer: dict[int, list[int]] = {}
    for e in sw_entries:
        sw_rows_by_layer.setdefault(int(e["layer"]), []).append(int(e["row"]))

    n_layers = len(model.core.transformer_blocks)
    layers_out = []

    print(f"\n{'layer':>5} {'d_model':>8} {'d_ffn':>6} {'median':>10} "
          f"{'max':>10} {'ratio':>7} {'top1_row':>9} {'sw_row':>8} {'sw_rank':>9}")
    print("-" * 90)

    for li in range(n_layers):
        blk = model.core.transformer_blocks[li]
        Wfc1 = blk.fc1.weight.detach().float().cpu().numpy()
        Wd   = blk.fc2.weight.detach().float().cpu().numpy()
        two_dffn, d_model = Wfc1.shape
        d_ffn = two_dffn // 2
        Wg = Wfc1[:d_ffn, :]
        Wu = Wfc1[d_ffn:, :]

        ng2 = (Wg ** 2).sum(1)
        nu2 = (Wu ** 2).sum(1)
        frob = np.sqrt((Wd ** 2 * (ng2 * nu2)[None, :]).sum(1))

        order = np.argsort(-frob)
        median = float(np.median(frob))
        mx     = float(frob.max())
        sw_here = sw_rows_by_layer.get(li, [])
        sw_row = sw_here[0] if sw_here else None
        sw_rank = (int(np.where(order == sw_row)[0][0] + 1)
                   if sw_row is not None else None)
        sw_pct  = (100.0 * (1.0 - (sw_rank - 1) / d_model)
                   if sw_rank is not None else None)
        sw_uk   = (float(frob[sw_row]) if sw_row is not None else None)

        print(f"{li:5d} {d_model:8d} {d_ffn:6d} {median:10.3f} {mx:10.3f} "
              f"{mx/median:7.2f} {int(order[0]):9d} "
              f"{str(sw_row):>8} {str(sw_rank):>9}")

        layers_out.append({
            "layer":          li,
            "d_model":        int(d_model),
            "d_ffn":          int(d_ffn),
            "frob_median":    median,
            "frob_max":       mx,
            "max_over_median": mx / median if median > 0 else None,
            "top1_row":       int(order[0]),
            "top1_uk":        float(frob[order[0]]),
            "sw_row":         sw_row,
            "sw_uk":          sw_uk,
            "sw_rank":        sw_rank,
            "sw_percentile":  sw_pct,
            # store full frob array (cheap, 1536 floats per layer)
            "frob_uk":        frob.tolist(),
        })

    out = {
        "model":    "ntv3",
        "n_layers": n_layers,
        "layers":   layers_out,
    }
    out_path = ROOT / "results/ntv3_uk_per_layer.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[ntv3_uk_per_layer] wrote {out_path}")


if __name__ == "__main__":
    main()
