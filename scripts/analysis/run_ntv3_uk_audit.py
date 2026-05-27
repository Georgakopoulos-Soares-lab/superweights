"""
scripts/analysis/run_ntv3_uk_audit.py
-------------------------------------
Mechanistic ||U_k||_F audit for NTv3 (encoder, GLU-FFN), closing the
mechanism story across three independent gated-FFN architectures
(GENERator, DNABERT-2, NTv3).

NTv3's SelfAttentionBlock.mlp is GLU:
    x_lin = fc1(x)                          # (..., 2*ffn_embed_dim)
    x1, x2 = split(x_lin, half, dim=-1)     # gate, up
    x = SiLU(x1) * x2                       # (..., ffn_embed_dim)
    out = fc2(x)                            # (..., embed_dim)

So:
    W_gate = fc1.weight[:ffn_embed_dim, :]   # (d_ffn, d_model)
    W_up   = fc1.weight[ffn_embed_dim:, :]   # (d_ffn, d_model)
    W_down = fc2.weight                       # (d_model, d_ffn)

    ||U_k||_F = sqrt( sum_i  W_down[k,i]^2 * ||W_gate[i,:]||^2 * ||W_up[i,:]||^2 )

Detected SW (from results/super_weight_index.json): layer 11, row 1472.

Output: results/sw_mechanistic_ntv3.json
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

    print("[uk_audit_ntv3] loading NTv3 …")
    model = AutoModelForMaskedLM.from_pretrained(
        "InstaDeepAI/NTv3_650M_pre",
        trust_remote_code=True,
        code_revision=_NTV3_CODE_REV,
        torch_dtype=torch.float32,
    )
    model.eval()
    n_layers = len(model.core.transformer_blocks)
    print(f"[uk_audit_ntv3] n_blocks = {n_layers}")

    sw_idx = json.loads((ROOT / "results/super_weight_index.json").read_text())
    sw_entries = sw_idx.get("ntv3", {})
    if isinstance(sw_entries, dict):
        sw_entries = sw_entries.get("results", [])
    sw_rows_by_layer: dict[int, list[int]] = {}
    for e in sw_entries:
        sw_rows_by_layer.setdefault(int(e["layer"]), []).append(int(e["row"]))
    print(f"[uk_audit_ntv3] SW rows by layer: {sw_rows_by_layer}")

    out: dict = {
        "model": "ntv3",
        "sw_rows_by_layer": {str(k): v for k, v in sw_rows_by_layer.items()},
        "frob_norm_uk_by_layer": {},
        "median_frob_by_layer": {},
        "max_frob_by_layer":   {},
        "concentration_ratio_by_layer": {},
        "sw_row_ranks": {},
        "sw_row_frob":  {},
        "top1_frob_by_layer": {},
    }

    for li in range(n_layers):
        blk = model.core.transformer_blocks[li]
        Wfc1 = blk.fc1.weight.detach().float().cpu().numpy()  # (2*d_ffn, d_model)
        Wd   = blk.fc2.weight.detach().float().cpu().numpy()  # (d_model, d_ffn)

        two_dffn, d_model = Wfc1.shape
        d_ffn = two_dffn // 2
        assert Wd.shape == (d_model, d_ffn)

        Wg = Wfc1[:d_ffn, :]
        Wu = Wfc1[d_ffn:, :]
        if li == 0:
            print(f"[uk_audit_ntv3] d_model={d_model}  d_ffn={d_ffn}")
            out["d_model"] = int(d_model)
            out["d_ffn"]   = int(d_ffn)

        norm_g2 = (Wg ** 2).sum(axis=1)
        norm_u2 = (Wu ** 2).sum(axis=1)
        weight  = norm_g2 * norm_u2
        frob    = np.sqrt((Wd ** 2 * weight[None, :]).sum(axis=1))  # (d_model,)

        order = np.argsort(-frob)
        rank_of = {int(r): int(np.where(order == r)[0][0] + 1)
                   for r in range(d_model)}

        median = float(np.median(frob))
        mx     = float(frob.max())
        out["frob_norm_uk_by_layer"][str(li)] = frob.tolist()
        out["median_frob_by_layer"][str(li)]  = median
        out["max_frob_by_layer"][str(li)]     = mx
        out["concentration_ratio_by_layer"][str(li)] = mx / median if median > 0 else float("nan")
        out["top1_frob_by_layer"][str(li)]    = int(order[0])

        sw_here = sw_rows_by_layer.get(li, [])
        if sw_here:
            ranks = {int(r): rank_of[r] for r in sw_here}
            fvals = {int(r): float(frob[r]) for r in sw_here}
            out["sw_row_ranks"][str(li)] = ranks
            out["sw_row_frob"][str(li)]  = fvals
            for r in sw_here:
                pct = 100.0 * (1.0 - (ranks[r] - 1) / d_model)
                print(f"  layer {li:2d}  SW row {r:5d}  ||U_k||_F = {frob[r]:9.3f}  "
                      f"rank {ranks[r]:5d}/{d_model}  median={median:.3f}  "
                      f"max={mx:.3f}  percentile={pct:.2f}")

    out_path = ROOT / "results/sw_mechanistic_ntv3.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[uk_audit_ntv3] wrote {out_path}")


if __name__ == "__main__":
    main()
