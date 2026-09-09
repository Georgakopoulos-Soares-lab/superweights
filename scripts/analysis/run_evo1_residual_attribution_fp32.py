"""
scripts/analysis/run_evo1_residual_attribution_fp32.py
------------------------------------------------------
Same per-block residual attribution as run_evo1_residual_attribution.py but
in float32 throughout — no autocast — so the trace stays finite through all
32 blocks instead of overflowing at the layer-10 spike.

Writes results/sw_residual_attribution_evo1_fp32.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--row",   type=int, default=3776)
    p.add_argument("--out",   default="results/sw_residual_attribution_evo1_fp32.json")
    p.add_argument("--probe", default="actb_500")
    args = p.parse_args()

    import sys; sys.path.insert(0, str(ROOT))
    from src.dna_probes import get_probe
    from evo import Evo

    print("[evo1_resid_attr fp32] loading Evo1 (bfloat16 — fp32-range exponent) …")
    evo = Evo("evo-1-8k-base", device="cuda")
    # Cannot use pure fp32: StripedHyena's flash-attn inner attn asserts
    # fp16/bf16. bf16 has the same exponent range as fp32, so the layer-10
    # +165 spike does not overflow as it did in fp16.
    model = evo.evo_model if False else evo.model
    model.to_bfloat16_except_poles_residues = lambda: None  # already done by loader
    # Force everything that isn't poles/residues to bf16 (loader did fp32 by
    # default after the snapshot patch).
    for n, p in model.named_parameters():
        if any(k in n for k in ("poles", "residues")):
            continue
        p.data = p.data.to(torch.bfloat16)
    tokenizer = evo.tokenizer
    n_blocks = len(model.blocks)
    print(f"[evo1_resid_attr fp32] n_blocks = {n_blocks}, target row = {args.row}")

    seq = get_probe(args.probe)
    device = next(model.parameters()).device
    ids = torch.tensor(list(tokenizer.tokenize(seq)),
                       dtype=torch.long, device=device).unsqueeze(0)

    records: list[dict] = [{} for _ in range(n_blocks)]

    def make_block_hook(li):
        def hook(module, inputs, output):
            res_in  = inputs[0]
            res_out = output[0] if isinstance(output, tuple) else output
            r = args.row
            ri = res_in[0]  if res_in.dim()  == 3 else res_in
            ro = res_out[0] if res_out.dim() == 3 else res_out
            sw_pos = int(ro[:, r].abs().argmax().item())
            records[li]["block_residual_in"]  = float(ri[sw_pos, r].item())
            records[li]["block_residual_out"] = float(ro[sw_pos, r].item())
            records[li]["block_delta"]        = float((ro[sw_pos, r] - ri[sw_pos, r]).item())
            records[li]["sw_pos"]             = sw_pos
            records[li]["max_abs_at_pos"]     = float(ro[sw_pos].abs().max().item())
            records[li]["argmax_at_pos"]      = int(ro[sw_pos].abs().argmax().item())
            records[li]["res_out_global_max"] = float(ro.abs().max().item())
        return hook

    def make_mlp_hook(li):
        def hook(module, inputs, output):
            r = args.row
            mlp_out = output[0] if isinstance(output, tuple) else output
            m = mlp_out[0] if mlp_out.dim() == 3 else mlp_out
            records[li]["_mlp_at_row_per_pos"]  = m[:, r].detach().cpu().tolist()
            records[li]["_mlp_argmax_per_pos"]  = m.abs().argmax(dim=-1).detach().cpu().tolist()
            records[li]["_mlp_max_abs_per_pos"] = m.abs().max(dim=-1).values.detach().cpu().tolist()
        return hook

    handles = []
    for li in range(n_blocks):
        blk = model.blocks[li]
        handles.append(blk.register_forward_hook(make_block_hook(li)))
        handles.append(blk.mlp.register_forward_hook(make_mlp_hook(li)))

    with torch.no_grad():
        model(ids)

    for h in handles:
        h.remove()

    for li, r in enumerate(records):
        if not r:
            continue
        sw_pos = r["sw_pos"]
        per_pos       = r.pop("_mlp_at_row_per_pos",  [])
        max_per_pos   = r.pop("_mlp_max_abs_per_pos", [])
        argmx_per_pos = r.pop("_mlp_argmax_per_pos",  [])
        r["mlp_out_at_row"]    = per_pos[sw_pos]      if sw_pos < len(per_pos)      else None
        r["mlp_max_at_pos"]    = max_per_pos[sw_pos]  if sw_pos < len(max_per_pos)  else None
        r["mlp_argmax_at_pos"] = argmx_per_pos[sw_pos] if sw_pos < len(argmx_per_pos) else None
        r["layer"]             = li

    print(f"{'layer':>5} {'pos':>5} {'res_in':>12} {'mlp@row':>12} "
          f"{'block_delta':>12} {'res_out':>12} {'|mlp|max':>10} {'argmax':>8}")
    print("-" * 90)
    for r in records:
        if not r:
            continue
        print(f"{r['layer']:5d} {r['sw_pos']:5d} "
              f"{r['block_residual_in']:12.3f} "
              f"{(r.get('mlp_out_at_row') or 0.0):12.3f} "
              f"{r['block_delta']:12.3f} "
              f"{r['block_residual_out']:12.3f} "
              f"{(r.get('mlp_max_at_pos') or 0.0):10.3f} "
              f"{(r.get('mlp_argmax_at_pos') or -1):8d}")

    out_doc = {
        "model": "evo1",
        "probe": args.probe,
        "row":   args.row,
        "dtype": "float32",
        "per_layer": records,
    }
    out_path = ROOT / args.out
    out_path.write_text(json.dumps(out_doc, indent=2))
    print(f"\n[evo1_resid_attr fp32] wrote {out_path}")


if __name__ == "__main__":
    main()
