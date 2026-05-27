"""
scripts/analysis/run_evo1_residual_attribution.py
-------------------------------------------------
Resolves the "necessary but not sufficient" Evo1 ||U_k||_F result.

Question: does Evo1's high-||U_k||_F row at the detected layer (layer 11)
actually receive a large MLP write at the SW token position, and does that
write persist in the residual stream, or does the StripedHyena conv mixer
either (a) never produce a large MLP output at that row or (b) immediately
disperse the would-be SW so the next layer's residual carries nothing?

Test: forward pass on the standard probe, record per-block at row r (one of
the detected rows, default 3776) the MLP contribution at the SW token
position and the residual carry-in. If both stay small or mlp_out gets
absorbed (residual_out << residual_in + mlp_out), confirms the
"mixer absorbs" explanation.

Output: results/sw_residual_attribution_evo1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--row",     type=int, default=3776)
    p.add_argument("--out",     default="results/sw_residual_attribution_evo1.json")
    p.add_argument("--probe",   default="actb_500")
    args = p.parse_args()

    import sys; sys.path.insert(0, str(ROOT))
    from probes.dna_probes import get_probe
    from evo import Evo

    print("[evo1_resid_attr] loading Evo1 …")
    evo = Evo("evo-1-8k-base", device="cuda")
    model = evo.model
    tokenizer = evo.tokenizer
    n_blocks = len(model.blocks)
    print(f"[evo1_resid_attr] n_blocks = {n_blocks}, target row = {args.row}")

    seq = get_probe(args.probe)
    device = next(model.parameters()).device
    ids = torch.tensor(list(tokenizer.tokenize(seq)),
                       dtype=torch.long, device=device).unsqueeze(0)

    records: list[dict] = [{} for _ in range(n_blocks)]

    # Hook each block.mlp.l3 (the down projection). Pre-hook captures the
    # input to l3 — that's the post-GLU activation. We also need the
    # block's residual stream before and after; the simplest proxy is to
    # also hook the block forward. Cleanest: hook each block forward to
    # see residual_in (input) and residual_out (output).
    def make_block_hook(li):
        def hook(module, inputs, output):
            # inputs[0] is residual stream entering the block (B, L, H)
            # output is residual stream exiting the block (potentially tuple)
            res_in  = inputs[0]
            res_out = output[0] if isinstance(output, tuple) else output
            r = args.row
            if res_in.dim() == 3:
                ri = res_in[0]; ro = res_out[0]
            else:
                ri = res_in; ro = res_out
            # SW token position: where |res_out[:, r]| is largest
            sw_pos = int(ro[:, r].abs().argmax().item())
            records[li]["block_residual_in"]  = float(ri[sw_pos, r].item())
            records[li]["block_residual_out"] = float(ro[sw_pos, r].item())
            records[li]["block_delta"]        = float((ro[sw_pos, r] - ri[sw_pos, r]).item())
            records[li]["sw_pos"]             = sw_pos
            records[li]["max_abs_at_pos"]     = float(ro[sw_pos].abs().max().item())
            records[li]["argmax_at_pos"]      = int(ro[sw_pos].abs().argmax().item())
        return hook

    def make_mlp_hook(li):
        def hook(module, inputs, output):
            # module = block.mlp; output is the MLP block's contribution
            # before residual add (StripedHyena's mlp is a pure feedforward).
            r = args.row
            mlp_out = output[0] if isinstance(output, tuple) else output
            if mlp_out.dim() == 3:
                m = mlp_out[0]
            else:
                m = mlp_out
            # Use the same SW position as the block-hook will record
            # (block hook runs after, so we save the row 603 value here for
            # later cross-reference at every token position; we'll pick out
            # the SW position when the block hook fires).
            records[li]["_mlp_at_row_per_pos"] = m[:, r].detach().cpu().tolist()
            records[li]["_argmax_row_per_pos"] = m.abs().argmax(dim=-1).detach().cpu().tolist()
            records[li]["_mlp_max_abs_per_pos"] = m.abs().max(dim=-1).values.detach().cpu().tolist()
        return hook

    handles = []
    for li in range(n_blocks):
        blk = model.blocks[li]
        handles.append(blk.register_forward_hook(make_block_hook(li)))
        handles.append(blk.mlp.register_forward_hook(make_mlp_hook(li)))

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        model(ids)

    for h in handles:
        h.remove()

    # Post-process: at each layer, pick mlp_out[row, sw_pos], drop the bulky
    # per-pos arrays, keep the scalars.
    for li, r in enumerate(records):
        if not r:
            continue
        sw_pos = r["sw_pos"]
        per_pos = r.pop("_mlp_at_row_per_pos", [])
        max_per_pos = r.pop("_mlp_max_abs_per_pos", [])
        argmax_per_pos = r.pop("_argmax_row_per_pos", [])
        r["mlp_out_at_row"]      = per_pos[sw_pos] if sw_pos < len(per_pos) else None
        r["mlp_max_at_pos"]      = max_per_pos[sw_pos] if sw_pos < len(max_per_pos) else None
        r["mlp_argmax_at_pos"]   = argmax_per_pos[sw_pos] if sw_pos < len(argmax_per_pos) else None
        r["layer"]               = li

    print(f"{'layer':>5} {'pos':>5} {'res_in':>10} {'block_delta':>12} {'res_out':>10}"
          f" {'mlp_at_row':>12} {'mlp_max':>10} {'mlp_argmax':>10}")
    print("-" * 100)
    for r in records:
        if not r:
            continue
        print(f"{r['layer']:5d} {r['sw_pos']:5d} "
              f"{r['block_residual_in']:10.3f} "
              f"{r['block_delta']:12.3f} "
              f"{r['block_residual_out']:10.3f} "
              f"{(r.get('mlp_out_at_row') or 0.0):12.3f} "
              f"{(r.get('mlp_max_at_pos') or 0.0):10.3f} "
              f"{(r.get('mlp_argmax_at_pos') or -1):10d}")

    out_doc = {
        "model": "evo1",
        "probe": args.probe,
        "row":   args.row,
        "per_layer": records,
        "note": ("block_residual_in/out are the residual stream values at "
                 "(row, sw_pos). block_delta = res_out - res_in is the total "
                 "block contribution (attn/conv mixer + MLP combined). "
                 "mlp_at_row is the MLP's signed contribution at (row, sw_pos). "
                 "If block_delta is small even when mlp_at_row is large, the "
                 "conv mixer + LN are actively suppressing the row.")
    }
    out_path = ROOT / args.out
    out_path.write_text(json.dumps(out_doc, indent=2))
    print(f"\n[evo1_resid_attr] wrote {out_path}")


if __name__ == "__main__":
    main()
