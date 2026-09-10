#!/usr/bin/env python
"""
Is the group-wise quantisation advantage SUPER-WEIGHT-specific, or general outlier handling?

The question
------------
We found that group-wise RTN (g=64) quantises DNABERT-2 to INT3 for -0.19 pp, while per-row
costs -5.93 pp and per-tensor -26.67 pp, and we explained it mechanistically: each
super-weight is the max of its own 64-element block, so group-wise preserves it exactly and
for free. That explanation makes a testable prediction.

  If the advantage is SUPER-WEIGHT-SPECIFIC, models with a genuine dominant super weight
  should show a LARGER group-wise advantage than models without one.

  If the advantage is GENERAL OUTLIER HANDLING, models with and without super weights should
  show a SIMILAR advantage -- group-wise helps because weight matrices are heavy-tailed, and
  the super weight merely rides along.

Both outcomes are informative, and the second is the honest null: group-wise helps with any
outlier structure, not only with super weights. Without this contrast we cannot distinguish
"super weights make group-wise necessary" from "group-wise is good and super weights benefit
incidentally".

Why this engages Yu et al. precisely
------------------------------------
Yu prescribes holding the super weight out of quantisation. If (a) the group-wise advantage
tracks super-weight presence and (b) SW exemption adds nothing on top of group-wise, then
Yu's exemption is redundant *specifically because group-wise already self-preserves the SW* --
a mechanistic statement about a published method, not a generic compression result.

Design
------
  readout      : LM loss on fixed probe sequences (causal LM or MLM), so all model families
                 are on one comparable scale. Reported as relative loss increase.
  granularity  : per_row | group_64 | group_128 | per_tensor
  bits         : 8, 4, 3
  weight stats : data-free outlier descriptors computed from down_proj alone --
                   sw_dominance      = global max|w| / median row-max|w|   (is there ONE row
                                       that dwarfs the rest?)
                   block_concentration = mean over g-blocks of (max|w| / mean|w|)  (are
                                       outliers isolated within blocks, i.e. is there
                                       anything for group-wise to preserve?)
  advantage    : adv_vs_tensor = damage(per_tensor) - damage(group_64)
                 adv_vs_row    = damage(per_row)    - damage(group_64)

Decision rule
-------------
  Compare adv_vs_tensor between SW-bearing models (health check verdict GENUINE) and non-SW
  models (verdict INERT / none). If the SW group's advantage is materially larger AND
  correlates with sw_dominance, the effect is super-weight-specific. If the two groups are
  comparable, the claim narrows to general outlier handling -- report that.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))

from run_sw_health_check import _encode, _loss_from, _tok_len, PROBES  # noqa: E402

SEED = 42


def quantize(W, bits, gran):
    qmax = 2 ** (bits - 1) - 1
    if gran.startswith("group_"):
        g = int(gran.split("_")[1])
        R, C = W.shape
        pad = (-C) % g
        Wp = torch.nn.functional.pad(W, (0, pad))
        Wb = Wp.view(R, -1, g)
        s = Wb.abs().amax(dim=2, keepdim=True).clamp(min=1e-12) / qmax
        return (torch.clamp(torch.round(Wb / s), -qmax - 1, qmax) * s).view(R, -1)[:, :C].contiguous()
    if gran == "per_row":
        s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / qmax
    else:
        s = (W.abs().max() / qmax).clamp(min=1e-12)
    return torch.clamp(torch.round(W / s), -qmax - 1, qmax) * s


def weight_stats(mats, g=64):
    """Data-free outlier descriptors over the down_proj matrices."""
    dom, conc = [], []
    for W in mats:
        W = W.float()
        rowmax = W.abs().amax(dim=1)
        dom.append(float(W.abs().max() / rowmax.median().clamp(min=1e-12)))
        R, C = W.shape
        pad = (-C) % g
        Wb = torch.nn.functional.pad(W, (0, pad)).view(R, -1, g).abs()
        conc.append(float((Wb.amax(dim=2) / Wb.mean(dim=2).clamp(min=1e-12)).mean()))
    return {"sw_dominance": float(np.mean(dom)), "sw_dominance_max": float(np.max(dom)),
            "block_concentration": float(np.mean(conc))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--env_note", default="")
    ap.add_argument("--bits", nargs="+", type=int, default=[8, 4, 3])
    ap.add_argument("--granularities", nargs="+",
                    default=["per_row", "group_64", "group_128", "per_tensor"])
    ap.add_argument("--pad_multiple", type=int, default=0)
    ap.add_argument("--max_layers", type=int, default=0,
                    help="0 = all layers; else quantise only the first N (memory guard)")
    ap.add_argument("--out", default="results/mechanism/granularity_multimodel.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from models import WRAPPER_MAP
    cfg = yaml.safe_load((ROOT / f"configs/{args.model}.yaml").read_text())
    w = WRAPPER_MAP[args.model](cfg)
    w.load()
    model, tok = w.model, w.tokenizer
    device = next(model.parameters()).device
    mods = dict(model.named_modules())

    nl = cfg["num_layers"]
    layers = list(range(nl if not args.max_layers else min(nl, args.max_layers)))
    targets = []
    for li in layers:
        k = cfg["down_proj_pattern"].format(i=li)
        m = mods.get(k)
        if m is not None and hasattr(m, "weight"):
            targets.append((k, m))
    if not targets:
        raise SystemExit(f"[fatal] no down_proj modules matched for {args.model}")
    print(f"[{args.model}] {len(targets)} down_proj matrices, shape {tuple(targets[0][1].weight.shape)}")

    stats = weight_stats([m.weight.data for _k, m in targets])
    print(f"  weight stats: sw_dominance={stats['sw_dominance']:.1f} "
          f"(max {stats['sw_dominance_max']:.1f})  block_conc={stats['block_concentration']:.2f}")

    # fixed probe batch, same for every condition
    probe_ids = []
    for name, seq in PROBES.items():
        s = seq * 3
        if args.pad_multiple:
            while _tok_len(tok, s) % args.pad_multiple:
                s = s + "A"
        probe_ids.append(_encode(tok, s, device)[0])

    def loss_now():
        return float(np.mean([_loss_from(model, ids) for ids in probe_ids]))

    with torch.no_grad():
        base = loss_now()
    print(f"  baseline LM loss = {base:.5f}")

    runs = {}
    for gran in args.granularities:
        for bits in args.bits:
            saved = [(m, m.weight.data.clone()) for _k, m in targets]
            for _k, m in targets:
                m.weight.data = quantize(m.weight.data, bits, gran)
            with torch.no_grad():
                lq = loss_now()
            for m, wt in saved:
                m.weight.data = wt
            rel = (lq - base) / max(abs(base), 1e-9)
            runs[f"{gran}_int{bits}"] = {"loss": lq, "rel_loss_increase": rel}
            print(f"  {gran:12s} INT{bits}  loss={lq:9.5f}  rel_increase={rel:+8.4f}")

    adv = {}
    for bits in args.bits:
        gw = runs.get(f"group_64_int{bits}")
        pt = runs.get(f"per_tensor_int{bits}")
        pr = runs.get(f"per_row_int{bits}")
        if gw and pt:
            adv[f"int{bits}_adv_vs_tensor"] = pt["rel_loss_increase"] - gw["rel_loss_increase"]
        if gw and pr:
            adv[f"int{bits}_adv_vs_row"] = pr["rel_loss_increase"] - gw["rel_loss_increase"]
    print("  group-wise advantage (positive = group_64 better):")
    for k, v in adv.items():
        print(f"    {k:24s} {v:+.4f}")

    import transformers as _tf
    rec = {"model": args.model, "env_note": args.env_note, "n_matrices": len(targets),
           "baseline_loss": base, "weight_stats": stats, "runs": runs, "advantage": adv,
           "provenance": {"seed": SEED, "torch": torch.__version__,
                          "transformers": _tf.__version__, "bits": args.bits,
                          "granularities": args.granularities}}
    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(op.read_text()) if op.exists() else {"models": {}}
    payload["models"][args.model] = rec
    op.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved → {op}")


if __name__ == "__main__":
    main()
