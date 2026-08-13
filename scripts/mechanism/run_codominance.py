#!/usr/bin/env python
"""
E2 — Does JOINT CRITICALITY require CO-DOMINANCE? (tested by construction, not by hunting)

The observation to explain
--------------------------
DNABERT-2 L9 has TWO co-dominant channels (norm 8.83 and 9.81, ranks 1-2 of 768) and
ablating both costs -33.76 pp. NTv3 L11 has ONE dominant channel (2369.66 vs 80.57 next,
rank 1 of 1536 with a 29.4x gap) and ablating it costs -0.02 pp. NTv3's channel is MORE
dominant, so dominance alone does not explain criticality.

Hypothesis: joint criticality requires TWO CO-DOMINANT carriers, so that either can sustain
the representation alone and only their joint removal breaks it. A lone dominant singleton
has no partner to hide behind -- the model must already be robust to it, or it would be
fragile to a single-channel failure.

Two constructive tests (no new models needed)
---------------------------------------------
  MAKE-PAIR (NTv3)  : upscale the runner-up channel to co-dominance with r1472, then ablate
                      both. If criticality APPEARS, co-dominance is sufficient.
  BREAK-PAIR (DNABERT-2): upscale one member and downscale the other so one dominates
                      (same total norm, redistributed). If criticality DISAPPEARS,
                      co-dominance is necessary.

Both keep total pair norm approximately constant, so the manipulation is a REDISTRIBUTION
of magnitude between two channels rather than an overall change of scale -- which is what
isolates co-dominance from magnitude.

Kill condition
--------------
If MAKE-PAIR does not create criticality AND BREAK-PAIR does not remove it, co-dominance is
not the operative variable and the DNABERT-2/NTv3 difference remains unexplained. Report
that plainly rather than reaching for a third story.
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
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

from run_gue_multiseed import _load_model  # noqa: E402
from run_gue_ablation import (  # noqa: E402
    GUEDataset, _resolve_module, evaluate, _task_key, _MAX_LEN, collate_fn,
)

SEED = 42


def channel_norms(model, pattern, layer, ds, device, n_batches=5, batch=8):
    mod = _resolve_module(model, pattern, layer)
    store = {}
    hh = mod.register_forward_hook(
        lambda _m, _i, o, _s=store: _s.__setitem__(
            "h", (o[0] if isinstance(o, tuple) else o).detach().float()))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, collate_fn=collate_fn)
    acc, n = None, 0
    with torch.no_grad():
        for i, b in enumerate(loader):
            if i >= n_batches:
                break
            model(input_ids=b["input_ids"].to(device),
                  attention_mask=b["attention_mask"].to(device))
            h = store["h"]
            h = h if h.dim() == 2 else h.reshape(-1, h.shape[-1])
            s = h.abs().sum(0).cpu().numpy()
            acc = s if acc is None else acc + s
            n += h.shape[0]
    hh.remove()
    return acc / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["make_pair", "break_pair"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="splice/reconstructed")
    ap.add_argument("--gue_root", default="/data/nvidia/data/gue/GUE")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--anchor", type=int, required=True,
                    help="dominant channel (NTv3 r1472) or first member (DNABERT-2 r264)")
    ap.add_argument("--partner", type=int, default=-1,
                    help="second channel; -1 = auto-pick the runner-up by norm")
    ap.add_argument("--max_length", type=int, default=None)
    ap.add_argument("--ratios", nargs="+", type=float, default=[1.0, 0.5, 0.25, 0.1],
                   help="break_pair: partner/anchor norm ratio to impose (1.0 = co-dominant)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / f"configs/{args.model}.yaml").read_text())
    pattern = cfg["down_proj_pattern"]
    maxlen = args.max_length or _MAX_LEN.get(_task_key(args.task), 512)
    model, tok = _load_model(cfg["model_id"], 3, None, args.device)
    tok.model_max_length = maxlen
    ds = GUEDataset(f"{args.gue_root}/{args.task}/test.csv", tok, maxlen)
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location="cpu"), strict=False)
    model.eval()
    mod = _resolve_module(model, pattern, args.layer)
    pristine = mod.weight.data.clone()

    cn = channel_norms(model, pattern, args.layer, ds, args.device)
    order = np.argsort(-cn)
    A = args.anchor
    B = args.partner if args.partner >= 0 else int([c for c in order if c != A][0])
    base = evaluate(model, ds, device=args.device)["accuracy"]
    print(f"[{args.mode}] {args.model} L{args.layer} baseline={base:.4f}")
    print(f"  anchor r{A} norm={cn[A]:.3f} (rank {int(np.where(order==A)[0][0])+1})")
    print(f"  partner r{B} norm={cn[B]:.3f} (rank {int(np.where(order==B)[0][0])+1})  "
          f"ratio partner/anchor = {cn[B]/max(cn[A],1e-9):.4f}")

    def measure(scale_a, scale_b, label):
        mod.weight.data = pristine.clone()
        mod.weight.data[A] = pristine[A] * scale_a
        mod.weight.data[B] = pristine[B] * scale_b
        rb = evaluate(model, ds, device=args.device)["accuracy"]

        def acc_zero(zs):
            sv = [(r, mod.weight.data[r].clone()) for r in zs]
            for r in zs:
                mod.weight.data[r].zero_()
            a = evaluate(model, ds, device=args.device)["accuracy"]
            for r, w in sv:
                mod.weight.data[r] = w
            return a
        da = (acc_zero([A]) - rb) * 100
        db = (acc_zero([B]) - rb) * 100
        dab = (acc_zero([A, B]) - rb) * 100
        mod.weight.data = pristine.clone()
        e = dab - (da + db)
        print(f"  {label:26s} base={rb:.4f}  A={da:+7.2f} B={db:+7.2f} "
              f"AB={dab:+7.2f}  epistasis={e:+7.2f}")
        return {"label": label, "scale_a": scale_a, "scale_b": scale_b,
                "rescaled_baseline": rb, "d_a": da, "d_b": db, "d_ab": dab,
                "epistasis": e}

    runs = [measure(1.0, 1.0, "unmodified")]
    total = float(cn[A] + cn[B])
    for ratio in args.ratios:
        # impose partner/anchor = ratio while keeping the summed norm ~constant
        target_a = total / (1.0 + ratio)
        target_b = total - target_a
        sa = target_a / max(float(cn[A]), 1e-9)
        sb = target_b / max(float(cn[B]), 1e-9)
        runs.append(measure(sa, sb, f"ratio_partner/anchor={ratio}"))

    import transformers as _tf
    out = {"task": f"E2_{args.mode}", "model": args.model, "layer": args.layer,
           "anchor": A, "partner": B, "baseline": base,
           "anchor_norm": float(cn[A]), "partner_norm": float(cn[B]),
           "observed_ratio": float(cn[B] / max(cn[A], 1e-9)),
           "total_pair_norm": total, "runs": runs,
           "provenance": {"seed": SEED, "torch": torch.__version__,
                          "transformers": _tf.__version__, "max_length": maxlen}}
    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    csvp = str(op).replace(".json", ".csv")
    with open(csvp, "w") as f:
        f.write("label,scale_a,scale_b,rescaled_baseline,d_a,d_b,d_ab,epistasis\n")
        for r in runs:
            f.write(f"{r['label']},{r['scale_a']:.6g},{r['scale_b']:.6g},"
                    f"{r['rescaled_baseline']:.4f},{r['d_a']:.3f},{r['d_b']:.3f},"
                    f"{r['d_ab']:.3f},{r['epistasis']:.3f}\n")
    print(f"\nsaved → {op}")


if __name__ == "__main__":
    main()
