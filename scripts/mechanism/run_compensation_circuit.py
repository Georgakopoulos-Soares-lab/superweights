#!/usr/bin/env python
"""
T1.2 — HOW does compensation work? The circuit behind the redundant pair.

Two geometries, two predictions
-------------------------------
SAME-ROW (depth redundancy, e.g. row 603 written at layers 3/5/6/7):
    All four layers write to the SAME residual channel. If one write is removed the
    others still deposit onto the wire, so the channel VALUE AT READOUT should degrade
    gracefully with the number of surviving writes -- not collapse.
    Prediction: readout value tracks the surviving write count.

SAME-LAYER (spatial redundancy, e.g. L9 r264 + r294):
    Two DIFFERENT channels in the same layer. Compensation cannot be on the wire; it must
    be at the READER. If the fine-tuned classifier head places comparable weight on both
    coordinates, either channel alone can carry the signal, and only removing both breaks
    it.
    Prediction: |head weight| on both SW coords >> on random coords.

Predictive check (the point of the task)
----------------------------------------
Use the measured single-ablation logit shifts + the head geometry to PREDICT the
double-ablation effect, and compare to the observed value. A circuit that predicts an
effect it was not fitted to is evidence the mechanism is right.

Kill condition (stated first)
-----------------------------
If the same-row channel value does NOT track surviving writes, AND the head does not read
both same-layer coordinates preferentially, then compensation is NOT explained by these
geometries -- report the mechanism actually found (e.g. routing through attention or a
third channel). Do not force the redundancy story.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

from run_gue_multiseed import _load_model  # noqa: E402
from run_gue_ablation import (  # noqa: E402
    GUEDataset, _resolve_module, _save_row, _zero_row, _restore_row,
    evaluate, _task_key, _MAX_LEN, collate_fn,
)

SEED = 42


def _last_hidden_channel(model, batch, channel, device):
    """Mean |value| on `channel` of the final encoder hidden state (the readout wire)."""
    store = {}

    def hk(_m, _i, o, _s=store):
        h = o[0] if isinstance(o, tuple) else o
        _s["h"] = h.detach().float()

    enc = model.bert.encoder if hasattr(model, "bert") else model.backbone
    last = enc.layer[-1] if hasattr(enc, "layer") else list(enc.children())[-1]
    hh = last.register_forward_hook(hk)
    try:
        with torch.no_grad():
            model(input_ids=batch["input_ids"].to(device),
                  attention_mask=batch["attention_mask"].to(device))
    finally:
        hh.remove()
    h = store["h"]
    v = h[:, channel] if h.dim() == 2 else h[..., channel]
    return float(v.abs().mean())


def _logits(model, batch, device):
    with torch.no_grad():
        return model(input_ids=batch["input_ids"].to(device),
                     attention_mask=batch["attention_mask"].to(device)).logits.float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dnabert2")
    ap.add_argument("--task", default="splice/reconstructed")
    ap.add_argument("--gue_root", default="/data/nvidia/data/gue/GUE")
    ap.add_argument("--ckpt", default="results/gue_checkpoints_multiseed/"
                                      "dnabert2_reconstructed/seed_0/model_state.pt")
    ap.add_argument("--same_row", default="603", help="channel written at multiple layers")
    ap.add_argument("--same_row_layers", nargs="+", type=int, default=[3, 5, 6, 7])
    ap.add_argument("--same_layer_pair", nargs=2, default=["9,264", "9,294"])
    ap.add_argument("--n_batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n_random_coords", type=int, default=200)
    ap.add_argument("--out", default="results/mechanism/compensation_circuit.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    pair = [tuple(int(x) for x in v.split(",")) for v in args.same_layer_pair]
    ch_sr = int(args.same_row)
    cfg = yaml.safe_load((ROOT / f"configs/{args.model}.yaml").read_text())
    pattern = cfg["down_proj_pattern"]
    max_length = _MAX_LEN.get(_task_key(args.task), 512)

    model, tok = _load_model(cfg["model_id"], 3, None, args.device)
    tok.model_max_length = max_length
    ds = GUEDataset(f"{args.gue_root}/{args.task}/test.csv", tok, max_length)
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location="cpu"), strict=False)
    model.eval()

    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch,
                                         collate_fn=collate_fn, shuffle=False)
    batches = []
    for i, b in enumerate(loader):
        if i >= args.n_batches:
            break
        batches.append(b)
    print(f"[circuit] {len(batches)} batches x {args.batch}")

    out = {"model": args.model, "task": args.task, "ckpt": args.ckpt, "seed": SEED}

    # ── PART A: same-row (depth) redundancy on the wire ──────────────────────
    print("\n[A] SAME-ROW: does the readout value track surviving writes?")
    L = args.same_row_layers
    rows = []
    for k in range(len(L) + 1):
        for combo in combinations(L, k):
            saves = [(l, ch_sr, _save_row(model, pattern, l, ch_sr)) for l in combo]
            for l in combo:
                _zero_row(model, pattern, l, ch_sr)
            vals = [_last_hidden_channel(model, b, ch_sr, args.device) for b in batches]
            for l, r, s in saves:
                _restore_row(model, pattern, l, r, s)
            rows.append({"ablated_layers": list(combo), "n_ablated": k,
                         "n_surviving": len(L) - k,
                         "readout_abs_mean": float(np.mean(vals))})
            print(f"   ablate {str(list(combo)):<16} surviving={len(L)-k}  "
                  f"|readout ch{ch_sr}| = {np.mean(vals):10.3f}")
    base_v = [r for r in rows if r["n_ablated"] == 0][0]["readout_abs_mean"]
    for r in rows:
        r["frac_of_baseline"] = r["readout_abs_mean"] / max(base_v, 1e-9)
    byk = {}
    for r in rows:
        byk.setdefault(r["n_surviving"], []).append(r["frac_of_baseline"])
    grade = {k: float(np.mean(v)) for k, v in sorted(byk.items())}
    print(f"   mean fraction of baseline by surviving-write count: "
          f"{ {k: round(v,3) for k,v in grade.items()} }")
    # monotone in surviving writes?
    ks = sorted(grade)
    mono = all(grade[ks[i]] <= grade[ks[i + 1]] + 1e-6 for i in range(len(ks) - 1))
    out["same_row"] = {"channel": ch_sr, "layers": L, "records": rows,
                       "frac_by_surviving": grade, "monotone_in_surviving": bool(mono)}

    # ── PART B: same-layer (spatial) redundancy at the reader ────────────────
    print("\n[B] SAME-LAYER: does the classifier head read BOTH coordinates?")
    head = None
    for name in ("classifier", "cls", "score"):
        if hasattr(model, name):
            mod = getattr(model, name)
            if hasattr(mod, "weight"):
                head = mod.weight.data.float()
                break
    if head is None:
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and m.out_features <= 8:
                head = m.weight.data.float()
                print(f"   (head found at {n})")
                break
    rng = random.Random(SEED)
    d = head.shape[1]
    rand_coords = rng.sample([c for c in range(d) if c not in (pair[0][1], pair[1][1])],
                             args.n_random_coords)
    w_a = float(head[:, pair[0][1]].abs().sum())
    w_b = float(head[:, pair[1][1]].abs().sum())
    w_r = [float(head[:, c].abs().sum()) for c in rand_coords]
    pct_a = float(np.mean([w_a > x for x in w_r]) * 100)
    pct_b = float(np.mean([w_b > x for x in w_r]) * 100)
    print(f"   |head w| coord {pair[0][1]} = {w_a:.4f}  (percentile vs random: {pct_a:.0f})")
    print(f"   |head w| coord {pair[1][1]} = {w_b:.4f}  (percentile vs random: {pct_b:.0f})")
    print(f"   random coords: mean {np.mean(w_r):.4f}  sd {np.std(w_r):.4f}")
    out["same_layer"] = {"pair": [list(p) for p in pair],
                         "head_w_abs_a": w_a, "head_w_abs_b": w_b,
                         "head_w_abs_random_mean": float(np.mean(w_r)),
                         "head_w_abs_random_sd": float(np.std(w_r)),
                         "percentile_a": pct_a, "percentile_b": pct_b,
                         "ratio_a_to_random": w_a / max(float(np.mean(w_r)), 1e-9),
                         "ratio_b_to_random": w_b / max(float(np.mean(w_r)), 1e-9)}

    # ── PART C: predictive check ─────────────────────────────────────────────
    print("\n[C] PREDICTIVE CHECK: predict double-ablation from singles")
    base_acc = evaluate(model, ds, device=args.device)["accuracy"]

    def acc_with(coords):
        saves = [(l, r, _save_row(model, pattern, l, r)) for l, r in coords]
        for l, r in coords:
            _zero_row(model, pattern, l, r)
        a = evaluate(model, ds, device=args.device)["accuracy"]
        for l, r, s in saves:
            _restore_row(model, pattern, l, r, s)
        return a

    a_a = acc_with([pair[0]])
    a_b = acc_with([pair[1]])
    a_ab = acc_with(list(pair))
    d_a, d_b, d_ab = (a_a - base_acc) * 100, (a_b - base_acc) * 100, (a_ab - base_acc) * 100
    additive = d_a + d_b
    print(f"   baseline acc={base_acc:.4f}")
    print(f"   single A: {d_a:+.2f} pp   single B: {d_b:+.2f} pp   additive: {additive:+.2f} pp")
    print(f"   OBSERVED double: {d_ab:+.2f} pp   -> epistasis {d_ab-additive:+.2f} pp")

    # logit-level prediction: zero each channel in the readout and measure logit shift
    def logit_shift(coords):
        L0 = torch.cat([_logits(model, b, args.device) for b in batches])
        saves = [(l, r, _save_row(model, pattern, l, r)) for l, r in coords]
        for l, r in coords:
            _zero_row(model, pattern, l, r)
        L1 = torch.cat([_logits(model, b, args.device) for b in batches])
        for l, r, s in saves:
            _restore_row(model, pattern, l, r, s)
        return float((L1 - L0).abs().mean())

    s_a, s_b, s_ab = logit_shift([pair[0]]), logit_shift([pair[1]]), logit_shift(list(pair))
    pred_ab = s_a + s_b
    print(f"   logit shift  A={s_a:.4f}  B={s_b:.4f}  A+B(additive pred)={pred_ab:.4f}  "
          f"observed AB={s_ab:.4f}   ratio obs/pred = {s_ab/max(pred_ab,1e-9):.2f}x")
    out["predictive"] = {"baseline_acc": base_acc,
                         "d_a_pp": d_a, "d_b_pp": d_b, "d_ab_pp": d_ab,
                         "additive_pp": additive, "epistasis_pp": d_ab - additive,
                         "logit_shift_a": s_a, "logit_shift_b": s_b,
                         "logit_shift_ab_observed": s_ab,
                         "logit_shift_ab_additive_prediction": pred_ab,
                         "obs_over_pred_ratio": s_ab / max(pred_ab, 1e-9)}

    import transformers as _tf
    out["provenance"] = {"torch": torch.__version__, "transformers": _tf.__version__,
                         "n_batches": len(batches), "batch": args.batch}
    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    csvp = str(op).replace(".json", ".csv")
    with open(csvp, "w") as f:
        f.write("part,key,value\n")
        for k, v in grade.items():
            f.write(f"same_row,frac_baseline_surviving_{k},{v:.6f}\n")
        for k in ("head_w_abs_a", "head_w_abs_b", "head_w_abs_random_mean",
                  "ratio_a_to_random", "ratio_b_to_random"):
            f.write(f"same_layer,{k},{out['same_layer'][k]:.6f}\n")
        for k, v in out["predictive"].items():
            f.write(f"predictive,{k},{v:.6f}\n")
    print(f"\nsaved → {op}\n        {csvp}")


if __name__ == "__main__":
    main()
