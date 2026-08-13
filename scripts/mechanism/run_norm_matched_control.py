#!/usr/bin/env python
"""
FALSIFICATION TEST — is the splice collapse about SUPER WEIGHTS, or just about NORM?

Motivation
----------
We found the compensation mechanism is *joint norm carriage*: DNABERT-2's L9 pair
(r264+r294) together carry 58% of the layer-9 residual-stream norm, and removing both
collapses the norm (17.20 -> 7.14) and hence the downstream LayerNorm scaling.

That mechanism is a MAGNITUDE property. It says nothing about super-weights per se. And our
random-pair control drew uniformly from all 768 rows -- overwhelmingly low-norm channels --
so it cannot distinguish "the SW pair is special" from "any two big-norm channels are".

This is the same class of confound we already caught once (the "shadow redundancy" result
turned out to be layer depth, because prox_far selected all of layer 0). Applying the same
discipline to our own strongest claim.

Design
------
  1. Rank all layer-9 channels by their contribution to the residual-stream norm
     (mean |activation| over the splice test set). Report where the SW rows rank.
  2. Build NORM-MATCHED control pairs: non-SW channel pairs whose combined norm
     contribution is closest to the SW pair's.
  3. Compare epistasis: SW pair vs norm-matched pairs vs uniform-random pairs.

Decision rule
-------------
  * SW pair >> norm-matched pairs  -> the effect is super-weight-specific. Central claim
    survives its strongest test.
  * SW pair ~= norm-matched pairs  -> KILL. The effect is a norm effect; "super weight" is
    the wrong frame and the paper must be rewritten around high-norm channel pairs. The SW
    detector would merely be an (imperfect) high-norm channel finder.
"""
from __future__ import annotations

import argparse
import json
import random
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
    GUEDataset, _resolve_module, _save_row, _zero_row, _restore_row,
    evaluate, _task_key, _MAX_LEN, collate_fn,
)

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dnabert2")
    ap.add_argument("--task", default="splice/reconstructed")
    ap.add_argument("--gue_root", default="/data/nvidia/data/gue/GUE")
    ap.add_argument("--ckpt_root", default="results/gue_checkpoints_multiseed/dnabert2_reconstructed")
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--pair", nargs=2, default=["264", "294"])
    ap.add_argument("--n_matched", type=int, default=8)
    ap.add_argument("--n_uniform", type=int, default=8)
    ap.add_argument("--max_seeds", type=int, default=3)
    ap.add_argument("--profile_batches", type=int, default=6)
    ap.add_argument("--out", default="results/mechanism/norm_matched_control.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    A, B = int(args.pair[0]), int(args.pair[1])
    cfg = yaml.safe_load((ROOT / f"configs/{args.model}.yaml").read_text())
    pattern = cfg["down_proj_pattern"]
    maxlen = _MAX_LEN.get(_task_key(args.task), 512)

    seed_dirs = sorted([d for d in (ROOT / args.ckpt_root).iterdir()
                        if d.is_dir() and (d / "model_state.pt").exists()])[:args.max_seeds]
    out = {"task": "norm_matched_falsification", "layer": args.layer,
           "sw_pair": [A, B], "seed": SEED, "per_seed": {}}

    for sd in seed_dirs:
        model, tok = _load_model(cfg["model_id"], 3, None, args.device)
        tok.model_max_length = maxlen
        ds = GUEDataset(f"{args.gue_root}/{args.task}/test.csv", tok, maxlen)
        model.load_state_dict(torch.load(sd / "model_state.pt", map_location="cpu"),
                              strict=False)
        model.eval()
        base = evaluate(model, ds, device=args.device)["accuracy"]

        # ── 1. per-channel norm contribution at the target layer ──────────────
        mod = _resolve_module(model, pattern, args.layer)
        store = {}
        hh = mod.register_forward_hook(
            lambda _m, _i, o, _s=store: _s.__setitem__(
                "h", (o[0] if isinstance(o, tuple) else o).detach().float()))
        loader = torch.utils.data.DataLoader(ds, batch_size=32, collate_fn=collate_fn)
        acc = None
        n = 0
        with torch.no_grad():
            for i, b in enumerate(loader):
                if i >= args.profile_batches:
                    break
                model(input_ids=b["input_ids"].to(args.device),
                      attention_mask=b["attention_mask"].to(args.device))
                h = store["h"]
                h = h if h.dim() == 2 else h.reshape(-1, h.shape[-1])
                s = h.abs().sum(0).cpu().numpy()
                acc = s if acc is None else acc + s
                n += h.shape[0]
        hh.remove()
        chan_norm = acc / max(n, 1)
        order = np.argsort(-chan_norm)
        rank = {int(c): int(np.where(order == c)[0][0]) + 1 for c in (A, B)}
        pair_norm = chan_norm[A] + chan_norm[B]
        print(f"\n{sd.name}: baseline={base:.4f}")
        print(f"  channel-norm rank: r{A}={rank[A]}/{len(chan_norm)}  "
              f"r{B}={rank[B]}/{len(chan_norm)}   pair_norm={pair_norm:.3f}")
        print(f"  top-5 channels by norm: "
              f"{[(int(c), round(float(chan_norm[c]),2)) for c in order[:5]]}")

        # ── 2. norm-matched control pairs ─────────────────────────────────────
        cands = [int(c) for c in order if c not in (A, B)]
        pairs = []
        for i in range(len(cands)):
            for j in range(i + 1, min(i + 12, len(cands))):
                p, q = cands[i], cands[j]
                pairs.append((abs((chan_norm[p] + chan_norm[q]) - pair_norm), p, q))
        pairs.sort()
        matched = [(p, q) for _d, p, q in pairs[:args.n_matched]]
        rng = random.Random(SEED)
        uniform = [tuple(rng.sample(cands, 2)) for _ in range(args.n_uniform)]

        def eps(p, q):
            def acc_with(cs):
                sv = [(args.layer, c, _save_row(model, pattern, args.layer, c)) for c in cs]
                for c in cs:
                    _zero_row(model, pattern, args.layer, c)
                a = evaluate(model, ds, device=args.device)["accuracy"]
                for l, c, s in sv:
                    _restore_row(model, pattern, l, c, s)
                return a
            da = (acc_with([p]) - base) * 100
            db = (acc_with([q]) - base) * 100
            dab = (acc_with([p, q]) - base) * 100
            return dab - (da + db), dab, da, db

        e_sw, dab_sw, da_sw, db_sw = eps(A, B)
        print(f"  SW pair          epistasis={e_sw:+8.2f}  (double={dab_sw:+.2f})")
        m_res = []
        for p, q in matched:
            e, dab, da, db = eps(p, q)
            m_res.append({"pair": [p, q],
                          "pair_norm": float(chan_norm[p] + chan_norm[q]),
                          "epistasis_pp": e, "double_pp": dab})
            print(f"  norm-matched {p:>4},{q:<4} epistasis={e:+8.2f}  "
                  f"(double={dab:+.2f}, norm={chan_norm[p]+chan_norm[q]:.3f})")
        u_res = []
        for p, q in uniform:
            e, dab, da, db = eps(p, q)
            u_res.append({"pair": [p, q], "epistasis_pp": e, "double_pp": dab})
        print(f"  uniform-random   epistasis mean="
              f"{np.mean([r['epistasis_pp'] for r in u_res]):+8.2f}")

        out["per_seed"][sd.name] = {
            "baseline": base, "sw_rank": rank, "sw_pair_norm": float(pair_norm),
            "top_channels": [[int(c), float(chan_norm[c])] for c in order[:10]],
            "sw_epistasis_pp": e_sw, "sw_double_pp": dab_sw,
            "norm_matched": m_res, "uniform_random": u_res}
        del model
        torch.cuda.empty_cache()

    sw = [v["sw_epistasis_pp"] for v in out["per_seed"].values()]
    nm = [r["epistasis_pp"] for v in out["per_seed"].values() for r in v["norm_matched"]]
    un = [r["epistasis_pp"] for v in out["per_seed"].values() for r in v["uniform_random"]]
    out["aggregate"] = {
        "sw_epistasis_mean": float(np.mean(sw)),
        "norm_matched_mean": float(np.mean(nm)), "norm_matched_sd": float(np.std(nm, ddof=1)),
        "norm_matched_min": float(np.min(nm)),
        "uniform_mean": float(np.mean(un)),
        "sw_over_norm_matched": float(np.mean(sw) / (np.mean(nm) if abs(np.mean(nm)) > 1e-9 else 1e-9)),
        "n_matched_more_extreme": int(sum(1 for x in nm if x <= np.mean(sw))),
        "n_matched_total": len(nm)}
    a = out["aggregate"]
    print("\n=== VERDICT ===")
    print(f"  SW pair epistasis      : {a['sw_epistasis_mean']:+8.2f} pp")
    print(f"  norm-matched pairs     : {a['norm_matched_mean']:+8.2f} +- {a['norm_matched_sd']:.2f} pp "
          f"(most extreme {a['norm_matched_min']:+.2f})")
    print(f"  uniform-random pairs   : {a['uniform_mean']:+8.2f} pp")
    print(f"  norm-matched pairs at least as extreme as SW: "
          f"{a['n_matched_more_extreme']}/{a['n_matched_total']}")

    import transformers as _tf
    out["provenance"] = {"torch": torch.__version__, "transformers": _tf.__version__}
    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    csvp = str(op).replace(".json", ".csv")
    with open(csvp, "w") as f:
        f.write("seed,condition,pair,pair_norm,epistasis_pp,double_pp\n")
        for s, v in out["per_seed"].items():
            f.write(f"{s},sw,\"{A},{B}\",{v['sw_pair_norm']:.4f},"
                    f"{v['sw_epistasis_pp']:.4f},{v['sw_double_pp']:.4f}\n")
            for r in v["norm_matched"]:
                f.write(f"{s},norm_matched,\"{r['pair'][0]},{r['pair'][1]}\","
                        f"{r['pair_norm']:.4f},{r['epistasis_pp']:.4f},{r['double_pp']:.4f}\n")
            for r in v["uniform_random"]:
                f.write(f"{s},uniform,\"{r['pair'][0]},{r['pair'][1]}\",,"
                        f"{r['epistasis_pp']:.4f},{r['double_pp']:.4f}\n")
    print(f"\nsaved → {op}\n        {csvp}")


if __name__ == "__main__":
    main()
