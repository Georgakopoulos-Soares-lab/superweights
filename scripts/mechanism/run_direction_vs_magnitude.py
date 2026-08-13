#!/usr/bin/env python
"""
E1 — DIRECTION or MAGNITUDE? The intervention that breaks the norm/super-weight confound.

The problem this solves
-----------------------
Our falsification attempt could not build a norm-matched control: DNABERT-2's SW channels
rank 1 and 2 of 768 by residual-norm contribution, and the best available non-SW pair falls
5.2-6.3x short of their combined norm. So "the SW pair is special" and "the two biggest-norm
channels are special" were not separable by SELECTION.

They are separable by INTERVENTION. Rescaling a down_proj row by a scalar changes that
channel's magnitude while preserving its direction exactly (the row is the vector written
into the residual stream; alpha * row writes alpha * the same direction).

  Condition DOWN : scale the SW rows so their residual-norm contribution matches a typical
                   channel. Direction preserved, magnitude normalised. Then ablate the pair.
                   -> if the collapse SURVIVES, the effect is carried by the DIRECTION
                      (a feature), not the norm.
  Condition UP   : scale two random non-SW rows up to the SW pair's norm. Then ablate them.
                   -> if a collapse APPEARS, the effect is carried by MAGNITUDE (norm), and
                      "super weight" is the wrong frame.

Both conditions are measured against their OWN rescaled baseline, because rescaling itself
perturbs the model; the quantity of interest is the epistasis, not the raw accuracy.

Decision rule
-------------
  DOWN collapses AND UP does not      -> DIRECTION / feature effect. Encoder claim defensible.
  UP collapses AND DOWN does not      -> MAGNITUDE / norm effect. "Super weight" is the wrong
                                          frame; rewrite around dominant norm carriers.
  BOTH collapse                       -> magnitude is sufficient; direction may add nothing.
  NEITHER collapses                   -> the effect needs both, i.e. it is genuinely the
                                          specific high-norm channels (report as such).

Reported per seed and aggregated. Seed 42 for all random choices.
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


def channel_norms(model, pattern, layer, ds, device, n_batches=6, batch=32):
    """Mean |activation| per residual channel at the layer output."""
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
    ap.add_argument("--model", default="dnabert2")
    ap.add_argument("--task", default="splice/reconstructed")
    ap.add_argument("--gue_root", default="/data/nvidia/data/gue/GUE")
    ap.add_argument("--ckpt_root", default="results/gue_checkpoints_multiseed/dnabert2_reconstructed")
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--pair", nargs=2, type=int, default=[264, 294])
    ap.add_argument("--n_random_pairs", type=int, default=5)
    ap.add_argument("--typical_percentile", type=float, default=50.0,
                    help="percentile of the channel-norm distribution used as 'typical'")
    ap.add_argument("--max_seeds", type=int, default=3)
    ap.add_argument("--out", default="results/mechanism/direction_vs_magnitude.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    A, B = args.pair
    cfg = yaml.safe_load((ROOT / f"configs/{args.model}.yaml").read_text())
    pattern = cfg["down_proj_pattern"]
    maxlen = _MAX_LEN.get(_task_key(args.task), 512)

    seed_dirs = sorted([d for d in (ROOT / args.ckpt_root).iterdir()
                        if d.is_dir() and (d / "model_state.pt").exists()])[:args.max_seeds]
    out = {"task": "E1_direction_vs_magnitude", "model": args.model, "layer": args.layer,
           "sw_pair": [A, B], "seed": SEED, "per_seed": {}}

    for sd in seed_dirs:
        model, tok = _load_model(cfg["model_id"], 3, None, args.device)
        tok.model_max_length = maxlen
        ds = GUEDataset(f"{args.gue_root}/{args.task}/test.csv", tok, maxlen)
        model.load_state_dict(torch.load(sd / "model_state.pt", map_location="cpu"),
                              strict=False)
        model.eval()
        mod = _resolve_module(model, pattern, args.layer)
        pristine = mod.weight.data.clone()

        cn = channel_norms(model, pattern, args.layer, ds, args.device)
        typical = float(np.percentile(cn, args.typical_percentile))
        base = evaluate(model, ds, device=args.device)["accuracy"]
        print(f"\n=== {sd.name}  baseline={base:.4f} ===")
        print(f"  norm: SW r{A}={cn[A]:.3f} r{B}={cn[B]:.3f} | "
              f"typical(p{args.typical_percentile:.0f})={typical:.4f} | "
              f"SW/typical = {cn[A]/typical:.0f}x, {cn[B]/typical:.0f}x")

        def epistasis(rows, scales=None):
            """Rescale `rows` by `scales` (in-place), then measure epistasis of ablating them.

            Returns (rescaled_baseline, d_a, d_b, d_ab, epistasis).
            """
            mod.weight.data = pristine.clone()
            if scales:
                for r, s in zip(rows, scales):
                    mod.weight.data[r] = pristine[r] * s
            rb = evaluate(model, ds, device=args.device)["accuracy"]

            def acc_zero(zs):
                sv = [(r, mod.weight.data[r].clone()) for r in zs]
                for r in zs:
                    mod.weight.data[r].zero_()
                a = evaluate(model, ds, device=args.device)["accuracy"]
                for r, w in sv:
                    mod.weight.data[r] = w
                return a
            da = (acc_zero([rows[0]]) - rb) * 100
            db = (acc_zero([rows[1]]) - rb) * 100
            dab = (acc_zero(list(rows)) - rb) * 100
            mod.weight.data = pristine.clone()
            return rb, da, db, dab, dab - (da + db)

        rec = {"baseline": base, "typical_norm": typical,
               "sw_norm": [float(cn[A]), float(cn[B])]}

        # --- reference: unmodified SW pair -----------------------------------
        rb, da, db, dab, e = epistasis((A, B))
        rec["reference"] = {"rescaled_baseline": rb, "d_a": da, "d_b": db,
                            "d_ab": dab, "epistasis": e}
        print(f"  REFERENCE (unmodified SW)   base={rb:.4f} A={da:+.2f} B={db:+.2f} "
              f"AB={dab:+.2f}  epistasis={e:+.2f}")

        # --- DOWN: SW direction preserved, magnitude normalised --------------
        sA, sB = typical / float(cn[A]), typical / float(cn[B])
        rb, da, db, dab, e = epistasis((A, B), (sA, sB))
        rec["down_scaled_sw"] = {"scales": [sA, sB], "rescaled_baseline": rb,
                                 "d_a": da, "d_b": db, "d_ab": dab, "epistasis": e}
        print(f"  DOWN  (SW -> typical norm)  scales=({sA:.2e},{sB:.2e}) base={rb:.4f} "
              f"A={da:+.2f} B={db:+.2f} AB={dab:+.2f}  epistasis={e:+.2f}")

        # --- UP: random pair raised to SW magnitude --------------------------
        rng = random.Random(SEED)
        pool = [c for c in range(pristine.shape[0]) if c not in (A, B)]
        ups = []
        for k in range(args.n_random_pairs):
            p, q = rng.sample(pool, 2)
            up_p = float(cn[A]) / max(float(cn[p]), 1e-9)
            up_q = float(cn[B]) / max(float(cn[q]), 1e-9)
            rb2, da2, db2, dab2, e2 = epistasis((p, q), (up_p, up_q))
            ups.append({"pair": [p, q], "scales": [up_p, up_q], "rescaled_baseline": rb2,
                        "d_a": da2, "d_b": db2, "d_ab": dab2, "epistasis": e2})
            print(f"  UP    (rand {p:>3},{q:<3} -> SW norm) scales=({up_p:.1f},{up_q:.1f}) "
                  f"base={rb2:.4f} AB={dab2:+.2f}  epistasis={e2:+.2f}")
        rec["up_scaled_random"] = ups
        out["per_seed"][sd.name] = rec
        del model
        torch.cuda.empty_cache()

    # ── aggregate + verdict ──────────────────────────────────────────────────
    ref = [v["reference"]["epistasis"] for v in out["per_seed"].values()]
    dn = [v["down_scaled_sw"]["epistasis"] for v in out["per_seed"].values()]
    up = [r["epistasis"] for v in out["per_seed"].values() for r in v["up_scaled_random"]]
    ref_m, dn_m, up_m = float(np.mean(ref)), float(np.mean(dn)), float(np.mean(up))
    retained = dn_m / ref_m if abs(ref_m) > 1e-9 else float("nan")
    induced = up_m / ref_m if abs(ref_m) > 1e-9 else float("nan")
    out["aggregate"] = {"reference_epistasis": ref_m, "down_scaled_epistasis": dn_m,
                        "up_scaled_epistasis_mean": up_m,
                        "up_scaled_epistasis_min": float(np.min(up)),
                        "fraction_retained_when_norm_removed": retained,
                        "fraction_induced_by_norm_alone": induced}
    if retained > 0.5 and induced < 0.2:
        verdict = "DIRECTION (feature) effect — encoder claim defensible"
    elif induced > 0.5 and retained < 0.2:
        verdict = "MAGNITUDE (norm) effect — 'super weight' is the wrong frame"
    elif retained > 0.5 and induced > 0.5:
        verdict = "BOTH — magnitude is sufficient; direction may add nothing"
    else:
        verdict = "NEITHER alone — effect requires these specific high-norm channels"
    out["verdict"] = verdict

    print("\n=== VERDICT ===")
    print(f"  reference epistasis (unmodified SW pair) : {ref_m:+8.2f} pp")
    print(f"  DOWN  (SW at typical norm, direction kept): {dn_m:+8.2f} pp "
          f"-> {100*retained:.1f}% retained")
    print(f"  UP    (random pair at SW norm)            : {up_m:+8.2f} pp "
          f"-> {100*induced:.1f}% induced  (most extreme {np.min(up):+.2f})")
    print(f"  ==> {verdict}")

    import transformers as _tf
    out["provenance"] = {"torch": torch.__version__, "transformers": _tf.__version__}
    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    csvp = str(op).replace(".json", ".csv")
    with open(csvp, "w") as f:
        f.write("seed,condition,pair,rescaled_baseline,d_a,d_b,d_ab,epistasis\n")
        for s, v in out["per_seed"].items():
            r = v["reference"]
            f.write(f"{s},reference,\"{A},{B}\",{r['rescaled_baseline']:.4f},"
                    f"{r['d_a']:.3f},{r['d_b']:.3f},{r['d_ab']:.3f},{r['epistasis']:.3f}\n")
            r = v["down_scaled_sw"]
            f.write(f"{s},down_scaled_sw,\"{A},{B}\",{r['rescaled_baseline']:.4f},"
                    f"{r['d_a']:.3f},{r['d_b']:.3f},{r['d_ab']:.3f},{r['epistasis']:.3f}\n")
            for r in v["up_scaled_random"]:
                f.write(f"{s},up_scaled_random,\"{r['pair'][0]},{r['pair'][1]}\","
                        f"{r['rescaled_baseline']:.4f},{r['d_a']:.3f},{r['d_b']:.3f},"
                        f"{r['d_ab']:.3f},{r['epistasis']:.3f}\n")
    print(f"\nsaved → {op}\n        {csvp}")


if __name__ == "__main__":
    main()
