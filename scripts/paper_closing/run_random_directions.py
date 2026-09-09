#!/usr/bin/env python3
"""
EXPERIMENT 8 — multiple damage-matched random directions at L4/r2371.

One random direction cannot support a direction-independence claim. This samples N unit
directions, scale-searches each toward the full-ablation damage level, and measures the
GC phenotype at the closest achievable damage.

Question: P(GC shift | damage matched, learned direction absent).
  nearly all matched random directions reproduce the GC shift -> generic damage-linked
  heterogeneous GC at equal damage                            -> weaken the damage-linked claim
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path("/home/nvidia/superweights")
D = ROOT / "experiments/frozen/E12_generator_degradation_control"
sys.path.insert(0, str(D))
import run_bos_mediation as bm
import e12_lib as e12

OUT = ROOT / "results/paper_closing"; OUT.mkdir(parents=True, exist_ok=True)
N_DIR, N_DMG, N_GC = 20, 40, 12
SCALES = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
TOL = 0.15


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def gc_of(seqs):
    g = [(s.count("G") + s.count("C")) / max(len(s), 1) for s in seqs if s]
    return float(np.mean(g)) if g else float("nan")


def main():
    t0 = time.time()
    model, tok, resolved = bm.load_model()
    prompts, dmg, meta = e12.build_corpora(seed=42, n_prompt=96, prompt_win_bp=170,
                                           n_damage=100, damage_win_bp=512)
    dwin, pwin = dmg[:N_DMG], prompts[:N_GC]
    mod = dict(model.named_modules())[bm.PATTERN.format(i=bm.LAYER)]
    orig = mod.weight.data[bm.ROW].clone()
    nrm = float(orig.norm())

    intact = bm.damage(model, tok, dwin, "A_intact")
    saved = e12.set_row_alpha(model, bm.PATTERN, bm.LAYER, bm.ROW, 0.0)
    ablated = bm.damage(model, tok, dwin, "A_intact")
    gc_abl = gc_of(bm.generate_for_condition(model, tok, [w[2] for w in pwin], "A_intact")
                   if False else [])
    e12.restore_row(model, bm.PATTERN, bm.LAYER, bm.ROW, saved)
    log(f"intact={intact:.5f} ablated={ablated:.5f} target_delta={ablated-intact:.5f}")

    def gc_now():
        outs = []
        for _c, _s, seq in pwin:
            ids = bm._tokenize(tok, seq)[:, :30]
            with torch.no_grad():
                o = model.generate(input_ids=ids, max_new_tokens=48, do_sample=True,
                                   top_k=50, temperature=1.0,
                                   pad_token_id=getattr(tok, "pad_token_id", 0) or 0)
            t = tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)
            outs.append("".join(c for c in t.upper() if c in "ACGT"))
        return gc_of(outs)

    torch.manual_seed(42)
    gc_intact = gc_now()
    saved = e12.set_row_alpha(model, bm.PATTERN, bm.LAYER, bm.ROW, 0.0)
    torch.manual_seed(42); gc_ablated = gc_now()
    e12.restore_row(model, bm.PATTERN, bm.LAYER, bm.ROW, saved)
    log(f"GC intact={gc_intact:.4f} ablated={gc_ablated:.4f}")

    rng = np.random.default_rng(42)
    rows = []
    for k in range(N_DIR):
        v = torch.tensor(rng.normal(size=orig.shape[0]), dtype=orig.dtype, device=orig.device)
        v = v / v.norm()
        best = None
        for c in SCALES:
            with torch.no_grad():
                mod.weight.data[bm.ROW] = v * nrm * c
            d = bm.damage(model, tok, dwin, "A_intact")
            rel = abs((d - intact) - (ablated - intact)) / max(abs(ablated - intact), 1e-9)
            if best is None or rel < best["rel_gap"]:
                best = dict(scale=c, nll=d, rel_gap=rel)
        with torch.no_grad():
            mod.weight.data[bm.ROW] = v * nrm * best["scale"]
        torch.manual_seed(42); g = gc_now()
        with torch.no_grad():
            mod.weight.data[bm.ROW] = orig.clone()
        rows.append(dict(direction=k, best_scale=best["scale"], nll=best["nll"],
                         delta_nll=best["nll"] - intact, rel_damage_gap=best["rel_gap"],
                         damage_matched=bool(best["rel_gap"] <= TOL), gc=g,
                         delta_gc_vs_intact=g - gc_intact))
        log(f"dir {k:2d} scale={best['scale']:.2f} dNLL={best['nll']-intact:+.4f} "
            f"gap={best['rel_gap']:.3f} matched={rows[-1]['damage_matched']} GC={g:.4f}")

    matched = [r for r in rows if r["damage_matched"]]
    tgt = gc_ablated - gc_intact
    repro = [r for r in matched if (tgt < 0 and r["delta_gc_vs_intact"] <= 0.5 * tgt) or
                                   (tgt > 0 and r["delta_gc_vs_intact"] >= 0.5 * tgt)]
    summ = dict(n_directions=N_DIR, n_damage_matched=len(matched),
                tolerance=TOL, gc_intact=gc_intact, gc_ablated=gc_ablated,
                target_delta_gc=tgt,
                frac_matched_reproducing_gc=(len(repro) / len(matched)) if matched else None,
                gc_matched_mean=float(np.mean([r["gc"] for r in matched])) if matched else None,
                gc_matched_sd=float(np.std([r["gc"] for r in matched])) if matched else None)
    log(f"SUMMARY {summ}")
    json.dump(dict(experiment="EXP8_random_directions", revision=resolved,
                   intact_nll=intact, ablated_nll=ablated, corpus_meta=meta,
                   rows=rows, summary=summ, elapsed_seconds=time.time() - t0),
              open(OUT / "generator_random_direction_replicates.json", "w"), indent=2)
    cols = ["direction","best_scale","nll","delta_nll","rel_damage_gap","damage_matched",
            "gc","delta_gc_vs_intact"]
    with open(OUT / "generator_random_direction_replicates.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    log("saved")


if __name__ == "__main__":
    main()
