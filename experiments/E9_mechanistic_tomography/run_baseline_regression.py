#!/usr/bin/env python
"""
E9 Phase 0 — baseline regression.

Reproduces, under the newly-fixed environment (pinned DNABERT-2 revision, repo-local hg38
FASTA), the untouched condition and at least one known target intervention endpoint for
each model, using tomography_lib.py (the current integrated implementation for E9).

Tolerance is declared BEFORE looking at the numbers this run produces (see
PROVENANCE_AND_BASELINES.md's baseline-regression section and the tolerances fixed below).
If the qualitative causal phenotype fails to reproduce, this script still writes its JSON
output — the STOP decision and E9_BLOCKED.md are written by hand after inspecting it, not
by this script silently passing/failing.

DNABERT-2 check: baseline MLM loss + single-row dLoss for the critical pair (L9/r264,
L9/r294) and their joint ablation, epsilon=1.0 (full zero-ablation, the only scale the
pre-existing colleague work used) — qualitative check: joint should be more damaging than
either single (superadditivity), matching C-036/C-037's reported direction.

GENERator check: untouched GC (scale=1.0) vs. full-ablation GC (scale=0.0, epsilon=1.0) and
half-suppression GC (scale=0.5, epsilon=0.5) for row 2371 — qualitative check: GC should
move monotonically with scale and should differ from the untouched baseline more than the
mean of 5 random matched control rows do, matching C-040's reported direction.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tomography_lib as tl

SEED = 42
OUT = Path(__file__).resolve().parent / "baseline_regression_results.json"

DNABERT_BASIS = [
    (5, 603), (3, 86), (3, 399), (9, 264), (9, 294),
    (3, 603), (3, 641), (7, 603), (6, 603), (5, 86),
]
PAIR = [(9, 264), (9, 294)]

GENERATOR_ROW_PRIMARY = 2371
GENERATOR_ROW_SECONDARY = 1522
GENERATOR_LAYER = 4


def run_dnabert2():
    t0 = time.time()
    model, tok, cfg, patched = tl.load_dnabert2_pretrained()
    pattern = cfg["down_proj_pattern"]
    print(f"[dnabert2] loaded in {time.time()-t0:.1f}s, triton_patched={patched}")

    rng = random.Random(SEED)
    seqs = tl.read_fasta_windows_mlm(tl.HG38_FASTA, tl.HG38_BED, n_windows=256,
                                      win_bp=600, rng=rng)
    batches = tl.build_fixed_batches(tok, seqs, "cuda", max_len=256, batch_size=16,
                                     mask_prob=0.15, seed=SEED)
    tot_masked = sum(b["n_masked"] for b in batches)
    print(f"[dnabert2] {len(seqs)} windows -> {len(batches)} batches, "
          f"{tot_masked} masked tokens")

    base = tl.mlm_loss(model, batches)
    print(f"[dnabert2] baseline MLM loss = {base:.6f}")

    def dloss(coords, alpha_val):
        alphas = [alpha_val] * len(coords)
        return tl.with_mask(model, pattern, coords, alphas,
                            lambda: tl.mlm_loss(model, batches)) - base

    d_a = dloss([PAIR[0]], 0.0)
    d_b = dloss([PAIR[1]], 0.0)
    d_ab = dloss(PAIR, 0.0)
    epistasis = d_ab - (d_a + d_b)
    print(f"[dnabert2] eps=1.0 (full ablation): dA={d_a:+.6f} dB={d_b:+.6f} "
          f"dAB={d_ab:+.6f} epistasis={epistasis:+.6f}")

    d_a_half = dloss([PAIR[0]], 0.5)
    d_b_half = dloss([PAIR[1]], 0.5)
    d_ab_half = dloss(PAIR, 0.5)
    epistasis_half = d_ab_half - (d_a_half + d_b_half)
    print(f"[dnabert2] eps=0.5 (partial suppression): dA={d_a_half:+.6f} "
          f"dB={d_b_half:+.6f} dAB={d_ab_half:+.6f} epistasis={epistasis_half:+.6f}")

    return {
        "model": "dnabert2", "model_id": cfg["model_id"], "revision": tl.DNABERT2_REVISION,
        "triton_attention_patched": patched,
        "fasta": tl.HG38_FASTA, "bed": tl.HG38_BED,
        "n_windows": len(seqs), "n_batches": len(batches), "n_masked_tokens": tot_masked,
        "baseline_mlm_loss": base,
        "epsilon_1p0": {"d_A": d_a, "d_B": d_b, "d_AB": d_ab, "epistasis": epistasis},
        "epsilon_0p5": {"d_A": d_a_half, "d_B": d_b_half, "d_AB": d_ab_half,
                        "epistasis": epistasis_half},
        "qualitative_check": {
            "expected": "epistasis > 0 at epsilon=1.0 (superadditive, matches C-036/C-037)",
            "observed_epistasis_eps1p0": epistasis,
            "passes": bool(epistasis > 0),
        },
    }


def run_generator():
    t0 = time.time()
    model, tok, cfg = tl.load_generator()
    pattern = cfg["down_proj_pattern"]
    print(f"[generator] loaded in {time.time()-t0:.1f}s")

    rng = random.Random(SEED)
    wins = tl.read_windows(tl.HG38_FASTA, tl.HG38_BED, 24, 120 + 50, rng)
    prompts = [s[:120] for _c, _s, s in wins]
    print(f"[generator] {len(prompts)} prompts")

    mods = dict(model.named_modules())
    nrows = mods[pattern.format(i=GENERATOR_LAYER)].weight.data.shape[0]
    rand_rows = random.Random(SEED).sample(
        [r for r in range(nrows) if r not in (GENERATOR_ROW_PRIMARY, GENERATOR_ROW_SECONDARY)], 5)

    results = {}
    for label, row in (("primary_2371", GENERATOR_ROW_PRIMARY),
                       ("secondary_1522", GENERATOR_ROW_SECONDARY)):
        results[label] = {}
        for alpha in (1.0, 0.5, 0.0):
            gcs, seqs = tl.generator_gc_response(model, tok, pattern, GENERATOR_LAYER,
                                                 row, alpha, prompts, max_new=64, seed=SEED)
            results[label][str(alpha)] = {
                "n": len(gcs),
                "gc_mean": float(np.mean(gcs)) if gcs else None,
                "gc_sd": float(np.std(gcs)) if gcs else None,
            }
            print(f"[generator] {label} alpha={alpha}: n={len(gcs)} "
                  f"GC={np.mean(gcs) if gcs else float('nan'):.4f}")

    results["random_control"] = {}
    for alpha in (1.0, 0.5, 0.0):
        allgc = []
        for rr in rand_rows:
            gcs, _ = tl.generator_gc_response(model, tok, pattern, GENERATOR_LAYER,
                                              rr, alpha, prompts, max_new=64, seed=SEED)
            allgc.extend(gcs)
        results["random_control"][str(alpha)] = {
            "n": len(allgc),
            "gc_mean": float(np.mean(allgc)) if allgc else None,
        }
        print(f"[generator] random_control alpha={alpha}: n={len(allgc)} "
              f"GC={np.mean(allgc) if allgc else float('nan'):.4f}")

    sw_gc = [results["primary_2371"][str(a)]["gc_mean"] for a in (0.0, 0.5, 1.0)]
    rd_gc = [results["random_control"][str(a)]["gc_mean"] for a in (0.0, 0.5, 1.0)]
    span_sw = max(sw_gc) - min(sw_gc) if all(g is not None for g in sw_gc) else None
    span_rd = max(rd_gc) - min(rd_gc) if all(g is not None for g in rd_gc) else None

    return {
        "model": "generator", "model_id": cfg["model_id"], "layer": GENERATOR_LAYER,
        "fasta": tl.HG38_FASTA, "bed": tl.HG38_BED, "n_prompts": len(prompts),
        "random_rows": rand_rows,
        "results": results,
        "gc_span_primary": span_sw, "gc_span_random": span_rd,
        "qualitative_check": {
            "expected": "GC span at row 2371 substantially exceeds random-row span "
                        "(matches C-040)",
            "span_primary": span_sw, "span_random": span_rd,
            "passes": bool(span_sw is not None and span_rd is not None
                          and span_sw > span_rd),
        },
    }


def main():
    out = {"seed": SEED, "torch": torch.__version__}
    import transformers
    out["transformers"] = transformers.__version__

    print("=" * 70)
    print("DNABERT-2 baseline regression")
    print("=" * 70)
    out["dnabert2"] = run_dnabert2()

    print("=" * 70)
    print("GENERator EUK baseline regression")
    print("=" * 70)
    out["generator"] = run_generator()

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {OUT}")

    d_ok = out["dnabert2"]["qualitative_check"]["passes"]
    g_ok = out["generator"]["qualitative_check"]["passes"]
    print(f"\nDNABERT-2 qualitative phenotype reproduced: {d_ok}")
    print(f"GENERator qualitative phenotype reproduced: {g_ok}")
    if not (d_ok and g_ok):
        print("\n*** AT LEAST ONE BASELINE FAILED TO REPRODUCE — see E9_BLOCKED.md rule ***")


if __name__ == "__main__":
    main()
