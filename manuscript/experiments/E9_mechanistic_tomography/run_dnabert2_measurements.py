#!/usr/bin/env python
"""
E9 — DNABERT-2 mask-response measurement collection.

Runs AFTER baseline regression passes (BASELINE_REGRESSION.md) and the prereg lock
(PREREG_mechanistic_tomography_E9.md, locked). Measures the pretrained MLM loss under
every mask in masks_dnabert2.json (singletons, fit, calibration, held_out), at both frozen
epsilon values (0.5, 1.0), on the SAME fixed batches used in baseline regression (same seed,
same windows) so every condition's delta is paired against one baseline.

Does not fit anything -- that is run_fit_observers.py. This script only produces the raw
per-mask response table, which is the artifact fitting reads.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tomography_lib as tl

SEED = 42
HERE = Path(__file__).resolve().parent
BASIS = [
    (5, 603), (3, 86), (3, 399), (9, 264), (9, 294),
    (3, 603), (3, 641), (7, 603), (6, 603), (5, 86),
]
EPSILONS = [0.5, 1.0]
OUT = HERE / "dnabert2_mask_responses.json"


def main():
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

    baseline_per_batch = tl.mlm_loss_per_batch(model, batches)
    baseline = tl.aggregate_per_batch(baseline_per_batch)
    print(f"[dnabert2] baseline MLM loss = {baseline:.6f}")

    masks = json.loads((HERE / "masks_dnabert2.json").read_text())
    responses = {"pools": {}}
    t_start = time.time()
    n_done = 0
    n_total = sum(len(v) for v in masks["pools"].values()) * len(EPSILONS)

    # H6 (Phase 7): layer-9 channel norm, captured only for held_out masks -- the pool
    # H6's covariation check is evaluated on. Channels = the unique output rows in BASIS
    # (down_proj out_features == hidden_size, so "row" IS the residual channel index).
    norm_channels = sorted(set(r for _l, r in BASIS))
    hook = tl.ChannelNormHook(model, norm_channels, layer=9)

    for pool_name, vecs in masks["pools"].items():
        responses["pools"][pool_name] = []
        for a in vecs:
            row_entry = {"a": a}
            for eps in EPSILONS:
                pb = tl.dnabert2_response_per_batch(model, pattern, BASIS, a, eps, batches)
                loss = tl.aggregate_per_batch(pb)
                row_entry[f"loss_eps{eps}"] = loss
                row_entry[f"dloss_eps{eps}"] = loss - baseline
                row_entry[f"per_batch_eps{eps}"] = pb
                if pool_name == "held_out":
                    row_entry[f"layer9_channel_norm_eps{eps}"] = hook.read()
                else:
                    hook.read()  # discard, keep the accumulator from leaking across conditions
                n_done += 1
            responses["pools"][pool_name].append(row_entry)
            if n_done % 20 == 0:
                elapsed = time.time() - t_start
                print(f"  [{n_done}/{n_total}] elapsed={elapsed:.1f}s")

    _ = tl.mlm_loss_per_batch(model, batches)  # one clean pass to capture untouched norms
    baseline_norms = hook.read()
    hook.remove()

    out = {
        "seed": SEED, "basis": [list(c) for c in BASIS], "epsilons": EPSILONS,
        "model_id": cfg["model_id"], "revision": tl.DNABERT2_REVISION,
        "triton_attention_patched": patched,
        "fasta": tl.HG38_FASTA, "bed": tl.HG38_BED,
        "n_windows": len(seqs), "n_batches": len(batches), "n_masked_tokens": tot_masked,
        "baseline_mlm_loss": baseline,
        "baseline_per_batch": baseline_per_batch,
        "layer9_norm_channels": norm_channels,
        "layer9_norm_baseline": baseline_norms,
        "responses": responses,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {OUT}  ({n_done} conditions, {time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
