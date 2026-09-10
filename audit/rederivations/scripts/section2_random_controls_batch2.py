#!/usr/bin/env python
"""Task 2 (full-22 Fig1C): sample 5 random same-layer control rows for the 10 batch-2
models (not in the original 12-model E11 panel that already has scale_ladder_controls.csv)
using the EXACT SAME SeedSequence(42) child-stream convention the E13 census uses
(build_candidate_manifest.py), then compute each control row's q1 via row_spectral_metrics.

Reuses section4a_batch2.py's MODELS list, load_model(), gate_up_for() unchanged -- same
loading order (cheapest first), same GPU-cleanup pattern.
"""
from __future__ import annotations

import csv
import sys
import time
import gc
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
E7_DIR = ROOT / "experiments/frozen/E7_exact_dimensionality"
E13_DIR = ROOT / "experiments/frozen/E13_full_cohort_causal_census"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(E7_DIR))
sys.path.insert(0, str(E13_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from spectral_lib import row_spectral_metrics  # noqa: E402
from panel import PANEL_ORDER  # noqa: E402
import section4a_batch2 as s4a  # noqa: E402

OUT = ROOT / "audit/rederivations/section2_random_control_q1_batch2.csv"

# d_model per model (from results/experiments/E11/scale_ladder.csv), needed for the control pool.
D_MODEL = {
    "ModernBERT-base": 768, "DNABERT-2": 768, "GENERator-EUK-3B": 3072,
    "Llama-7B": 4096, "Mistral-7B": 4096, "OLMo-7B-0724-hf": 4096,
    "MosaicBERT": 768, "GenomeOcean-4B": 3072, "Qwen2.5-7B": 3584, "NTv3": 1536,
}


def already_done(key):
    if not OUT.exists():
        return False
    return any(r["model"] == key for r in csv.DictReader(open(OUT)))


def append_csv(path, fields, rows):
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def control_rows_for(model, layer, excluded_row, d_model):
    panel_index = PANEL_ORDER.index(model)
    seed_seqs = np.random.SeedSequence(42).spawn(len(PANEL_ORDER))
    rng = np.random.default_rng(seed_seqs[panel_index])
    pool = np.setdiff1d(np.arange(d_model), np.asarray([excluded_row]), assume_unique=True)
    return sorted(int(x) for x in rng.choice(pool, size=5, replace=False))


def main():
    for spec in s4a.MODELS:
        key = spec["key"]
        if already_done(key):
            print(f"skip {key} (checkpointed)")
            continue

        t0 = time.time()
        model, tok = s4a.load_model(spec)
        dev = next(model.parameters()).device
        print(f"loaded {key} in {time.time()-t0:.1f}s")

        mods = dict(model.named_modules())
        layer, cand_row = spec["layer"], spec["row"]
        d_model = D_MODEL[key]
        ctrl_rows = control_rows_for(key, layer, cand_row, d_model)
        print(f"  {key}: L{layer} control rows (SeedSequence(42) convention) = {ctrl_rows}")

        gate, up = s4a.gate_up_for(mods, spec["pattern"], layer, spec["ffn_kind"])
        down_mod = mods[spec["pattern"].format(i=layer)]
        gate64, up64 = gate.double(), up.double()

        q1_rows = []
        for row in ctrl_rows:
            d_row = down_mod.weight[row].detach().cpu().double()
            metrics, _ = row_spectral_metrics(gate64, up64, d_row, device="cpu")
            q1_rows.append(dict(model=key, layer=layer, row=row, q1=metrics.q1,
                                 pr_spec=metrics.pr_spec, norm=metrics.frob_norm))
        append_csv(OUT, ["model", "layer", "row", "q1", "pr_spec", "norm"], q1_rows)
        ctrl_q1_strs = [f"{r['q1']:.4f}" for r in q1_rows]
        print(f"  {key}: control q1s = {ctrl_q1_strs}  "
              f"mean={np.mean([r['q1'] for r in q1_rows]):.4f}")
        print(f"[{time.time()-t0:.1f}s total for {key}]")

        del model, mods, tok, down_mod, gate, up, gate64, up64
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    print("=== BATCH2 RANDOM CONTROLS COMPLETE ===")


if __name__ == "__main__":
    main()
