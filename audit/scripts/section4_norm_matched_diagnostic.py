#!/usr/bin/env python
"""Section 4 cheap diagnostic: full within-layer ||U_k||_F distribution for the
smallest 3 cached E11-panel models, reusing existing, already-validated code
(E11_scale_ladder/run_model.py's loader + gate_up_down_weights, E5_dimensionality's
exact_uk_all_rows Gram-identity implementation). No new experiment, no new model code.
Read-only with respect to the repo; only loads model weights into memory/GPU.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
E11_DIR = ROOT / "manuscript/experiments/E11_scale_ladder"
E5_DIR = ROOT / "manuscript/experiments/E5_dimensionality"
sys.path.insert(0, str(E11_DIR))
sys.path.insert(0, str(E5_DIR))
sys.path.insert(0, str(ROOT / "manuscript/experiments/E7_exact_dimensionality"))

import run_model as rm  # noqa: E402
from dimensionality_lib import exact_uk_all_rows  # noqa: E402

TARGETS = [
    ("smollm2-135m", 11, 507),
    ("qwen25-0.5b", 21, 62),
    ("smollm2-360m", 3, 87),
]

CONTROLS_CSV = ROOT / "results/E11/scale_ladder_controls.csv"


def main():
    controls_by_model = {}
    for r in csv.DictReader(open(CONTROLS_CSV)):
        controls_by_model.setdefault(r["model"], []).append(int(r["row"]))

    key_to_full = {"smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
                   "qwen25-0.5b": "Qwen/Qwen2.5-0.5B",
                   "smollm2-360m": "HuggingFaceTB/SmolLM2-360M"}

    results = []
    for key, layer, cand_row in TARGETS:
        t0 = time.time()
        spec = rm.PANEL[key]
        model, tok, cfg, device, dtype_used, resolved_rev = rm.load_model_and_tokenizer(spec)
        t_load = time.time() - t0

        stack = rm.get_decoder_stack(model, spec["model_class"])
        layer_mod = stack[layer]
        Wg, Wu, Wd = rm.gate_up_down_weights(layer_mod, spec["ffn_kind"])

        t1 = time.time()
        norms = exact_uk_all_rows(Wg, Wu, Wd, device=device)
        t_compute = time.time() - t1
        norms = norms.numpy()

        d_model = norms.shape[0]
        cand_norm = float(norms[cand_row])
        layer_median = float(np.median(norms))
        layer_max = float(np.max(norms))
        rank = int((norms > cand_norm).sum()) + 1
        percentile = 100.0 * (norms <= cand_norm).sum() / d_model
        n_within_10 = int(np.sum(np.abs(norms - cand_norm) <= 0.10 * cand_norm)) - 1
        n_within_20 = int(np.sum(np.abs(norms - cand_norm) <= 0.20 * cand_norm)) - 1

        full_key = key_to_full[key]
        ctrl_rows = controls_by_model.get(full_key, [])
        ctrl_within_20 = [r for r in ctrl_rows if abs(norms[r] - cand_norm) <= 0.20 * cand_norm]

        print(f"\n=== {full_key} (layer {layer}, d_model={d_model}) ===")
        print(f"  load: {t_load:.1f}s   Gram-identity compute (all {d_model} rows): {t_compute:.3f}s")
        print(f"  candidate row {cand_row}: norm={cand_norm:.4f}  layer_median={layer_median:.4f}  "
              f"layer_max={layer_max:.4f}  rank={rank}/{d_model}  percentile={percentile:.2f}")
        print(f"  rows within +/-10% of candidate norm: {n_within_10}   within +/-20%: {n_within_20}")
        print(f"  existing 5 control rows: {ctrl_rows}")
        print(f"  control rows within +/-20% of candidate norm: {ctrl_within_20}")

        for r in ctrl_rows:
            print(f"    control row {r}: norm={norms[r]:.4f}  pct_of_candidate={100*norms[r]/cand_norm:.2f}%")

        results.append(dict(model=full_key, layer=layer, candidate_row=cand_row, d_model=int(d_model),
                             candidate_norm=cand_norm, layer_median_norm=layer_median,
                             layer_max_norm=layer_max, candidate_rank=rank,
                             candidate_percentile=percentile, n_within_10pct=n_within_10,
                             n_within_20pct=n_within_20, control_rows=ctrl_rows,
                             control_rows_within_20pct=ctrl_within_20,
                             load_seconds=t_load, compute_seconds=t_compute, device=device,
                             dtype=dtype_used))

        del model, Wg, Wu, Wd, norms
        if device == "cuda":
            torch.cuda.empty_cache()

    out = ROOT / "audit" / "norm_matched_feasibility.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "layer", "candidate_row", "d_model", "candidate_norm",
                    "layer_median_norm", "layer_max_norm", "candidate_rank", "candidate_percentile",
                    "n_rows_within_10pct", "n_rows_within_20pct", "control_rows",
                    "control_rows_within_20pct", "load_seconds", "compute_seconds", "device", "dtype"])
        for r in results:
            w.writerow([r["model"], r["layer"], r["candidate_row"], r["d_model"],
                        r["candidate_norm"], r["layer_median_norm"], r["layer_max_norm"],
                        r["candidate_rank"], r["candidate_percentile"], r["n_within_10pct"],
                        r["n_within_20pct"], ";".join(map(str, r["control_rows"])),
                        ";".join(map(str, r["control_rows_within_20pct"])),
                        r["load_seconds"], r["compute_seconds"], r["device"], r["dtype"]])
    print(f"\nwrote {out}")
    (ROOT / "audit" / "section4_diagnostic_raw.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
