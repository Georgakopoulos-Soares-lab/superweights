#!/usr/bin/env python
"""Author-requested (per-model structural vs causal comparison): for every 4b model
completed so far, report the structural top-5-by-norm q1 gap (Section 4a) side by side
with the causal top-5-by-norm gap (4b) and the causal random-control gap from the
original E13 census, at both epsilons. Rerun as more 4b models land -- reads
results/experiments/E13/raw_4b/*.json dynamically, so no model list to maintain by hand.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW_4B = ROOT / "results/experiments/E13/raw_4b"
OUT = ROOT / "audit/rederivations/structural_vs_causal_gap.csv"

FILE_TO_MODEL = {
    "dnabert2": "DNABERT-2", "generator-euk-3b": "GENERator-EUK-3B",
    "modernbert-base": "ModernBERT-base", "llama": "Llama-7B", "mistral": "Mistral-7B",
    "olmo": "OLMo-7B-0724-hf", "genomeocean-4b": "GenomeOcean-4B",
    "qwen25-7b": "Qwen2.5-7B", "mosaicbert": "MosaicBERT", "ntv3": "NTv3",
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M", "eurobert-210m": "EuroBERT/EuroBERT-210m",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M",
    "modernbert-large": "answerdotai/ModernBERT-large",
    "qwen25-0.5b": "Qwen/Qwen2.5-0.5B", "eurobert-610m": "EuroBERT/EuroBERT-610m",
    "generator-prok-1.2b": "GenerTeam/GENERator-v2-prokaryote-1.2b-base",
    "qwen25-1.5b": "Qwen/Qwen2.5-1.5B", "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B",
    "eurobert-2.1b": "EuroBERT/EuroBERT-2.1B", "qwen25-3b": "Qwen/Qwen2.5-3B",
    "generator-prok-3b": "GenerTeam/GENERator-v2-prokaryote-3b-base",
}

structural = {r["model"]: float(r["topk_gap"])
              for r in csv.DictReader(open(ROOT / "audit/rederivations/full22_topk_gap_confounds.csv"))}
census = {r["model"]: r
          for r in csv.DictReader(open(ROOT / "results/experiments/E13/part2_22_model_results.csv"))}


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 == 1 else (xs[n // 2 - 1] + xs[n // 2]) / 2


rows_out = []
for fpath in sorted(RAW_4B.glob("*.json")):
    slug = fpath.stem
    model = FILE_TO_MODEL.get(slug)
    if model is None:
        continue
    d = json.loads(fpath.read_text())
    conds = d["conditions"]
    for eps in [0.5, 1.0]:
        cand = [c for c in conds if c["kind"] == "candidate" and c["epsilon"] == eps][0]
        ctrls = [c for c in conds if c["kind"] == "control" and c["epsilon"] == eps]
        med_topk = median([c["relative_loss_change"] for c in ctrls])
        causal_topk_gap = cand["relative_loss_change"] - med_topk
        eps_tag = "eps0p5" if eps == 0.5 else "eps1p0"
        causal_random_gap = float(census[model][f"candidate_minus_median_control_relative_{eps_tag}"])
        rows_out.append(dict(
            model=model, epsilon=eps,
            structural_topk_gap_q1=structural.get(model, ""),
            causal_topk_gap=causal_topk_gap,
            causal_random_control_gap=causal_random_gap,
            candidate_relative_loss_change=cand["relative_loss_change"],
            median_topk_control_relative_loss_change=med_topk,
            median_random_control_relative_loss_change=float(census[model][f"median_control_relative_loss_change_{eps_tag}"]),
        ))

rows_out.sort(key=lambda r: (r["model"], r["epsilon"]))
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)
print(f"wrote {OUT} ({len(rows_out)} rows, {len(set(r['model'] for r in rows_out))} models)")
for r in rows_out:
    print(f"  {r['model']:20s} eps={r['epsilon']}  structural_gap={r['structural_topk_gap_q1']:>+8.4f}  "
          f"causal_topk_gap={r['causal_topk_gap']:>+9.4f}  causal_random_gap={r['causal_random_control_gap']:>+9.4f}")
