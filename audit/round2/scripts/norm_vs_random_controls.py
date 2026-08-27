#!/usr/bin/env python
"""Author-requested: test whether top-5-by-norm control rows are causally distinguishable
from random control rows -- i.e. whether within-layer operator magnitude confers any
causal importance at all. Per model, per epsilon: median top-5-by-norm control effect
(4b) vs median random-control effect (original E13 census). Paired across models (same
model measured both ways), Wilcoxon signed-rank test since n is small; also reports the
raw distribution of paired differences directly, since n may be too small to trust a
single p-value.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
RAW_4B = ROOT / "results/E13/raw_4b"
OUT = ROOT / "audit/round2/norm_controls_vs_random_controls.csv"

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

census = {r["model"]: r
          for r in csv.DictReader(open(ROOT / "results/E13/part2_22_model_results.csv"))}


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
        ctrls = [c for c in conds if c["kind"] == "control" and c["epsilon"] == eps]
        med_topk = median([c["relative_loss_change"] for c in ctrls])
        eps_tag = "eps0p5" if eps == 0.5 else "eps1p0"
        med_random = float(census[model][f"median_control_relative_loss_change_{eps_tag}"])
        rows_out.append(dict(
            model=model, epsilon=eps,
            median_topk_control_effect=med_topk,
            median_random_control_effect=med_random,
            difference_topk_minus_random=med_topk - med_random,
        ))

rows_out.sort(key=lambda r: (r["epsilon"], r["model"]))
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)
print(f"wrote {OUT} ({len(rows_out)} rows, {len(set(r['model'] for r in rows_out))} models)")

for eps in [0.5, 1.0]:
    sub = [r for r in rows_out if r["epsilon"] == eps]
    diffs = np.array([r["difference_topk_minus_random"] for r in sub])
    n = len(diffs)
    print(f"\n--- eps={eps}, n={n} models ---")
    print(f"  paired differences (topK control - random control): {[f'{d:+.5f}' for d in diffs]}")
    print(f"  mean diff={diffs.mean():.5f}  median diff={np.median(diffs):.5f}  std={diffs.std():.5f}")
    print(f"  median topK control effect: {np.median([r['median_topk_control_effect'] for r in sub]):.5f}")
    print(f"  median random control effect: {np.median([r['median_random_control_effect'] for r in sub]):.5f}")
    if n >= 6 and not np.all(diffs == 0):
        try:
            stat, p = stats.wilcoxon(diffs)
            print(f"  Wilcoxon signed-rank test (paired, topK vs random): statistic={stat:.3f}, p={p:.4f}")
        except ValueError as e:
            print(f"  Wilcoxon test not applicable: {e}")
    else:
        print(f"  n={n} too small (or all-zero diffs) for a paired test -- reporting distribution only")
