#!/usr/bin/env python
"""Full 22-model cohort summary for Section 4a: top-K-by-norm gap, architecture and
domain confound checks, using census_master.csv for architecture/domain/scale metadata.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
Q1_SUMMARY = ROOT / "audit/round2/topk_norm_q1_summary.csv"
CENSUS_MASTER = ROOT / "audit/census_master.csv"
OUT = ROOT / "audit/round2/full22_topk_gap_confounds.csv"

# census_master.csv model_id doesn't always match topk_norm_q1_summary.csv's "model" key
# (E11-batch uses full HF-style names, batch2/3 uses census display names) -- map both
# to the census_master.csv model_id.
NAME_MAP = {
    "Qwen/Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B": "Qwen/Qwen2.5-3B",
    "HuggingFaceTB/SmolLM2-135M": "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-360M": "HuggingFaceTB/SmolLM2-360M",
    "HuggingFaceTB/SmolLM2-1.7B": "HuggingFaceTB/SmolLM2-1.7B",
    "GenerTeam/GENERator-v2-prokaryote-1.2b-base": "GenerTeam/GENERator-v2-prokaryote-1.2b-base",
    "GenerTeam/GENERator-v2-prokaryote-3b-base": "GenerTeam/GENERator-v2-prokaryote-3b-base",
    "EuroBERT/EuroBERT-210m": "EuroBERT/EuroBERT-210m", "EuroBERT/EuroBERT-610m": "EuroBERT/EuroBERT-610m",
    "EuroBERT/EuroBERT-2.1B": "EuroBERT/EuroBERT-2.1B", "answerdotai/ModernBERT-large": "answerdotai/ModernBERT-large",
    "ModernBERT-base": "ModernBERT-base", "DNABERT-2": "DNABERT-2", "GENERator-EUK-3B": "GENERator-EUK-3B",
    "Llama-7B": "Llama-7B", "Mistral-7B": "Mistral-7B", "OLMo-7B-0724-hf": "OLMo-7B-0724-hf",
    "MosaicBERT": "MosaicBERT", "GenomeOcean-4B": "GenomeOcean-4B", "Qwen2.5-7B": "Qwen2.5-7B", "NTv3": "NTv3",
}

census = {r["model_id"]: r for r in csv.DictReader(open(CENSUS_MASTER))}

rows = list(csv.DictReader(open(Q1_SUMMARY)))
by_model = defaultdict(dict)
for r in rows:
    by_model[r["model"]].append(r) if False else None
by_model = defaultdict(dict)
for r in rows:
    by_model[r["model"]][r["coord_role"]] = float(r["q1"])

out_rows = []
for qkey, cid in NAME_MAP.items():
    d = by_model[qkey]
    cand_q1 = d["candidate"]
    topk = [v for k, v in d.items() if k != "candidate"]
    gap = cand_q1 - max(topk)
    rank = sorted([cand_q1] + topk, reverse=True).index(cand_q1) + 1
    crow = census.get(cid, {})
    out_rows.append(dict(
        model=cid, candidate_q1=cand_q1, topk_gap=gap, rank_of_6=rank,
        architecture=crow.get("architecture", ""), domain=crow.get("domain", ""),
        non_embed_params=crow.get("n_nonembedding_params", ""),
    ))

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)
print(f"wrote {OUT} ({len(out_rows)} rows)")

gaps = [r["topk_gap"] for r in out_rows]
ranks = [r["rank_of_6"] for r in out_rows]
print(f"\n=== FULL 22-MODEL COHORT ===")
print(f"rank1/6: {sum(1 for r in ranks if r==1)}/22   rank<=2/6: {sum(1 for r in ranks if r<=2)}/22")
print(f"median gap: {np.median(gaps):.4f}   mean gap: {np.mean(gaps):.4f}   range: [{min(gaps):.4f},{max(gaps):.4f}]")
print(f"n negative gap: {sum(1 for g in gaps if g<0)}/22")

print("\n=== per-model, sorted by gap ===")
for r in sorted(out_rows, key=lambda r: r["topk_gap"]):
    print(f"  {r['model']:45s} gap={r['topk_gap']:>+8.4f}  rank={r['rank_of_6']}/6  "
          f"arch={r['architecture']:8s} domain={r['domain']}")

print("\n=== CONFOUND 1: architecture (n=22) ===")
for arch in ("decoder", "encoder"):
    vals = [r["topk_gap"] for r in out_rows if r["architecture"] == arch]
    print(f"  {arch}: n={len(vals)}  median={np.median(vals):.4f}  range=[{min(vals):.4f},{max(vals):.4f}]  "
          f"n_negative={sum(1 for v in vals if v<0)}")

print("\n=== CONFOUND 1b: domain (n=22) ===")
for dom in ("text", "genomic"):
    vals = [r["topk_gap"] for r in out_rows if r["domain"] == dom]
    print(f"  {dom}: n={len(vals)}  median={np.median(vals):.4f}  range=[{min(vals):.4f},{max(vals):.4f}]  "
          f"n_negative={sum(1 for v in vals if v<0)}")

print("\n=== CONFOUND 1c: architecture x domain (n=22) ===")
for dom in ("text", "genomic"):
    for arch in ("decoder", "encoder"):
        vals = [r["topk_gap"] for r in out_rows if r["domain"] == dom and r["architecture"] == arch]
        if vals:
            print(f"  {dom}/{arch}: n={len(vals)}  median={np.median(vals):.4f}  range=[{min(vals):.4f},{max(vals):.4f}]")

print("\n=== CONFOUND 2: scale (n=22) ===")
log_params = [np.log10(float(r["non_embed_params"])) for r in out_rows]
rho, p = spearmanr(log_params, gaps)
print(f"  Spearman rho(log10 non_embed_params, topK gap) = {rho:.4f}  p={p:.4f}  n=22")
