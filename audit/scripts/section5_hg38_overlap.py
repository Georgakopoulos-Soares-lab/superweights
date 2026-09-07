#!/usr/bin/env python
"""Section 5 point 6: independently recompute the interval intersection between the
E12 96-window generation-prompt pool and the 100-window damage pool, from the stored
sampled coordinates (results/E12/raw/corpora.json), rather than trusting build_corpora's
own live assertion or E12_summary.md's docstring claim.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
d = json.loads((ROOT / "results/E12/raw/corpora.json").read_text())

print("meta:", d["meta"])
print("n_prompt_windows:", len(d["prompt_windows"]))
print("n_damage_windows:", len(d["damage_windows"]))
print("example prompt window entry:", d["prompt_windows"][0])
print("example damage window entry:", d["damage_windows"][0])


def span(w):
    # each window entry is [chrom, start, sequence] per _sample()
    c, st, sq = w[0], w[1], w[2]
    return (c, st, st + len(sq))


prompt_spans = [span(w) for w in d["prompt_windows"]]
damage_spans = [span(w) for w in d["damage_windows"]]


def overlaps(a, b):
    return a[0] == b[0] and a[1] < b[2] and b[1] < a[2]


conflicts = []
for i, dsp in enumerate(damage_spans):
    for j, psp in enumerate(prompt_spans):
        if overlaps(dsp, psp):
            conflicts.append((i, j, dsp, psp))

print(f"\nRECOMPUTED pairwise interval intersection: {len(conflicts)} overlapping pairs "
      f"out of {len(damage_spans)}x{len(prompt_spans)}={len(damage_spans)*len(prompt_spans)} checked")
if conflicts:
    print("CONFLICTS FOUND:", conflicts[:5])
else:
    print("Confirmed non-overlapping: genuinely disjoint, independently verified.")
