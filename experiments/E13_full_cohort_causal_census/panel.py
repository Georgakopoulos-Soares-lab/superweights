"""
experiments/E13_full_cohort_causal_census/panel.py

The single fixed 23-model panel ordering declared in
docs/prereg/PREREG_full_cohort_causal_census.md ("Fixed cohort and panel ordering (frozen)").
This literal list is the authoritative source of truth for `panel_index` in every
`SeedSequence(42).spawn(23)` same-layer control draw this experiment makes -- the display
table in the PREREG doc is illustrative only; this file is what the code actually uses.

Generated verbatim from `results/experiments/E11/scale_ladder.csv` row order at protocol-authoring time
(2026-08-24), which already contains one row per model for the exact 23-model cohort
part_prompt.md specifies (verified name-for-name). Do not reorder after the PREREG lock.
"""
from __future__ import annotations

import csv
from pathlib import Path

SALVAGE = Path(__file__).resolve().parents[2]
ROOT = SALVAGE  # repo root; harness paths are written relative to it
SCALE_LADDER_CSV = ROOT / "results" / "experiments" / "E11" / "scale_ladder.csv"

# Frozen 2026-08-24. Matches results/experiments/E11/scale_ladder.csv row order exactly.
PANEL_ORDER = [
    "Llama-7B",
    "Mistral-7B",
    "OLMo-7B-0724-hf",
    "Phi-3-mini-4k-instruct",
    "Qwen2.5-7B",
    "MosaicBERT",
    "ModernBERT-base",
    "NTv3",
    "DNABERT-2",
    "GENERator-EUK-3B",
    "GenomeOcean-4B",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-360M",
    "HuggingFaceTB/SmolLM2-1.7B",
    "GenerTeam/GENERator-v2-prokaryote-1.2b-base",
    "GenerTeam/GENERator-v2-prokaryote-3b-base",
    "EuroBERT/EuroBERT-210m",
    "EuroBERT/EuroBERT-610m",
    "EuroBERT/EuroBERT-2.1B",
    "answerdotai/ModernBERT-large",
]
N_PANEL = len(PANEL_ORDER)
assert N_PANEL == 23


def _verify_against_csv() -> None:
    """Defensive check: PANEL_ORDER must match scale_ladder.csv's row order exactly. Run at
    import time so any future drift (e.g. someone re-sorting the CSV) is caught immediately
    rather than silently shifting every model's control-row draw."""
    if not SCALE_LADDER_CSV.exists():
        return
    rows = list(csv.DictReader(SCALE_LADDER_CSV.open()))
    csv_order = [r["model"] for r in rows]
    if csv_order != PANEL_ORDER:
        raise RuntimeError(
            "panel.py PANEL_ORDER has drifted from results/experiments/E11/scale_ladder.csv row order. "
            f"csv={csv_order}\npanel={PANEL_ORDER}\n"
            "This must not happen silently -- it would shift every model's SeedSequence(42) "
            "control-row draw. Fix by hand and re-check, do not auto-sort."
        )


_verify_against_csv()


def panel_index(model_key: str) -> int:
    return PANEL_ORDER.index(model_key)


if __name__ == "__main__":
    for i, m in enumerate(PANEL_ORDER):
        print(i, m)
    print(f"N_PANEL = {N_PANEL}")
