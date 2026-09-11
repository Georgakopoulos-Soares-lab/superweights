#!/usr/bin/env python3
"""
experiments/E13_full_cohort_causal_census/build_backfilled_csv.py

Merges results/experiments/E11/scale_ladder.csv (untouched, frozen) with
results/experiments/E13/layer_median_backfill.json to produce results/experiments/E11/scale_ladder_backfilled.csv:
every row from the original CSV, plus filled-in layer_median_frob_norm /
candidate_frob_norm_ratio_to_layer_median for the 11 "cited" rows that originally lacked
them, plus a new `layer_median_source` column documenting exactly how each value was
obtained (verbatim from E11's own measurement for the 12 "measured" rows; one of the two
backfill methods -- "precomputed" read-back or "new" weight-only computation -- for the 11
"cited" rows, per backfill_layer_median.py's own per-model `method` string).

Phi-3-mini-4k-instruct is a special case: its scale_ladder.csv row lists 6 published rows
across 2 layers as one entry, so it has no single layer-relative ||U_k||_F scalar. Its
per-row breakdown is recorded in a separate note file
(results/experiments/E13/phi3_layer_median_breakdown.json) and the CSV row's
layer_median_frob_norm/candidate_frob_norm_ratio_to_layer_median are left blank with
`layer_median_source` pointing at that file, rather than fabricating one averaged number for
a 6-row/2-layer entry.

Does not touch scale_ladder.csv itself -- it is treated as a frozen, already-committed
artifact of E11.
"""
import csv
import json
from pathlib import Path

SALVAGE = Path(__file__).resolve().parents[2]
ROOT = SALVAGE  # repo root; harness paths are written relative to it
CSV_IN = ROOT / "results" / "experiments" / "E11" / "scale_ladder.csv"
CSV_OUT = ROOT / "results" / "experiments" / "E11" / "scale_ladder_backfilled.csv"
BACKFILL_JSON = ROOT / "results" / "experiments" / "E13" / "layer_median_backfill.json"
PHI3_NOTE = ROOT / "results" / "experiments" / "E13" / "phi3_layer_median_breakdown.json"

# csv_model -> backfill.json key
MODEL_TO_KEY = {
    "Llama-7B": "llama-7b",
    "Mistral-7B": "mistral-7b",
    "OLMo-7B-0724-hf": "olmo-7b",
    "Phi-3-mini-4k-instruct": "phi3-mini",
    "Qwen2.5-7B": "qwen25-7b",
    "MosaicBERT": "mosaicbert",
    "ModernBERT-base": "modernbert-base",
    "GENERator-EUK-3B": "generator-euk-3b",
    "GenomeOcean-4B": "genomeocean-4b",
    "DNABERT-2": "dnabert2",
    "NTv3": "ntv3",
}


def main():
    rows = list(csv.DictReader(CSV_IN.open()))
    backfill = json.loads(BACKFILL_JSON.read_text())

    fieldnames = list(rows[0].keys()) + ["layer_median_source"]

    n_filled, n_missing, n_special = 0, 0, 0
    for row in rows:
        row["layer_median_source"] = ""
        if row["source_type"] != "cited":
            row["layer_median_source"] = f"E11 own measurement ({row['source']})"
            continue
        key = MODEL_TO_KEY.get(row["model"])
        if key is None:
            n_missing += 1
            row["layer_median_source"] = "NOT BACKFILLED -- no key mapping"
            continue
        entry = backfill.get(key)
        if entry is None or "error" in entry:
            n_missing += 1
            err = entry.get("error") if entry else "no backfill entry"
            row["layer_median_source"] = f"BACKFILL FAILED: {err}"
            continue
        if key == "phi3-mini":
            # Special case: 6 rows / 2 layers, no single scalar -- see PHI3_NOTE.
            n_special += 1
            row["layer_median_source"] = (
                f"see {PHI3_NOTE.relative_to(ROOT)} (6 published rows split across "
                f"layers 2 and 4; no single layer-relative value applies to this CSV row)")
            continue
        row["layer_median_frob_norm"] = entry["layer_median_frob_norm"]
        row["candidate_frob_norm_ratio_to_layer_median"] = \
            entry["candidate_frob_norm_ratio_to_layer_median"]
        row["layer_median_source"] = f"E13 backfill: {entry['method']}"
        n_filled += 1

    with CSV_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {CSV_OUT}")
    print(f"  filled: {n_filled}  special (phi3): {n_special}  missing/failed: {n_missing}")

    # Phi-3 per-row breakdown note
    phi3 = backfill.get("phi3-mini")
    if phi3 and "per_row" in phi3:
        PHI3_NOTE.write_text(json.dumps(phi3, indent=2))
        print(f"wrote {PHI3_NOTE}")


if __name__ == "__main__":
    main()
