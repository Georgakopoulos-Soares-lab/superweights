#!/usr/bin/env python3
"""Build the frozen E13 structural-candidate manifest from E11 artifacts only.

E11 has one primary detected candidate per cohort row.  Phi-3 is the sole row whose E11
entry explicitly aggregates a six-component published structural basis; those six rows are
expanded here from ``results/E7/e7_phi3_spectral.json`` and their layer-relative norms are
joined from ``results/E13/phi3_layer_median_breakdown.json``.  No causal artifact is read.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from panel import PANEL_ORDER

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "results" / "E11" / "scale_ladder_backfilled.csv"
PHI_SPECTRAL = ROOT / "results" / "e7_phi3_spectral.json"
PHI_NORMS = ROOT / "results" / "E13" / "phi3_layer_median_breakdown.json"
OUT = ROOT / "results" / "E13" / "candidate_manifest.json"


def finite_or_none(value: str):
    value = value.strip()
    if not value or value.lower() == "nan":
        return None
    return float(value)


def main() -> None:
    rows = list(csv.DictReader(SOURCE.open()))
    assert [r["model"] for r in rows] == PANEL_ORDER
    candidates = []
    for panel_index, row in enumerate(rows):
        if row["model"] == "Phi-3-mini-4k-instruct":
            continue
        candidates.append({
            "model": row["model"],
            "panel_index": panel_index,
            "candidate_index": 0,
            "primary": True,
            "layer": int(row["layer"]),
            "row": int(row["row"]),
            "q1": finite_or_none(row["q1"]),
            "frob_norm": finite_or_none(row["frob_norm"]),
            "layer_median_frob_norm": finite_or_none(row["layer_median_frob_norm"]),
            "frob_ratio_to_layer_median": finite_or_none(
                row["candidate_frob_norm_ratio_to_layer_median"]
            ),
            "source": row["source"],
        })

    spectral = json.loads(PHI_SPECTRAL.read_text())
    norms = json.loads(PHI_NORMS.read_text())
    norm_rows = norms.get("per_row", norms)
    phi_idx = PANEL_ORDER.index("Phi-3-mini-4k-instruct")
    for candidate_index, row in enumerate(spectral["rows"]):
        key = f"L{row['layer']}/r{row['row']}"
        norm = norm_rows[key]
        candidates.append({
            "model": "Phi-3-mini-4k-instruct",
            "panel_index": phi_idx,
            "candidate_index": candidate_index,
            "primary": candidate_index == 0,
            "layer": int(row["layer"]),
            "row": int(row["row"]),
            "q1": float(row["q1"]),
            "frob_norm": float(row["frob_norm"]),
            "layer_median_frob_norm": float(norm["layer_median_frob_norm"]),
            "frob_ratio_to_layer_median": float(
                norm["candidate_frob_norm_ratio_to_layer_median"]
            ),
            "source": str(PHI_SPECTRAL.relative_to(ROOT)),
        })

    candidates.sort(key=lambda x: (x["panel_index"], x["candidate_index"]))
    counts = {m: 0 for m in PANEL_ORDER}
    for c in candidates:
        counts[c["model"]] += 1
    assert len(candidates) == 28
    assert counts["Phi-3-mini-4k-instruct"] == 6
    assert all(n == 1 for m, n in counts.items() if m != "Phi-3-mini-4k-instruct")

    d_model = {r["model"]: int(r["d_model"]) for r in rows}
    control_sets = []
    seed_seqs = np.random.SeedSequence(42).spawn(len(PANEL_ORDER))
    for panel_index, model in enumerate(PANEL_ORDER):
        # One deterministic stream per model, consumed in increasing layer order.  For the
        # only multi-layer basis (Phi-3), this gives distinct panels at L2 and L4 while
        # retaining the preregistered spawn(23)[panel_index] convention.
        rng = np.random.default_rng(seed_seqs[panel_index])
        layers = sorted({c["layer"] for c in candidates if c["model"] == model})
        for layer in layers:
            excluded = sorted(
                c["row"] for c in candidates if c["model"] == model and c["layer"] == layer
            )
            pool = np.setdiff1d(np.arange(d_model[model]), np.asarray(excluded), assume_unique=True)
            controls = sorted(int(x) for x in rng.choice(pool, size=5, replace=False))
            control_sets.append({
                "model": model,
                "panel_index": panel_index,
                "layer": layer,
                "excluded_candidate_rows": excluded,
                "control_rows": controls,
            })

    payload = {
        "provenance": "E11 structural inputs only; built before Stage 1 causal measurement",
        "source_csv": str(SOURCE.relative_to(ROOT)),
        "panel_order": PANEL_ORDER,
        "candidate_counts": counts,
        "n_models": len(counts),
        "n_candidates": len(candidates),
        "candidates": candidates,
        "control_sets": control_sets,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}: {len(counts)} models, {len(candidates)} candidates")


if __name__ == "__main__":
    main()
