#!/usr/bin/env python
"""Section 6 point 5/6: detector_provenance.csv for all 22 census models.

For each model, determine whether its frozen candidate was selected fresh under the
current ratio>=5.0 rule (source JSON carries an explicit ratio field at/above
threshold) or grandfathered from the older e7_legacy_reanalysis.json path (which
carries no ratio field at all -- confirmed by direct inspection, see AUDIT_REPORT.md
Section 6). Also records HF revision recoverability (requested vs resolved).

This does NOT re-run detection for the 21 non-DNABERT-2 models (that would require
loading all 22 models fresh) -- it reports what the existing source artifacts do and
do not contain, honestly marking NOT FOUND where a genuine activation-vs-ratio
argmax comparison would require new compute this audit did not run.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CENSUS = ROOT / "results/experiments/E13/part2_22_model_results.csv"
OUT = ROOT / "audit/detector_provenance.csv"

LEGACY_SOURCE = "results/experiments/E7/e7_legacy_reanalysis.json"


def main():
    rows = list(csv.DictReader(open(CENSUS)))
    out_rows = []
    for r in rows:
        source = r["candidate_source"]
        is_legacy = LEGACY_SOURCE in source
        src_path = ROOT / source
        ratio_field = None
        selection_rule = ""
        frozen_is_ratio_argmax = ""
        frozen_is_activation_argmax = ""
        note = ""

        if is_legacy:
            selection_rule = "activation-argmax (grandfathered, pre-ratio-rule protocol; e7_legacy_reanalysis.json carries no ratio field at all)"
            frozen_is_activation_argmax = "TRUE (by construction of the legacy selection method)"
            if r["model"] == "DNABERT-2":
                frozen_is_ratio_argmax = "FALSE (this audit's diagnostic: ratio-argmax is L8/r603 at ratio 302.02, not the frozen L5/r603 at ratio 152.73)"
                note = "Fully reconciled in this audit -- see Section 6, DNABERT2_EXECUTION_PATH_DIAGNOSTIC.md"
            else:
                frozen_is_ratio_argmax = "NOT FOUND (not verified this audit -- would require a full per-row detection re-run, not performed for the other 5 legacy models)"
                note = "Same legacy-source pattern as DNABERT-2; not independently re-verified against the current ratio rule in this audit"
        else:
            # current-protocol models (E11/E8/E7-phase1 raw JSONs) store an explicit
            # ratio/detection_ratio field at the selected candidate, at/above threshold,
            # confirming ratio-argmax selection by construction of the detector code.
            selection_rule = "ratio-argmax (current protocol: global max of out_max/median(channel_max), accept if >=5.0)"
            frozen_is_ratio_argmax = "TRUE (by construction of the current detector code)"
            frozen_is_activation_argmax = "NOT FOUND (not verified this audit -- source JSON does not retain the full per-row activation table needed to check whether the ratio-argmax coordinate also happens to be the activation-argmax coordinate)"

        out_rows.append({
            "model": r["model"], "candidate_layer": r["candidate_layer"],
            "candidate_row": r["candidate_row"], "candidate_source": source,
            "selection_rule": selection_rule,
            "frozen_candidate_is_ratio_argmax": frozen_is_ratio_argmax,
            "frozen_candidate_is_activation_argmax": frozen_is_activation_argmax,
            "requested_revision": r["requested_revision"],
            "resolved_revision": r["resolved_revision"],
            "revision_recoverable": "FALSE (no resolved commit recorded)" if r["resolved_revision"].startswith("unpinned") else "TRUE",
            "note": note,
        })

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {OUT} ({len(out_rows)} rows)")

    n_legacy = sum(1 for r in out_rows if "grandfathered" in r["selection_rule"])
    n_unrecoverable = sum(1 for r in out_rows if r["revision_recoverable"].startswith("FALSE"))
    print(f"legacy (activation-argmax, grandfathered) models: {n_legacy}/22")
    print(f"models with unrecoverable HF revision: {n_unrecoverable}/22")
    for r in out_rows:
        if r["revision_recoverable"].startswith("FALSE"):
            print(f"  unrecoverable: {r['model']}")


if __name__ == "__main__":
    main()
