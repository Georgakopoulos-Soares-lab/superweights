#!/usr/bin/env python3
"""Import the exactly matching locked E10b Phi-3 tomography into E13 Stage 3.

The import is permitted reuse, not a substituted analysis: this script refuses to proceed
unless the E13 multi-component structural basis exactly equals E10b's ordered basis.  The
other 22 models each have a one-component E11 basis and are recorded as non-identifiable.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R = ROOT / "results"
E13 = R / "E13"


def main():
    manifest = json.loads((E13 / "candidate_manifest.json").read_text())
    fit = json.loads((R / "e10b_phi3_fit_results.json").read_text())
    raw = json.loads((R / "e10b_phi3_tomography_responses.json").read_text())
    design = json.loads((ROOT / "paper-salvage" / "experiments" / "E10b_phi3_tomography" /
                         "masks_phi3.json").read_text())
    phi = "Phi-3-mini-4k-instruct"
    e13_basis = [[c["layer"], c["row"]] for c in manifest["candidates"] if c["model"] == phi]
    assert e13_basis == fit["basis"] == raw["basis"] == design["basis"]
    assert raw["resolved_revision"] == "f39ac1d28e925b323eae81227eaba4464caced4e"
    assert raw["epsilons"] == [0.5, 1.0] and raw["n_windows"] == 100
    assert len(design["pools"]["fit"]) == 36
    assert len(design["pools"]["held_out"]) == 10
    rows = []
    counts = manifest["candidate_counts"]
    for model in manifest["panel_order"]:
        if model != phi:
            for eps in (0.5, 1.0):
                rows.append({"model":model,"epsilon":eps,"n_candidates":counts[model],
                             "interaction_identifiable":False,"F2_heldout_mae":"",
                             "F2_heldout_normalized_mae":"","F2_adequate":"",
                             "F3_heldout_mae":"","F3_heldout_normalized_mae":"",
                             "F3_adequate":"","F2_to_F3_relative_mae_improvement":"",
                             "bootstrap_ci95_low":"","bootstrap_ci95_high":"",
                             "bootstrap_excludes_zero":"","pair_terms_materially_improved":"",
                             "decision":"NOT_IDENTIFIABLE_ONE_COMPONENT_BASIS",
                             "provenance":"E13 candidate_manifest.json"})
            continue
        for eps in ("0.5", "1.0"):
            x = fit["by_epsilon"][eps]
            boot = x["F2_vs_F3_bootstrap"]
            f2_mae = x["F2"]["metrics"]["mae"]
            threshold = fit["thresholds"]["normalized_mae_adequate"]
            rows.append({"model":model,"epsilon":float(eps),"n_candidates":6,
                         "interaction_identifiable":True,
                         "F2_heldout_mae":x["F2"]["metrics"]["mae"],
                         "F2_heldout_normalized_mae":x["F2"]["metrics"]["normalized_mae"],
                         "F2_adequate":x["F2"]["metrics"]["normalized_mae"] <= threshold,
                         "F3_heldout_mae":x["F3"]["metrics"]["mae"],
                         "F3_heldout_normalized_mae":x["F3"]["metrics"]["normalized_mae"],
                         "F3_adequate":x["F3"]["metrics"]["normalized_mae"] <= threshold,
                         "F2_to_F3_relative_mae_improvement":x["F2_vs_F3_relative_MAE_improvement"],
                         # Existing bootstrap is on absolute MAE difference.  Divide by
                         # fixed observed F2 MAE to express the CI on the same relative-
                         # improvement scale as the reported point estimate.
                         "bootstrap_ci95_low":boot["ci_2.5"] / f2_mae,
                         "bootstrap_ci95_high":boot["ci_97.5"] / f2_mae,
                         "bootstrap_excludes_zero":boot["excludes_zero"],
                         "pair_terms_materially_improved":x["mechanical_decision"] == "PAIR_TERMS_REQUIRED",
                         "decision":x["mechanical_decision"],
                         "provenance":"results/e10b_phi3_fit_results.json + locked raw responses"})
    out = E13 / "tomography_results.csv"
    with out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    provenance={"reuse_reason":"exact basis/revision/endpoint/epsilon identity",
                "basis":e13_basis,"fit_masks":36,"calibration_masks":10,"held_out_masks":10,
                "raw_source":"results/e10b_phi3_tomography_responses.json",
                "fit_source":"results/e10b_phi3_fit_results.json",
                "design_source":"paper-salvage/experiments/E10b_phi3_tomography/masks_phi3.json"}
    (E13 / "tomography_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(f"wrote {out} ({len(rows)} model-epsilon rows)")


if __name__ == "__main__": main()
