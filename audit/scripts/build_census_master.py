#!/usr/bin/env python
"""Build audit/census_master.csv and audit/census_controls_structural.csv.

Source: results/experiments/E13/part2_22_model_results.csv (STORED, gitignored, mtime/sha256
provenance only -- see audit/provenance.json) + results/experiments/E13/raw/*.json (endpoint
metadata) + results/experiments/E13/candidate_manifest.json (control seed stream formula).

Fields activation_ratio / activation_max / selection_statistic_used are left
blank here by design -- they require reconciling multiple, sometimes-disagreeing
detector code paths per model, which Section 6's detector_provenance.csv does
properly and is not duplicated here (see AUDIT_REPORT.md Section 2 note).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CENSUS = ROOT / "results/experiments/E13/part2_22_model_results.csv"
RAW_DIR = ROOT / "results/experiments/E13/raw"
MANIFEST = ROOT / "results/experiments/E13/candidate_manifest.json"
OUT = ROOT / "audit"

FIELDS = ["model_id", "hf_repo", "hf_revision", "domain", "architecture",
          "n_nonembedding_params", "candidate_layer", "candidate_row",
          "selection_statistic_used", "activation_ratio", "activation_max", "q1",
          "PR_spec", "frobenius_norm", "layer_relative_frobenius", "endpoint_type",
          "n_eval_units", "batch_size", "baseline_loss",
          "R_cand_eps0.5", "R_cand_eps1.0",
          "control_1_row", "control_2_row", "control_3_row", "control_4_row", "control_5_row",
          "R_ctrl1_eps0.5", "R_ctrl2_eps0.5", "R_ctrl3_eps0.5", "R_ctrl4_eps0.5", "R_ctrl5_eps0.5",
          "R_ctrl1_eps1.0", "R_ctrl2_eps1.0", "R_ctrl3_eps1.0", "R_ctrl4_eps1.0", "R_ctrl5_eps1.0",
          "median_control_eps0.5", "median_control_eps1.0", "G_eps0.5", "G_eps1.0",
          "control_seed_stream", "source_file", "git_commit", "obtained_via"]


def main():
    rows = list(csv.DictReader(open(CENSUS)))
    manifest = json.loads(MANIFEST.read_text())
    panel_index = {c["model"]: c["panel_index"] for c in manifest["candidates"]}

    out_rows = []
    for r in rows:
        raw_path = RAW_DIR / Path(r["raw_artifact"]).name if r.get("raw_artifact") else None
        endpoint = {}
        if raw_path and raw_path.exists():
            raw = json.loads(raw_path.read_text())
            endpoint = raw.get("endpoint", {})
        n_eval = endpoint.get("n_windows") or endpoint.get("n_prompts") or endpoint.get("n_units")
        row = {
            "model_id": r["model"], "hf_repo": r["repo"], "hf_revision": r["resolved_revision"],
            "domain": r["domain"], "architecture": r["architecture"],
            "n_nonembedding_params": r["non_embed_params"],
            "candidate_layer": r["candidate_layer"], "candidate_row": r["candidate_row"],
            "selection_statistic_used": "", "activation_ratio": "", "activation_max": "",
            "q1": r["q1"], "PR_spec": r["pr_spec"], "frobenius_norm": r["frob_norm"],
            "layer_relative_frobenius": r["frob_ratio_to_layer_median"],
            "endpoint_type": endpoint.get("mask_prob") is not None and "MLM" or "causal_LM_NLL",
            "n_eval_units": n_eval, "batch_size": endpoint.get("batch_size"),
            "baseline_loss": r["baseline_loss"],
            "R_cand_eps0.5": r["candidate_relative_loss_change_eps0p5"],
            "R_cand_eps1.0": r["candidate_relative_loss_change_eps1p0"],
            "median_control_eps0.5": r["median_control_relative_loss_change_eps0p5"],
            "median_control_eps1.0": r["median_control_relative_loss_change_eps1p0"],
            "G_eps0.5": r["candidate_minus_median_control_relative_eps0p5"],
            "G_eps1.0": r["candidate_minus_median_control_relative_eps1p0"],
            "control_seed_stream": f"SeedSequence(42).spawn(23)[{panel_index.get(r['model'], '?')}]",
            "source_file": str(CENSUS.relative_to(ROOT)),
            "git_commit": "UNTRACKED (results/ gitignored; see audit/provenance.json for mtime/sha256)",
            "obtained_via": "STORED",
        }
        for i in range(1, 6):
            row[f"control_{i}_row"] = r.get(f"control{i}_row", "")
            row[f"R_ctrl{i}_eps0.5"] = r.get(f"control{i}_relative_loss_change_eps0p5", "")
            row[f"R_ctrl{i}_eps1.0"] = r.get(f"control{i}_relative_loss_change_eps1p0", "")
        out_rows.append(row)

    with open(OUT / "census_master.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {OUT/'census_master.csv'} ({len(out_rows)} rows)")

    # census_controls_structural.csv: q1 and ||U_k||_F per control row, where available.
    # part2_22_model_results.csv does not carry per-control q1/frob_norm (only causal
    # R values) -- those structural quantities exist only for the 12-model E11 cohort
    # (scale_ladder_controls.csv) plus the 2 recovered legacy models (MosaicBERT,
    # ModernBERT-base), per PART1_STRUCTURAL_MANUSCRIPT_PACKET.md Sections 2-3.
    ladder_controls = ROOT / "results/experiments/E11/scale_ladder_controls.csv"
    struct_rows = []
    if ladder_controls.exists():
        by_model = {}
        for cr in csv.DictReader(open(ladder_controls)):
            by_model.setdefault(cr["model"], []).append(cr)
        for r in rows:
            lkey = r["model"]
            if lkey and lkey in by_model:
                for i, cr in enumerate(sorted(by_model[lkey], key=lambda z: int(z["row"])), start=1):
                    struct_rows.append({"model": r["model"], "control_index": i, "control_row": cr["row"],
                                         "q1": cr["q1"], "frob_norm_U_k": cr.get("frob_norm", "")})
            else:
                struct_rows.append({"model": r["model"], "control_index": "", "control_row": "",
                                     "q1": "NOT FOUND (only 12-model E11 cohort has per-control structural spectra)",
                                     "frob_norm_U_k": ""})
    with open(OUT / "census_controls_structural.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "control_index", "control_row", "q1", "frob_norm_U_k"])
        w.writeheader()
        w.writerows(struct_rows)
    n_covered = len({r["model"] for r in struct_rows if r["control_index"] != ""})
    print(f"wrote {OUT/'census_controls_structural.csv'} ({n_covered}/22 models have per-control structural spectra)")


if __name__ == "__main__":
    main()
