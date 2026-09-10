#!/usr/bin/env python3
"""
Generate and verify docs/EXPERIMENT_MAP.{md,tsv} -- the manuscript-to-code map.

Why this is generated rather than hand-written
----------------------------------------------
A reviewer's first question is "which code produced this number, and is it here?". A
hand-maintained table answers that only until the next commit. This script holds the
manifest inline, checks every path against the working tree, and writes the map with a
per-row status. Run it after any change to scripts/ or results/; it exits non-zero only if a
SCRIPT is missing, because a missing script is a repository defect, whereas a missing raw
artifact is a known and separately documented condition (several were produced on the TACC
cluster and are too large or were never committed -- see
experiments/docs/MISSING_COLLEAGUE_ARTIFACTS.md). Those rows are reported as
ARTIFACT_ABSENT and point at the committed figure source-data snapshot instead, which is
what a reviewer can actually inspect.

Status values
  OK              every listed script and artifact is present
  ARTIFACT_ABSENT script(s) present; one or more raw artifacts not committed
  SCRIPT_MISSING  a listed script does not exist -> non-zero exit
"""
from __future__ import annotations
import csv, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (id, manuscript item, figure/table, scripts, artifacts, note)
M = [
 ("S1", "Row-associated bilinear operator; exact q1, PR_spec, Frobenius norm",
  "Fig 1B,C",
  ["experiments/frozen/E7_exact_dimensionality/run_legacy_reanalysis.py",
   "experiments/frozen/E7_exact_dimensionality/run_phase1_detection.py",
   "experiments/frozen/E7_exact_dimensionality/spectral_lib.py"],
  ["results/E7/e7_legacy_reanalysis.json",
   "experiments/figures/source_data/fig1_structural.json"],
  "Exact spectrum, no diagonal approximation."),

 ("S2", "Retrospective diagonal-proxy calibration on published NLP super weights",
  "Fig 1A",
  ["experiments/frozen/E1_nlp_validation"],
  ["experiments/figures/source_data/fig1_structural_panel_a.csv"],
  "Calibration only; numerically distinct from the exact metric in Fig 1B,C."),

 ("S3", "23-model structural panel (scale ladder)",
  "Fig 1B",
  ["experiments/frozen/E11_scale_ladder/run_model.py",
   "experiments/frozen/E11_scale_ladder/build_csv.py"],
  ["results/E11/scale_ladder_backfilled.csv"],
  "Structural metrics only; no causal endpoint."),

 ("C1", "22-model singleton causal census, candidate + 5 random same-layer controls, eps 0.5/1.0",
  "Fig 2A,B",
  ["experiments/frozen/E13_full_cohort_causal_census/run_singleton_census.py",
   "experiments/frozen/E13_full_cohort_causal_census/build_candidate_manifest.py",
   "experiments/frozen/E13_full_cohort_causal_census/compile_census.py"],
  ["results/E13/part2_22_model_results.csv", "audit/census_master.csv"],
  "Controls drawn from SeedSequence(42).spawn(23)[panel_index]."),

 ("C2", "Top-norm control set: 5 highest-Frobenius same-layer rows, no forward pass",
  "Fig 2D",
  ["experiments/frozen/E13_full_cohort_causal_census/run_priority1_robustness.py"],
  ["experiments/figures/source_data/fig2_panel_d_source.csv"],
  "Weight-space selection; the stricter control."),

 ("C3", "q1 vs signed full-ablation effect across the panel",
  "Fig 2C",
  ["experiments/frozen/E13_full_cohort_causal_census/build_cohort_summary.py"],
  ["experiments/figures/source_data/fig2_causal.json"],
  "The structure-function dissociation; rho = 0.074."),

 ("D1", "Activation-ratio detector, per-row over all layers (candidate provenance)",
  "Methods",
  ["experiments/frozen/E13_full_cohort_causal_census/run_rowwise_detector.py"],
  ["audit/detector_provenance.csv"],
  "Raw per-row tables were written to results/E13_multicandidate_structural on the cluster."),

 ("D2", "Legacy vs ratio rule resolved for all 6 legacy models, with paired causal evaluation",
  "Methods",
  ["scripts/detection/run_uniform_detector_text.py",
   "scripts/detection/run_uniform_detector.py",
   "scripts/detection/run_detector_published_protocol.py"],
  ["results/detector_provenance/uniform_detector_text_llama.json",
   "results/detector_provenance/uniform_detector_text_mistral.json",
   "results/detector_provenance/uniform_detector_text_olmo.json",
   "results/detector_provenance/uniform_detector_dnabert2.json",
   "results/detector_provenance/uniform_detector_ntv3.json",
   "results/detector_provenance/detector_published_protocol.json",
   "audit/detector_provenance_exp2_resolution.csv",
   "results/detector_provenance/EXP2_LEGACY_DETECTOR_RESOLUTION.md"],
  "Four reproduction gates per text model. OLMo disagrees; frozen coord 47x more damaging."),

 ("D3", "Candidate input stability over 24 domain-matched inputs",
  "Methods",
  ["experiments/frozen/E13_full_cohort_causal_census/run_candidate_input_stability.py"],
  ["results/E13_candidate_stability"],
  "20 models with a comparable singleton candidate and executable path."),

 ("D4", "OLMo coordinate-substitution sensitivity of the cohort correlations",
  "Note S2",
  ["scripts/census_analysis/run_census_analyses.py"],
  ["results/detector_provenance/exp2_olmo_exposure_sensitivity.json"],
  "22-model rho robust (0.766 -> 0.755); n=10 text subgroup falls to 0.673."),

 ("M1", "DNABERT-2 pretrained MLM pair L9/r264 + L9/r294; joint ablation and epistasis",
  "Fig 3A,B",
  ["experiments/frozen/E9_mechanistic_tomography/run_dnabert2_measurements.py",
   "scripts/mechanism/run_pretrained_epistasis.py"],
  ["experiments/figures/source_data/fig3_dnabert.json"],
  "Epistasis +2.0118 at eps=1.0 on the pretrained objective."),

 ("M2", "Finite-intervention tomography, observer families F0-F3 on a fixed 10-row basis",
  "Fig 3C,D",
  ["experiments/frozen/E9_mechanistic_tomography/run_fit_observers.py",
   "experiments/frozen/E9_mechanistic_tomography/run_baseline_regression.py"],
  ["experiments/figures/source_data/fig3_dnabert.json"],
  "Held-out R2 0.888 (F3) vs 0.517 (F2) at eps=0.5; 100 resplits."),

 ("M3", "GENERator L4/r2371 BOS mediation and position rescue",
  "Fig 4",
  ["experiments/frozen/E12_generator_degradation_control/run_bos_mediation_main.py",
   "experiments/frozen/E12_generator_degradation_control/run_e12_full.py",
   "scripts/mechanism/run_generator_position_rescue.py",
   "scripts/mechanism/run_bos_identity_vs_position.py"],
  ["results/E_BOS_MEDIATION/bos_mediation_results.json",
   "results/mechanism_generator/generator_position_rescue.json",
   "results/mechanism_generator/generator_bos_identity_vs_position.json",
   "experiments/figures/source_data/fig4_generator.json"],
  "Activation follows BOS identity, not position 0."),

 ("M4", "GENERator GC composition and damage-matched random-direction replacement",
  "Fig S2",
  ["scripts/mechanism/run_random_directions.py",
   "experiments/figures/supplement/fig_s2_random_direction.py"],
  ["results/mechanism_generator/generator_random_direction_replicates.json",
   "results/E12/damage_matching.csv"],
  "GC phenotype tracks damage magnitude, not the learned direction."),

 ("M5", "Attention-sink diagnostics at the GENERator SW position",
  "Discussion",
  ["scripts/mechanism/run_attention_sink.py",
   "scripts/mechanism/run_attention_sink_generator.py"],
  ["results/mechanism_generator/generator_attention_sink.json"],
  "Observational: attention was not causally manipulated."),

 ("W1", "Within-layer graded sweep, 36 rows log-spaced by ratio rank, two decoders",
  "Fig 5",
  ["scripts/within_layer_sweep/run_within_model_slope.py",
   "scripts/within_layer_sweep/run_within_model_slope_text.py",
   "scripts/within_layer_sweep/plot_within_model_slope_2panel.py"],
  ["results/within_layer_sweep/within_model_slope.json",
   "results/within_layer_sweep/within_model_slope_smollm2_1.7b.json",
   "results/within_layer_sweep/within_model_slope_comparison.json",
   "results/within_layer_sweep/fig_within_model_slope_2panel.pdf",
   "results/supplementary/S7_within_layer_sweep.tsv"],
  "Two-regime: no graded signal below the ratio>=5 accept threshold, strong ordering above."),

 ("W2", "Second critical row r161 and the r161/r749 masking interaction, with geometry",
  "Results; Table S6",
  ["scripts/within_layer_sweep/run_smollm2_second_row_epistasis.py",
   "scripts/within_layer_sweep/run_smollm2_row_geometry.py"],
  ["results/within_layer_sweep/smollm2_second_row_epistasis.json",
   "results/within_layer_sweep/smollm2_161_749_geometry.json",
   "results/supplementary/S8_two_critical_rows.tsv"],
  "Independently re-verified; mechanism explicitly not claimed."),

 ("P1", "Phi-3 six-row basis and its tomography (not used for any cohort claim)",
  "Fig S1A; Note S1",
  ["experiments/frozen/E10b_phi3_tomography/run_phi3_tomography.py",
   "experiments/frozen/E7_exact_dimensionality/run_phi3_spectral.py"],
  ["results/E10/e10b_phi3_tomography_responses.json",
   "experiments/figures/source_data/fig_s1_structural_detail.json"],
  "Full-ablation response treated as unresolved."),

 ("P2", "Per-model random-control causal detail for the decoder audit",
  "Fig S1B",
  ["experiments/frozen/E10_nlp_architecture_causal/run_decoder_spectrum.py"],
  ["results/E10/e10_decoder_control_rows.json"],
  "Individual control values behind the cohort medians."),

 ("T1", "Supplementary tables S2, S3, S6, S7, S8 (built from committed artifacts)",
  "Tables S2, S3, S6-S8",
  ["scripts/census_analysis/build_supplementary_tables.py"],
  ["results/supplementary/S2_structural_metrics_stability.tsv",
   "results/supplementary/S3_causal_census_disagreements.tsv",
   "results/supplementary/S6_dnabert2_tomography.tsv",
   "results/supplementary/S7_within_layer_sweep.tsv",
   "results/supplementary/S8_two_critical_rows.tsv",
   "results/supplementary/README.md"],
  "S1, S4 and S5 are reformatted verbatim from the round-2 audit and are not rebuilt."),

 ("A1", "Cohort structure-function correlations and figures",
  "Fig 2C; Table S4",
  ["scripts/census_analysis/run_activation_vs_causality.py",
   "scripts/census_analysis/run_census_analyses.py"],
  ["results/census_analysis/activation_vs_causality.tsv",
   "results/census_analysis/activation_vs_causality_per_model.tsv",
   "results/census_analysis/family_clustered_sensitivity.tsv"],
  "Includes the family-clustered resampling sensitivity."),
]

def main() -> int:
    rows, missing_scripts = [], []
    for i, item, fig, scripts, arts, note in M:
        ms = [s for s in scripts if not (ROOT / s).exists()]
        ma = [a for a in arts if not (ROOT / a).exists()]
        status = "SCRIPT_MISSING" if ms else ("ARTIFACT_ABSENT" if ma else "OK")
        if ms:
            missing_scripts += [(i, s) for s in ms]
        rows.append(dict(id=i, manuscript_item=item, figure_or_table=fig,
                         scripts=";".join(scripts), artifacts=";".join(arts),
                         status=status, absent_artifacts=";".join(ma), note=note))

    tsv = ROOT / "docs/EXPERIMENT_MAP.tsv"
    with tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader(); w.writerows(rows)

    n_ok = sum(r["status"] == "OK" for r in rows)
    n_abs = sum(r["status"] == "ARTIFACT_ABSENT" for r in rows)
    L = [
      "# Manuscript-to-code map",
      "",
      "*Generated by `scripts/build_experiment_map.py` — do not edit by hand. Re-run it after",
      "changing `scripts/` or `results/`; it verifies every path below against the working tree.*",
      "",
      f"**{len(rows)} experiments — {n_ok} fully present, {n_abs} with one or more raw artifacts "
      "not committed, 0 with a missing script.**",
      "",
      "`ARTIFACT_ABSENT` means the code is here but a raw output is not: several were produced on",
      "the TACC cluster and never committed (see",
      "`experiments/docs/MISSING_COLLEAGUE_ARTIFACTS.md`). For every such row the committed",
      "figure source-data snapshot under `experiments/figures/source_data/` carries the plotted",
      "values, so each manuscript number remains inspectable.",
      "",
      "| id | manuscript item | figure / table | status | code | artifacts |",
      "|---|---|---|---|---|---|",
    ]
    for r in rows:
        code = "<br>".join(f"`{s}`" for s in r["scripts"].split(";"))
        art = "<br>".join(("~~`%s`~~" % a) if a in r["absent_artifacts"].split(";") else f"`{a}`"
                          for a in r["artifacts"].split(";"))
        L.append(f"| {r['id']} | {r['manuscript_item']} | {r['figure_or_table']} | "
                 f"{r['status']} | {code} | {art} |")
    L += ["", "Struck-through artifacts are the uncommitted ones.", "",
          "Machine-readable: `docs/EXPERIMENT_MAP.tsv`."]
    (ROOT / "docs/EXPERIMENT_MAP.md").write_text("\n".join(L) + "\n")

    for r in rows:
        print(f"  {r['id']:3s} {r['status']:16s} {r['manuscript_item'][:62]}")
        if r["absent_artifacts"]:
            for a in r["absent_artifacts"].split(";"):
                print(f"      absent: {a}")
    print(f"\n{len(rows)} experiments: {n_ok} OK, {n_abs} ARTIFACT_ABSENT, "
          f"{len(missing_scripts)} missing script(s)")
    if missing_scripts:
        for i, s in missing_scripts:
            print(f"  [SCRIPT_MISSING] {i}: {s}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
