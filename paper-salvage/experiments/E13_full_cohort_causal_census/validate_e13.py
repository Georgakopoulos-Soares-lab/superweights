#!/usr/bin/env python3
"""Strict requirement-by-requirement completion audit for E13."""
from __future__ import annotations

import csv,json,subprocess,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
SALVAGE=HERE.parents[1]
ROOT=SALVAGE.parent
E13=ROOT/"results"/"E13"


def rows(path):
    with path.open() as f:return list(csv.DictReader(f))


def require(ok,msg):
    if not ok:raise AssertionError(msg)
    print(f"OK  {msg}")


def require_columns(table, expected, name):
    require(bool(table), f"{name} is non-empty")
    missing = sorted(set(expected) - set(table[0]))
    require(not missing, f"{name} required columns present (missing={missing})")


def main():
    lock=subprocess.run([sys.executable,str(SALVAGE/"src"/"prereg_lock.py"),"verify",
                         str(SALVAGE/"docs"/"prereg"/"PREREG_full_cohort_causal_census.md")],
                        capture_output=True,text=True)
    require(lock.returncode==0 and "OK" in lock.stdout,"preregistration lock verifies")
    gate=json.loads((E13/"stage0_gate_decision.json").read_text())
    require(gate.get("decision") == "PASS","Stage 0 reproduction gate passed")
    estimate=HERE/"STAGE1_RESOURCE_ESTIMATE.md"
    require(estimate.exists() and estimate.stat().st_size>500,
            "pre-launch GPU-hour/disk/special-loader estimate exists")
    manifest=json.loads((E13/"candidate_manifest.json").read_text())
    require(manifest["n_models"]==23,"fixed cohort has exactly 23 models")
    require(manifest["n_candidates"]==28,"structural basis has exactly 28 candidates")
    require(len(manifest["control_sets"])==24,"24 candidate-layer control panels are frozen")
    raw={}
    for p in (E13/"raw").glob("*.json"):
        d=json.loads(p.read_text()); require(d["model"] not in raw,f"unique raw model {d['model']}")
        raw[d["model"]]=d
    require(set(raw)==set(manifest["panel_order"]),"raw responses cover all and only 23 models")
    total_cand=total_ctrl=0
    for model in manifest["panel_order"]:
        d=raw[model]; base=d["baseline_per_unit"]
        require(len(base)>0 and all(len(x)==2 and x[1]>0 for x in base),f"{model}: valid baseline units")
        expected=32 if model=="Phi-3-mini-4k-instruct" else 12
        require(len(d["conditions"])==expected,f"{model}: complete candidate/control conditions")
        keys=set()
        for c in d["conditions"]:
            k=(c["kind"],c["layer"],c["row"],c["epsilon"])
            require(k not in keys,f"{model}: condition unique {k}");keys.add(k)
            require(len(c["per_unit"])==len(base),f"{model}: paired raw units {k}")
            require(all(len(x)==2 and x[1]>0 for x in c["per_unit"]),f"{model}: valid perturbed units {k}")
            total_cand+=c["kind"]=="candidate";total_ctrl+=c["kind"]=="control"
    require(total_cand==56,"56 candidate effects (28 candidates x 2 epsilon)")
    require(total_ctrl==240,"240 control effects (24 layers x 5 rows x 2 epsilon)")
    candidate_table=rows(E13/"candidate_effects.csv")
    control_table=rows(E13/"control_effects.csv")
    census_table=rows(E13/"causal_census.csv")
    require(len(candidate_table)==56,"candidate_effects.csv complete")
    require(len(control_table)==240,"control_effects.csv complete")
    require(len(census_table)==296,"causal_census.csv complete")
    common_columns={"model","domain","architecture","total_params","layer","row","epsilon",
                    "baseline_loss","perturbed_loss","absolute_delta_loss",
                    "relative_loss_change","relative_ci95_low","relative_ci95_high"}
    require_columns(candidate_table, common_columns | {"candidate_index","primary","q1",
                    "frob_norm","layer_median_frob_norm","frob_ratio_to_layer_median"},
                    "candidate_effects.csv")
    require_columns(candidate_table,{"candidate_control_structural_q1_gap"},
                    "candidate_effects.csv structural-control gap")
    require(any(r["candidate_control_structural_q1_gap"] not in ("", "None")
                for r in candidate_table),
            "candidate structural-control q1 gaps retained where available")
    require_columns(control_table, common_columns | {"control_index"},"control_effects.csv")
    require_columns(census_table, common_columns | {"kind"},"causal_census.csv")
    tomo=rows(E13/"tomography_results.csv")
    require(len(tomo)==46,"tomography has 23 models x 2 epsilon")
    require(sum(x["interaction_identifiable"]=="True" for x in tomo)==2,
            "only the two Phi-3 epsilon cells are interaction-identifiable")
    corr=json.loads((E13/"structure_function_correlations.json").read_text())
    require(len(corr["analyses"])==4,"four preregistered structure-function analyses")
    cohort=json.loads((E13/"cohort_summary.json").read_text())
    require(cohort["n_models"]==23 and not cohort["missing_models"],"cohort_summary covers 23/23")
    cohort_table=rows(E13/"cohort_summary.csv")
    require(len(cohort_table)==23,"one-row-per-model cohort CSV")
    require_columns(cohort_table,{"model","architecture","domain","total_params",
                    "n_structural_candidates","primary_candidate_q1",
                    "candidate_control_structural_q1_gap",
                    "strongest_singleton_relative_loss_change_eps0p5",
                    "strongest_singleton_relative_loss_change_eps1p0",
                    "candidate_minus_median_control_effect_eps0p5",
                    "candidate_minus_median_control_effect_eps1p0","interaction_identifiable",
                    "F2_heldout_mae_eps0p5","F2_heldout_mae_eps1p0",
                    "F3_heldout_mae_eps0p5","F3_heldout_mae_eps1p0",
                    "F2_to_F3_relative_mae_improvement_eps0p5",
                    "F2_to_F3_relative_mae_improvement_eps1p0",
                    "bootstrap_ci95_low_eps0p5","bootstrap_ci95_high_eps0p5",
                    "bootstrap_ci95_low_eps1p0","bootstrap_ci95_high_eps1p0",
                    "pair_terms_materially_improved_eps0p5",
                    "pair_terms_materially_improved_eps1p0",
                    "second_order_observer_adequate_eps0p5",
                    "second_order_observer_adequate_eps1p0"},"cohort_summary.csv")
    for name in ("figure_A_structure_vs_causal","figure_B_full_cohort_census",
                 "figure_C_interaction_complexity"):
        for ext in ("pdf","png"):
            p=E13/"figures"/f"{name}.{ext}"
            require(p.exists() and p.stat().st_size>10_000,f"publication figure {p.name}")
    report=(E13/"FULL_COHORT_CAUSAL_SUMMARY.md").read_text()
    for section in ("Exact methods","Technical failures and missing models","Full cohort table",
                    "Structure–function associations","Tomography","Answers to Q1–Q6",
                    "What the results support, falsify, and leave unresolved"):
        require(section in report,f"final report section: {section}")
    print("\nE13 COMPLETION AUDIT PASSED")


if __name__=="__main__":main()
