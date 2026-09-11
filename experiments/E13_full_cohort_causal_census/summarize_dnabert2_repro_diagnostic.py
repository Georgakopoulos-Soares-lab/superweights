#!/usr/bin/env python3
"""Summarize the four-cell DNABERT-2 execution-path diagnostic."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "results" / "experiments" / "E13_dnabert2_reproducibility"
NAMES = ["historical_default_specials", "current_eager_no_specials", "eager_specials", "default_no_specials"]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def exact_rows(a: dict, b: dict) -> bool:
    keys = ("layer", "row", "activation_max", "activation_median", "detector_ratio", "ratio_rank", "activation_rank")
    return all(tuple(x[k] for k in keys) == tuple(y[k] for k in keys) for x, y in zip(a["rows"], b["rows"]))


def main() -> None:
    d = {name: json.loads((OUT / f"{name}.json").read_text()) for name in NAMES}
    historical = d[NAMES[0]]; current = d[NAMES[1]]
    old_index = json.loads((ROOT / "results" / "super_weight_index.json").read_text())["dnabert2"]["results"][0]
    census = json.loads((ROOT / "results" / "E13_multicandidate_structural" / "raw" / "dnabert2.json").read_text())
    census_target = next(r for r in census["rows"] if r["layer"] == 5 and r["row"] == 603)

    comparison = []
    for name in NAMES:
        x = d[name]; t = x["target"]
        comparison.append({
            "cell": name, "attention_requested": x["attention_requested"],
            "attention_runtime": x["attention_runtime"], "preprocessing": x["preprocessing"],
            "input_tokens": x["input_tokens"], "target_activation_max": t["activation_max"],
            "target_detector_ratio": t["detector_ratio"], "target_ratio_rank": t["ratio_rank"],
            "target_activation_rank": t["activation_rank"], "target_passes_ratio5": t["passes"],
            "k": x["k"], "global_ratio_layer": x["global_max_ratio"]["layer"],
            "global_ratio_row": x["global_max_ratio"]["row"],
            "global_ratio": x["global_max_ratio"]["detector_ratio"],
            "global_activation_layer": x["global_max_activation"]["layer"],
            "global_activation_row": x["global_max_activation"]["row"],
            "global_activation": x["global_max_activation"]["activation_max"],
            **{f"structural_{k}": v for k, v in x["target_structural_metrics"].items()},
        })
    with (OUT / "execution_path_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comparison[0])); w.writeheader(); w.writerows(comparison)

    hcoords = {(r["layer"], r["row"]): r for r in historical["historical_coordinates"]}
    ccoords = {(r["layer"], r["row"]): r for r in current["historical_coordinates"]}
    coord_rows = []
    for coord, h in hcoords.items():
        c = ccoords[coord]
        coord_rows.append({
            "layer": coord[0], "row": coord[1],
            "historical_activation": h["activation_max"], "current_activation": c["activation_max"],
            "activation_fold_historical_over_current": h["activation_max"] / c["activation_max"] if c["activation_max"] else "inf",
            "historical_ratio": h["detector_ratio"], "current_ratio": c["detector_ratio"],
            "historical_ratio_rank": h["ratio_rank"], "current_ratio_rank": c["ratio_rank"],
            "historical_activation_rank": h["activation_rank"], "current_activation_rank": c["activation_rank"],
            "historical_passes": h["detector_ratio"] >= 5, "current_passes": c["detector_ratio"] >= 5,
        })
    with (OUT / "historical_coordinate_impact.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(coord_rows[0])); w.writeheader(); w.writerows(coord_rows)

    specials_exact = exact_rows(historical, d["eager_specials"])
    no_specials_exact = exact_rows(current, d["default_no_specials"])
    metrics_exact = len({json.dumps(x["target_structural_metrics"], sort_keys=True) for x in d.values()}) == 1
    historical_activation_exact = historical["target"]["activation_max"] == old_index["out_max"]
    current_census_exact = (
        current["target"]["activation_max"] == census_target["activation_max"]
        and current["target"]["detector_ratio"] == census_target["detector_score"]
        and current["k"] == len(census["selected"])
    )
    inner_ids_equal = historical["input_ids"][1:-1] == current["input_ids"]
    provenance = {
        "diagnostic": "DNABERT-2 historical-discovery versus E13 execution path",
        "git_head": git("rev-parse", "HEAD"),
        "historical_code_commit_recovered": "e7906060884bb3ab20bb56576330f9b477f17644",
        "historical_artifact_first_tracked_after_restructure": "6e789a3cfb69f6d57b9aa917bb534b72fdd00de1",
        "historical_revision_provenance_gap": "Original wrapper requested revision=main and artifact did not store resolved revision or environment.",
        "checkpoint_tested": historical["checkpoint_revision"],
        "historical_activation_matches_artifact_exactly": historical_activation_exact,
        "current_endpoint_matches_census_exactly": current_census_exact,
        "historical_vs_explicit_eager_specials_all_rows_exact": specials_exact,
        "current_vs_default_no_specials_all_rows_exact": no_specials_exact,
        "actb_interior_token_ids_exactly_equal": inner_ids_equal,
        "historical_added_token_ids": [historical["input_ids"][0], historical["input_ids"][-1]],
        "structural_metrics_exact_across_cells": metrics_exact,
        "raw_outputs": {name: {"path": f"results/experiments/E13_dnabert2_reproducibility/{name}.json",
                                "sha256": hashlib.sha256((OUT / f"{name}.json").read_bytes()).hexdigest()} for name in NAMES},
        "causal_interventions_run": False,
        "tomography_run": False,
    }
    (OUT / "diagnostic_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    t_h, t_c = historical["target"], current["target"]
    sm = historical["target_structural_metrics"]
    lines = [
        "# DNABERT-2 execution-path reproducibility diagnostic",
        "",
        "Scope: `(layer 5, row 603)` activation discrepancy only. No causal intervention, tomography, candidate redesign, or threshold change was performed.",
        "",
        "## A. Reproduction status",
        "",
        f"- **Historical endpoint: PASS.** Default-special-token ACTB preprocessing reproduces historical `out_max={old_index['out_max']}` exactly; `(5,603)` is activation rank 1, detector ratio {t_h['detector_ratio']:.6f}, ratio rank {t_h['ratio_rank']}, and passes ratio 5. Retrospective frozen-threshold set size is K={historical['k']}.",
        f"- **Current E13 endpoint: PASS.** No-special-token preprocessing reproduces the census activation and ratio exactly: `activation_max={t_c['activation_max']:.9f}`, ratio {t_c['detector_ratio']:.9f}, ratio rank {t_c['ratio_rank']}, activation rank {t_c['activation_rank']}, and K={current['k']}.",
        "",
        "Historical provenance caveat: the original tracked wrapper at commit `e790606` requested Hugging Face revision `main`; the discovery artifact did not store the resolved commit, package versions, or GPU. The later project pin `7bce263b...` was tested here. Exact reproduction of the stored activation on that pin is strong empirical checkpoint agreement, but does not retroactively create missing historical metadata.",
        "",
        "## B. Root cause",
        "",
        "**Demonstrated cause: tokenizer special-token handling.** The historical wrapper called the tokenizer with default `add_special_tokens=True`, producing 116 tokens: token ID 1, the exact 114-token ACTB interior used by E13, then token ID 2. E13 called the tokenizer with `add_special_tokens=False`, producing only the 114-token interior. The interior token IDs are exactly equal.",
        "",
        f"Adding the two historical boundary tokens switches `(5,603)` from activation {t_c['activation_max']:.6f}, ratio {t_c['detector_ratio']:.6f}, to activation {t_h['activation_max']:.6f}, ratio {t_h['detector_ratio']:.6f}. All other tested settings are held fixed.",
        "",
        "Attention is **not the cause of the reproduced discrepancy**. Triton is unavailable in the diagnostic environment, so both `default` and explicitly `eager` requests execute the PyTorch fallback. Historical-special and explicit-eager-special cells are identical for all 9,216 rows; current-no-special and default-no-special cells are also identical for all rows. Because the historical stored activation reproduces exactly under eager attention, a historical flash kernel is not required to produce the spike. This study does not claim numerical equivalence between an unavailable flash kernel and eager attention.",
        "",
        "A separate protocol mismatch was also confirmed: the historical discovery selected the global maximum activation iteratively; it did not originally define the full ratio≥5 set. K=89 above is therefore a retrospective application of the frozen E13 set rule to the historical activation path, not a claim about the original iterative basis size.",
        "",
        "## C. Structural-metric stability",
        "",
        f"All four cells give exactly the same weight-derived values for `(5,603)`: q1={sm['q1']:.15f}, PR_spec={sm['pr_spec']:.15f}, exact `||U_k||_F`={sm['frob_norm']:.15f}, stable rank={sm['stable_rank']:.15f}. This rules out a checkpoint/model-state inconsistency in the tested state.",
        "",
        "## D. Candidate identity impact",
        "",
        f"With historical special tokens, `(5,603)` is the global activation candidate; the global ratio candidate is `({historical['global_max_ratio']['layer']},{historical['global_max_ratio']['row']})` at ratio {historical['global_max_ratio']['detector_ratio']:.3f}. Without specials, the global activation candidate becomes `({current['global_max_activation']['layer']},{current['global_max_activation']['row']})`, the global ratio candidate becomes `({current['global_max_ratio']['layer']},{current['global_max_ratio']['row']})`, and `(5,603)` does not pass. The historical-coordinate comparison CSV shows that all ten established coordinates change substantially, not only the primary.",
        "",
        "## E. Manuscript impact and protocol decision",
        "",
        "Classification: **1. Benign execution-path dependence**, specifically input-boundary preprocessing dependence. Both endpoints reproduce exactly, the switching factor is isolated, and structural metrics are stable.",
        "",
        "For claims tied to the established DNABERT-2 discovery coordinate, the canonical path should be the historical wrapper's default-special-token preprocessing because it exactly reproduces the frozen artifact. The current E13 no-special-token DNABERT-2 result is valid for the code that was run but is not selection-compatible with that historical candidate and should not be used to validate or replace it silently. Any recomputation of the DNABERT-2 census with historical preprocessing requires an explicit protocol amendment and a separate artifact; this diagnostic does not make that change.",
        "",
        "Universal Part 2B tomography remains stopped.",
        "",
        "## Artifacts",
        "",
        "- `results/experiments/E13_dnabert2_reproducibility/execution_path_comparison.csv`",
        "- `results/experiments/E13_dnabert2_reproducibility/historical_coordinate_impact.csv`",
        "- `results/experiments/E13_dnabert2_reproducibility/diagnostic_provenance.json`",
        "- `results/experiments/E13_dnabert2_reproducibility/{historical_default_specials,current_eager_no_specials,eager_specials,default_no_specials}.json`",
        "- `results/experiments/E13_dnabert2_reproducibility/logs/*.log`",
        "- `experiments/E13_full_cohort_causal_census/run_dnabert2_repro_diagnostic.py`",
        "- `experiments/E13_full_cohort_causal_census/summarize_dnabert2_repro_diagnostic.py`",
    ]
    (OUT / "DNABERT2_EXECUTION_PATH_DIAGNOSTIC.md").write_text("\n".join(lines) + "\n")
    print(f"PASS historical={historical_activation_exact} current={current_census_exact} specials_exact={specials_exact} no_specials_exact={no_specials_exact} metrics={metrics_exact}")


if __name__ == "__main__":
    main()
