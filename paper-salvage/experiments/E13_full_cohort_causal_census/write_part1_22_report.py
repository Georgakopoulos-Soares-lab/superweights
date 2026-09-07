#!/usr/bin/env python3
"""Compile and report the user-confirmed 22-model Part-1 causal census.

Phi-3 is intentionally outside this report's Part-1 cohort.  The locked 23-model E13
manifest and strict validator are left untouched; outputs from this script carry a
``part1_22`` prefix so they cannot be mistaken for the preregistered 23-model artifacts.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

import numpy as np
from scipy.stats import spearmanr

import compile_census as cc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E13 = ROOT / "results" / "E13"
EXCLUDED = "Phi-3-mini-4k-instruct"
N_BOOT = 5000
SEED = 42


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(x: float) -> str:
    return f"{100*x:+.2f}%"


def bootstrap_median_ci(values: list[float], seed: int) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = [np.median(rng.choice(a, len(a), replace=True)) for _ in range(N_BOOT)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def spearman_ci(rows: list[dict], xkey: str, seed: int) -> dict:
    x = np.asarray([float(r[xkey]) for r in rows])
    y = np.asarray([float(r["relative_loss_change"]) for r in rows])
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(N_BOOT):
        ix = rng.integers(0, len(rows), len(rows))
        value = spearmanr(x[ix], y[ix]).statistic
        if np.isfinite(value):
            boot.append(float(value))
    return {"x": xkey, "n_models": len(rows), "spearman_rho": float(rho),
            "asymptotic_p": float(p),
            "bootstrap_ci95": [float(np.percentile(boot, 2.5)),
                               float(np.percentile(boot, 97.5))],
            "n_boot": N_BOOT, "seed": seed}


def main() -> None:
    manifest = json.loads((E13 / "candidate_manifest.json").read_text())
    structural = {r["model"]: r for r in csv.DictReader(
        (ROOT / "results" / "E11" / "scale_ladder_backfilled.csv").open())}
    panel = [m for m in manifest["panel_order"] if m != EXCLUDED]
    raw = {}
    for path in (E13 / "raw").glob("*.json"):
        payload = json.loads(path.read_text())
        raw[payload["model"]] = payload
    missing = [m for m in panel if m not in raw]
    if missing:
        raise SystemExit(f"missing Part-1 raw models: {missing}")

    candidate_meta = {(c["model"], c["layer"], c["row"]): c
                      for c in manifest["candidates"] if c["model"] in panel}
    seed_seq = np.random.SeedSequence(SEED).spawn(1000)
    candidate_rows, control_rows = [], []
    seed_i = 0
    for model in panel:
        payload, sr = raw[model], structural[model]
        if len(payload["conditions"]) != 12:
            raise AssertionError(f"{model}: expected 12 conditions")
        for condition in payload["conditions"]:
            ci = cc.paired_relative_ci(payload["baseline_per_unit"],
                                       condition["per_unit"], seed_seq[seed_i])
            seed_i += 1
            row = {
                "model": model, "domain": payload["domain"],
                "architecture": payload["architecture"],
                "total_params": int(sr["total_params"]),
                "layer": int(condition["layer"]), "row": int(condition["row"]),
                "epsilon": float(condition["epsilon"]),
                "baseline_loss": float(payload["baseline_loss"]),
                "perturbed_loss": float(condition["perturbed_loss"]),
                "relative_loss_change": float(condition["relative_loss_change"]),
                "relative_ci95_low": ci[0], "relative_ci95_high": ci[1],
            }
            if condition["kind"] == "candidate":
                meta = candidate_meta[(model, row["layer"], row["row"])]
                candidate_rows.append({**row, "q1": float(meta["q1"]),
                                       "frob_ratio_to_layer_median":
                                           float(meta["frob_ratio_to_layer_median"])})
            else:
                control_rows.append(row)

    if len(candidate_rows) != 44 or len(control_rows) != 220:
        raise AssertionError((len(candidate_rows), len(control_rows)))
    write_csv(E13 / "part1_22_candidate_effects.csv", candidate_rows)
    write_csv(E13 / "part1_22_control_effects.csv", control_rows)

    summary = []
    for model in panel:
        sr = structural[model]
        row = {"model": model, "architecture": sr["architecture"],
               "domain": sr["domain"], "total_params": int(sr["total_params"])}
        for eps, tag in ((0.5, "eps0p5"), (1.0, "eps1p0")):
            cand = next(r for r in candidate_rows
                        if r["model"] == model and r["epsilon"] == eps)
            controls = [r["relative_loss_change"] for r in control_rows
                        if r["model"] == model and r["epsilon"] == eps]
            row[f"candidate_effect_{tag}"] = cand["relative_loss_change"]
            row[f"candidate_ci_low_{tag}"] = cand["relative_ci95_low"]
            row[f"candidate_ci_high_{tag}"] = cand["relative_ci95_high"]
            row[f"median_control_effect_{tag}"] = median(controls)
            row[f"candidate_minus_control_{tag}"] = (
                cand["relative_loss_change"] - median(controls))
        summary.append(row)
    write_csv(E13 / "part1_22_cohort_summary.csv", summary)

    full = [r for r in candidate_rows if r["epsilon"] == 1.0]
    correlations = [spearman_ci(full, "q1", SEED + 1),
                    spearman_ci(full, "frob_ratio_to_layer_median", SEED + 2)]
    (E13 / "part1_22_structure_function_correlations.json").write_text(
        json.dumps({"cohort": "Part 1", "n_models": 22,
                    "excluded": [EXCLUDED], "analyses": correlations}, indent=2) + "\n")

    gaps05 = [r["candidate_minus_control_eps0p5"] for r in summary]
    gaps10 = [r["candidate_minus_control_eps1p0"] for r in summary]
    ci05 = bootstrap_median_ci(gaps05, SEED + 5)
    ci10 = bootstrap_median_ci(gaps10, SEED + 10)
    ranked = sorted(summary, key=lambda r: r["candidate_effect_eps1p0"], reverse=True)
    groups = {}
    for row in summary:
        groups.setdefault((row["domain"], row["architecture"]), []).append(
            row["candidate_effect_eps1p0"])

    lines = [
        "# Part 1 Causal Census — 22-Model Report", "",
        "## Scope and completion", "",
        "This report summarizes the completed Part 1 cohort of 22 models. Phi-3 is "
        "excluded by explicit scope clarification and is not treated as a missing model. "
        "All 22 included models have complete atomic raw responses: one frozen structural "
        "candidate, five seeded same-layer ordinary controls, two intervention strengths "
        "(epsilon 0.5 and 1.0), and paired per-unit evaluation data.", "",
        "The resulting census contains 44 candidate effects and 220 control effects. "
        "Text decoders use WikiText-2 causal-LM NLL, text encoders fixed-mask WikiText-2 "
        "MLM loss, genomic decoders fixed hg38 teacher-forced NLL, and genomic encoders "
        "fixed-mask hg38 MLM loss. Confidence intervals use 5,000 paired bootstrap resamples.", "",
        "## Main results", "",
        f"At epsilon 0.5, {sum(x > 0 for x in gaps05)}/22 candidates exceeded their "
        f"same-layer median control; the median candidate-minus-control effect was "
        f"{pct(median(gaps05))} (95% model-bootstrap CI {pct(ci05[0])} to {pct(ci05[1])}).", "",
        f"At full ablation (epsilon 1.0), {sum(x > 0 for x in gaps10)}/22 candidates "
        f"exceeded their same-layer median control; the median candidate-minus-control "
        f"effect was {pct(median(gaps10))} (95% model-bootstrap CI {pct(ci10[0])} to "
        f"{pct(ci10[1])}).", "",
    ]
    for corr in correlations:
        label = "q1" if corr["x"] == "q1" else "layer-relative Frobenius magnitude"
        lines.append(f"Across the 22 frozen candidates, {label} versus full-ablation "
                     f"causal effect had Spearman rho={corr['spearman_rho']:.3f} "
                     f"(95% bootstrap CI {corr['bootstrap_ci95'][0]:.3f} to "
                     f"{corr['bootstrap_ci95'][1]:.3f}; p={corr['asymptotic_p']:.3g}).")
        lines.append("")
    lines += ["## Full cohort", "",
              "| Model | Type | Candidate effect eps=.5 | Candidate effect eps=1 | "
              "Candidate - median control eps=1 |",
              "|---|---|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['model']} | {row['domain']}/{row['architecture']} | "
                     f"{pct(row['candidate_effect_eps0p5'])} | "
                     f"{pct(row['candidate_effect_eps1p0'])} | "
                     f"{pct(row['candidate_minus_control_eps1p0'])} |")
    lines += ["", "## Largest full-ablation candidate effects", ""]
    for i, row in enumerate(ranked[:5], 1):
        lines.append(f"{i}. {row['model']}: {pct(row['candidate_effect_eps1p0'])} "
                     f"(candidate-minus-control {pct(row['candidate_minus_control_eps1p0'])}).")
    lines += ["", "## Descriptive architecture/domain medians", ""]
    for (domain, architecture), values in groups.items():
        lines.append(f"- {domain}/{architecture}: n={len(values)}, median full-ablation "
                     f"candidate effect {pct(median(values))}.")
    lines += ["", "## Interpretation", "",
              "The Part 1 census supports the claim that frozen structurally prominent rows are "
              "usually more causally consequential than ordinary rows in the same layer: this "
              "holds for 18/22 models at partial suppression and 20/22 at full ablation, with "
              "positive cohort-median candidate-minus-control gaps at both strengths. The result "
              "is not universal: NTv3 and EuroBERT-2.1B do not show a positive full-ablation gap.", "",
              "The data do not support q1 as a cross-model predictor of native-objective damage "
              "(rho=0.074 and a bootstrap interval spanning substantial negative and positive "
              "values). They do support a modest positive association for layer-relative "
              "Frobenius magnitude (rho=0.441; bootstrap CI 0.062 to 0.717), although this should "
              "be interpreted as one fixed-panel association rather than a universal law.", "",
              "Causal magnitude is highly heterogeneous: the median full-ablation effect is "
              "+85.47% for text decoders, versus +0.65% for text encoders, +0.68% for genomic "
              "decoders, and +0.02% for genomic encoders. A few rows produce catastrophic loss "
              "increases while many produce sub-percent changes, so the evidence falsifies a "
              "single common effect-size regime across architectures. These native-objective "
              "measurements do not establish downstream-task mechanisms or interaction order.", "",
              "## Artifacts", "",
              "- `results/E13/part1_22_candidate_effects.csv`", "",
              "- `results/E13/part1_22_control_effects.csv`", "",
              "- `results/E13/part1_22_cohort_summary.csv`", "",
              "- `results/E13/part1_22_structure_function_correlations.json`", "",
              "- Raw paired responses: `results/E13/raw/`", ""]
    report = E13 / "PART1_CAUSAL_CENSUS_SUMMARY.md"
    report.write_text("\n".join(lines))
    print(f"wrote {report} ({len(summary)} models)")


if __name__ == "__main__":
    main()
