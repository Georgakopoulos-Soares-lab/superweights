#!/usr/bin/env python3
"""Priority-1 robustness audit for okstillnotlast.md Section A.

Primary analysis: the 16 models whose E13 candidate coordinate was originally selected via
the activation-based detector (ratio >= 5.0, `candidate.ratio` in the E7/E8/E11 per-model
detection JSONs). The other 6 candidates in the 22-model causal cohort (Llama-7B, Mistral-7B,
OLMo-7B-0724-hf, NTv3, DNABERT-2, GENERator-EUK-3B) were carried over from
`results/E7/e7_legacy_reanalysis.json`, which has no activation-ratio field at all -- those
coordinates came from published/structural provenance, not the prospective ratio detector.
Phi-3 is excluded throughout, matching `part1_22_structure_function_correlations.json`.

Existing artifacts only -- no new model runs. Sources:
  - results/E11/scale_ladder_backfilled.csv        (detection_ratio, when backfilled)
  - results/E7/e7_phase1_detection_qwen25.json         (candidate.ratio, Qwen2.5-7B)
  - results/e8_detection_mosaicbert.json            (candidate.ratio, MosaicBERT)
  - results/e8_detection_modernbert.json            (candidate.ratio, ModernBERT-base)
  - results/E7/e7_phase1_detection_genomeocean.json    (candidate.ratio, GenomeOcean-4B)
  - results/E13/part1_22_candidate_effects.csv      (signed relative-loss effect, both eps)
  - results/E13/part1_22_cohort_summary.csv         (candidate-minus-random-control gap)
  - audit/round2/structural_vs_causal_gap.csv       (candidate-minus-top-norm-control gap)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "E13" / "priority1_robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEGACY_NOT_RATIO_SELECTED = {
    "Llama-7B", "Mistral-7B", "OLMo-7B-0724-hf", "NTv3", "DNABERT-2", "GENERator-EUK-3B",
}
EXCLUDED = {"Phi-3-mini-4k-instruct"}

# model -> (family label for leave-one-family-out; singletons get their own label)
FAMILY = {
    "Qwen2.5-7B": "Qwen2.5", "Qwen/Qwen2.5-0.5B": "Qwen2.5", "Qwen/Qwen2.5-1.5B": "Qwen2.5",
    "Qwen/Qwen2.5-3B": "Qwen2.5",
    "HuggingFaceTB/SmolLM2-135M": "SmolLM2", "HuggingFaceTB/SmolLM2-360M": "SmolLM2",
    "HuggingFaceTB/SmolLM2-1.7B": "SmolLM2",
    "EuroBERT/EuroBERT-210m": "EuroBERT", "EuroBERT/EuroBERT-610m": "EuroBERT",
    "EuroBERT/EuroBERT-2.1B": "EuroBERT",
    "GenerTeam/GENERator-v2-prokaryote-1.2b-base": "GENERator-prok",
    "GenerTeam/GENERator-v2-prokaryote-3b-base": "GENERator-prok",
    "ModernBERT-base": "ModernBERT", "answerdotai/ModernBERT-large": "ModernBERT",
    "MosaicBERT": "MosaicBERT-singleton",
    "GenomeOcean-4B": "GenomeOcean-singleton",
}

RATIO_SOURCE_JSON = {
    "Qwen2.5-7B": RESULTS / "e7_phase1_detection_qwen25.json",
    "MosaicBERT": RESULTS / "e8_detection_mosaicbert.json",
    "ModernBERT-base": RESULTS / "e8_detection_modernbert.json",
    "GenomeOcean-4B": RESULTS / "e7_phase1_detection_genomeocean.json",
}


def load_activation_ratios() -> dict[str, float]:
    ratios: dict[str, float] = {}
    with (RESULTS / "E11" / "scale_ladder_backfilled.csv").open() as f:
        for row in csv.DictReader(f):
            v = row.get("detection_ratio", "").strip()
            if v:
                ratios[row["model"]] = float(v)
    for model, path in RATIO_SOURCE_JSON.items():
        d = json.loads(path.read_text())
        ratios[model] = float(d["candidate"]["ratio"])
    return ratios


def load_candidate_effects() -> dict[str, dict[float, dict]]:
    """model -> epsilon -> {relative_loss_change, frob_ratio_to_layer_median}"""
    out: dict[str, dict[float, dict]] = {}
    with (RESULTS / "E13" / "part1_22_candidate_effects.csv").open() as f:
        for row in csv.DictReader(f):
            m, eps = row["model"], float(row["epsilon"])
            out.setdefault(m, {})[eps] = {
                "relative_loss_change": float(row["relative_loss_change"]),
                "frob_ratio_to_layer_median": float(row["frob_ratio_to_layer_median"]),
            }
    return out


def load_random_control_gap() -> dict[str, dict[float, float]]:
    """model -> epsilon -> candidate_minus_control"""
    out: dict[str, dict[float, float]] = {}
    with (RESULTS / "E13" / "part1_22_cohort_summary.csv").open() as f:
        for row in csv.DictReader(f):
            m = row["model"]
            out[m] = {
                0.5: float(row["candidate_minus_control_eps0p5"]),
                1.0: float(row["candidate_minus_control_eps1p0"]),
            }
    return out


def load_topnorm_control_gap() -> dict[str, dict[float, float]]:
    """model -> epsilon -> candidate_relative_loss_change - median_topk_control_relative_loss_change"""
    out: dict[str, dict[float, float]] = {}
    with (ROOT / "audit" / "round2" / "structural_vs_causal_gap.csv").open() as f:
        for row in csv.DictReader(f):
            m, eps = row["model"], float(row["epsilon"])
            gap = float(row["candidate_relative_loss_change"]) - float(row["median_topk_control_relative_loss_change"])
            out.setdefault(m, {})[eps] = gap
    return out


def spearman(x: np.ndarray, y: np.ndarray, seed: int, n_boot: int = 5000) -> dict:
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) == 0 or np.std(yb) == 0:
            continue
        r, _ = stats.spearmanr(xb, yb)
        if np.isfinite(r):
            boots.append(r)
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else [None, None]
    return {
        "n_models": n,
        "spearman_rho": float(rho),
        "asymptotic_p": float(p),
        "bootstrap_ci95": ci,
        "n_boot": n_boot,
        "n_boot_finite": len(boots),
        "seed": seed,
    }


def main() -> None:
    ratios = load_activation_ratios()
    effects = load_candidate_effects()
    rand_gap = load_random_control_gap()
    topnorm_gap = load_topnorm_control_gap()

    all_22 = sorted(set(effects) - EXCLUDED)
    primary_16 = sorted(m for m in all_22 if m not in LEGACY_NOT_RATIO_SELECTED)
    assert len(primary_16) == 16, f"expected 16 ratio-selected models, got {len(primary_16)}: {primary_16}"
    assert len(all_22) == 22, f"expected 22-model cohort, got {len(all_22)}"
    for m in primary_16:
        assert m in ratios, f"missing activation ratio for prospectively ratio-selected model {m}"

    report: dict = {
        "primary_16": primary_16,
        "excluded_not_ratio_selected": sorted(LEGACY_NOT_RATIO_SELECTED),
        "excluded_other": sorted(EXCLUDED),
    }

    x16 = np.array([ratios[m] for m in primary_16])
    y16_eps1 = np.array([effects[m][1.0]["relative_loss_change"] for m in primary_16])
    y16_eps0p5 = np.array([effects[m][0.5]["relative_loss_change"] for m in primary_16])

    # --- Main analysis, eps=1.0 (headline) ---
    main_eps1 = spearman(x16, y16_eps1, seed=1001)
    report["main_activation_ratio_vs_signed_effect_eps1p0"] = main_eps1

    # --- Item 4: repeat at eps=0.5 ---
    main_eps0p5 = spearman(x16, y16_eps0p5, seed=1002)
    report["main_activation_ratio_vs_signed_effect_eps0p5"] = main_eps0p5

    # --- Item 1: leave-one-model-out (eps=1.0) ---
    loo = {}
    for i, held_out in enumerate(primary_16):
        mask = np.arange(16) != i
        rho, _ = stats.spearmanr(x16[mask], y16_eps1[mask])
        loo[held_out] = float(rho)
    loo_rhos = list(loo.values())
    full_rho = main_eps1["spearman_rho"]
    most_influential = max(loo, key=lambda m: abs(loo[m] - full_rho))
    report["leave_one_model_out"] = {
        "full_16_rho": full_rho,
        "per_model_rho_with_that_model_removed": loo,
        "min_rho": min(loo_rhos),
        "max_rho": max(loo_rhos),
        "most_influential_model": most_influential,
        "most_influential_delta_rho": loo[most_influential] - full_rho,
    }

    # --- Item 2: leave-one-family-out (eps=1.0) ---
    families = sorted(set(FAMILY[m] for m in primary_16))
    lofo = {}
    for fam in families:
        mask = np.array([FAMILY[m] != fam for m in primary_16])
        n_remaining = int(mask.sum())
        n_removed = 16 - n_remaining
        if n_remaining < 4:
            lofo[fam] = {"n_removed": n_removed, "n_remaining": n_remaining, "rho": None,
                         "note": "too few models remaining for a meaningful Spearman rho"}
            continue
        rho, p = stats.spearmanr(x16[mask], y16_eps1[mask])
        lofo[fam] = {"n_removed": n_removed, "n_remaining": n_remaining,
                     "rho": float(rho), "asymptotic_p": float(p)}
    report["leave_one_family_out"] = lofo

    # --- Item 3: correlate activation ratio with three effect definitions (eps=1.0) ---
    y_rand_gap = np.array([rand_gap[m][1.0] for m in primary_16])
    y_topnorm_gap = np.array([topnorm_gap[m][1.0] for m in primary_16])
    report["activation_ratio_vs_candidate_minus_random_control_eps1p0"] = spearman(x16, y_rand_gap, seed=1003)
    report["activation_ratio_vs_candidate_minus_topnorm_control_eps1p0"] = spearman(x16, y_topnorm_gap, seed=1004)
    report["activation_ratio_vs_signed_full_ablation_effect_eps1p0"] = main_eps1  # alias, item 3's first bullet == main analysis

    # --- Item 5: sensitivity only -- absolute magnitude (eps=1.0) ---
    y_abs = np.abs(y16_eps1)
    report["SENSITIVITY_ONLY_activation_ratio_vs_absolute_effect_magnitude_eps1p0"] = spearman(x16, y_abs, seed=1005)

    # --- 22-model sensitivity context: pointer to the pre-existing structural-ratio correlation ---
    report["twentytwo_model_sensitivity_note"] = (
        "The 6 non-ratio-selected models (Llama-7B, Mistral-7B, OLMo-7B-0724-hf, NTv3, "
        "DNABERT-2, GENERator-EUK-3B) have no activation-ratio detector value at all -- their "
        "candidates came from results/E7/e7_legacy_reanalysis.json (published/structural "
        "provenance), which never recorded an activation ratio. A 22-model activation-ratio "
        "correlation therefore cannot be constructed. The existing 22-model sensitivity check "
        "on the adjacent structural quantity (frob_ratio_to_layer_median) is already computed "
        "in part1_22_structure_function_correlations.json: rho=0.441, asymptotic p=0.040, "
        "bootstrap 95% CI [0.062, 0.717], n=22 (Phi-3 excluded)."
    )

    out_path = OUT_DIR / "priority1_robustness_results.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print("WROTE", out_path)
    print(json.dumps({k: v for k, v in report.items() if k not in (
        "leave_one_model_out",)}, indent=2, default=str)[:2000])


if __name__ == "__main__":
    main()
