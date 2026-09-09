#!/usr/bin/env python3
"""Build the final supported Figure 4 from stored E7, mechanism, and E12 artifacts.

No model inference is performed. Panel-C uncertainty is recomputed from the stored E12
per-prompt generation records using the E12 percentile prompt-bootstrap convention.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "results"
E12 = RESULTS / "E12"
OUT = HERE / "main" / "fig4_generator_specificity"
PANEL_C_CSV = E12 / "figure4_generator_specificity_panel_c.csv"
NOTE = E12 / "FIGURE4_GENERATOR_SPECIFICITY_NOTE.md"

sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from _figstyle import apply_style, panel_label  # noqa: E402
from _paper_encoding import DOMAIN_COLOR  # noqa: E402

N_BOOT = 5000
BOOT_SEED = 42
N_PROMPTS = 96
PRIMARY_ROW = 2371


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def prompt_gc(records: list[dict], label: str, row: int, value: float) -> tuple[np.ndarray, list[int]]:
    """Average GC over the three sampling seeds within each aligned prompt."""
    conds = [x for x in records if x["label"] == label and x["row"] == row
             and np.isclose(float(x["value"]), value) and x["do_sample"]]
    seeds = sorted(int(x["seed"]) for x in conds)
    assert seeds == [42, 43, 44], (label, row, value, seeds)
    a = np.full((len(conds), N_PROMPTS), np.nan)
    for i, cond in enumerate(sorted(conds, key=lambda x: x["seed"])):
        for rec in cond["records"]:
            a[i, int(rec["prompt_idx"])] = float(rec["gc"])
    assert np.isfinite(a).all(), (label, row, value)
    return np.mean(a, axis=0), seeds


def mean_ci(a: np.ndarray) -> tuple[float, float, float]:
    """E12-style 5,000-draw percentile bootstrap over the 96 prompt indices."""
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    means = np.mean(a[idx], axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(a)), float(lo), float(hi)


def build_panel_c_data() -> tuple[list[dict], dict]:
    matches = list(csv.DictReader((E12 / "damage_matching.csv").open()))
    by_label = {x["label"]: x for x in matches}
    gen = read_jsonl(E12 / "raw" / "gen_records.jsonl")
    damage = read_jsonl(E12 / "raw" / "damage_evals.jsonl")

    def damage_at(label: str, row: int, value: float) -> float:
        vals = [float(x["damage"]) for x in damage if x["label"] == label
                and x["row"] == row and np.isclose(float(x["value"]), value)]
        assert len(vals) == 1, (label, row, value, vals)
        return vals[0]

    specs = [
        ("Baseline (row intact)", "baseline", "row2371_grid", "row2371", PRIMARY_ROW, 1.0),
        ("L4/r2371 ablation", "primary_ablation", "row2371_grid", "row2371", PRIMARY_ROW, 0.0),
    ]
    for label in ["control_row_2621", "control_row_456", "control_row_102",
                  "control_row_3039", "control_row_1126"]:
        m = by_label[label]
        row, value = int(m["row"]), float(m["matched_scale"])
        specs.append((f"Non-2371 row {row}", "non2371_control", f"matched_{label}",
                      label, row, value))
    rd = by_label["random_direction"]
    specs.append(("Random direction at L4/r2371", "same_location_random_direction",
                  "matched_random_direction", "random_direction", PRIMARY_ROW,
                  float(rd["matched_scale"])))

    rows = []
    for name, role, gen_label, damage_label, row, value in specs:
        a, seeds = prompt_gc(gen, gen_label, row, value)
        mean, lo, hi = mean_ci(a)
        if role in {"non2371_control", "same_location_random_direction"}:
            nll = float(by_label[damage_label]["matched_damage"])
        else:
            nll = damage_at(damage_label, row, value)
        rows.append({
            "condition": name, "role": role, "layer": 4, "row": row,
            "perturbation_kind": "direction" if role == "same_location_random_direction" else "alpha",
            "scale": value, "native_nll": nll, "gc_mean": mean,
            "gc_ci95_low": lo, "gc_ci95_high": hi, "n_prompts": len(a),
            "generation_seeds": ";".join(map(str, seeds)), "bootstrap_draws": N_BOOT,
            "bootstrap_seed": BOOT_SEED,
            "damage_match_reachable": ("" if role in {"baseline", "primary_ablation"}
                                       else by_label[damage_label]["reachable"]),
            "source_generation_label": gen_label, "source_damage_label": damage_label,
        })

    # Cross-artifact validation: published final CSV/summary values must reconstruct.
    target = float(by_label["random_direction"]["target_damage"])
    primary = next(x for x in rows if x["role"] == "primary_ablation")
    random_dir = next(x for x in rows if x["role"] == "same_location_random_direction")
    assert np.isclose(primary["native_nll"], target)
    # Known final-output bug: damage_matching.csv selected c=.0125 for NLL but its GC
    # writer filtered generation records by label only, pooling the earlier c=0 records
    # with c=.0125. Use the explicitly scale-filtered raw records for the plotted point.
    pooled_csv_gc = float(rd["matched_gc_mean"])
    assert np.isclose(random_dir["gc_mean"], 0.3068485966435185)
    assert not np.isclose(random_dir["gc_mean"], pooled_csv_gc)
    for x in rows:
        if x["role"] == "non2371_control":
            label = x["source_damage_label"]
            assert np.isclose(x["gc_mean"], float(by_label[label]["matched_gc_mean"]))
    max_non2371 = max(max(float(p["damage"]) for p in json.loads(m["grid_json"]))
                      for m in matches if m["label"].startswith("control_row_"))
    anchors = {
        "baseline_gc": next(x["gc_mean"] for x in rows if x["role"] == "baseline"),
        "primary_ablation_nll": primary["native_nll"],
        "primary_ablation_gc": primary["gc_mean"],
        "random_direction_scale": random_dir["scale"],
        "random_direction_nll": random_dir["native_nll"],
        "random_direction_gc": random_dir["gc_mean"],
        "buggy_damage_matching_csv_random_gc": pooled_csv_gc,
        "random_minus_primary_nll": random_dir["native_nll"] - primary["native_nll"],
        "random_minus_primary_gc": random_dir["gc_mean"] - primary["gc_mean"],
        "maximum_non2371_nll_any_tested_scale": max_non2371,
    }
    assert np.isclose(anchors["baseline_gc"], 0.4204011140046296)
    assert np.isclose(anchors["primary_ablation_nll"], 8.754319605827332)
    assert np.isclose(anchors["primary_ablation_gc"], 0.3064959490740741)
    return rows, anchors


def write_csv(rows: list[dict]) -> None:
    with PANEL_C_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def make_figure(rows: list[dict], anchors: dict) -> tuple[dict, dict]:
    apply_style()
    structural = json.loads((RESULTS / "e7_legacy_reanalysis.json").read_text())["GENERator EUK"]
    sink = json.loads((RESULTS / "mechanism" / "attention_sink_implicit_bias.json").read_text())["generator"]
    assert np.isclose(structural["q1"], 0.9688691252375834)
    att = sink["attention"]
    bias = sink["implicit_bias"]
    assert np.isclose(bias["pos0_real_mean"], bias["max_real_mean"])

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(10.6, 3.65),
                                  gridspec_kw={"width_ratios": [0.62, 1.0, 1.75]})

    orange = DOMAIN_COLOR["genomic"]
    a.bar([0], [structural["q1"]], width=.52, color=orange, edgecolor="black", linewidth=.8)
    a.set_ylim(0, 1.05)
    a.set_xticks([0])
    a.set_xticklabels(["GENERator EUK\nL4/r2371"])
    a.set_ylabel(r"exact spectral concentration $q_1$")
    a.text(0, structural["q1"] - .08, f"{structural['q1']:.3f}", ha="center",
           color="white", fontweight="bold", fontsize=9)
    a.text(.5, .05, "structural descriptor only", transform=a.transAxes,
           ha="center", color=".35", fontsize=7)
    panel_label(a, "A")

    bars = [100 * att["sink_share_pos0"], 100 * att["uniform_expectation"]]
    b.bar([0, 1], bars, width=.58, color=[orange, ".76"], edgecolor="black", linewidth=.7)
    b.set_xticks([0, 1])
    b.set_xticklabels(["position 0", "uniform\nexpectation"])
    b.set_ylabel("mean incoming attention (%)")
    b.set_ylim(0, 49)
    b.text(0, bars[0] + 1.1, f"{bars[0]:.1f}%", ha="center", fontweight="bold")
    b.text(.98, .96, f"{att['sink_over_uniform']:.1f}× uniform\n"
                     f"{100*att['frac_heads_argmax_pos0']:.1f}% argmax at pos. 0\n"
                     "activation maximum at pos. 0\nco-occurrence only",
           transform=b.transAxes, ha="right", va="top", fontsize=7.2,
           bbox={"facecolor": "white", "edgecolor": ".75", "pad": 3})
    panel_label(b, "B")

    styles = {
        "baseline": dict(marker="^", color=".25", s=54, label="intact baseline", zorder=4),
        "primary_ablation": dict(marker="o", color=orange, s=68, label="L4/r2371 ablation", zorder=6),
        "non2371_control": dict(marker="o", facecolors="none", edgecolors=".42", s=55,
                                label="five non-2371 endpoints", zorder=3),
        "same_location_random_direction": dict(marker="D", facecolors="none",
                                                edgecolors="#7A5195", linewidths=1.8,
                                                s=145, label="random direction at L4/r2371", zorder=7),
    }
    seen = set()
    control_offsets = {2621: (8, -15), 456: (12, 18), 102: (16, 8),
                       3039: (18, -5), 1126: (12, -24)}
    for x in rows:
        role = x["role"]
        yerr = np.array([[x["gc_mean"] - x["gc_ci95_low"]],
                         [x["gc_ci95_high"] - x["gc_mean"]]])
        c.errorbar(x["native_nll"], x["gc_mean"], yerr=yerr, fmt="none", ecolor=".62",
                   elinewidth=.85, capsize=2.2, zorder=2)
        kw = dict(styles[role])
        if role in seen:
            kw.pop("label", None)
        c.scatter(x["native_nll"], x["gc_mean"], **kw)
        if role == "non2371_control":
            c.annotate(f"r{x['row']}", (x["native_nll"], x["gc_mean"]),
                       xytext=control_offsets[x["row"]], textcoords="offset points",
                       fontsize=5.8, color=".35",
                       arrowprops={"arrowstyle": "-", "color": ".65", "lw": .45})
        seen.add(role)
    c.set_xlabel("native NLL on held-out damage pool")
    c.set_ylabel("generated GC fraction")
    c.set_xlim(6.25, 8.90)
    c.set_ylim(.275, .455)
    c.grid(alpha=.17)
    c.text(.98, .04, "same location, nearly equal damage:\n"
                     f"ΔNLL={anchors['random_minus_primary_nll']:+.4f}; "
                     f"ΔGC={anchors['random_minus_primary_gc']:+.4f}",
           transform=c.transAxes, ha="right", va="bottom", fontsize=7.2,
           bbox={"facecolor": "white", "edgecolor": ".78", "pad": 3})
    c.legend(loc="upper right", frameon=False, fontsize=7.1, handletextpad=.4)
    panel_label(c, "C")

    fig.tight_layout(w_pad=2.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"))
    plt.close(fig)
    return structural, sink


def write_note(rows: list[dict], anchors: dict, structural: dict, sink: dict) -> None:
    caption = (
        "**Figure 4. Functional consequences of disrupting a concentrated high-gain "
        "location in GENERator EUK.** (A) Exact spectral concentration of the primary "
        "high-gain row L4/r2371. (B) Reproduced BOS-centered attention/activation "
        "phenotype. Mean incoming attention mass at position 0 is 37.9% (33.0× uniform), "
        "78.8% of layer-head observations have their maximum there, and the high-gain "
        "activation maximum is also at position 0. These measurements establish "
        "co-occurrence only, not that L4/r2371 causes the attention sink. (C) Generated "
        "GC fraction versus native NLL for intact baseline, L4/r2371 ablation, the five "
        "highest-damage non-2371 endpoints from the preregistered sweep, and a fixed "
        "random-direction replacement at L4/r2371. The nonzero random-direction endpoint "
        "nearly overlaps ablation in damage and GC, so the low-GC effect does not require "
        "the learned row-2371 direction. The other rows never reached comparable native "
        "damage and therefore do not establish uniqueness to this location. Error bars "
        "are 95% percentile intervals from 5,000 bootstrap resamples of 96 prompt-level "
        "GC values after averaging the three generation seeds per prompt."
    )
    lines = [
        "# Figure 4 GENERator specificity: validation and provenance", "",
        "No model inference or new experiment was run. The figure is reconstructed from stored final artifacts.", "",
        "## Numerical anchors", "",
        f"- Exact q1, L4/r2371: `{structural['q1']:.15f}`.",
        f"- Baseline generated GC: `{anchors['baseline_gc']:.15f}`.",
        f"- L4/r2371 ablation: native NLL `{anchors['primary_ablation_nll']:.15f}`; GC `{anchors['primary_ablation_gc']:.15f}`.",
        f"- Selected random-direction scale: `c={anchors['random_direction_scale']}` (closest tested nonzero condition).",
        f"- Random-direction endpoint: native NLL `{anchors['random_direction_nll']:.15f}`; GC `{anchors['random_direction_gc']:.15f}`.",
        f"- The uncorrected `damage_matching.csv` GC field is `{anchors['buggy_damage_matching_csv_random_gc']:.15f}` and is not used in the plot.",
        f"- Random minus ablation: ΔNLL `{anchors['random_minus_primary_nll']:+.15f}`; ΔGC `{anchors['random_minus_primary_gc']:+.15f}`.",
        f"- Maximum native NLL reached at any tested scale by any of the five non-2371 rows: `{anchors['maximum_non2371_nll_any_tested_scale']:.15f}`.",
        f"- GC estimates use `{N_PROMPTS}` aligned prompts and generation seeds 42, 43, 44. Error bars use `{N_BOOT}` prompt-bootstrap draws with seed `{BOOT_SEED}`.", "",
        "The random-direction matching table marks the endpoint as a fallback because the target was not bracketed by distinct nonzero directions. `c=0` is the zero vector and therefore coincides with row ablation; the selected `c=0.0125` value is the closest tested nonzero replacement.", "",
        "### Documented aggregation bug", "",
        "`run_e12_full.py::write_damage_matching_csv` filters generation records by `label` but not by the selected `value`. After the direction-fix rerun left both `c=0` and `c=0.0125` records under `matched_random_direction`, the CSV pooled all six condition records. Its value `0.3066722728587963` is exactly the mean of the scale-specific `c=0` GC (`0.3064959490740741`) and `c=0.0125` GC (`0.3068485966435185`). Figure 4 and its plotting CSV use only the three stored `c=0.0125` records (seeds 42, 43, and 44). The original E12 artifacts are not altered.", "",
        "## Exact sources", "",
        "- `results/e7_legacy_reanalysis.json` — exact q1.",
        "- `results/mechanism/attention_sink_implicit_bias.json` — BOS attention and activation co-occurrence.",
        "- `results/E12/damage_matching.csv` — selected scales, native NLL, final GC means, reachability, and full tested grids.",
        "- `results/E12/raw/damage_evals.jsonl` — native-NLL measurements.",
        "- `results/E12/raw/gen_records.jsonl` — per-seed, per-prompt generated GC measurements.",
        "- `results/E12/E12_summary.md` — final interpretation and paired-difference bootstrap results.",
        "- `manuscript/experiments/E12_generator_degradation_control/run_e12_full.py` and `e12_lib.py` — intervention, damage, generation, and bootstrap implementation.", "",
        "## Caption", "", caption, "",
        "## Outputs", "",
        "- `manuscript/figures/main/fig4_generator_specificity.png`",
        "- `manuscript/figures/main/fig4_generator_specificity.pdf`",
        "- `results/E12/figure4_generator_specificity_panel_c.csv`",
        "- `manuscript/figures/fig4_generator_specificity.py`", "",
    ]
    NOTE.write_text("\n".join(lines))


def main() -> None:
    rows, anchors = build_panel_c_data()
    write_csv(rows)
    structural, sink = make_figure(rows, anchors)
    write_note(rows, anchors, structural, sink)
    print(json.dumps(anchors, indent=2))
    print(f"wrote {OUT}.png/.pdf, {PANEL_C_CSV}, and {NOTE}")


if __name__ == "__main__":
    main()
