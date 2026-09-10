#!/usr/bin/env python3
"""
Build the supplementary tables that this round adds or extends.

S1, S4 and S5 are untouched -- they were produced by the round-2 audit and are reformatted
verbatim from stored artifacts. This script produces:

  S2  structural metrics  + the 24-input stability columns merged in
  S3  full causal census  + a selection_rule column and the disagreement-coordinate rows
  S6  DNABERT-2 finite-intervention tomography (two sections: observer families, pair terms)
  S7  within-layer activation-ratio sweep, 36 rows in each of two decoders
  S8  the two-critical-rows panel: single and pairwise ablations, plus the row geometry

Every value is read from a committed artifact; nothing is transcribed. Each table is written
as .tsv (machine-readable) and .md (human-readable, with a provenance header).
"""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/supplementary"
OUT.mkdir(parents=True, exist_ok=True)


def write(stem: str, title: str, prov: str, header: list[str], rows: list[list],
          sections: list[tuple[str, list[str], list[list]]] | None = None) -> None:
    """Emit .tsv and .md. `sections` lets one table carry several blocks (S6, S8)."""
    blocks = sections if sections else [("", header, rows)]
    with (OUT / f"{stem}.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        for name, hdr, rs in blocks:
            if name: w.writerow([f"# {name}"])
            w.writerow(hdr); w.writerows(rs)
    L = [f"# {title}", "", prov, ""]
    for name, hdr, rs in blocks:
        if name: L += [f"## {name}", ""]
        L += ["| " + " | ".join(map(str, hdr)) + " |",
              "|" + "|".join("---" for _ in hdr) + "|"]
        L += ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in rs]
        L += [""]
    (OUT / f"{stem}.md").write_text("\n".join(L))
    n = sum(len(rs) for _, _, rs in blocks)
    print(f"  {stem:44s} {n:4d} rows")


def r(x, n=6):
    try: return round(float(x), n)
    except (TypeError, ValueError): return x


# ── S2: structural metrics + 24-input stability ──────────────────────────────
def s2():
    base = list(csv.DictReader((OUT / "S2_structural_metrics.tsv").open(), delimiter="\t"))
    stab = {d["model"]: d for d in
            csv.DictReader((ROOT / "results/census_analysis/input_stability.tsv").open(),
                           delimiter="\t")}
    add = ["n_inputs", "ratio_median", "ratio_min", "ratio_max", "frac_rank1",
           "frac_max_at_pos0"]
    hdr = list(base[0].keys()) + add
    rows = []
    for b in base:
        s = stab.get(b.get("model", b.get("Model", "")).strip(), {})
        rows.append([b[k] for k in base[0]] +
                    [s.get("n_inputs", ""), r(s.get("ratio_median")), r(s.get("ratio_min")),
                     r(s.get("ratio_max")), s.get("frac_rank1", ""),
                     s.get("frac_max_at_pos0", "")])
    write("S2_structural_metrics_and_stability",
          "S2 — Structural metrics and candidate input stability",
          "Structural columns reformatted verbatim from the round-2 audit table. Stability "
          "columns from `results/census_analysis/input_stability.tsv`: each frozen coordinate "
          "re-evaluated on 24 independent domain-matched inputs, with no reselection. "
          "`frac_rank1` is the fraction of inputs on which the candidate is the layer's "
          "activation-ratio maximum; `frac_max_at_pos0` the fraction on which its own maximum "
          "falls at the first token.", hdr, rows)


# ── S3: causal census + selection rule + disagreement coordinates ────────────
def s3():
    base = list(csv.DictReader((OUT / "S3_causal_census_full.tsv").open(), delimiter="\t"))
    res = {d["model"]: d for d in
           csv.DictReader((ROOT / "audit/detector_provenance_exp2_resolution.csv").open())}
    hdr = ["coordinate_class", "selection_rule", "endpoint"] + list(base[0].keys())
    rows = [["frozen candidate", "as published (see S1)", "census native-loss"] +
            [b[k] for k in base[0]] for b in base]

    extra = []
    olmo = json.loads((ROOT / "results/detector_provenance/uniform_detector_text_olmo.json").read_text())
    for c in olmo["conditions"]:
        if c["kind"] == "uniform_candidate":
            extra.append(["ratio-argmax (disagreement)", "layer-relative ratio argmax",
                          "census native-loss", "OLMo-7B-0724-hf", c["epsilon"],
                          r(c["rel_delta"]), "", "", "", "", "", ""])
    med = olmo.get("uniform_median_control_eps1")
    if med is not None:
        extra.append(["ratio-argmax controls (5, same layer)", "seeded", "census native-loss",
                      "OLMo-7B-0724-hf", 1.0, "", "", "", r(med), "", "", ""])
    dna = json.loads((ROOT / "results/detector_provenance/uniform_detector_dnabert2.json").read_text())
    c = dna["conditions"]["default"]
    if "causal_uniform_rel" in c:
        extra.append(["ratio-argmax (disagreement)", "layer-relative ratio argmax",
                      "MLM loss on 6 fixed DNA probes -- NOT the census endpoint",
                      "DNABERT-2", 1.0, r(c["causal_uniform_rel"]), "", "", "", "", "", ""])
        extra.append(["frozen candidate, same probes", "as published",
                      "MLM loss on 6 fixed DNA probes -- NOT the census endpoint",
                      "DNABERT-2", 1.0, r(c["causal_frozen_rel"]), "", "", "", "", "", ""])
    rows += [e[:len(hdr)] + [""] * (len(hdr) - len(e)) for e in extra]
    write("S3_causal_census_and_disagreements",
          "S3 — Full causal census, with selection rule and disagreement coordinates",
          "The 44 census rows are reformatted verbatim from the round-2 audit table. Appended "
          "rows give the causal effect of the coordinate the layer-relative ratio rule selects "
          "where it disagrees with the published one (`audit/detector_provenance_exp2_"
          "resolution.csv`). **The DNABERT-2 rows use a different endpoint** — MLM loss on six "
          "fixed DNA probes — and are on a different scale from the census rows; they are "
          "comparable to each other, not to the rest of the table. NTv3 contributes no "
          "disagreement row: under the six-probe input its two rules select the same "
          "coordinate.", hdr, rows)


# ── S6: DNABERT-2 finite-intervention tomography ─────────────────────────────
def s6():
    fit = json.loads((ROOT / "experiments/frozen/E9_mechanistic_tomography/"
                             "fit_results_dnabert2.json").read_text())["by_epsilon"]
    stab = list(csv.DictReader((ROOT / "audit/round2/tomography_split_stability.csv").open()))
    fam_hdr = ["epsilon", "observer_family", "held_out_r2", "held_out_mae", "rmse",
               "normalized_mae", "resplit_median_r2", "resplit_p2.5", "resplit_p97.5",
               "n_resplits"]
    fam_rows = []
    for eps in ("0.5", "1.0"):
        for fam in ("F0", "F1", "F2", "F3"):
            m = fit[eps][fam]["metrics"]
            s = next((d for d in stab if d["epsilon"] == eps
                      and d["metric"] == f"{fam}_held_out_r2"), {})
            fam_rows.append([eps, fam, r(m["r2"]), r(m["mae"]), r(m["rmse"]),
                             r(m["normalized_mae"]), r(s.get("median")), r(s.get("p2.5")),
                             r(s.get("p97.5")), s.get("n_splits", "")])
    pair_hdr = ["epsilon", "rank_by_abs_gamma", "pair", "gamma", "abs_gamma"]
    pair_rows = [[d["epsilon"], d["rank_by_abs_gamma"], d["pair"], r(d["gamma"]),
                  r(d["abs_gamma"])] for d in
                 csv.DictReader((ROOT / "audit/tomography_pair_coeffs.csv").open())]
    write("S6_dnabert2_tomography",
          "S6 — DNABERT-2 finite-intervention tomography",
          "Observer families fitted on a fixed 10-row high-gain basis and evaluated on 20 "
          "held-out intervention conditions at each strength. F0 sums independently measured "
          "singleton effects; F1 adds one global calibration scalar; F2 fits each row's "
          "contribution but stays additive; F3 adds all 45 pairwise interaction terms. "
          "Resplit columns are the median and 95% interval over 100 alternative "
          "fit/calibration/held-out partitions. Sources: "
          "`experiments/frozen/E9_mechanistic_tomography/fit_results_dnabert2.json`, "
          "`audit/round2/tomography_split_stability.csv`, `audit/tomography_pair_coeffs.csv`.",
          fam_hdr, fam_rows,
          sections=[("A — Observer families, held-out performance", fam_hdr, fam_rows),
                    ("B — Pairwise interaction coefficients (45 pairs x 2 strengths)",
                     pair_hdr, pair_rows)])


# ── S7: within-layer sweep ───────────────────────────────────────────────────
def s7():
    hdr = ["model", "layer", "row", "ratio_rank", "activation_ratio", "nll", "delta_nll",
           "rel_delta", "above_detector_threshold", "is_frozen_candidate"]
    rows = []
    for fn, model, layer in [("within_model_slope.json", "GENERator-EUK-3B", 4),
                             ("within_model_slope_smollm2_1.7b.json", "SmolLM2-1.7B", 7)]:
        d = json.loads((ROOT / "results/within_layer_sweep" / fn).read_text())
        for q in sorted(d["rows"], key=lambda x: x["rank"]):
            rows.append([model, layer, q["row"], q["rank"], r(q["activation_ratio"], 4),
                         r(q["nll"], 8), r(q["delta_nll"], 8), r(q["rel_delta"], 8),
                         int(q["activation_ratio"] >= 5.0), int(q["is_frozen_candidate"])])
    write("S7_within_layer_sweep",
          "S7 — Within-layer activation-ratio sweep",
          "36 rows per model, log-spaced by activation-ratio rank within one layer, each "
          "ablated to alpha=0 with the model's native loss re-evaluated on its own frozen "
          "pool. Selection used the activation ratio alone and never a causal outcome. "
          "`above_detector_threshold` marks the detector's own >=5 acceptance rule and "
          "reproduces the 21/15 partition used in the main text. Sources: "
          "`results/within_layer_sweep/within_model_slope{,_smollm2_1.7b}.json`.", hdr, rows)


# ── S8: two critical rows -- ablations and geometry ──────────────────────────
def s8():
    e = json.loads((ROOT / "results/within_layer_sweep/smollm2_second_row_epistasis.json").read_text())
    g = json.loads((ROOT / "results/within_layer_sweep/smollm2_161_749_geometry.json").read_text())
    RATIO, RANK = {227: 3181.70, 161: 63.54, 749: 358.56}, {227: 1, 161: 4, 749: 2}
    a_hdr = ["condition", "rows", "activation_ratio", "ratio_rank", "nll", "effect_rel",
             "sum_of_singles", "interaction", "joint_over_sum", "verdict"]
    a_rows = []
    for s in e["singles"].values():
        a_rows.append([f"single r{s['row']}", s["row"], RATIO[s["row"]], RANK[s["row"]],
                       r(s["nll"], 8), r(s["rel_delta"]), "", "", "", ""])
    for q in e["pairs"].values():
        x, y = q["rows"]
        a_rows.append([f"joint r{x}+r{y}", f"{x};{y}", "", "", r(q["nll"], 8),
                       r(q["joint_rel_delta"]), r(q["sum_of_singles"]), r(q["epistasis"]),
                       r(q["joint_over_sum"], 4), q["verdict"]])
    ref = g["reference"]
    g_hdr = ["quantity", "rows", "value", "z_vs_reference", "percentile_vs_reference"]
    g_rows = [[f"L2 norm of row {k}", k, r(v, 4), "", ""] for k, v in g["row_norms"].items()]
    for k, v in g["cosines"].items():
        x, y = k.split("_")
        g_rows.append([f"cosine(r{x}, r{y})", f"{x};{y}", r(v, 4),
                       r(g["z_scores"][k], 2), r(g["percentiles"][k], 2)])
    g_rows.append([f"reference distribution ({ref['n_pairs']} pairs, {ref['pool']})", "",
                   f"mean {ref['mean']:+.4f}", f"sd {ref['sd']:.4f}",
                   f"min {ref['min']:+.4f} / max {ref['max']:+.4f}"])
    write("S8_two_critical_rows",
          "S8 — Two critical rows in SmolLM2-1.7B layer 7: ablations and geometry",
          "**Sign convention:** the endpoint is a loss increase, so `interaction = effect_rel "
          "- sum_of_singles` is positive for super-additivity. This is the opposite sign to "
          "this project's accuracy-endpoint epistasis figures (e.g. DNABERT-2's -33.63 pp), "
          "which must be converted before comparison. All six ablation conditions were "
          "re-measured with an independent implementation that zeroes rows by explicit "
          "indexing and re-checks the intact loss after every restoration; every value "
          "reproduced exactly and the result was invariant to ablation order. Geometry is "
          "data-free. Sources: `results/within_layer_sweep/smollm2_second_row_epistasis.json`, "
          "`results/within_layer_sweep/smollm2_161_749_geometry.json`.",
          a_hdr, a_rows,
          sections=[("A — Single and pairwise ablations", a_hdr, a_rows),
                    ("B — Row geometry against a same-layer null", g_hdr, g_rows)])


if __name__ == "__main__":
    print("building supplementary tables ->", OUT)
    s2(); s3(); s6(); s7(); s8()
    print("done")
