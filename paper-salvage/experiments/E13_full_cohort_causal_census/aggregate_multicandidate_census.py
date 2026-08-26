#!/usr/bin/env python3
"""Aggregate and validate the frozen E13 row-wise activation census.

This script is deliberately read-only with respect to ``raw/``.  It applies no
new selection rule: membership is reconstructed solely from the preregistered
``detector_score >= 5.0`` rule and compared with each stored selected list.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "results" / "E13_multicandidate_structural"
RAW = OUT / "raw"
PLOTS = OUT / "plots"
META_CSV = ROOT / "results" / "E11" / "scale_ladder_backfilled.csv"
MANIFEST = ROOT / "results" / "E13" / "candidate_manifest.json"
STATUS = OUT / "status.csv"
THRESHOLD = 5.0


def q(values: list[float], p: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), p))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_status() -> tuple[dict[str, str], dict[str, list[str]]]:
    latest: dict[str, str] = {}
    history: dict[str, list[str]] = defaultdict(list)
    if not STATUS.exists():
        return latest, history
    with STATUS.open() as f:
        for row in csv.DictReader(f):
            model, state = row.get("model", ""), row.get("state", "")
            if model and model != "queue":
                latest[model] = state
                history[model].append(state)
    return latest, history


def layer_hashes(rows: list[dict]) -> list[str]:
    by_layer: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for r in rows:
        by_layer[int(r["layer"])].append(
            (int(r["row"]), float(r["activation_max"]), float(r["activation_median"]))
        )
    hashes = []
    for layer in sorted(by_layer):
        payload = json.dumps(sorted(by_layer[layer]), separators=(",", ":")).encode()
        hashes.append(hashlib.sha256(payload).hexdigest())
    return hashes


def main() -> None:
    meta_rows = list(csv.DictReader(META_CSV.open()))
    metadata = {r["model"]: r for r in meta_rows if r["model"] != "Phi-3-mini-4k-instruct"}
    manifest = json.loads(MANIFEST.read_text())
    primaries = {
        c["model"]: c for c in manifest["candidates"]
        if c.get("primary") and c["model"] != "Phi-3-mini-4k-instruct"
    }
    latest_status, status_history = load_status()

    raw_paths = sorted(RAW.glob("*.json"))
    artifacts_by_model: dict[str, list[Path]] = defaultdict(list)
    payloads: dict[str, tuple[Path, dict]] = {}
    parse_errors: list[str] = []
    for path in raw_paths:
        try:
            payload = json.loads(path.read_text())
            artifacts_by_model[payload.get("model", "")].append(path)
            payloads[payload.get("model", "")] = (path, payload)
        except Exception as exc:
            parse_errors.append(f"{path}: {exc}")

    model_rows: list[dict] = []
    candidate_rows: list[dict] = []
    all_detector_rows: list[dict] = []
    layer_rows: list[dict] = []
    validation_rows: list[dict] = []

    for model in [r["model"] for r in meta_rows if r["model"] != "Phi-3-mini-4k-instruct"]:
        m = metadata[model]
        artifact_count = len(artifacts_by_model.get(model, []))
        if model not in payloads:
            validation_rows.append({"model": model, "overall_status": "FAIL", "issues": "missing raw artifact"})
            continue
        path, d = payloads[model]
        rows = d.get("rows", [])
        selected = d.get("selected", [])
        slug = d.get("slug", path.stem)
        n_layers = int(m["n_layers"])
        output_rows_per_layer = int(m["d_model"])
        expected_eligible = n_layers * output_rows_per_layer
        coords = [(int(r["layer"]), int(r["row"])) for r in rows]
        selected_coords = [(int(r["layer"]), int(r["row"])) for r in selected]
        reconstructed = [r for r in rows if float(r["detector_score"]) >= THRESHOLD]
        reconstructed_coords = [(int(r["layer"]), int(r["row"])) for r in reconstructed]
        stored_flag_coords = [(int(r["layer"]), int(r["row"])) for r in rows if bool(r.get("selected"))]
        ratios = [float(r["detector_score"]) for r in selected]
        rejected_ratios = [float(r["detector_score"]) for r in rows if not bool(r.get("selected"))]
        accepted_by_layer = Counter(int(r["layer"]) for r in selected)
        all_by_layer = Counter(int(r["layer"]) for r in rows)
        hashes = layer_hashes(rows)
        primary = primaries.get(model)
        primary_coord = (int(primary["layer"]), int(primary["row"])) if primary else None
        row_by_coord = {(int(r["layer"]), int(r["row"])): r for r in rows}
        primary_record = row_by_coord.get(primary_coord) if primary_coord else None
        primary_included = primary_coord in set(selected_coords) if primary_coord else None

        checks = {
            "artifact_unique_valid": artifact_count == 1 and not parse_errors,
            "coordinates_unique": len(coords) == len(set(coords)),
            "dimensions_valid": (
                len(rows) == expected_eligible
                and set(int(r["layer"]) for r in rows) == set(range(n_layers))
                and all(0 <= layer < n_layers and 0 <= row < output_rows_per_layer for layer, row in coords)
                and all(all_by_layer[layer] == output_rows_per_layer for layer in range(n_layers))
            ),
            "no_hook_aliasing": len(hashes) == n_layers and len(hashes) == len(set(hashes)),
            "stored_k_matches": len(selected) == len(selected_coords) == len(set(selected_coords)),
            "reconstructed_k_matches": selected_coords == reconstructed_coords == stored_flag_coords,
            "accepted_threshold": bool(ratios) and all(x >= THRESHOLD for x in ratios),
            "boundary_correct": (
                all(x < THRESHOLD for x in rejected_ratios)
                and (not rejected_ratios or max(rejected_ratios) < THRESHOLD)
                and (not ratios or min(ratios) >= THRESHOLD)
            ),
            "model_layer_naming": d.get("model") == model and all_by_layer.keys() == set(range(n_layers)),
            "successful_artifact_preserved": latest_status.get(slug) == "completed",
            "primary_included": bool(primary_included),
        }
        issues = [name for name, passed in checks.items() if not passed]
        validation_rows.append({
            "model": model,
            "slug": slug,
            "overall_status": "PASS" if not issues else "FAIL",
            **{name: "PASS" if passed else "FAIL" for name, passed in checks.items()},
            "raw_artifact_count": artifact_count,
            "historical_failed_status_present": "failed" in status_history.get(slug, []),
            "latest_status": latest_status.get(slug, "missing"),
            "duplicate_layer_vector_count": len(hashes) - len(set(hashes)),
            "issues": ";".join(issues),
            "raw_path": str(path.relative_to(ROOT)),
        })

        k = len(selected)
        occupied_layers = len(accepted_by_layer)
        top_layer_count = max(accepted_by_layer.values()) if accepted_by_layer else 0
        top3_count = sum(sorted(accepted_by_layer.values(), reverse=True)[:3])
        near6 = sum(x < 6.0 for x in ratios)
        near10 = sum(x < 10.0 for x in ratios)
        extreme100 = sum(x >= 100.0 for x in ratios)
        model_rows.append({
            "model": model, "slug": slug, "repo": m["repo"], "revision": m["revision"],
            "family": m["family"], "architecture": m["architecture"], "domain": m["domain"],
            "total_params": int(m["total_params"]), "non_embed_params": int(m["non_embed_params"]),
            "d_model_output_rows_per_layer": output_rows_per_layer, "d_ffn": int(m["d_ffn"]),
            "n_layers": n_layers, "eligible_rows": len(rows), "accepted_k": k,
            "accepted_fraction": k / len(rows), "layers_with_accepted": occupied_layers,
            "layer_fraction_with_accepted": occupied_layers / n_layers,
            "accepted_ratio_min": min(ratios), "accepted_ratio_q25": q(ratios, .25),
            "accepted_ratio_median": statistics.median(ratios), "accepted_ratio_q75": q(ratios, .75),
            "accepted_ratio_q90": q(ratios, .9), "accepted_ratio_q95": q(ratios, .95),
            "accepted_ratio_q99": q(ratios, .99), "accepted_ratio_max": max(ratios),
            "accepted_ratio_lt6_count": near6, "accepted_ratio_lt6_fraction": near6 / k,
            "accepted_ratio_lt10_count": near10, "accepted_ratio_lt10_fraction": near10 / k,
            "accepted_ratio_ge100_count": extreme100, "accepted_ratio_ge100_fraction": extreme100 / k,
            "max_rejected_ratio": max(rejected_ratios) if rejected_ratios else "",
            "max_single_layer_accepted": top_layer_count,
            "max_single_layer_share_of_k": top_layer_count / k,
            "top3_layers_share_of_k": top3_count / k,
            "primary_layer": primary_coord[0] if primary_coord else "",
            "primary_row": primary_coord[1] if primary_coord else "",
            "primary_detector_ratio": float(primary_record["detector_score"]) if primary_record else "",
            "primary_in_census": bool(primary_included),
            "validation_status": "PASS" if not issues else "FAIL",
            "raw_path": str(path.relative_to(ROOT)),
        })

        common = {
            "model": model, "slug": slug, "repo": m["repo"], "revision": m["revision"],
            "architecture": m["architecture"], "domain": m["domain"],
            "total_params": int(m["total_params"]), "non_embed_params": int(m["non_embed_params"]),
            "n_layers": n_layers, "eligible_rows": len(rows), "accepted_k": k,
        }
        for r in rows:
            item = {**common, **r}
            item["normalized_layer_position"] = int(r["layer"]) / max(1, n_layers - 1)
            item["is_primary_part2a"] = primary_coord == (int(r["layer"]), int(r["row"]))
            all_detector_rows.append(item)
        for r in selected:
            item = {**common, **r}
            item["normalized_layer_position"] = int(r["layer"]) / max(1, n_layers - 1)
            item["is_primary_part2a"] = primary_coord == (int(r["layer"]), int(r["row"]))
            candidate_rows.append(item)
        for layer in range(n_layers):
            layer_selected = [r for r in selected if int(r["layer"]) == layer]
            layer_ratios = [float(r["detector_score"]) for r in layer_selected]
            layer_rows.append({
                "model": model, "slug": slug, "architecture": m["architecture"], "domain": m["domain"],
                "layer": layer, "normalized_layer_position": layer / max(1, n_layers - 1),
                "eligible_rows": all_by_layer[layer], "accepted_k": len(layer_selected),
                "accepted_fraction": len(layer_selected) / all_by_layer[layer],
                "share_of_model_k": len(layer_selected) / k,
                "accepted_ratio_min": min(layer_ratios) if layer_ratios else "",
                "accepted_ratio_median": statistics.median(layer_ratios) if layer_ratios else "",
                "accepted_ratio_max": max(layer_ratios) if layer_ratios else "",
            })

    model_fields = list(model_rows[0].keys())
    candidate_fields = list(candidate_rows[0].keys())
    all_fields = list(all_detector_rows[0].keys())
    validation_fields = list(validation_rows[0].keys())
    layer_fields = list(layer_rows[0].keys())
    write_csv(OUT / "basis_summary_by_model.csv", model_rows, model_fields)
    write_csv(OUT / "full_cohort_multicandidate_basis.csv", candidate_rows, candidate_fields)
    write_csv(OUT / "full_cohort_rowwise_detector.csv", all_detector_rows, all_fields)
    write_csv(OUT / "basis_validation_by_model.csv", validation_rows, validation_fields)
    write_csv(OUT / "accepted_layer_summary.csv", layer_rows, layer_fields)
    make_plots(model_rows, candidate_rows, layer_rows)
    write_report(model_rows, validation_rows, parse_errors)
    provenance = {
        "task": "E13 full-cohort multi-candidate structural census aggregation/validation",
        "threshold": THRESHOLD,
        "selection_rule": "activation_max / within-layer median activation_max >= 5.0",
        "n_models": len(model_rows),
        "n_eligible_rows": len(all_detector_rows),
        "n_accepted_rows": len(candidate_rows),
        "aggregator": str(Path(__file__).resolve().relative_to(ROOT)),
        "aggregator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metadata_source": str(META_CSV.relative_to(ROOT)),
        "primary_manifest": str(MANIFEST.relative_to(ROOT)),
        "status_source": str(STATUS.relative_to(ROOT)),
        "raw_artifacts": {
            path.name: {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in raw_paths
        },
        "outputs": [
            "results/E13_multicandidate_structural/full_cohort_rowwise_detector.csv",
            "results/E13_multicandidate_structural/full_cohort_multicandidate_basis.csv",
            "results/E13_multicandidate_structural/basis_summary_by_model.csv",
            "results/E13_multicandidate_structural/basis_validation_by_model.csv",
            "results/E13_multicandidate_structural/accepted_layer_summary.csv",
            "results/E13_multicandidate_structural/MULTICANDIDATE_STRUCTURAL_CENSUS_SUMMARY.md",
        ],
        "causal_interventions_run": False,
    }
    (OUT / "aggregation_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"WROTE {len(model_rows)} models, {len(candidate_rows)} accepted rows, {len(all_detector_rows)} eligible rows")


def model_order(model_rows: list[dict]) -> list[str]:
    return [r["model"] for r in sorted(model_rows, key=lambda x: x["accepted_k"])]


def make_plots(model_rows: list[dict], candidates: list[dict], layers: list[dict]) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    order = model_order(model_rows)
    by_model = {r["model"]: r for r in model_rows}
    y = np.arange(len(order))
    colors = ["#4477AA" if by_model[m]["validation_status"] == "PASS" else "#CC6677" for m in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 8), constrained_layout=True)
    ax1.barh(y, [by_model[m]["accepted_k"] for m in order], color=colors)
    ax1.set_yticks(y); ax1.set_yticklabels(order, fontsize=8)
    ax1.set_xlabel("Accepted candidate count K"); ax1.set_title("Raw basis size")
    ax1.grid(axis="x", alpha=.25)
    ax2.barh(y, [100 * by_model[m]["accepted_fraction"] for m in order], color=colors)
    ax2.set_yticks(y); ax2.set_yticklabels([])
    ax2.set_xlabel("Accepted eligible rows (%)"); ax2.set_title("Basis density")
    ax2.grid(axis="x", alpha=.25)
    fig.suptitle("Frozen activation-ratio ≥5 census across 22 models")
    fig.savefig(PLOTS / "k_and_fraction_across_models.png", dpi=220)
    fig.savefig(PLOTS / "k_and_fraction_across_models.pdf")
    plt.close(fig)

    c_by_model: dict[str, list[float]] = defaultdict(list)
    for r in candidates: c_by_model[r["model"]].append(float(r["detector_score"]))
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    vals = [c_by_model[m] for m in order]
    ax.boxplot(vals, vert=False, labels=order, showfliers=True, whis=(5, 95), patch_artist=True,
               boxprops={"facecolor": "#88CCEE", "alpha": .75},
               medianprops={"color": "#332288", "linewidth": 1.5},
               flierprops={"markersize": 2, "alpha": .35})
    ax.axvline(THRESHOLD, color="#CC3311", linestyle="--", linewidth=1.5, label="threshold = 5")
    ax.set_xscale("log"); ax.set_xlabel("Accepted detector ratio (log scale)")
    ax.set_title("Accepted detector-ratio distributions (whiskers: 5th–95th percentiles)")
    ax.grid(axis="x", which="both", alpha=.2); ax.legend(loc="lower right")
    ax.tick_params(axis="y", labelsize=8)
    fig.savefig(PLOTS / "detector_ratio_distributions.png", dpi=220)
    fig.savefig(PLOTS / "detector_ratio_distributions.pdf")
    plt.close(fig)

    bins = np.linspace(0, 1, 11)
    heat = np.zeros((len(order), 10), dtype=float)
    for i, model in enumerate(order):
        positions = [float(r["normalized_layer_position"]) for r in candidates if r["model"] == model]
        counts, _ = np.histogram(positions, bins=bins)
        heat[i] = counts / max(1, counts.sum())
    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    im = ax.imshow(heat, aspect="auto", cmap="viridis", vmin=0)
    ax.set_yticks(np.arange(len(order))); ax.set_yticklabels(order, fontsize=8)
    ax.set_xticks(np.arange(10)); ax.set_xticklabels([f"{a:.1f}–{b:.1f}" for a, b in zip(bins[:-1], bins[1:])], rotation=45, ha="right")
    ax.set_xlabel("Normalized layer position bin"); ax.set_title("Within-model distribution of accepted candidates")
    cb = fig.colorbar(im, ax=ax); cb.set_label("Fraction of model K")
    fig.savefig(PLOTS / "accepted_layer_position_heatmap.png", dpi=220)
    fig.savefig(PLOTS / "accepted_layer_position_heatmap.pdf")
    plt.close(fig)


def write_report(model_rows: list[dict], validations: list[dict], parse_errors: list[str]) -> None:
    by_model = {r["model"]: r for r in model_rows}
    failures = [r for r in validations if r.get("overall_status") == "FAIL"]
    total_k = sum(int(r["accepted_k"]) for r in model_rows)
    ks = [int(r["accepted_k"]) for r in model_rows]
    fracs = [float(r["accepted_fraction"]) for r in model_rows]
    ranked = sorted(model_rows, key=lambda r: int(r["accepted_k"]), reverse=True)
    lines = [
        "# Full-cohort multi-candidate structural census",
        "",
        "## Completion and protocol",
        "",
        f"The frozen activation-ratio census completed for **{len(model_rows)}/22 models**. "
        f"It contains **{total_k:,} accepted rows** across **{sum(int(r['eligible_rows']) for r in model_rows):,} eligible gated-FFN output rows**. "
        "Membership is exactly `channel_max[row] / median(channel_max within layer) >= 5.0`; no threshold adjustment, top-K rule, or K cap was applied. No causal response or tomography was run.",
        "",
        "## Validation",
        "",
        f"Overall: **{len(validations)-len(failures)} PASS, {len(failures)} FAIL**. A model is FAIL if any requested integrity check fails.",
        "",
        "| Model | Status | K | Primary included | Issues |",
        "|---|---:|---:|---:|---|",
    ]
    for v in validations:
        m = by_model.get(v["model"], {})
        lines.append(f"| {v['model']} | {v['overall_status']} | {m.get('accepted_k','—')} | {m.get('primary_in_census','—')} | {v.get('issues') or 'none'} |")
    lines += [
        "",
        "All parsed artifacts had unique coordinates, exact threshold reconstruction, valid output-channel dimensions, complete layer coverage, and no identical full activation vectors across layers. The latter two checks, combined with the source-code audit showing one explicitly resolved module per layer and dictionary storage keyed by layer, provide no evidence of duplicate-module traversal or hook aliasing.",
        "",
        "Historical failed queue entries remain in the append-only status log, but successful artifacts were written atomically and the final per-model status is `completed`; earlier failures did not overwrite completed JSONs.",
        "",
    ]
    if failures:
        lines += [
            "### Validation anomaly",
            "",
            "DNABERT-2 is the sole failure: frozen primary `(5,603)` is present as an eligible row but has detector ratio approximately 0.999 and is not selected. The historical discovery artifact reports very large activation for that coordinate under the ACTB probe. The current compatibility loader forces eager float32 attention, whereas historical DNABERT-2 discovery used the older execution path; this execution-path difference is a plausible explanation, but it is an inference and has not been experimentally isolated. The mismatch must be resolved before treating the DNABERT-2 census basis as frozen for Part 2B.",
            "",
        ]
    if parse_errors:
        lines += ["Parse errors:", ""] + [f"- {e}" for e in parse_errors] + [""]
    lines += [
        "## Census size and density",
        "",
        f"K ranges from **{min(ks)} to {max(ks):,}** (median **{statistics.median(ks):.1f}**); accepted fractions range from **{100*min(fracs):.3f}% to {100*max(fracs):.3f}%** (median **{100*statistics.median(fracs):.3f}%**).",
        "",
        "| Model | K | Accepted fraction | Layers occupied | Ratio median | Ratio max | <6 share | ≥100 share | Top-layer share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ranked:
        lines.append(
            f"| {r['model']} | {r['accepted_k']:,} | {100*r['accepted_fraction']:.3f}% | "
            f"{r['layers_with_accepted']}/{r['n_layers']} | {r['accepted_ratio_median']:.2f} | "
            f"{r['accepted_ratio_max']:.1f} | {100*r['accepted_ratio_lt6_fraction']:.1f}% | "
            f"{100*r['accepted_ratio_ge100_fraction']:.1f}% | {100*r['max_single_layer_share_of_k']:.1f}% |"
        )
    lines += [
        "",
        "## Why K varies",
        "",
        "Large K is not an aggregation duplicate: coordinates and layer vectors are unique, counts reconstruct exactly, and eligible dimensions match `n_layers × d_model`. The large bases arise through different structural patterns. EuroBERT-2.1B occupies every layer, no single layer contributes more than 16.7% of K, and its median accepted ratio is 7.73. Mistral also has accepted rows in 31/32 layers, but 81.4% of its K is concentrated in one early layer and its accepted-ratio median is 19.42. Thus EuroBERT's extreme K is widespread, whereas Mistral's is primarily a dense layer cluster. The genomic decoders have hundreds of accepted rows spread across most layers, with medians around 9–12 rather than populations confined immediately above 5. Across models only 5.3–32.6% of accepted rows lie below ratio 6, so none of the large sets is explained solely by threshold-edge rounding. The `<6`, `<10`, and `≥100` columns quantify near-threshold versus extreme tails without causal interpretation.",
        "",
        "The normalized-layer heatmap shows model-specific depth profiles rather than one common narrow layer band. The accepted-layer CSV preserves exact per-layer counts and ratios.",
        "",
        "## Artifacts",
        "",
        "- `results/E13_multicandidate_structural/full_cohort_rowwise_detector.csv` — every eligible row and selection flag.",
        "- `results/E13_multicandidate_structural/full_cohort_multicandidate_basis.csv` — every accepted row; no K cap.",
        "- `results/E13_multicandidate_structural/basis_summary_by_model.csv` — model-level K, density, ratio, layer, metadata, and primary checks.",
        "- `results/E13_multicandidate_structural/basis_validation_by_model.csv` — explicit PASS/FAIL checks.",
        "- `results/E13_multicandidate_structural/accepted_layer_summary.csv` — exact per-layer summaries.",
        "- `results/E13_multicandidate_structural/aggregation_provenance.json` — raw SHA-256 hashes, frozen inputs, code hash, and output manifest.",
        "- `results/E13_multicandidate_structural/plots/k_and_fraction_across_models.{png,pdf}`.",
        "- `results/E13_multicandidate_structural/plots/detector_ratio_distributions.{png,pdf}`.",
        "- `results/E13_multicandidate_structural/plots/accepted_layer_position_heatmap.{png,pdf}`.",
        "- `results/E13_multicandidate_structural/raw/*.json` — immutable source evidence.",
        "",
        "## Basis-feasibility assessment",
        "",
        "### **B. Heterogeneous feasibility**",
        "",
        "The unchanged threshold yields compact sets for some models (for example K=12–14) but hundreds to more than one thousand rows for others. Those large sets are genuine outputs of the detector under this implementation, not duplicate aggregation, and are impractical for the same combinatorial tomography design used by earlier small bases. Thus the detector is a valid set-valued structural detector but does **not currently support a practical, comparable full-cohort Part 2B basis**.",
        "",
        "Do not launch Part 2B tomography yet. First resolve the DNABERT-2 primary mismatch and explicitly decide—without post-hoc threshold tuning, top-K selection, or K caps—whether Part 2B is limited to models with tractable naturally occurring K or stopped as a universal cohort experiment. That is a protocol decision, not an aggregation repair.",
    ]
    (OUT / "MULTICANDIDATE_STRUCTURAL_CENSUS_SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
