#!/usr/bin/env python3
"""Compile E13 raw model checkpoints into preregistered tables and correlations."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "results" / "experiments" / "E13"
MANIFEST = RESULTS / "candidate_manifest.json"
STRUCTURAL = ROOT / "results" / "experiments" / "E11" / "scale_ladder_backfilled.csv"
N_BOOT = 5000
SEED = 42


def weighted(pb, indices=None):
    if indices is None:
        indices = range(len(pb))
    vals = [pb[i] for i in indices]
    return sum(x[0] for x in vals) / max(sum(x[1] for x in vals), 1)


def paired_relative_ci(base, pert, seed):
    rng = np.random.default_rng(seed)
    n = len(base)
    vals = np.empty(N_BOOT)
    for i in range(N_BOOT):
        ix = rng.integers(0, n, n)
        b, p = weighted(base, ix), weighted(pert, ix)
        vals[i] = (p - b) / b
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"refusing to write empty {path}")
    fields = list(rows[0])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def candidate_control_q1_gap(meta):
    """Return the locked structural q1 gap when the source measured controls.

    Older E7 artifacts did not measure structural control spectra.  Those cells stay
    missing rather than being reconstructed after causal responses are available.
    """
    source = ROOT / meta["source"]
    if not source.exists():
        return None
    payload = json.loads(source.read_text())
    controls = payload.get("control_rows", [])
    control_q1 = [float(row["q1"]) for row in controls if row.get("q1") is not None]
    if not control_q1 or meta.get("q1") is None:
        return None
    return float(meta["q1"]) - median(control_q1)


def correlation_payload(rows, xkey, primary_only):
    selected = [r for r in rows if (r["primary"] or not primary_only) and r[xkey] is not None]
    x = np.asarray([r[xkey] for r in selected], float)
    y = np.asarray([r["relative_loss_change"] for r in selected], float)
    rho, p = spearmanr(x, y)
    # Model-cluster bootstrap: draw models, then retain all their candidate rows.  Duplicate
    # sampled models are represented as duplicate clusters, never as independent rows.
    models = sorted({r["model"] for r in selected})
    by_model = {m: [r for r in selected if r["model"] == m] for m in models}
    rng = np.random.default_rng(SEED + (1 if primary_only else 2) + (10 if xkey == "q1" else 20))
    boot = []
    for _ in range(N_BOOT):
        draw = rng.choice(models, len(models), replace=True)
        rs = [r for m in draw for r in by_model[m]]
        rr = spearmanr([r[xkey] for r in rs], [r["relative_loss_change"] for r in rs]).statistic
        if np.isfinite(rr): boot.append(float(rr))
    return {"x": xkey, "analysis": "one-primary-per-model" if primary_only else "all-candidate_model-clustered",
            "n_models": len(models), "n_candidates": len(selected), "spearman_rho": float(rho),
            "asymptotic_p": float(p), "model_cluster_bootstrap_ci95":
            [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "n_boot": N_BOOT}


def main():
    manifest = json.loads(MANIFEST.read_text())
    structural = {r["model"]: r for r in csv.DictReader(STRUCTURAL.open())}
    raw_paths = sorted((RESULTS / "raw").glob("*.json"))
    raw = {json.loads(p.read_text())["model"]: json.loads(p.read_text()) for p in raw_paths}
    missing = [m for m in manifest["panel_order"] if m not in raw]
    if missing:
        raise SystemExit(f"missing raw cohort models ({len(missing)}): {missing}")
    candidates_by_coord = {(c["model"],c["layer"],c["row"]):c for c in manifest["candidates"]}
    cand_rows, ctrl_rows, all_rows = [], [], []
    seed_seq = np.random.SeedSequence(SEED).spawn(10000)
    si = 0
    for model in manifest["panel_order"]:
        d, sr = raw[model], structural[model]
        base = d["baseline_per_unit"]
        for c in d["conditions"]:
            ci = paired_relative_ci(base, c["per_unit"], seed_seq[si]); si += 1
            common = {"model":model,"domain":d["domain"],"architecture":d["architecture"],
                      "total_params":int(sr["total_params"]),"layer":int(c["layer"]),
                      "row":int(c["row"]),"epsilon":float(c["epsilon"]),
                      "baseline_loss":float(d["baseline_loss"]),
                      "perturbed_loss":float(c["perturbed_loss"]),
                      "absolute_delta_loss":float(c["absolute_delta_loss"]),
                      "relative_loss_change":float(c["relative_loss_change"]),
                      "relative_ci95_low":ci[0],"relative_ci95_high":ci[1]}
            if c["kind"] == "candidate":
                meta = candidates_by_coord[(model,int(c["layer"]),int(c["row"]))]
                row = {**common,"candidate_index":int(meta["candidate_index"]),
                       "primary":bool(meta["primary"]),"q1":meta["q1"],
                       "frob_norm":meta["frob_norm"],
                       "layer_median_frob_norm":meta["layer_median_frob_norm"],
                       "frob_ratio_to_layer_median":meta["frob_ratio_to_layer_median"],
                       "candidate_control_structural_q1_gap":candidate_control_q1_gap(meta)}
                cand_rows.append(row)
            else:
                row = {**common,"control_index":None}
                ctrl_rows.append(row)
            all_rows.append({"kind":c["kind"],**common})
    # Stable control index within each (model, layer, epsilon) group.
    groups = {}
    for r in ctrl_rows:
        key=(r["model"],r["layer"],r["epsilon"]); groups.setdefault(key,[]).append(r)
    for rs in groups.values():
        for i,r in enumerate(sorted(rs,key=lambda z:z["row"])): r["control_index"]=i
    write_csv(RESULTS / "candidate_effects.csv", cand_rows)
    write_csv(RESULTS / "control_effects.csv", ctrl_rows)
    write_csv(RESULTS / "causal_census.csv", all_rows)
    full = [r for r in cand_rows if r["epsilon"] == 1.0]
    corr = {"n_boot":N_BOOT,"seed":SEED,"analyses":[]}
    for key in ("q1","frob_ratio_to_layer_median"):
        for primary in (True,False): corr["analyses"].append(correlation_payload(full,key,primary))
    (RESULTS / "structure_function_correlations.json").write_text(json.dumps(corr,indent=2)+"\n")
    print(f"compiled {len(cand_rows)} candidate and {len(ctrl_rows)} control effects")


if __name__ == "__main__": main()
