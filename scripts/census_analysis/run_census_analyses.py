#!/usr/bin/env python3
"""
EXPERIMENTS 11, 12, N, O — analyses computable from already-recorded census outputs.

  EXP11 extreme-effect audit        : are the tail effects (incl. the ~+700% case) real?
  EXP12 family-clustered sensitivity: do correlations survive one-model-per-family?
  N     threshold sensitivity       : candidate counts at ratio > 3 / 5 / 10
  O     input stability             : rank + ratio across 22 models x 24 inputs

No model is executed; nothing is reselected.
"""
from __future__ import annotations
import csv, json, glob, os, re
import numpy as np
from scipy import stats

ROOT = "/home/nvidia/superweights"
OUT = os.path.join(ROOT, "results/analyses/census_analysis")
SEED = 42

per = list(csv.DictReader(open(os.path.join(OUT, "activation_vs_causality_per_model.tsv")), delimiter="\t"))
cen = list(csv.DictReader(open(os.path.join(ROOT, "audit/census_master.csv"))))
cmap = {c["model_id"]: c for c in cen}
def fl(x):
    try:
        v = float(x); return v if np.isfinite(v) else np.nan
    except Exception: return np.nan

# ── family assignment ────────────────────────────────────────────────────────
def family(mid):
    m = mid.lower()
    for k in ("qwen", "smollm", "eurobert", "modernbert", "mosaicbert", "generator",
              "llama", "mistral", "olmo", "dnabert", "ntv3", "genomeocean"):
        if k in m: return k
    return re.split(r"[-/_]", m)[0]

for r in per:
    r["family"] = family(r["model_id"])

# ── EXP 11: extreme-effect audit ─────────────────────────────────────────────
rows11 = []
eff = sorted(per, key=lambda r: -fl(r["R_cand_eps1.0"]))
top3 = eff[:3]
negs = [r for r in per if fl(r["R_cand_eps1.0"]) < 0]
for r in top3 + negs:
    c = cmap[r["model_id"]]
    base = fl(c["baseline_loss"]); rel = fl(c["R_cand_eps1.0"])
    ctrl = [fl(c[f"R_ctrl{i}_eps1.0"]) for i in range(1, 6)]
    rows11.append(dict(
        model_id=r["model_id"], tail=("positive_top3" if r in top3 else "negative"),
        baseline_loss=base, relative_delta_eps1_0=rel,
        perturbed_loss_implied=base * (1 + rel) if np.isfinite(base) and np.isfinite(rel) else np.nan,
        endpoint=c["endpoint_type"], n_eval_units=c["n_eval_units"], batch_size=c["batch_size"],
        median_control_eps1_0=fl(c["median_control_eps1.0"]),
        control_spread=float(np.nanmax(ctrl) - np.nanmin(ctrl)) if any(np.isfinite(ctrl)) else np.nan,
        G_eps1_0=fl(c["G_eps1.0"]),
        baseline_is_small=bool(np.isfinite(base) and base < 0.5),
        activation_ratio=fl(r["activation_ratio"]),
        note=("tiny denominator risk" if np.isfinite(base) and base < 0.5 else "baseline in normal range")))

# ── EXP 12: family-clustered sensitivity ─────────────────────────────────────
PRED = ["activation_ratio", "q1", "layer_relative_frobenius"]
END = "R_cand_eps1.0"
fams = sorted({r["family"] for r in per})
rows12 = []
for pred in PRED:
    x = np.array([fl(r[pred]) for r in per]); y = np.array([fl(r[END]) for r in per])
    m = np.isfinite(x) & np.isfinite(y)
    rho_all, p_all = stats.spearmanr(x[m], y[m])
    # (a) deterministic: first model per family (alphabetical, no outcome peeking)
    rep = {}
    for r in sorted(per, key=lambda r: r["model_id"]):
        rep.setdefault(r["family"], r)
    xs = np.array([fl(v[pred]) for v in rep.values()]); ys = np.array([fl(v[END]) for v in rep.values()])
    mm = np.isfinite(xs) & np.isfinite(ys)
    rho_rep, p_rep = stats.spearmanr(xs[mm], ys[mm]) if mm.sum() >= 5 else (np.nan, np.nan)
    # (b) repeated random one-per-family
    rng = np.random.default_rng(SEED)
    byf = {}
    for r in per: byf.setdefault(r["family"], []).append(r)
    draws = []
    for _ in range(2000):
        pick = [byf[f][rng.integers(0, len(byf[f]))] for f in fams]
        xa = np.array([fl(v[pred]) for v in pick]); ya = np.array([fl(v[END]) for v in pick])
        k = np.isfinite(xa) & np.isfinite(ya)
        if k.sum() >= 5:
            draws.append(stats.spearmanr(xa[k], ya[k]).statistic)
    draws = np.array([d for d in draws if np.isfinite(d)])
    rows12.append(dict(predictor=pred, endpoint=END,
                       rho_all=float(rho_all), p_all=float(p_all), n_all=int(m.sum()),
                       rho_one_per_family_det=float(rho_rep), p_one_per_family_det=float(p_rep),
                       n_families=len(fams),
                       rho_resample_mean=float(draws.mean()) if draws.size else np.nan,
                       rho_resample_lo=float(np.percentile(draws, 2.5)) if draws.size else np.nan,
                       rho_resample_hi=float(np.percentile(draws, 97.5)) if draws.size else np.nan,
                       frac_draws_positive=float((draws > 0).mean()) if draws.size else np.nan))

# ── N: threshold sensitivity + O: input stability ────────────────────────────
stab_rows, all_ratios, all_ranks = [], [], []
for f in sorted(glob.glob(os.path.join(ROOT, "results/experiments/E13_candidate_stability/*.json"))):
    j = json.load(open(f)); s = j["summary"]
    rr = [p["activation_ratio"] for p in j["per_input"]]
    rk = [p["rank"] for p in j["per_input"]]
    pos0 = [p.get("max_position") == 0 for p in j["per_input"]]
    all_ratios += rr; all_ranks += rk
    stab_rows.append(dict(model=s["model"], slug=s["slug"], domain=s["domain"],
                          architecture=s["architecture"], layer=s["layer"], row=s["row"],
                          n_inputs=len(rr), ratio_min=float(np.min(rr)),
                          ratio_median=float(np.median(rr)), ratio_max=float(np.max(rr)),
                          frac_rank1=float(np.mean([r == 1 for r in rk])),
                          frac_max_at_pos0=float(np.mean(pos0)),
                          gt3=int(np.sum(np.array(rr) > 3)), gt5=int(np.sum(np.array(rr) > 5)),
                          gt10=int(np.sum(np.array(rr) > 10))))
med_by_model = np.array([r["ratio_median"] for r in stab_rows])
thresh = dict(n_models=len(stab_rows),
              models_gt3=int((med_by_model > 3).sum()),
              models_gt5=int((med_by_model > 5).sum()),
              models_gt10=int((med_by_model > 10).sum()),
              min_median_ratio=float(med_by_model.min()),
              max_median_ratio=float(med_by_model.max()),
              n_obs=len(all_ratios),
              frac_rank1_overall=float(np.mean([r == 1 for r in all_ranks])))

def w(name, rows, cols):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols, delimiter="\t"); wr.writeheader()
        for r in rows: wr.writerow({c: ("" if r.get(c) is None else r[c]) for c in cols})
    print(f"wrote {name} ({len(rows)} rows)")

w("extreme_effect_audit.tsv", rows11, list(rows11[0]))
w("family_clustered_sensitivity.tsv", rows12, list(rows12[0]))
w("input_stability.tsv", stab_rows, list(stab_rows[0]))
json.dump(thresh, open(os.path.join(OUT, "threshold_sensitivity.json"), "w"), indent=2)

print("\n=== EXP11 extreme effects ===")
for r in rows11:
    print(f"  {r['model_id']:26s} {r['tail']:14s} base={r['baseline_loss']:.4f} "
          f"rel={r['relative_delta_eps1_0']:+8.3f} G={r['G_eps1_0']:+8.3f}  {r['note']}")
print("\n=== EXP12 family-clustered (endpoint R_cand_eps1.0) ===")
for r in rows12:
    print(f"  {r['predictor']:26s} rho_all={r['rho_all']:+.3f}  1/family(det)={r['rho_one_per_family_det']:+.3f}"
          f"  resample={r['rho_resample_mean']:+.3f} [{r['rho_resample_lo']:+.3f},{r['rho_resample_hi']:+.3f}]"
          f"  P(rho>0)={r['frac_draws_positive']:.3f}")
print(f"\n=== N threshold: {thresh}")
