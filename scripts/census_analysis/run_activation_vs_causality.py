#!/usr/bin/env python
"""
EXPERIMENT 1 — does activation extremeness PREDICT causal severity, or only enrich for it?

The census selected rows by activation extremeness, then tested q1 and Frobenius against
causal damage. The selection variable itself was never tested. This does that, using only
already-recorded raw outputs (audit/census_master.csv x results/E13_candidate_stability),
so no model is re-run and no candidate is re-selected.

Predictors (all from the frozen candidate, no reselection):
  activation_max            discovery absolute activation maximum
  activation_ratio          max_abs(candidate) / median same-layer max_abs
  layer_relative_rank       same-layer rank (1 = most extreme)
  ratio_mean_24 / median_24 independent-input mean/median ratio over the 24-input assay
  ratio_cv_24               coefficient of variation across the 24 inputs
  q1                        spectral concentration            (incumbent predictor)
  layer_relative_frobenius  weight magnitude                  (incumbent predictor)

Endpoints (paired candidate-vs-control, as recorded):
  R_cand_eps0.5 / eps1.0    signed relative native-loss change
  G_eps0.5 / G_eps1.0       candidate minus median random control
  G_topnorm_*               candidate minus top-norm control (if available)

Statistics: Spearman rho with BCa-free percentile bootstrap CI (10k resamples, seed 42),
computed on all models and on the text-decoder subset (most homogeneous endpoint).
n=22 is small and the models are not independent families, so only univariate comparisons
are reported -- no multivariate fit.
"""
from __future__ import annotations
import csv, json, glob, os, sys
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED = 42
NBOOT = 10000

def fnum(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None

# ---- load census -------------------------------------------------------------
cen = list(csv.DictReader(open(os.path.join(ROOT, "audit/census_master.csv"))))

# ---- load 24-input stability assay, keyed by (layer,row) + slug --------------
stab = {}
for f in glob.glob(os.path.join(ROOT, "results/experiments/E13_candidate_stability/*.json")):
    j = json.load(open(f))
    s = j["summary"]
    per = j.get("per_input", [])
    ratios = [p["activation_ratio"] for p in per if p.get("activation_ratio") is not None]
    amax   = [p["activation_max"]   for p in per if p.get("activation_max") is not None]
    ranks  = [p["rank"]             for p in per if p.get("rank") is not None]
    pos    = [p.get("max_position") for p in per]
    stab[s["slug"]] = {
        "model": s["model"], "layer": s["layer"], "row": s["row"],
        "domain": s["domain"], "architecture": s["architecture"],
        "n_inputs": len(per),
        "ratio_mean_24":   float(np.mean(ratios)) if ratios else None,
        "ratio_median_24": float(np.median(ratios)) if ratios else None,
        "ratio_cv_24":     float(np.std(ratios) / np.mean(ratios)) if ratios and np.mean(ratios) else None,
        "act_max_mean_24": float(np.mean(amax)) if amax else None,
        "rank_median_24":  float(np.median(ranks)) if ranks else None,
        "frac_rank1":      s.get("frac_rank1"),
        "max_position_at_0_fraction": s.get("max_position_at_0_fraction"),
    }

# ---- match census rows to stability slugs -----------------------------------
def norm(x): return "".join(ch for ch in x.lower() if ch.isalnum())
by_lr = {}
for slug, v in stab.items():
    by_lr.setdefault((v["layer"], v["row"]), []).append(slug)

recs = []
unmatched = []
for r in cen:
    L, R = fnum(r["candidate_layer"]), fnum(r["candidate_row"])
    slug = None
    cands = by_lr.get((int(L), int(R)), []) if L is not None and R is not None else []
    if len(cands) == 1:
        slug = cands[0]
    else:
        for s, v in stab.items():
            if norm(v["model"]) == norm(r["model_id"]) or norm(s) == norm(r["model_id"]):
                slug = s; break
        if slug is None and cands:
            for s in cands:
                if norm(stab[s]["model"])[:6] == norm(r["model_id"])[:6]:
                    slug = s; break
    if slug is None:
        unmatched.append(r["model_id"]); continue
    v = stab[slug]
    recs.append({
        "model_id": r["model_id"], "slug": slug,
        "domain": r["domain"], "architecture": r["architecture"],
        "candidate_layer": int(L), "candidate_row": int(R),
        "activation_max": v["act_max_mean_24"],
        "activation_ratio": v["ratio_mean_24"],
        "ratio_median_24": v["ratio_median_24"],
        "ratio_cv_24": v["ratio_cv_24"],
        "rank_median_24": v["rank_median_24"],
        "frac_rank1": v["frac_rank1"],
        "max_position_at_0_fraction": v["max_position_at_0_fraction"],
        "q1": fnum(r["q1"]),
        "layer_relative_frobenius": fnum(r["layer_relative_frobenius"]),
        "R_cand_eps0.5": fnum(r["R_cand_eps0.5"]),
        "R_cand_eps1.0": fnum(r["R_cand_eps1.0"]),
        "G_eps0.5": fnum(r["G_eps0.5"]),
        "G_eps1.0": fnum(r["G_eps1.0"]),
        "endpoint_type": r["endpoint_type"],
        "baseline_loss": fnum(r["baseline_loss"]),
    })
print(f"[join] matched {len(recs)}/{len(cen)} census models to the stability assay")
if unmatched:
    print(f"[join] UNMATCHED (excluded, recorded): {unmatched}")

# ---- spearman + bootstrap CI -------------------------------------------------
def spearman_boot(x, y, nboot=NBOOT, seed=SEED):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 5:
        return dict(n=n, rho=None, p=None, lo=None, hi=None)
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(nboot):
        i = rng.integers(0, n, n)
        if len(np.unique(x[i])) < 3 or len(np.unique(y[i])) < 3:
            continue
        bs.append(stats.spearmanr(x[i], y[i]).statistic)
    lo, hi = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else (None, None)
    return dict(n=int(n), rho=float(rho), p=float(p), lo=lo, hi=hi)

PRED = ["activation_ratio", "activation_max", "ratio_median_24", "ratio_cv_24",
        "rank_median_24", "q1", "layer_relative_frobenius"]
ENDP = ["R_cand_eps0.5", "R_cand_eps1.0", "G_eps0.5", "G_eps1.0"]
COHORTS = {
    "all": lambda r: True,
    "text_decoder": lambda r: r["domain"] == "text" and r["architecture"] == "decoder",
    "text_encoder": lambda r: r["domain"] == "text" and r["architecture"] == "encoder",
    "genomic": lambda r: r["domain"] == "genomic",
}

out_rows = []
for cname, filt in COHORTS.items():
    sub = [r for r in recs if filt(r)]
    for pred in PRED:
        for end in ENDP:
            x = [r[pred] for r in sub]; y = [r[end] for r in sub]
            st = spearman_boot(x, y)
            out_rows.append(dict(cohort=cname, n=st["n"], predictor=pred, endpoint=end,
                                 spearman_rho=st["rho"], p_value=st["p"],
                                 ci95_lo=st["lo"], ci95_hi=st["hi"]))

os.makedirs(os.path.join(ROOT, "results/analyses/census_analysis"), exist_ok=True)
tsv = os.path.join(ROOT, "results/analyses/census_analysis/activation_vs_causality.tsv")
with open(tsv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["cohort","n","predictor","endpoint","spearman_rho",
                                      "p_value","ci95_lo","ci95_hi"], delimiter="\t")
    w.writeheader()
    for r in out_rows:
        w.writerow({k: ("" if r[k] is None else r[k]) for k in r})

per = os.path.join(ROOT, "results/analyses/census_analysis/activation_vs_causality_per_model.tsv")
cols = ["model_id","slug","domain","architecture","candidate_layer","candidate_row",
        "activation_max","activation_ratio","ratio_median_24","ratio_cv_24","rank_median_24",
        "frac_rank1","max_position_at_0_fraction","q1","layer_relative_frobenius",
        "R_cand_eps0.5","R_cand_eps1.0","G_eps0.5","G_eps1.0","endpoint_type","baseline_loss"]
with open(per, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, delimiter="\t"); w.writeheader()
    for r in recs: w.writerow({c: ("" if r.get(c) is None else r[c]) for c in cols})

print(f"\nwrote {tsv}\nwrote {per}")
print("\n=== PREDICTOR COMPARISON (cohort=all, endpoint=R_cand_eps1.0) ===")
print(f"{'predictor':26s}{'n':>4s}{'rho':>8s}{'p':>9s}{'95% CI':>22s}")
for r in out_rows:
    if r["cohort"]=="all" and r["endpoint"]=="R_cand_eps1.0":
        ci = f"[{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}]" if r["ci95_lo"] is not None else "n/a"
        print(f"{r['predictor']:26s}{r['n']:>4d}{r['spearman_rho']:>+8.3f}{r['p_value']:>9.4f}{ci:>22s}")
print("\n=== text-decoder subset (endpoint=R_cand_eps1.0) ===")
for r in out_rows:
    if r["cohort"]=="text_decoder" and r["endpoint"]=="R_cand_eps1.0":
        ci = f"[{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}]" if r["ci95_lo"] is not None else "n/a"
        print(f"{r['predictor']:26s}{r['n']:>4d}{r['spearman_rho']:>+8.3f}{r['p_value']:>9.4f}{ci:>22s}")
