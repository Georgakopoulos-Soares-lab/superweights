"""
scripts/analysis/run_composition_vs_motif.py
------------------------------------------------
Does the super weight encode COMPOSITION or MOTIFS? (strong form)

The manuscript currently supports "composition, not annotated regulatory
grammar" with a motif-enrichment test over ~9 hand-curated IUPAC motifs that
all came back q = 1.00. That is a weak negative: nine arbitrary motifs is low
power, and "we looked and found nothing" is not the same as "composition
explains it."

This replaces it with a positive, composition-controlled analysis over the
FULL 4,096-hexamer activation scan already on disk:

  PART 1  Variance decomposition. Regress SW activation on composition
          features only (GC, mononucleotide freqs, 16 dinucleotide freqs,
          CpG count, Shannon entropy, longest homopolymer run). R^2 is the
          fraction of SW activation explained by composition alone.

  PART 2  Motif test BEYOND composition. For every motif, compare its
          hexamers against a GC-MATCHED background drawn from non-motif
          hexamers (not a uniform background), via permutation. This asks
          the question that matters: does motif membership predict SW
          activation once composition is held fixed?

  PART 3  Residual structure. Are the largest regression residuals
          themselves motif-enriched? If composition is the whole story the
          residuals should be motif-agnostic.

No external motif database is required (none is available offline), and no
GPU -- this runs on the existing sw_kmer_scan.json outputs.

Usage:
  python scripts/analysis/run_composition_vs_motif.py --scan results/sw_kmer_scan.json --label euk
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "interpretability"))

SEED = 42
BASES = "ACGT"


# ─────────────────────────────────────────────────────────────────────────────
# Composition features
# ─────────────────────────────────────────────────────────────────────────────

def features(kmers: list[str]):
    dinucs = ["".join(p) for p in product(BASES, repeat=2)]
    names = (["gc", "cpg", "entropy", "max_run", "purine"]
             + [f"mono_{b}" for b in BASES]
             + [f"di_{d}" for d in dinucs])
    X = np.zeros((len(kmers), len(names)), dtype=np.float64)
    for i, km in enumerate(kmers):
        L = len(km)
        gc = sum(km.count(b) for b in "GC") / L
        cpg = km.count("CG")
        counts = np.array([km.count(b) for b in BASES], dtype=float)
        p = counts[counts > 0] / L
        ent = float(-(p * np.log2(p)).sum())
        run, best = 1, 1
        for a, b in zip(km, km[1:]):
            run = run + 1 if a == b else 1
            best = max(best, run)
        pur = sum(km.count(b) for b in "AG") / L
        row = [gc, cpg, ent, best, pur] + list(counts / L)
        dc = [0.0] * len(dinucs)
        idx = {d: j for j, d in enumerate(dinucs)}
        for j in range(L - 1):
            dc[idx[km[j:j + 2]]] += 1
        row += [c / (L - 1) for c in dc]
        X[i] = row
    return X, names


def ols_r2(X, y):
    """OLS with intercept; returns R^2 and residuals."""
    A = np.hstack([np.ones((len(X), 1)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    resid = y - pred
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return (1 - ss_res / ss_tot if ss_tot > 0 else float("nan")), resid, pred


# ─────────────────────────────────────────────────────────────────────────────
# GC-matched permutation test
# ─────────────────────────────────────────────────────────────────────────────

def gc_matched_test(kmers, act, gc, member_mask, rng, n_perm=10000):
    """Is mean activation of motif hexamers different from GC-MATCHED
    non-motif hexamers? Matching is done by GC bin so the comparison holds
    composition fixed."""
    n_mem = int(member_mask.sum())
    if n_mem == 0:
        return None
    obs = float(act[member_mask].mean())
    gc_bins = np.round(gc * 6).astype(int)          # 6-mers -> 0..6 GC bases
    pool = {b: np.where((~member_mask) & (gc_bins == b))[0] for b in np.unique(gc_bins)}
    mem_bins = gc_bins[member_mask]
    null = np.empty(n_perm)
    for t in range(n_perm):
        pick = []
        for b in mem_bins:
            cand = pool.get(b)
            if cand is None or len(cand) == 0:
                continue
            pick.append(cand[rng.randint(len(cand))])
        null[t] = act[pick].mean() if pick else np.nan
    null = null[~np.isnan(null)]
    if null.size == 0:
        return None
    p = float((np.abs(null - null.mean()) >= abs(obs - null.mean())).mean())
    z = float((obs - null.mean()) / null.std()) if null.std() > 0 else float("nan")
    return {"n_members": n_mem, "obs_mean_activation": obs,
            "gcmatched_null_mean": float(null.mean()),
            "gcmatched_null_sd": float(null.std()),
            "z": z, "p_perm": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n_perm", type=int, default=10000)
    args = ap.parse_args()

    d = json.loads(Path(args.scan).read_text())
    recs = d["results"]
    kmers = [r["kmer"] for r in recs]
    act = np.array([r["activation_mean"] for r in recs], dtype=float)
    gc = np.array([r["gc_frac"] for r in recs], dtype=float)
    print(f"[comp-vs-motif] {args.label}: {len(kmers)} k-mers, "
          f"layer={d.get('sw_layer')} rows={d.get('sw_rows')}")
    print(f"[comp-vs-motif] activation: mean={act.mean():.2f} sd={act.std():.2f} "
          f"range=[{act.min():.1f},{act.max():.1f}]")

    # ---- PART 1: how much does composition explain? ----
    X, names = features(kmers)
    r2_full, resid, pred = ols_r2(X, act)
    r2_gc, _, _ = ols_r2(gc.reshape(-1, 1), act)
    print(f"\n[PART 1] composition-only variance explained")
    print(f"   R^2 (GC alone)                = {r2_gc:.4f}")
    print(f"   R^2 (full composition model)  = {r2_full:.4f}   [{len(names)} features]")

    # robustness: activation here is heavy-tailed on some models, so also
    # report the rank-space version
    r2_rank, resid_rank, _ = ols_r2(X, stats.rankdata(act))
    print(f"   R^2 (rank-transformed target) = {r2_rank:.4f}   <- outlier-robust")

    # ---- PART 2: motifs beyond composition ----
    from analyze_sw_kmer_motifs import _build_motif_dict
    motifs = _build_motif_dict()
    rng = np.random.RandomState(SEED)
    print(f"\n[PART 2] motif vs GC-MATCHED background ({len(motifs)} motifs, "
          f"{args.n_perm} permutations)")
    print(f"   {'motif':<18} {'n':>4} {'obs_act':>12} {'gc_null':>12} {'z':>7} {'p_perm':>9}")
    motif_res = {}
    for name, kset in sorted(motifs.items()):
        mask = np.array([k in kset for k in kmers])
        r = gc_matched_test(kmers, act, gc, mask, rng, args.n_perm)
        if r is None:
            continue
        motif_res[name] = r
        print(f"   {name:<18} {r['n_members']:>4} {r['obs_mean_activation']:>12.2f} "
              f"{r['gcmatched_null_mean']:>12.2f} {r['z']:>7.2f} {r['p_perm']:>9.4f}")

    # BH correction across motifs
    if motif_res:
        ks = list(motif_res)
        ps = np.array([motif_res[k]["p_perm"] for k in ks])
        order = np.argsort(ps)
        q = np.empty_like(ps)
        m = len(ps)
        prev = 1.0
        for rank, i in enumerate(order[::-1]):
            prev = min(prev, ps[i] * m / (m - rank))
            q[i] = prev
        for k, qq in zip(ks, q):
            motif_res[k]["q_bh"] = float(qq)
        n_sig = int((q < 0.05).sum())
        print(f"   -> {n_sig}/{m} motifs significant at BH q<0.05 "
              f"AFTER holding composition fixed")
    else:
        n_sig = 0

    # ---- PART 3: is residual structure motif-enriched? ----
    print(f"\n[PART 3] residual structure")
    top_res = np.argsort(-np.abs(resid))[:200]
    in_any = np.array([any(kmers[i] in s for s in motifs.values()) for i in top_res])
    base_rate = np.mean([any(k in s for s in motifs.values()) for k in kmers])
    print(f"   motif membership rate: top-200 residuals = {in_any.mean():.4f}  "
          f"vs all k-mers = {base_rate:.4f}")
    bt = stats.binomtest(int(in_any.sum()), len(in_any), base_rate)
    print(f"   binomial p = {bt.pvalue:.4f}")

    out = {
        "label": args.label, "scan": args.scan,
        "sw_layer": d.get("sw_layer"), "sw_rows": d.get("sw_rows"),
        "n_kmers": len(kmers), "seed": SEED, "n_perm": args.n_perm,
        "activation": {"mean": float(act.mean()), "sd": float(act.std()),
                       "min": float(act.min()), "max": float(act.max())},
        "part1_variance_explained": {
            "r2_gc_only": r2_gc, "r2_full_composition": r2_full,
            "r2_rank_transformed": r2_rank, "n_features": len(names),
            "features": names},
        "part2_motif_vs_gcmatched": motif_res,
        "part2_n_significant_bh05": n_sig,
        "part3_residual_motif_enrichment": {
            "top200_motif_rate": float(in_any.mean()),
            "background_motif_rate": float(base_rate),
            "binomial_p": float(bt.pvalue)},
        "interpretation": (
            f"Composition explains R^2={r2_full:.3f} of SW activation across all {len(kmers)} "
            f"hexamers ({r2_rank:.3f} rank-transformed). {n_sig} of {len(motif_res)} motifs retain "
            f"a significant association after GC matching (BH q<0.05). This is a POSITIVE test of "
            f"the composition account, replacing the earlier weak negative (9 motifs, all q=1.00 "
            f"against a uniform background)."),
    }
    p = ROOT / "results/mechanism" / f"composition_vs_motif_{args.label}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n[comp-vs-motif] wrote {p}")


if __name__ == "__main__":
    main()
