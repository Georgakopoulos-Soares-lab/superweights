"""
Analyze downstream compression data for the SW-aware tiered compression claim.

Compares:
  - PRUNING (zeroing rows) on DNABERT-2 prom_core: near-SW vs far-SW vs random
  - INT4 on DNABERT-2 splice and prom_core: near-SW vs random at matched counts
  - INT4 on DNABERT-2 splice: naive all-rows vs Yu heuristic vs SW-aware tiered

Key findings:
  1. Pruning — near-SW is safest at low fractions (≤10%); far-SW is
     catastrophically fragile at ≥20% (5.9x worse than near-SW at 20%).
     This supports a tiered pruning strategy: prune near-SW first,
     never prune far-SW rows.
  2. INT4 — at fractions up to 30%, INT4 perturbation on any subset of rows
     causes negligible MCC degradation (<0.002). Near-SW shows no consistent
     advantage over random selection at matched row counts.
     The "shadow redundancy" effect does not extend from pruning to INT4
     on these downstream classification tasks.
  3. A U_k-guided precision-allocation experiment is proposed as follow-up.

Output: prints comparison tables to stdout.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"


def load_json(name: str) -> dict:
    return json.load(open(RES / name))


# ═══════════════════════════════════════════════════════════════════════
# 1. U_k distribution
# ═══════════════════════════════════════════════════════════════════════
def show_uk_distribution():
    mech = load_json("sw_mechanistic_dnabert2.json")
    uk_all = []
    for fnorms in mech.get("frob_norm_uk_by_layer", {}).values():
        uk_all.extend(fnorms)
    uk_all = np.array(uk_all)

    print("=" * 72)
    print("1. U_k FROBENIUS NORM — DNABERT-2 (9216 rows across 12 layers)")
    print("=" * 72)
    print(f"  P50={np.percentile(uk_all, 50):.2f}  "
          f"P90={np.percentile(uk_all, 90):.2f}  "
          f"P95={np.percentile(uk_all, 95):.2f}  "
          f"P99={np.percentile(uk_all, 99):.2f}  "
          f"max={uk_all.max():.2f}")
    print()
    print("  SW rows (U_k / percentile):")
    sw_rows = mech.get("sw_rows_by_layer", {})
    for layer in sorted(sw_rows, key=int):
        layer_uk = np.array(mech["frob_norm_uk_by_layer"].get(layer, []))
        for r in sw_rows[layer]:
            if r < len(layer_uk):
                uk = layer_uk[r]
                pct = (layer_uk < uk).mean() * 100
                bar = "█" * min(int(pct / 5), 20)
                print(f"    L{layer:>2s} r{r:<4d}  U_k={uk:7.2f}  "
                      f"pct={pct:5.1f}%  {bar}")
    print()
    print("  → SW rows are massive U_k outliers (4–52× above P50).")
    print("    U_k reliably identifies candidate super-rows without forward pass.")
    print()


# ═══════════════════════════════════════════════════════════════════════
# 2. Pruning (zeroing rows) — DNABERT-2 prom_core
# ═══════════════════════════════════════════════════════════════════════
def show_pruning():
    p = load_json("compression_sweep_dnabert2_prom_core_notata.json")
    baseline = p["baseline"]
    print("=" * 72)
    print("2. PRUNING (ZEROING ROWS) — DNABERT-2 prom_core")
    print("=" * 72)
    print(f"  Baseline: ACC={baseline['accuracy']:.4f}  MCC={baseline['mcc']:.4f}")
    print(f"  SW rows: {p['n_sw']}  Total rows: {p['num_rows']}")
    print()
    hdr = (f"  {'Frac':>6s} {'N':>6s}  "
           f"{'near-SW':>10s} {'far-SW':>10s} "
           f"{'L1-low':>10s} {'L1-high':>10s} "
           f"{'random±std':>14s}  {'best':>10s}  {'near/far':>9s}")
    print(hdr)
    print(f"  {'':>6s} {'':>6s}  "
          f"{'Δacc%':>10s} {'Δacc%':>10s} "
          f"{'Δacc%':>10s} {'Δacc%':>10s} "
          f"{'Δacc%':>14s}  {'method':>10s}  {'ratio':>9s}")
    print("  " + "-" * 96)

    for i, frac in enumerate(p["fracs"]):
        near = p["curves"]["prox_near"][i]
        far = p["curves"]["prox_far"][i]
        l1lo = p["curves"]["l1_low"][i]
        l1hi = p["curves"]["l1_high"][i]
        rand = p["curves"]["random"][i]

        n = near["n_pruned"]
        vals = {
            "near-SW": near["delta_acc_pct"],
            "far-SW": far["delta_acc_pct"],
            "L1-low": l1lo["delta_acc_pct"],
            "L1-high": l1hi["delta_acc_pct"],
            "random": rand["delta_acc_pct"],
        }
        best = max(vals, key=vals.get)
        ratio = abs(far["delta_acc_pct"] / near["delta_acc_pct"]) if near["delta_acc_pct"] != 0 else float("inf")
        rand_str = f"{rand['delta_acc_pct']:+.2f}±{rand['accuracy_std']*100:.2f}"

        print(f"  {frac:>5.1f}% {n:>6d}  "
              f"{near['delta_acc_pct']:>+10.4f} {far['delta_acc_pct']:>+10.4f} "
              f"{l1lo['delta_acc_pct']:>+10.4f} {l1hi['delta_acc_pct']:>+10.4f} "
              f"{rand_str:>14s}  {best:>10s}  {ratio:>8.1f}x")

    print()
    print("  KEY FINDING: far-SW rows are uniquely fragile.")
    print("    At 20% pruning: far-SW loses -9.39% accuracy vs near-SW -1.60%")
    print("    (5.9× difference). At 30%: far-SW -8.15% vs near-SW -1.65% (4.9×).")
    print("    → Practical technique: rank rows by SW proximity,")
    print("      prune near-SW first (≤10%), NEVER prune far-SW rows.")
    print("    → Random pruning (-0.82% at 20%) outperforms near-SW pruning")
    print("      (-1.60% at 20%), so near-SW is NOT \"shadow-redundant.\"")
    print("      The effect is far-SW *fragility*, not near-SW *tolerance*.")
    print()


# ═══════════════════════════════════════════════════════════════════════
# 3. INT4 — DNABERT-2 splice and prom_core
# ═══════════════════════════════════════════════════════════════════════
def show_int4(task_label: str, filename: str):
    q = load_json(filename)
    baseline = q["baseline"]
    print("=" * 72)
    print(f"3. INT4 — DNABERT-2 {task_label}")
    print("=" * 72)
    print(f"  Baseline: ACC={baseline['accuracy']:.4f}  MCC={baseline['mcc']:.6f}")
    print()

    # Naive / Yu reference
    yu_all = q.get("yu_all", {})
    if "mcc" in yu_all:
        yu_dmcc = yu_all["mcc"] - baseline["mcc"]
        print(f"  Yu heuristic (all non-SW INT4): MCC={yu_all['mcc']:.6f}  "
              f"ΔMCC={yu_dmcc:+.6f}")
    sw = q.get("sw_fragility", {})
    print(f"  SW-only INT4: ΔMCC={sw.get('sw_delta_mcc', '?'):+.6f}"
          if isinstance(sw.get("sw_delta_mcc"), (int, float)) else
          f"  SW-only INT4: ΔMCC={sw.get('sw_delta_mcc', '?')}")
    print()

    print(f"  {'Frac':>6s} {'N':>6s}  "
          f"{'near-SW':>12s} {'random':>16s}  "
          f"{'Δ(n-r)':>10s} {'signif?':>8s}")
    print(f"  {'':>6s} {'':>6s}  "
          f"{'ΔMCC':>12s} {'ΔMCC±std':>16s}  "
          f"{'':>10s} {'':>8s}")
    print("  " + "-" * 66)

    for ns, rn in zip(q["near_sw"], q["random"]):
        frac = ns["frac"]
        n = ns["n_rows"]
        nd = ns["delta_mcc"]
        rd = rn["delta_mcc_mean"]
        rs = rn.get("mcc_std", rn.get("delta_mcc_std", 0.0))
        diff = nd - rd
        sig = "**" if abs(diff) > 2 * rs else ("*" if abs(diff) > rs else "")
        print(f"  {frac:>5.1f}% {n:>6d}  "
              f"{nd:>+12.6f} {rd:>+10.6f}±{rs:.6f}  "
              f"{diff:>+10.6f} {sig:>8s}")

    print()
    print("  → At all fractions up to 30%, near-SW and random INT4 are")
    print("    statistically indistinguishable. INT4 perturbation of any")
    print("    subset of rows causes negligible MCC degradation (<0.002).")
    print("    The \"shadow redundancy\" effect does NOT extend from pruning")
    print("    to INT4 on these downstream classification tasks.")
    print()


# ═══════════════════════════════════════════════════════════════════════
# 4. Tiered compression summary
# ═══════════════════════════════════════════════════════════════════════
def show_summary():
    print("=" * 72)
    print("4. SUMMARY — WHAT THE DATA SUPPORTS")
    print("=" * 72)
    print()
    print("  STRONG (pruning, DNABERT-2 prom_core):")
    print("    ✓ SW-proximity stratifies rows by pruning sensitivity")
    print("    ✓ far-SW rows are uniquely fragile (5.9× worse than near-SW at 20%)")
    print("    ✓ Tiered pruning technique: near-SW first, never far-SW")
    print("    ✗ near-SW is NOT \"shadow-redundant\" — random pruning is safer")
    print()
    print("  WEAK (INT4, DNABERT-2 splice + prom_core):")
    print("    ✗ near-SW INT4 shows no consistent advantage over random")
    print("    ✗ INT4 at ≤30% causes negligible degradation regardless of which rows")
    print("    ✗ No novel INT4 technique is supported by downstream classification data")
    print()
    print("  PROPOSED FOLLOW-UP:")
    print("    → U_k-guided INT4 precision allocation:")
    print("      Rank all rows by U_k Frobenius norm (weight-only, no forward pass).")
    print("      Protect high-U_k rows (fp16), aggressively INT4 low-U_k rows.")
    print("      Compare against random-row INT4 at matched compression ratios.")
    print("      Hypothesis: U_k predicts INT4 sensitivity better than SW proximity.")
    print("      This would make U_k a practical compression tool, not just a")
    print("      mechanistic descriptor.")
    print()


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    show_uk_distribution()
    show_pruning()
    show_int4("splice", "quant_ablation_dnabert2_splice_reconstructed_int4.json")
    show_int4("prom_core", "quant_ablation_dnabert2_prom_core_notata_int4.json")
    show_summary()
