"""Figure 3 — kingdom-asymmetric SW mechanism.

Clean redesign. 5 panels, no titles, no suptitle.

A — Mean |SW activation| by hexamer composition class
    (EUK and PROK side-by-side, single set of axes).
B — EUK causal-hexamer scatter: |SW activation| vs KL(clean ‖ ablated).
C — PROK causal-hexamer scatter: |SW activation| vs KL(clean ‖ ablated).
    (Headline kingdom asymmetry: r_EUK = +0.44 vs r_PROK = -0.71.)
D — Shuffle-control activation shift across 4 shuffle types (EUK vs PROK).
E — Motif enrichment in top-500 SW-dependent hexamers, EUK vs PROK
    drawn as a butterfly (mirrored bars) so the kingdom contrast is direct.

Inputs (unchanged):
  results/sw_kmer_scan.json
  results/sw_hexamer_causal.json
  results/sw_shuffle_controls.json
  results/sw_kmer_motifs.json
  results/prokaryote/<same files>

Output: paper/media/image_fig3.{png,pdf}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from _figstyle import apply_style, panel_label

apply_style()

_REPO = Path(__file__).resolve().parents[2]
RES = _REPO / "results"

C_EUK  = "#1f77b4"
C_PROK = "#d62728"


# ─── Hexamer composition class ─────────────────────────────────────────────
def _classify(kmer: str) -> str:
    s = kmer.upper()
    if any(s.count(b) >= 5 for b in "ACGT"):
        return "homopolymer"
    at = s.count("A") + s.count("T")
    gc = s.count("G") + s.count("C")
    if at >= 5:
        return "AT-rich"
    if gc >= 5:
        return "GC-rich"
    if "CG" in s and s.count("CG") >= 2:
        return "CpG-rich"
    return "balanced"


_CLASSES = ["AT-rich", "GC-rich", "CpG-rich", "homopolymer", "balanced"]


def _composition_means(scan_path: Path):
    rows = json.load(open(scan_path))["results"]
    groups = {c: [] for c in _CLASSES}
    for r in rows:
        groups[_classify(r["kmer"])].append(abs(r["activation_mean"]))
    means = np.array([np.mean(groups[c]) if groups[c] else 0.0 for c in _CLASSES])
    sems  = np.array([np.std(groups[c]) / np.sqrt(max(len(groups[c]), 1))
                      for c in _CLASSES])
    return means, sems


def panel_A(ax):
    eu_m, eu_s = _composition_means(RES / "sw_kmer_scan.json")
    pr_m, pr_s = _composition_means(RES / "prokaryote/sw_kmer_scan.json")
    x = np.arange(len(_CLASSES))
    w = 0.38
    ax.bar(x - w / 2, eu_m, w, yerr=eu_s, capsize=3, color=C_EUK,
           edgecolor="black", linewidth=0.5, label="EUK")
    ax.bar(x + w / 2, pr_m, w, yerr=pr_s, capsize=3, color=C_PROK,
           edgecolor="black", linewidth=0.5, label="PROK")
    ax.set_xticks(x)
    ax.set_xticklabels(_CLASSES, fontsize=7.5)
    ax.set_ylabel(r"mean $|$SW activation$|$")
    ax.set_yscale("log")
    ax.legend(loc="upper right", frameon=False)


# ─── Causal scatter ────────────────────────────────────────────────────────
def _scatter(ax, path: Path, color: str, *, kingdom: str):
    d = json.load(open(path))
    rows = d["results"]
    x = np.array([r["sw_activation"] for r in rows], dtype=float)
    y = np.array([r["kl_clean_ablated"] for r in rows], dtype=float)
    r = d.get("r_kl_sw_activation")
    ax.scatter(x, y, s=3, alpha=0.4, color=color, edgecolors="none",
               rasterized=True)
    if np.std(x) > 0:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m * xs + b, color="black", lw=1.0, ls="--", alpha=0.7)
    ax.set_xlabel(f"{kingdom} SW activation")
    ax.set_ylabel(r"KL(clean $\|$ ablated)")
    ax.text(0.04, 0.96, rf"$r = {r:+.3f}$",
            transform=ax.transAxes, va="top", fontsize=10, fontweight="bold",
            color=color)
    ax.grid(alpha=0.2)


def panel_B(ax):
    _scatter(ax, RES / "sw_hexamer_causal.json", C_EUK, kingdom="EUK")


def panel_C(ax):
    _scatter(ax, RES / "prokaryote/sw_hexamer_causal.json", C_PROK,
             kingdom="PROK")


# ─── Shuffle controls ──────────────────────────────────────────────────────
def panel_D(ax):
    shuffles = ["mono", "dinuc", "trinuc", "kmer_block"]
    pretty   = ["mono", "dinuc", "trinuc", "6-mer block"]
    x = np.arange(len(shuffles))
    w = 0.36
    for i, (path, color, label) in enumerate([
        (RES / "sw_shuffle_controls.json",            C_EUK,  "EUK"),
        (RES / "prokaryote/sw_shuffle_controls.json", C_PROK, "PROK"),
    ]):
        agg = json.load(open(path))["aggregate"]
        means = np.array([agg[s]["mean"] for s in shuffles])
        sems  = np.array([agg[s]["std"] / np.sqrt(agg[s]["n"]) for s in shuffles])
        ps    = [agg[s]["p_two_sided"] for s in shuffles]
        off   = (i - 0.5) * w
        ax.bar(x + off, means, w, yerr=sems, capsize=2.5,
               color=color, edgecolor="black", linewidth=0.5, label=label)
        for j, p in enumerate(ps):
            if p is None or np.isnan(p) or p >= 0.05:
                continue
            star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*"
            ax.text(j + off, means[j] + sems[j] + 1e-4, star,
                    ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(pretty)
    ax.set_ylabel(r"$\Delta$ SW activation"
                  "\n"
                  "(shuffled − original)")
    ax.legend(loc="upper left", frameon=False)


# ─── Motif enrichment butterfly ────────────────────────────────────────────
_MOTIFS = [
    "splice_donor", "splice_acceptor", "TATA_box", "Kozak",
    "GC_box_Sp1", "CpG_rich", "CCAAT_box", "E_box", "AP1_TRE", "NFkB",
    "Shine_Dalgarno", "AT_rich", "homopolymer_run",
]


def _motif_log2or(path: Path):
    d = json.load(open(path))
    by_name = {m["motif"]: m for m in d["motif_results"]}
    log2 = []
    qs = []
    for m in _MOTIFS:
        if m in by_name and by_name[m]["odds_ratio"] > 0:
            log2.append(np.log2(by_name[m]["odds_ratio"]))
            qs.append(by_name[m]["p_adj"])
        else:
            log2.append(np.nan)
            qs.append(np.nan)
    return np.array(log2), np.array(qs)


def panel_E(ax):
    eu_log, eu_q = _motif_log2or(RES / "sw_kmer_motifs.json")
    pr_log, pr_q = _motif_log2or(RES / "prokaryote/sw_kmer_motifs.json")

    y = np.arange(len(_MOTIFS))
    # EUK to the right (positive x), PROK to the left (negative x = -PROK_log)
    eu_x = np.where(np.isnan(eu_log), 0, eu_log)
    pr_x = np.where(np.isnan(pr_log), 0, -pr_log)
    eu_colors = [C_EUK  if not np.isnan(q) and q < 0.05 else "#aac8e6" for q in eu_q]
    pr_colors = [C_PROK if not np.isnan(q) and q < 0.05 else "#e6b0b0" for q in pr_q]
    ax.barh(y, eu_x, color=eu_colors, edgecolor="black", linewidth=0.4)
    ax.barh(y, pr_x, color=pr_colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([m.replace("_", " ") for m in _MOTIFS], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\log_{2}$(odds ratio)")
    # symmetric xlim
    span = max(np.nanmax(np.abs(eu_log)) if np.any(~np.isnan(eu_log)) else 1,
               np.nanmax(np.abs(pr_log)) if np.any(~np.isnan(pr_log)) else 1)
    ax.set_xlim(-span * 1.05, span * 1.05)
    # axis tick labels on left side should display abs value
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(t):.0f}" for t in xticks])
    # side labels
    ax.text(0.02, 1.02, "PROK \u2190", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8, fontweight="bold", color=C_PROK)
    ax.text(0.98, 1.02, "\u2192 EUK", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, fontweight="bold", color=C_EUK)
    ax.grid(axis="x", alpha=0.25)


# ─── Compose ──────────────────────────────────────────────────────────────
def build(out_path: Path):
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           height_ratios=[0.95, 1.0, 1.15],
                           hspace=0.55, wspace=0.30,
                           left=0.07, right=0.97, top=0.965, bottom=0.06)

    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])
    axD = fig.add_subplot(gs[2, 0])
    axE = fig.add_subplot(gs[2, 1])

    panel_A(axA); panel_label(axA, "A", x=-0.04)
    panel_B(axB); panel_label(axB, "B")
    panel_C(axC); panel_label(axC, "C")
    panel_D(axD); panel_label(axD, "D")
    panel_E(axE); panel_label(axE, "E", x=-0.20)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    print("saved:", out_path)
    print("saved:", out_path.with_suffix(".pdf"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="paper/media/image_fig3.png")
    args = parser.parse_args()
    p = Path(args.out)
    build(p if p.is_absolute() else _REPO / p)


if __name__ == "__main__":
    main()
