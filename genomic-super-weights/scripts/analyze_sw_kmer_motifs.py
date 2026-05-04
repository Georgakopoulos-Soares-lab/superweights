"""
scripts/analyze_sw_kmer_motifs.py
------------------------------------
Cross-reference the top SW-activating 6-mers (from run_sw_kmer_scan.py) against
a curated dictionary of known regulatory DNA motifs to ask:

  "Do the k-mers that maximally drive the GENERator super-weight correspond to
   known transcription start / splice / regulatory signals?"

Method
------
For each reference motif (given as a set of matching 6-mer strings or a regex):
  1. Compute its observed count in the top-K k-mers.
  2. Compute its expected count under the null that all 4096 6-mers are equally
     likely to appear in the top-K.
  3. Run a one-sided Fisher's exact test:
        foreground = top-K k-mers
        background = all 4096 k-mers
        contingency table:
            | in motif | not in motif |
            |----------|--------------|
     top-K  |   obs    |   top_K-obs  |
     rest   | exp_bg   | 4096-K-exp   |
  4. Report odds ratio + p-value (BH-corrected for multiple motifs).

Motif dictionary
----------------
Includes all canonical 6-mer exact matches for:
  - TATA box family (TATAAA, TATATA, TATAAG, TATATA, ...)
  - Kozak sequence core (GCCACC, GCCGCC, ACCATG, ...)
  - Splice donor consensus (GTAAGT, GTAAGC, GTGAGT, ...)
  - Splice acceptor consensus (TTTCAG, TTACAG, CTGCAG, ...)
  - Shine-Dalgarno (AGGAGG, AAGGAG, GAGGAG, ...) [prokaryote]
  - CCAAT box (GGCCAA, CCAATC, CCAAT[AG], ...)
  - GC box / Sp1 (GGGCGG, GGCGGG, CCGCCC, ...)
  - E-box (CANNTG → all matching 6-mers)
  - AP-1 (TGACTC, TGAGTC, TGACGC, ...)
  - CpG dinucleotide enriched k-mers (≥ 2 CG dinucs in a 6-mer)

Usage
-----
    python scripts/analyze_sw_kmer_motifs.py \\
        --kmer_scan results/sw_kmer_scan.json \\
        --top_k 200 \\
        --out   results/sw_kmer_motifs.json \\
        --plot  results/sw_kmer_motifs.png

No GPU required — pure analysis on the k-mer scan output.
"""

import argparse
import json
import re
import sys
from itertools import product
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Build motif dictionary: name → set of 6-mer strings that match the motif
# ─────────────────────────────────────────────────────────────────────────────

_IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "R": "AG",  "Y": "CT",  "S": "GC",  "W": "AT",
    "K": "GT",  "M": "AC",  "B": "CGT", "D": "AGT",
    "H": "ACT", "V": "ACG", "N": "ACGT",
}

ALL_6MERS = ["".join(k) for k in product("ACGT", repeat=6)]


def _expand_iupac(pattern: str) -> set:
    """Expand an IUPAC ambiguity pattern of length 6 to all matching 6-mers."""
    assert len(pattern) == 6, f"Expected 6-mer, got '{pattern}'"
    options = [_IUPAC.get(c.upper(), "N") for c in pattern.upper()]
    result = set()
    for combo in product(*options):
        result.add("".join(combo))
    return result


def _build_motif_dict() -> dict:
    motifs = {}

    # ── TATA box (TATAAA core + common variants) ──────────────────────────────
    motifs["TATA_box"] = (
        _expand_iupac("TATAWA")   # canonical TATA[A/T]A → TATAAA, TATATA, TATATA, TATAAA
        | {"TATAAG", "TATAAC", "TATAGT", "TATAAT",
           "TATAAA", "TATATA", "TATATT", "TATACG"}
    ) & set(ALL_6MERS)

    # ── Kozak consensus (GCC[A/G]CC surrounding start AUG) ───────────────────
    motifs["Kozak"] = (
        _expand_iupac("GCCRCC")   # GCCACC, GCCGCC (strict Kozak)
        | _expand_iupac("RCCATG") # ACCATG, GCCATG — position −3 to +3
        | _expand_iupac("NNATGN") # ATG context k-mers
    ) & set(ALL_6MERS)

    # ── Splice donor (GU/GT at exon-intron boundary, positions +1/+2) ─────────
    # Consensus: (A/C)AG|GURAGU; 6-mer spanning the junction is xGTAAGT etc.
    # We take the last 2 nt of exon + GT + first 2 nt of intron
    motifs["splice_donor"] = set()
    for prefix in ["AG", "GG", "TG", "CG", "AA", "GA", "CA", "TA"]:
        for suffix in ["AGT", "AGC", "GGT", "GGC", "AAG", "GAG"]:
            km = prefix + "GT" + suffix[:2]   # 6-mer = prefix(2) + GT + 2nt
            if len(km) == 6 and km in set(ALL_6MERS):
                motifs["splice_donor"].add(km)
    motifs["splice_donor"] |= {"GTAAGT", "GTAAGC", "GTGAGT", "GTATGT",
                                "GTAAGG", "GTACGT", "GTAGGT"}

    # ── Splice acceptor (pyrimidine tract + AG at intron-exon boundary) ───────
    motifs["splice_acceptor"] = set()
    for pref in ["TT", "CT", "TC", "CC"]:
        for suf in ["CAG", "TAG", "AAG", "GAG"]:
            km = pref + suf[:2] + "AG"
            if len(km) == 6 and km in set(ALL_6MERS):
                motifs["splice_acceptor"].add(km)
    motifs["splice_acceptor"] |= {"TTTCAG", "TTACAG", "CTGCAG", "CTTCAG",
                                   "TCCCAG", "CTCCAG", "TCTCAG", "TCCAAG"}

    # ── Shine-Dalgarno (prokaryote ribosome binding, complementary to 16S rRNA) ─
    motifs["Shine_Dalgarno"] = (
        _expand_iupac("AGGAGG")
        | _expand_iupac("AAGGAG")
        | _expand_iupac("GAGGAG")
        | {"AGGAGG", "AAGGAG", "GAGGAG", "AGGAGN"[:6],
           "AGAAGG", "GGAGGT", "AGGAGA"}
    ) & set(ALL_6MERS)

    # ── CCAAT box (eukaryotic promoter −80 element) ───────────────────────────
    motifs["CCAAT_box"] = (
        _expand_iupac("GGCCAA")   # complement
        | _expand_iupac("CCAATN")
        | {"CCAATC", "CCAATG", "CCAATA", "CCAATT",
           "GGCCAA", "TGGCCA", "ATTGGT", "GCCAAT"}
    ) & set(ALL_6MERS)

    # ── GC box / Sp1 binding site ─────────────────────────────────────────────
    motifs["GC_box_Sp1"] = (
        _expand_iupac("GGGCGG")
        | _expand_iupac("CCGCCC")
        | {"GGGCGG", "GGCGGG", "CCGCCC", "CCCGCC",
           "GCGGGC", "CGGGCG", "GGGCGC"}
    ) & set(ALL_6MERS)

    # ── E-box (CANNTG → CACNTG family) ────────────────────────────────────────
    # bHLH binding: CA[ACGT][ACGT]TG — within a 6-mer this is the exact match
    ebox = set()
    for n1 in "ACGT":
        for n2 in "ACGT":
            ebox.add(f"CA{n1}{n2}TG")
    motifs["E_box"] = ebox & set(ALL_6MERS)

    # ── AP-1 / TRE (TGA[CG]TC) ───────────────────────────────────────────────
    motifs["AP1_TRE"] = (
        _expand_iupac("TGASTC")   # TGACTC, TGAGTC
        | {"TGACTC", "TGAGTC", "TGACGC", "ATGACT",
           "CATGAC", "AGTCAG"}
    ) & set(ALL_6MERS)

    # ── NF-κB (GGGRNNTCC family, partial 6-mer matches) ─────────────────────
    motifs["NFkB"] = (
        _expand_iupac("GGGRNN")
        | _expand_iupac("NNTCCC")
        | {"GGGACT", "GGGATT", "GGAACC", "GGAGCC",
           "GGGCCC", "GGGACC", "GGGCCT", "GGGAAC"}
    ) & set(ALL_6MERS)

    # ── CpG-enriched (≥ 2 CG dinucleotides within the 6-mer) ─────────────────
    cpg_rich = set()
    for km in ALL_6MERS:
        cg_count = sum(1 for i in range(5) if km[i] == "C" and km[i+1] == "G")
        if cg_count >= 2:
            cpg_rich.add(km)
    motifs["CpG_rich"] = cpg_rich

    # ── AT-rich (≥ 5/6 A or T) ───────────────────────────────────────────────
    at_rich = {km for km in ALL_6MERS if (km.count("A") + km.count("T")) >= 5}
    motifs["AT_rich"] = at_rich

    # ── Homopolymer runs (≥ 4 consecutive identical nucleotides) ─────────────
    homopoly = set()
    for km in ALL_6MERS:
        for base in "ACGT":
            if base * 4 in km:
                homopoly.add(km)
                break
    motifs["homopolymer_run"] = homopoly

    return motifs


# ─────────────────────────────────────────────────────────────────────────────
# Fisher's exact test (one-sided, enrichment)
# ─────────────────────────────────────────────────────────────────────────────

def _fisher_exact_enrichment(obs: int, total_fg: int,
                              bg_in: int, total_bg: int) -> tuple:
    """
    One-sided Fisher's exact test for enrichment of a set in the foreground.

    Contingency table:
                 in_motif    not_in_motif
    foreground     obs        total_fg - obs
    background    bg_in       total_bg - bg_in

    Returns (odds_ratio, p_value).
    """
    try:
        from scipy.stats import fisher_exact
        table = [
            [obs,        total_fg - obs],
            [bg_in,      total_bg - bg_in],
        ]
        or_, p = fisher_exact(table, alternative="greater")
        return float(or_), float(p)
    except ImportError:
        # Manual approximation via hypergeometric PMF tail sum
        # P(X >= obs) where X ~ Hypergeometric(N=total, K=bg_in, n=total_fg)
        total = total_fg + total_bg
        K     = bg_in + obs  # motif k-mers in full universe
        n     = total_fg     # foreground size
        # mean
        mu = n * K / total
        or_ = (obs / max(total_fg - obs, 1e-9)) / (bg_in / max(total_bg - bg_in, 1e-9)) if bg_in > 0 else float("inf")
        # approximate p-value via normal approximation
        var = n * K * (total - K) * (total - n) / (total * total * (total - 1))
        if var <= 0:
            return or_, (1.0 if obs <= mu else 0.0)
        z = (obs - mu) / var ** 0.5
        # One-sided upper: Φ(z) approximation
        import math
        def _phi(z):
            return 0.5 * (1 + math.erf(z / math.sqrt(2)))
        p = 1 - _phi(z)
        return or_, p


def _bh_correction(p_values: list) -> list:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    ranks   = [None] * n
    for rank, (i, _) in enumerate(indexed, 1):
        ranks[i] = rank
    adj = [None] * n
    cummin = float("inf")
    for _, (i, p) in reversed(list(enumerate(indexed))):
        rank = ranks[i]
        corrected = p * n / rank
        cummin = min(cummin, corrected)
        adj[i] = min(cummin, 1.0)
    return adj


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

def _analyze(kmer_records: list, top_k: int, motif_dict: dict) -> list:
    """
    kmer_records: list of {kmer, activation} sorted descending by activation.
    top_k:        how many top k-mers to use as foreground.
    Returns list of enrichment result dicts, sorted by p_adj.
    """
    all_kmers   = {r["kmer"] for r in kmer_records}
    total_bg    = len(all_kmers) - top_k
    fg_set      = {r["kmer"] for r in kmer_records[:top_k]}

    results = []
    p_values = []
    entries  = []

    for motif_name, motif_kmers in motif_dict.items():
        # Restrict to k-mers that appear in our universe (should be all 4096 for DNA 6-mers)
        motif_in_universe = motif_kmers & all_kmers
        obs    = len(fg_set & motif_in_universe)
        bg_in  = len(motif_in_universe) - obs   # motif members in the background
        or_, p = _fisher_exact_enrichment(obs, top_k, bg_in, total_bg)
        p_values.append(p)
        entries.append({
            "motif":         motif_name,
            "obs_in_top_k":  obs,
            "expected":      round(top_k * len(motif_in_universe) / len(all_kmers), 2),
            "motif_size":    len(motif_in_universe),
            "odds_ratio":    round(or_, 4),
            "p_value":       round(p, 6),
            "p_adj":         None,   # filled below
            "top_k_members": sorted(fg_set & motif_in_universe),
        })

    adj = _bh_correction(p_values)
    for i, e in enumerate(entries):
        e["p_adj"] = round(adj[i], 6)

    return sorted(entries, key=lambda x: x["p_adj"])


# ─────────────────────────────────────────────────────────────────────────────
# Compute per-position nucleotide enrichment in top-K
# ─────────────────────────────────────────────────────────────────────────────

def _position_enrichment(fg_kmers: list, bg_kmers: list) -> dict:
    """
    For each of the 6 positions, compute the frequency of each nucleotide in fg
    vs. bg. Returns dict with position-level counts for plotting.
    """
    nt = list("ACGT")
    fg_freq = np.zeros((6, 4))
    bg_freq = np.zeros((6, 4))
    for km in fg_kmers:
        for pos, base in enumerate(km):
            if base in nt:
                fg_freq[pos, nt.index(base)] += 1
    for km in bg_kmers:
        for pos, base in enumerate(km):
            if base in nt:
                bg_freq[pos, nt.index(base)] += 1
    # normalise
    fg_freq /= max(fg_freq.sum(axis=1, keepdims=True).max(), 1)
    bg_freq_norm = bg_freq / max(bg_freq.sum(axis=1, keepdims=True).max(), 1)
    log2fc = np.log2((fg_freq + 1e-6) / (bg_freq_norm + 1e-6))
    return {
        "fg_freq":   fg_freq.tolist(),
        "bg_freq":   bg_freq_norm.tolist(),
        "log2fc":    log2fc.tolist(),
        "nt_order":  nt,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot(results: list, pos_enrich: dict, fg_kmers: list, top_k: int,
          all_records: list, out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("  [skip plot] matplotlib not available")
        return

    # Filter to motifs with at least 1 fg member
    sig   = [r for r in results if r["obs_in_top_k"] > 0]
    sig_sorted = sorted(sig, key=lambda x: -x["odds_ratio"])[:20]

    fig = plt.figure(figsize=(16, 11))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_or   = fig.add_subplot(gs[0, 0])
    ax_pval = fig.add_subplot(gs[0, 1])
    ax_pos  = fig.add_subplot(gs[1, 0])
    ax_act  = fig.add_subplot(gs[1, 1])

    # ── Odds ratio bar chart ──────────────────────────────────────────────────
    names = [r["motif"] for r in sig_sorted]
    ors   = [min(r["odds_ratio"], 20) for r in sig_sorted]
    padjs = [r["p_adj"] for r in sig_sorted]
    colours = ["#d73027" if p < 0.05 else "#4575b4" for p in padjs]
    ax_or.barh(range(len(names)), ors[::-1], color=colours[::-1])
    ax_or.set_yticks(range(len(names)))
    ax_or.set_yticklabels(names[::-1], fontsize=8)
    ax_or.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax_or.set_xlabel("Odds ratio (capped at 20)")
    ax_or.set_title(f"Motif enrichment in top-{top_k} SW-activating 6-mers\n(red = p_adj < 0.05)")
    ax_or.grid(True, alpha=0.3, axis="x")

    # ── -log10(p_adj) volcano-style ───────────────────────────────────────────
    all_names = [r["motif"] for r in results]
    all_ors   = [min(r["odds_ratio"], 20) for r in results]
    all_ps    = [-np.log10(max(r["p_adj"], 1e-10)) for r in results]
    sig_mask  = [r["p_adj"] < 0.05 for r in results]
    ax_pval.scatter([all_ors[i] for i in range(len(results)) if not sig_mask[i]],
                    [all_ps[i]  for i in range(len(results)) if not sig_mask[i]],
                    s=30, alpha=0.5, color="#4575b4", label="n.s.")
    ax_pval.scatter([all_ors[i] for i in range(len(results)) if sig_mask[i]],
                    [all_ps[i]  for i in range(len(results)) if sig_mask[i]],
                    s=50, alpha=0.8, color="#d73027", label="p_adj < 0.05")
    for i, r in enumerate(results):
        if r["p_adj"] < 0.05:
            ax_pval.annotate(r["motif"], (min(r["odds_ratio"], 20), all_ps[i]),
                             fontsize=7, ha="left", va="bottom")
    ax_pval.axhline(-np.log10(0.05), color="gray", linewidth=0.8, linestyle="--")
    ax_pval.axvline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax_pval.set_xlabel("Odds ratio")
    ax_pval.set_ylabel("-log₁₀(p_adj)")
    ax_pval.set_title("Enrichment significance")
    ax_pval.legend(fontsize=8)
    ax_pval.grid(True, alpha=0.3)

    # ── Position-level log2FC heatmap ─────────────────────────────────────────
    log2fc = np.array(pos_enrich["log2fc"])    # [6, 4]
    nt_order = pos_enrich["nt_order"]
    im = ax_pos.imshow(log2fc.T, aspect="auto", cmap="RdBu_r",
                       vmin=-2, vmax=2, interpolation="nearest")
    ax_pos.set_xticks(range(6))
    ax_pos.set_xticklabels([f"pos {i+1}" for i in range(6)])
    ax_pos.set_yticks(range(4))
    ax_pos.set_yticklabels(nt_order)
    ax_pos.set_title(f"Nucleotide log₂FC in top-{top_k} vs background\n(red=enriched, blue=depleted)")
    plt.colorbar(im, ax=ax_pos, label="log₂FC")

    # ── Activation score distribution with top k-mer annotations ─────────────
    all_acts = [r["activation_mean"] for r in all_records]
    ax_act.hist(all_acts, bins=60, color="#4575b4", alpha=0.7, label="all 4096 k-mers")
    fg_acts  = [r["activation_mean"] for r in all_records[:top_k]]
    ax_act.hist(fg_acts, bins=30, color="#d73027", alpha=0.7, label=f"top {top_k}")
    ax_act.axvline(np.percentile(all_acts, 95), color="orange", linestyle="--",
                   linewidth=1, label="95th percentile")
    ax_act.set_xlabel("Mean SW activation at k-mer position")
    ax_act.set_ylabel("Count")
    ax_act.set_title("SW activation score distribution")
    ax_act.legend(fontsize=8)
    ax_act.grid(True, alpha=0.3)

    plt.suptitle("Super-weight k-mer motif analysis — GENERator", fontsize=13, y=1.02)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-reference SW-activating k-mers against known regulatory motifs"
    )
    p.add_argument("--kmer_scan", default="results/sw_kmer_scan.json",
                   help="Output of run_sw_kmer_scan.py")
    p.add_argument("--top_k",    type=int, default=200,
                   help="Foreground size: top-K SW-activating k-mers to test")
    p.add_argument("--out",      default="results/sw_kmer_motifs.json")
    p.add_argument("--plot",     default="results/sw_kmer_motifs.png")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load k-mer scan ───────────────────────────────────────────────────────
    kmer_path = Path(args.kmer_scan)
    if not kmer_path.exists():
        print(f"K-mer scan file not found: {kmer_path}")
        print("Run scripts/run_sw_kmer_scan.py first.")
        sys.exit(1)

    scan = json.load(open(kmer_path))
    # Support both list format and dict-with-'results' format
    if isinstance(scan, dict):
        records = scan.get("results", scan.get("kmers", []))
    else:
        records = scan

    # Ensure sorted descending by activation
    records = sorted(records, key=lambda x: x.get("activation_mean", 0), reverse=True)
    print(f"  Loaded {len(records)} k-mer records")
    print(f"  Foreground: top {args.top_k} k-mers")

    if len(records) < args.top_k:
        print(f"  Warning: fewer than {args.top_k} k-mers available; using all {len(records)}")
        args.top_k = len(records)

    # ── Build motif dictionary ────────────────────────────────────────────────
    print("  Building motif dictionary ...", flush=True)
    motif_dict = _build_motif_dict()
    for name, s in sorted(motif_dict.items(), key=lambda x: -len(x[1])):
        print(f"    {name:20s}  {len(s):4d} 6-mers")

    # ── Run enrichment analysis ───────────────────────────────────────────────
    print(f"\n  Running Fisher enrichment for {len(motif_dict)} motif classes ...", flush=True)
    results = _analyze(records, args.top_k, motif_dict)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n  === Motif enrichment results (sorted by p_adj) ===")
    print(f"  {'Motif':22s}  {'obs':>5}  {'exp':>6}  {'OR':>7}  {'p':>8}  {'p_adj':>8}  sig")
    for r in results:
        sig = "***" if r["p_adj"] < 0.001 else ("**" if r["p_adj"] < 0.01
              else ("*" if r["p_adj"] < 0.05 else ""))
        print(f"  {r['motif']:22s}  {r['obs_in_top_k']:5d}  {r['expected']:6.1f}"
              f"  {min(r['odds_ratio'], 99.99):7.2f}  {r['p_value']:8.4f}  {r['p_adj']:8.4f}  {sig}")

    # ── Position enrichment ───────────────────────────────────────────────────
    fg_kmers = [r["kmer"] for r in records[:args.top_k]]
    bg_kmers = [r["kmer"] for r in records[args.top_k:]]
    pos_enrich = _position_enrichment(fg_kmers, bg_kmers)

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kmer_scan_file": str(kmer_path),
        "top_k":          args.top_k,
        "total_kmers":    len(records),
        "motif_results":  results,
        "position_enrichment": pos_enrich,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Results saved → {args.out}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot(results, pos_enrich, fg_kmers, args.top_k, records, args.plot)


if __name__ == "__main__":
    main()
