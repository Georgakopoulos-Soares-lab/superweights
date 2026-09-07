"""
experiments/E11_scale_ladder/plot.py

E11 primary plot (PREREG_E11_scale_ladder.md, Analysis #1): q1 vs log10(non-embedding
params), colored by architecture class (decoder/encoder), shaped by domain (text/genomic),
with within-family lines connecting sizes in the same ladder (Qwen2.5, SmolLM2, EuroBERT,
GENERator-PROK).

Colors: Okabe-Ito CVD-safe pair (blue #0072B2 decoder, vermillion #D55E00 encoder).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
LADDER_CSV = ROOT / "results" / "E11" / "scale_ladder.csv"
OUT = ROOT / "figures" / "E11" / "q1_vs_scale.pdf"

COLOR = {"decoder": "#0072B2", "encoder": "#D55E00"}
MARKER = {"text": "o", "genomic": "^"}
FAMILY_LADDERS = ["Qwen2.5", "SmolLM2", "EuroBERT", "GENERator-PROK"]


def main():
    rows = []
    with LADDER_CSV.open() as f:
        for r in csv.DictReader(f):
            try:
                q1 = float(r["q1"])
                nep = float(r["non_embed_params"])
            except (TypeError, ValueError):
                continue
            rows.append(dict(model=r["model"], family=r["family"], domain=r["domain"],
                              architecture=r["architecture"], q1=q1, logp=math.log10(nep)))

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # within-family connecting lines, drawn first (under the points)
    for fam in FAMILY_LADDERS:
        members = sorted([r for r in rows if r["family"] == fam], key=lambda r: r["logp"])
        if len(members) >= 2:
            arch = members[0]["architecture"]
            ax.plot([m["logp"] for m in members], [m["q1"] for m in members],
                    color=COLOR[arch], linewidth=1.1, alpha=0.55, zorder=1)

    seen_labels = set()
    for r in rows:
        label_key = (r["architecture"], r["domain"])
        label = None
        if label_key not in seen_labels:
            label = f"{r['architecture']} ({r['domain']})"
            seen_labels.add(label_key)
        ax.scatter(r["logp"], r["q1"], color=COLOR[r["architecture"]], marker=MARKER[r["domain"]],
                   s=64, edgecolor="white", linewidth=0.6, zorder=3, label=label)

    for r in rows:
        if r["family"] in FAMILY_LADDERS or r["family"] in ("Llama", "Mistral", "OLMo-v1", "Phi-3",
                                                              "MosaicBERT", "ModernBERT", "NTv3",
                                                              "DNABERT-2", "GENERator-EUK", "GenomeOcean"):
            ax.annotate(r["model"], (r["logp"], r["q1"]), fontsize=6, color="#444444",
                        xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(r"$\log_{10}$(non-embedding parameters)")
    ax.set_ylabel(r"$q_1$ (top singular-energy share)")
    ax.set_title("E11 scale ladder: $q_1$ vs. scale, by architecture and domain")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
