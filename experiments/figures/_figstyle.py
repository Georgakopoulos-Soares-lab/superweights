"""Shared matplotlib style for paper figures.

Principles:
  * No suptitles, no per-panel titles.
  * Compact bold 'A', 'B', ... labels in the upper-left corner of each axes.
  * Sans-serif, medium-weight, generous tick padding, no top/right spines.
  * Legends are placed *inside* axes in a corner that doesn't overlap data.
  * Short axis labels — keep numerical detail in the figure caption, not on the plot.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

BASE_RC = {
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size":  3.5,
    "ytick.major.size":  3.5,
    "savefig.dpi":       220,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,  # editable text in pdf
    "ps.fonttype":       42,
}


def apply_style() -> None:
    mpl.rcParams.update(BASE_RC)


def panel_label(ax, text: str, *, x: float = -0.10, y: float = 1.04,
                fontsize: int = 14, weight: str = "bold") -> None:
    """Put a bold panel label (A, B, …) in the corner of an axes."""
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=fontsize, fontweight=weight,
            va="bottom", ha="left", clip_on=False)
