"""Shared model-class visual encoding for all v3 main/supplement figures.

One consistent scheme across Figures 1-4, per FIGURE_PLAN_v3.md's binding rule:
"Do not invent a new palette independently in every script."

Axis of encoding -> visual channel (fixed, do not remix per-figure):
  architecture (encoder vs decoder)  -> MARKER SHAPE  (primary organizing axis per thesis)
  domain (text vs genomic)           -> COLOR          (secondary/scrutinized axis)
  high-gain / candidate row          -> FILLED marker, black edge
  ordinary / control / random row    -> OPEN marker (facecolor='none'), colored edge
  focal model (this paper's causal deep dive: DNABERT-2, GENERator)
                                      -> larger marker size + heavier edge

This module intentionally holds only encoding constants + tiny helpers, not full plot
logic -- each figure script still owns its own axes/layout. Import via:
    from _paper_encoding import DOMAIN_COLOR, ARCH_MARKER, kw
"""
from __future__ import annotations

# ---- domain -> color -------------------------------------------------------
DOMAIN_COLOR = {
    "text":    "#4C72B0",   # blue
    "genomic": "#DD8452",   # orange
}

# ---- architecture -> marker shape ------------------------------------------
ARCH_MARKER = {
    "decoder": "o",   # circle
    "encoder": "s",   # square
}

# ---- row role -> fill style -------------------------------------------------
ROLE_FILL = {
    "high_gain": dict(alpha=0.95),                       # filled, full alpha
    "control":   dict(facecolors="none", alpha=0.9),     # open marker
}

# ---- fixed per-model metadata (domain, architecture) used across figures ---
# Extend as needed; keep this the single source of truth for model -> class.
MODEL_CLASS = {
    # NLP decoders
    "Llama-7B":       ("text", "decoder"),
    "Mistral-7B":      ("text", "decoder"),
    "OLMo-7B":         ("text", "decoder"),
    "Phi-3-mini":      ("text", "decoder"),
    "Qwen2.5-7B":      ("text", "decoder"),
    # NLP encoders
    "MosaicBERT":      ("text", "encoder"),
    "ModernBERT":      ("text", "encoder"),
    # genomic decoders
    "GENERator EUK":   ("genomic", "decoder"),
    "GENERator PROK":  ("genomic", "decoder"),
    "GenomeOcean-4B":  ("genomic", "decoder"),
    "Evo2-7B":         ("genomic", "decoder"),   # structural null, see caveats
    # genomic encoders
    "DNABERT-2":       ("genomic", "encoder"),
    "NTv3":            ("genomic", "encoder"),
}


def style_for(model: str, *, role: str = "high_gain") -> dict:
    """Return a matplotlib scatter/plot kwargs dict for `model`, given its role
    ('high_gain' or 'control'). Raises KeyError for an unregistered model name
    -- extend MODEL_CLASS rather than silently falling back to a default look.
    """
    domain, arch = MODEL_CLASS[model]
    kw = dict(
        color=DOMAIN_COLOR[domain],
        marker=ARCH_MARKER[arch],
        edgecolor="black" if role == "high_gain" else DOMAIN_COLOR[domain],
        linewidths=0.9 if role == "high_gain" else 0.7,
    )
    kw.update(ROLE_FILL[role])
    return kw


LEGEND_NOTE = (
    "marker shape = architecture (o decoder, s encoder); "
    "color = domain (blue text, orange genomic); "
    "filled = detected high-gain row, open = ordinary/control row"
)
