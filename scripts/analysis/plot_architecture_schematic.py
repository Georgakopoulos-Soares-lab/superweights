"""
Architecture schematic figure for GENERator EUK and PROK.

Shows:
  - 30-layer transformer stack (left portion of each panel)
  - Highlighted superweight layer
  - Zoom-in of down_proj row strip with SW rows marked (right portion)

Output: results/architecture_schematic.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
NUM_LAYERS = 30
D_MODEL    = 3072

MODELS = [
    {
        "title":    "GENERator-EUK-3B",
        "subtitle": "(eukaryote)",
        "sw_layer": 4,
        "sw_rows":  [1522, 2371],
    },
    {
        "title":    "GENERator-PROK-3B",
        "subtitle": "(prokaryote)",
        "sw_layer": 2,
        "sw_rows":  [1927],
    },
]

# ── Colours ──────────────────────────────────────────────────────────────────
C_LAYER_BG   = "#DDE4EF"   # normal transformer layer fill
C_LAYER_EDGE = "#9AAAC8"   # normal layer border
C_SW_FILL    = "#4878CF"   # SW layer fill
C_SW_EDGE    = "#2A54A8"   # SW layer border
C_EMBED_FILL = "#B8D8B8"   # embedding / LM-head fill
C_EMBED_EDGE = "#6FA86F"
C_SW_ROW     = "#E8541A"   # superweight row stripe
C_NORM_ROW   = "#D8DEE8"   # normal row in strip
C_STRIP_EDGE = "#808080"
C_ARROW      = "#505050"

FONT = "DejaVu Sans"

# ── Helpers ──────────────────────────────────────────────────────────────────

def _draw_rounded_rect(ax, x, y, w, h, fc, ec, lw=1.0, radius=0.008, zorder=2, alpha=1.0):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0",
        fc=fc, ec=ec, lw=lw, zorder=zorder, alpha=alpha,
        transform=ax.transData,
    )
    ax.add_patch(box)


def _draw_panel(ax, model):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    sw_layer  = model["sw_layer"]
    sw_rows   = model["sw_rows"]

    # ── Title ─────────────────────────────────────────────────────────────
    ax.text(0.5, 0.98, model["title"],
            ha="center", va="top", fontsize=12, fontweight="bold",
            fontfamily=FONT, transform=ax.transAxes)
    ax.text(0.5, 0.94, model["subtitle"],
            ha="center", va="top", fontsize=9, color="#555555",
            fontfamily=FONT, transform=ax.transAxes)

    # ── Layout: fix all block heights, derive layer_h to fill remaining ────
    stack_x   = 0.08
    stack_w   = 0.38
    top_y     = 0.90   # top of embedding
    bot_y     = 0.17   # absolute bottom of LM head block
    embed_h   = 0.030
    lmhead_h  = 0.030
    gap       = 0.004  # gap between every adjacent block pair

    # Space available for layers = total - embed - lmhead - 3 inter-block gaps
    # (embed→layer1, layer29→lmhead, plus the gap already absorbed per layer)
    # Each layer slot = layer_h (NO internal gaps between layers — they touch)
    layer_area = (top_y - bot_y) - embed_h - lmhead_h - 3 * gap
    layer_h    = layer_area / NUM_LAYERS

    # Absolute y positions
    embed_y   = top_y - embed_h                 # bottom of embed block
    layers_y  = embed_y - gap                   # top of first layer
    lmh_y     = bot_y                           # bottom of LM head block

    # ── Embedding block ────────────────────────────────────────────────────
    _draw_rounded_rect(ax, stack_x, embed_y, stack_w, embed_h,
                       fc=C_EMBED_FILL, ec=C_EMBED_EDGE, lw=0.8)
    ax.text(stack_x + stack_w / 2, embed_y + embed_h / 2,
            "Embedding", ha="center", va="center",
            fontsize=7.5, fontfamily=FONT, color="#2A5A2A")

    # ── Transformer layers (no gaps between them — solid stack) ───────────
    layer_tops = []   # top-edge y of each layer

    for i in range(NUM_LAYERS):
        ly_top = layers_y - i * layer_h
        ly_bot = ly_top - layer_h
        layer_tops.append(ly_top)

        is_sw = (i == sw_layer)
        fc    = C_SW_FILL  if is_sw else C_LAYER_BG
        ec    = C_SW_EDGE  if is_sw else C_LAYER_EDGE
        lw    = 1.4        if is_sw else 0.5

        _draw_rounded_rect(ax, stack_x, ly_bot, stack_w, layer_h,
                           fc=fc, ec=ec, lw=lw, zorder=3 if is_sw else 2)

        if i % 5 == 0 or is_sw:
            colour = "white" if is_sw else "#444444"
            fw     = "bold"  if is_sw else "normal"
            ax.text(stack_x + stack_w / 2, ly_bot + layer_h / 2,
                    f"L{i}",
                    ha="center", va="center",
                    fontsize=6.5 if not is_sw else 7.5,
                    fontfamily=FONT, color=colour, fontweight=fw, zorder=4)

    last_layer_bot = layers_y - NUM_LAYERS * layer_h   # bottom of L29

    # ── LM head block ──────────────────────────────────────────────────────
    _draw_rounded_rect(ax, stack_x, lmh_y, stack_w, lmhead_h,
                       fc=C_EMBED_FILL, ec=C_EMBED_EDGE, lw=0.8)
    ax.text(stack_x + stack_w / 2, lmh_y + lmhead_h / 2,
            "LM head", ha="center", va="center",
            fontsize=7.5, fontfamily=FONT, color="#2A5A2A")

    # ── Down-proj row strip ────────────────────────────────────────────────
    strip_x   = 0.62
    strip_w   = 0.14
    strip_top = layers_y                  # top of first layer
    strip_bot = last_layer_bot            # bottom of last layer
    strip_h   = strip_top - strip_bot

    ax.add_patch(mpatches.Rectangle(
        (strip_x, strip_bot), strip_w, strip_h,
        fc=C_NORM_ROW, ec=C_STRIP_EDGE, lw=0.8, zorder=2))

    # SW row stripes
    row_stripe_h = max(strip_h / D_MODEL * 6, 0.010)

    for row in sw_rows:
        row_y = strip_bot + (D_MODEL - 1 - row) / D_MODEL * strip_h
        ax.add_patch(mpatches.Rectangle(
            (strip_x, row_y), strip_w, row_stripe_h,
            fc=C_SW_ROW, ec="none", zorder=3, lw=0))
        ax.annotate(
            f"  row {row}",
            xy=(strip_x + strip_w, row_y + row_stripe_h / 2),
            xytext=(strip_x + strip_w + 0.01, row_y + row_stripe_h / 2),
            ha="left", va="center",
            fontsize=7.5, fontfamily=FONT, color=C_SW_ROW, fontweight="bold",
            annotation_clip=False,
        )

    # Strip labels
    ax.text(strip_x + strip_w / 2, strip_top + 0.016,
            "down_proj", ha="center", va="bottom",
            fontsize=8, fontfamily=FONT, color="#333333", fontstyle="italic")
    ax.text(strip_x + strip_w / 2, strip_top + 0.042,
            f"({D_MODEL} rows)", ha="center", va="bottom",
            fontsize=7, fontfamily=FONT, color="#666666")
    ax.text(strip_x - 0.02, strip_top, "row 0", ha="right", va="center",
            fontsize=6, fontfamily=FONT, color="#666666")
    ax.text(strip_x - 0.02, strip_bot, f"row {D_MODEL-1}", ha="right", va="center",
            fontsize=6, fontfamily=FONT, color="#666666")
    ax.plot([strip_x - 0.01, strip_x], [strip_top, strip_top],
            color="#888888", lw=0.8, zorder=3)
    ax.plot([strip_x - 0.01, strip_x], [strip_bot, strip_bot],
            color="#888888", lw=0.8, zorder=3)

    # ── Arrow: SW layer → strip mid ────────────────────────────────────────
    sw_top     = layer_tops[sw_layer]
    sw_bot     = sw_top - layer_h
    sw_ctr_y   = (sw_top + sw_bot) / 2

    ax.annotate(
        "",
        xy=(strip_x, (strip_top + strip_bot) / 2),
        xytext=(stack_x + stack_w, sw_ctr_y),
        arrowprops=dict(
            arrowstyle="-|>",
            color=C_ARROW,
            lw=1.0,
            connectionstyle="arc3,rad=-0.10",
        ),
        zorder=5,
    )

    # Dot on SW layer
    ax.plot(stack_x + stack_w + 0.015, sw_ctr_y, "o",
            color=C_SW_ROW, markersize=5, zorder=6, clip_on=False)

    # ── Legend ────────────────────────────────────────────────────────────
    sw_patch = mpatches.Patch(color=C_SW_ROW, label="Superweight row")
    nl_patch = mpatches.Patch(color=C_NORM_ROW, ec=C_STRIP_EDGE, lw=0.6,
                               label="Other rows")
    sl_patch = mpatches.Patch(color=C_SW_FILL, label="Superweight layer")
    ax.legend(
        handles=[sl_patch, sw_patch, nl_patch],
        loc="lower left",
        fontsize=7.5,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        ncol=1,
        handlelength=1.4,
        handleheight=1.0,
        borderpad=0.7,
        bbox_to_anchor=(0.04, 0.02),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 8),
                             gridspec_kw={"wspace": 0.08})

    for ax, model in zip(axes, MODELS):
        _draw_panel(ax, model)

    fig.suptitle(
        "Superweight location in GENERator 3B architecture",
        fontsize=13, fontweight="bold", y=1.01, fontfamily=FONT,
    )

    out = "results/architecture_schematic.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
