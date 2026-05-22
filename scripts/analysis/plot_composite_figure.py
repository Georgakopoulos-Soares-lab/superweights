"""
Composite manuscript figure.
  A — model_ablation_comparison
  B — architecture_schematic
  C — activation_lifecycle

Figure size: 11 × 8.5 inches, 300 dpi.
Panel labels A/B/C at top-left of each panel.
Suptitles stripped by cropping the top of each source image.

Output: results/composite_figure.png
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np

# ── Source images ─────────────────────────────────────────────────────────────
SOURCES = [
    ("A", "results/model_ablation_comparison.png"),
    ("B", "results/architecture_schematic.png"),
    ("C", "results/activation_lifecycle.png"),
]

# Fraction to strip from the top of each image (removes suptitle text)
# and from the bottom (removes figure-level captions / whitespace)
CROP = {
    "A": dict(top=0.052, bot=0.010),   # strips "Super-weight behaviour..." title
    "B": dict(top=0.038, bot=0.005),   # strips "Superweight location..." suptitle
    "C": dict(top=0.000, bot=0.060),   # strips bottom annotation text
}


def _crop(img, top_frac, bot_frac):
    h = img.shape[0]
    t = int(h * top_frac)
    b = int(h * bot_frac)
    return img[t: h - b if b else h, :]


# ── Load & crop ───────────────────────────────────────────────────────────────
images = {}
for label, path in SOURCES:
    raw = mpimg.imread(path)
    c   = CROP[label]
    images[label] = _crop(raw, c["top"], c["bot"])

# Height ratios = natural h/w of each cropped image
# (so when displayed at the same width they are compressed equally)
hr = [images[lbl].shape[0] / images[lbl].shape[1] for lbl, _ in SOURCES]

# ── Build figure ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 8.5))

gs = gridspec.GridSpec(
    3, 1,
    figure=fig,
    height_ratios=hr,
    hspace=0.04,
    left=0.01,
    right=0.99,
    top=0.995,
    bottom=0.005,
)

axes = [fig.add_subplot(gs[i]) for i in range(3)]

for ax, (label, _) in zip(axes, SOURCES):
    img = images[label]
    ax.imshow(img, aspect="auto", interpolation="lanczos")
    ax.set_axis_off()

    # Panel label — bold, large, black, top-left corner
    ax.text(
        0.005, 0.97, label,
        transform=ax.transAxes,
        fontsize=18, fontweight="bold",
        va="top", ha="left",
        color="black",
        fontfamily="DejaVu Sans",
        zorder=10,
    )

# ── Save ──────────────────────────────────────────────────────────────────────
out = "results/composite_figure.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
