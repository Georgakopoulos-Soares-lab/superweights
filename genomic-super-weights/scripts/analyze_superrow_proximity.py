"""
analyze_superrow_proximity.py
------------------------------
Measures how spatially clustered the detected super-rows are compared to what
you would expect from 10 randomly chosen (layer, row) pairs.

Model space (DNABERT-2 mlp.wo):
    layers : 0 – 11      (12 total)
    rows   : 0 – 767     (768 total, out-channels of the down-proj)

The 10 super-rows from super_weight_index.json are mapped to 2-D coordinates
(layer, row) and compared against 50 000 Monte-Carlo random sets of 10 rows.

Metrics computed
----------------
mean_pairwise_dist  – average Euclidean distance between all C(10,2)=45 pairs
                      in the space normalised to [0,1]^2.
layer_span          – max_layer - min_layer  (raw, integer)
row_span            – max_row   - min_row    (raw, integer)
unique_rows         – how many distinct row indices appear
unique_layers       – how many distinct layer indices appear

Output
------
  - printed summary table
  - results/superrow_proximity.png  (scatter + distance distribution)

Run from the repo root:
    python scripts/analyze_superrow_proximity.py
"""

import json
import itertools
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
NUM_LAYERS   = 12
NUM_ROWS     = 768          # out-channels of mlp.wo
N_MONTE      = 50_000       # random control sets
SEED         = 42
INDEX_PATH   = Path("results/super_weight_index.json")
OUT_PNG      = Path("results/superrow_proximity.png")

# ── Load super-rows ───────────────────────────────────────────────────────────
index = json.loads(INDEX_PATH.read_text())
superrows = [
    (sw["layer"], sw["row"])
    for sw in index["dnabert2"]["results"]
]
n = len(superrows)
print(f"Loaded {n} DNABERT-2 super-rows:")
for layer, row in sorted(superrows):
    print(f"  layer={layer:2d}  row={row:4d}")

# ── Normalise coordinates to [0,1]^2 ─────────────────────────────────────────
def normalise(pts):
    """pts: list of (layer, row) → np.ndarray shape (k, 2)"""
    a = np.array(pts, dtype=float)
    a[:, 0] /= (NUM_LAYERS - 1)   # layer → [0, 1]
    a[:, 1] /= (NUM_ROWS   - 1)   # row   → [0, 1]
    return a

# ── Pairwise-distance statistics ──────────────────────────────────────────────
def mean_pairwise_dist(pts_norm):
    dists = []
    for i, j in itertools.combinations(range(len(pts_norm)), 2):
        d = np.linalg.norm(pts_norm[i] - pts_norm[j])
        dists.append(d)
    return float(np.mean(dists))

sw_norm  = normalise(superrows)
sw_mpd   = mean_pairwise_dist(sw_norm)

sw_layers = [p[0] for p in superrows]
sw_rows   = [p[1] for p in superrows]

print(f"\nSuper-row statistics (raw coordinates):")
print(f"  unique layers   : {sorted(set(sw_layers))}")
print(f"  unique rows     : {sorted(set(sw_rows))}")
print(f"  layer span      : {max(sw_layers) - min(sw_layers)} (layers {min(sw_layers)}–{max(sw_layers)})")
print(f"  row span        : {max(sw_rows)   - min(sw_rows)}   (rows {min(sw_rows)}–{max(sw_rows)})")
print(f"  mean pairwise dist (normalised): {sw_mpd:.4f}")

# ── Pairwise distance matrix between super-rows ───────────────────────────────
print("\nPairwise distances (normalised) between super-rows:")
labels = [f"L{l}R{r}" for l, r in superrows]
max_lbl = max(len(x) for x in labels)
header  = " " * (max_lbl + 2) + "  ".join(f"{x:>{max_lbl}}" for x in labels)
print(header)
for i, (li, ri) in enumerate(superrows):
    row_str = f"{labels[i]:<{max_lbl}}  "
    for j, (lj, rj) in enumerate(superrows):
        d = np.linalg.norm(sw_norm[i] - sw_norm[j])
        row_str += f"{d:{max_lbl}.3f}  "
    print(row_str)

# ── Monte-Carlo: random row sets ──────────────────────────────────────────────
all_coords = [(l, r) for l in range(NUM_LAYERS) for r in range(NUM_ROWS)]
rng = random.Random(SEED)

rand_mpds = []
for _ in range(N_MONTE):
    sample = rng.sample(all_coords, n)
    rand_mpds.append(mean_pairwise_dist(normalise(sample)))

rand_mpds = np.array(rand_mpds)
p_value   = float(np.mean(rand_mpds <= sw_mpd))   # fraction of random sets as or more clustered
percentile = float(np.mean(rand_mpds < sw_mpd)) * 100

print(f"\nMonte-Carlo results ({N_MONTE:,} random sets of {n} rows):")
print(f"  random mean pairwise dist  : {rand_mpds.mean():.4f} ± {rand_mpds.std():.4f}")
print(f"  super-row mean pairwise dist: {sw_mpd:.4f}")
z_score = (sw_mpd - rand_mpds.mean()) / rand_mpds.std()
print(f"  z-score                    : {z_score:+.2f}  (negative = more clustered than random)")
print(f"  percentile                 : {percentile:.1f}th  "
      f"({'more clustered' if z_score < 0 else 'more spread'} than {100 - percentile:.1f}% of random sets)")

if z_score < -1.5:
    verdict = ("CLUSTERED — the super-rows are significantly tighter than random. "
               "The random control is spread across more of the model. "
               "For a fair comparison, sample random rows with the SAME (layer, row) "
               "distribution structure (see --structured_rand flag suggestion below).")
elif z_score > 1.5:
    verdict = ("SPREAD — the super-rows are more spread out than random. "
               "The random control is naturally fairer (similar or more coverage).")
else:
    verdict = ("SIMILAR — super-row proximity is close to what you'd expect by chance. "
               "The current random control is already fair.")

print(f"\nVerdict: {verdict}")

# ── Structured-random suggestion ──────────────────────────────────────────────
print("\n── Structured-random control suggestion ──")
print("If you need proximity-matched random sets, sample random rows such that:")
layer_counts = {}
for l, _ in superrows:
    layer_counts[l] = layer_counts.get(l, 0) + 1
print(f"  • same layer distribution: {dict(sorted(layer_counts.items()))}")
row_center  = float(np.mean(sw_rows))
row_std_raw = float(np.std(sw_rows))
print(f"  • rows drawn from a window of ±{int(row_std_raw * 2)} around row {row_center:.0f}")
print("  • OR: shuffle only the row indices within each layer stratum.")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: scatter of super-rows in (layer, row) space
ax = axes[0]
# Background: all possible positions (tiny grey dots)
all_l = [c[0] for c in all_coords]
all_r = [c[1] for c in all_coords]
ax.scatter(all_l, all_r, c="lightgrey", s=0.3, alpha=0.2, zorder=0)

# Super-rows coloured by row index to reveal repeated rows
unique_row_ids = sorted(set(sw_rows))
cmap = plt.get_cmap("tab10", len(unique_row_ids))
row_color = {r: cmap(i) for i, r in enumerate(unique_row_ids)}

for layer, row in superrows:
    ax.scatter(layer, row, s=180, zorder=3,
               color=row_color[row],
               edgecolors="black", linewidths=0.8,
               label=f"row {row}")

# Remove duplicate legend entries
handles, lbls = ax.get_legend_handles_labels()
seen = {}
for h, l in zip(handles, lbls):
    if l not in seen:
        seen[l] = h
ax.legend(seen.values(), seen.keys(), title="row idx", fontsize=8,
          loc="upper left", markerscale=0.8)

ax.set_xlabel("Layer index", fontsize=11)
ax.set_ylabel("Row index (out-channel of mlp.wo)", fontsize=11)
ax.set_title("Super-row positions in (layer, row) space\n(colour = unique row index)", fontsize=11)
ax.set_xlim(-0.5, NUM_LAYERS - 0.5)
ax.set_ylim(-10, NUM_ROWS + 10)
ax.set_xticks(range(NUM_LAYERS))
ax.grid(True, alpha=0.3)

# Right: distribution of random mean pairwise dist vs super-row value
ax = axes[1]
ax.hist(rand_mpds, bins=80, color="steelblue", alpha=0.7,
        label=f"Random ({N_MONTE:,} sets)")
ax.axvline(sw_mpd, color="crimson", linewidth=2.5,
           label=f"Super-rows\n(MPD={sw_mpd:.3f}, z={z_score:+.2f})")
ax.axvline(rand_mpds.mean(), color="navy", linewidth=1.5, linestyle="--",
           label=f"Random mean\n({rand_mpds.mean():.3f})")
ax.set_xlabel("Mean pairwise distance (normalised [0,1]²)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Super-row clustering vs. random baseline\n"
             f"(z={z_score:+.2f}, {percentile:.0f}th percentile)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle("DNABERT-2 super-row proximity analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {OUT_PNG}")
