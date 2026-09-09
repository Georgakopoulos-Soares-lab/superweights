"""
E10b mask design -- E9/E10's disjoint-sampling algorithm reused for N=6 (Phi-3's frozen
basis size), imported directly rather than reimplemented: mask_to_vec-equivalent,
sample_disjoint, and check_design all come from E9's generate_masks.py.

K=6 has far fewer available masks than DNABERT-2/MosaicBERT/ModernBERT's K=10 (2^6=64 total
vs 2^10=1024), so E9's exact density/count numbers do not transfer -- they were re-derived
here (Phase 3 dry run, no response data involved) and validated against a full-rank,
well-conditioned lifted design before being frozen into the E10b prereg.

k=1 is reserved exclusively for the singleton pool (E9's convention); density buckets draw
only from k in {2,3,4,5}. Densities rho in {0.25,0.5,0.75} map to k=2 (2/6=.33, closest
available to .25 given k=1 is reserved), k=3 (exact 0.5), and k in {4,5} (mean 4.5/6=.75
exactly).
"""
from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "E9_mechanistic_tomography"))
import generate_masks as g9  # noqa: E402  (sample_disjoint, check_design -- reused verbatim)

N = 6
SEED = 20260823
DENSITIES = {"rho_0.25": [2], "rho_0.5": [3], "rho_0.75": [4, 5]}
# Frozen after the Phase 3 dry-run validated full rank (21/21), cond(XtX)=588.8,
# max|corr|=0.672, 14/15 pairs covered in held_out. Uses all 56 available non-singleton,
# non-trivial masks (15+20+15+6 across k=2..5) -- no slack left unused.
POOL_COUNTS = {
    "fit":         {"rho_0.25": 9, "rho_0.5": 14, "rho_0.75": 13},
    "calibration": {"rho_0.25": 3, "rho_0.5": 3,  "rho_0.75": 4},
    "held_out":    {"rho_0.25": 3, "rho_0.5": 3,  "rho_0.75": 4},
}


def mask_to_vec(m, n=N):
    v = [0] * n
    for i in m:
        v[i] = 1
    return v


def build_design():
    rng = random.Random(SEED)
    all_ks = sorted({k for ks in DENSITIES.values() for k in ks})
    pool_by_k = {k: list(itertools.combinations(range(N), k)) for k in all_ks}
    for k in pool_by_k:
        rng.shuffle(pool_by_k[k])

    used: set = set()
    pools = {}
    singleton_masks = [tuple([i]) for i in range(N)]
    for m in singleton_masks:
        used.add(m)
    pools["singletons"] = [mask_to_vec(m) for m in singleton_masks]

    for pool_name in ("fit", "calibration", "held_out"):
        pool_masks = []
        for density_name, k_choices in DENSITIES.items():
            n_needed = POOL_COUNTS[pool_name][density_name]
            pool_masks.extend(g9.sample_disjoint(rng, pool_by_k, k_choices, n_needed, used))
        pools[pool_name] = [mask_to_vec(m) for m in pool_masks]
    return pools


def design_matrix_additive(masks):
    import numpy as np
    return np.array(masks, dtype=float)


def design_matrix_lifted(masks):
    import numpy as np
    pairs = list(itertools.combinations(range(N), 2))
    rows = []
    for a in masks:
        rows.append(list(a) + [a[i] * a[j] for i, j in pairs])
    return np.array(rows, dtype=float)


def cooccurrence(vecs):
    s = set()
    for v in vecs:
        on = [i for i, x in enumerate(v) if x]
        for i, j in itertools.combinations(on, 2):
            s.add((i, j))
    return s
