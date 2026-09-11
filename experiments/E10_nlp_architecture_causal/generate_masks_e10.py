#!/usr/bin/env python
"""
E10 ARM B mask design -- E9's generate_masks.py reused by direct import.

Per the locked v2 prereg: n_E = 10 for both encoders, identical to E9's DNABERT-2 basis
size, so E9's mask-generation logic, seed, densities and pool counts are reused
**unmodified** (imported, not copied). One index-level mask design is produced and applied
to each encoder's OWN 10-row basis -- so mask index i means MosaicBERT L0/r287 for one
model and ModernBERT L15/r251 for the other. The physical row sets are therefore not
shared; the design matrix is, which makes the two models' tomography directly comparable
on an identical design.

Runs E9's own pre-fitting design checks (rank / condition number / column correlation of
the 10-col additive and 55-col lifted fit design, plus the fit-vs-held-out co-occurrence
comparison) BEFORE any response is measured, per Phase 3 / the design-size requirement.
"""
from __future__ import annotations

import itertools
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E9 = HERE.parent / "E9_mechanistic_tomography"
sys.path.insert(0, str(E9))

# E9's module, imported unmodified -- constants and helpers both come from there.
import generate_masks as g9  # noqa: E402

BASES = {
    "mosaicbert": [(0, 287), (0, 444), (1, 287), (11, 287), (0, 302),
                   (5, 287), (3, 287), (10, 666), (9, 287), (10, 79)],
    "modernbert": [(15, 251), (0, 251), (11, 251), (0, 67), (9, 251),
                   (4, 251), (1, 251), (5, 251), (10, 251), (1, 67)],
}
CANONICAL_INDEX = {"mosaicbert": 8, "modernbert": 0}  # E8 detector row's basis index


def build_design():
    """Exactly E9's main() procedure, returning the result instead of writing it."""
    rng = random.Random(g9.SEED)
    pool_by_k = {k: g9.all_masks_of_size(k) for ks in g9.DENSITIES.values() for k in ks}
    for k in pool_by_k:
        rng.shuffle(pool_by_k[k])

    used: set = set()
    pools = {}
    singleton_masks = [tuple([i]) for i in range(g9.N)]
    for m in singleton_masks:
        used.add(m)
    pools["singletons"] = [g9.mask_to_vec(m) for m in singleton_masks]

    for pool_name in ("fit", "calibration", "held_out"):
        pool_masks = []
        for density_name, k_choices in g9.DENSITIES.items():
            n_needed = g9.POOL_COUNTS[pool_name][density_name]
            pool_masks.extend(g9.sample_disjoint(rng, pool_by_k, k_choices, n_needed, used))
        pools[pool_name] = [g9.mask_to_vec(m) for m in pool_masks]
    return pools


def cooccurrence(vecs):
    s = set()
    for v in vecs:
        on = [i for i, x in enumerate(v) if x]
        for i, j in itertools.combinations(on, 2):
            s.add((i, j))
    return s


def main():
    pools = build_design()
    fit_vecs = pools["fit"]
    checks = [
        g9.check_design(g9.design_matrix_additive(fit_vecs), "fit_additive_10col"),
        g9.check_design(g9.design_matrix_lifted(fit_vecs), "fit_lifted_55col"),
    ]
    fit_pairs, held_pairs = cooccurrence(fit_vecs), cooccurrence(pools["held_out"])
    all_pairs = set(itertools.combinations(range(g9.N), 2))
    cochk = {
        "n_pairs_total": len(all_pairs),
        "n_pairs_seen_in_fit": len(fit_pairs),
        "n_pairs_seen_in_held_out": len(held_pairs),
        "n_pairs_seen_in_both": len(fit_pairs & held_pairs),
        "n_pairs_in_held_out_not_in_fit": len(held_pairs - fit_pairs),
    }

    print(f"pool sizes: singletons={len(pools['singletons'])} "
          f"fit={len(pools['fit'])} calibration={len(pools['calibration'])} "
          f"held_out={len(pools['held_out'])}")
    for c in checks:
        print(f"  [{c['label']}] rank={c['rank']}/{c['n_cols']} full_rank={c['full_rank']} "
              f"cond(XtX)={c['condition_number_XtX']:.3e} "
              f"max|corr|={c['max_abs_column_correlation']:.3f} "
              f"near_dup_pairs={c['n_near_duplicate_column_pairs']}")
    print(f"  cooccurrence: {cochk}")

    if not checks[1]["full_rank"]:
        raise SystemExit("*** LIFTED DESIGN RANK-DEFICIENT -- STOP per Phase 3 ***")

    for model, basis in BASES.items():
        out = {
            "model": model, "seed": g9.SEED, "n": g9.N,
            "densities": g9.DENSITIES, "pool_counts": g9.POOL_COUNTS,
            "basis": [list(b) for b in basis],
            "canonical_row_basis_index": CANONICAL_INDEX[model],
            "design_source": "E9 generate_masks.py, imported unmodified",
            "pools": pools, "design_checks": checks, "cooccurrence_check": cochk,
        }
        p = HERE / f"masks_{model}.json"
        p.write_text(json.dumps(out, indent=2))
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
