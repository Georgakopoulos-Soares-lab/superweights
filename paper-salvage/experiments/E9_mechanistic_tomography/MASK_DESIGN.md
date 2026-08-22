# MASK_DESIGN.md — E9 Phase 3

Produced by `generate_masks.py` (deterministic, `SEED = 20260822`, frozen with this file's
first commit — see `paper-salvage/docs/prereg/LOCKS.jsonl` after Phase 4 locking).
GENERator receives no mask design (`BASIS_FREEZE.md`: 1D dose-response only). This document
covers DNABERT-2 (`n_D = 10`) only.

## Pools (disjoint by construction — no mask realization appears in more than one pool)

| pool | n masks | densities |
|---|---|---|
| singletons | 10 | exhaustive (one-hot, all 10 basis components — trivial at this `n`) |
| fit | 78 | rho≈0.25 (k=2,3): 22 · rho≈0.5 (k=5): 34 · rho≈0.75 (k=7,8): 22 |
| calibration | 20 | rho≈0.25: 6 · rho≈0.5: 8 · rho≈0.75: 6 |
| held_out | 20 | rho≈0.25: 6 · rho≈0.5: 8 · rho≈0.75: 6 |

`held_out` is never touched by any fitting step, ridge-penalty selection, or interaction
selection (Phase 3's explicit prohibition). `calibration` is used only for F1's scalar gain
and for F2/F3 ridge-penalty selection.

**Mask density and total activation displacement**: per Phase 3's explicit instruction, no
renormalization is applied across densities — a rho=0.75 mask suppresses 7-8 of 10
components at the same per-coordinate `alpha`, so it necessarily displaces more total
activation than a rho=0.25 mask. This is deliberate and is not treated as a confound to
correct for; it is reported as-is in `RESULTS.md`.

## Design-matrix checks (Phase 3 required gate), computed on the `fit` pool

| design | rows | cols | rank | full rank? | cond(XᵀX) | max\|corr\| | near-dup col pairs |
|---|---|---|---|---|---|---|---|
| additive (F0-F2 basis) | 78 | 10 | 10 | yes | 22.0 | 0.251 | 0 |
| lifted (F3: 10 main + 45 pair) | 78 | 55 | 55 | **yes** | 4,521 | 0.739 | 0 |

**First attempt failed and was corrected before any response was measured**: an initial
fit-pool size of 40 masks gave the lifted design rank 40/55 (mathematically guaranteed
rank-deficient — 40 rows cannot span 55 columns regardless of which masks are drawn). Per
Phase 3's explicit STOP-and-redesign instruction, the fit-pool counts were increased
(rho=0.25: 12→22, rho=0.5: 16→34, rho=0.75: 12→22) and the check re-run until the lifted
design reached full column rank. This is a pre-data mask-count correction, not a
result-driven change — no response had been measured under either mask set at the time of
the correction, and `calibration`/`held_out` pool composition and counts were not altered.

## Co-occurrence / aliasing check (fit vs. held-out)

All 45 possible pairs among the 10 basis components appear at least once in both `fit` and
`held_out` (`n_pairs_seen_in_both = 45`). Since the two pools share zero mask realizations
(enforced by construction), held-out pair evaluations are never literal repeats of fit-pool
mask realizations even though the same pairs recur — this is the relevant identifiability
guarantee (Phase 3 asks that held-out masks differ enough to detect aliases, not that they
touch novel pairs no fit mask ever saw).

## What is NOT done here

No mask was dropped, reweighted, or reassigned between pools after seeing this design
check — the single correction above (fit-pool size) happened before any response
measurement and is fully disclosed. No response value has been computed as of this file's
commit.
