# BASIS_FREEZE.md — E9 Phase 1

Frozen before any E9 held-out response is measured. Selection rules below are applied
without reference to E9 causal-response strength, per Phase 1's requirement.

---

## DNABERT-2 basis — `n_D = 10`

Selection rule: **the already-established 10-row/high-gain ensemble** underlying the
existing pretrained-epistasis and fine-tuned ablation work (C-027/C-028), i.e. the full,
unmodified `results/super_weight_index.json["dnabert2"]["results"]` list, ordered by
`out_max` descending (matches `run_pretrained_epistasis.py:219`'s own convention). No row
was added, dropped, or reordered based on any E9 measurement — none exists yet.

| index `i` | layer | row |
|---|---|---|
| 0 | 5 | 603 |
| 1 | 3 | 86 |
| 2 | 3 | 399 |
| 3 | 9 | **264** |
| 4 | 9 | **294** |
| 5 | 3 | 603 |
| 6 | 3 | 641 |
| 7 | 7 | 603 |
| 8 | 6 | 603 |
| 9 | 5 | 86 |

The known critical pair (indices 3, 4 — L9/r264 and L9/r294) is **inside** this basis, as
required, and receives no special treatment during fitting — only a retrospective
comparison after the primary F0-F3 analysis is frozen (Phase 7, H5).

## GENERator EUK basis — scope-limited, `n_G = 1` primary (+1 real secondary)

Per `DECISIONS.md` D-025 and `PROVENANCE_AND_BASELINES.md`: the pre-existing candidate pool
for GENERator EUK has exactly two entries (layer 4/row 2371 rank 1, layer 4/row 1522 rank
2), both mapping to the same output column. This is below the ~6-12 component target and
cannot be padded to that range without introducing structurally-arbitrary rows never flagged
as high-gain — which Phase 1 explicitly instructs against ("do not force a tomography
comparison").

**Frozen scope**: GENERator EUK receives a **one-dimensional dose-response analysis** on
row 2371 (the primary/dominant candidate), with row 1522 run as an independent real
secondary dose-response curve for consistency only. **No F0-F3 observer-complexity ladder,
no pairwise lifting, no mask-density design is applied to GENERator.** Sections of this
experiment that describe "the observer family ladder" (Phase 5), "designed masks" (Phase 3),
and "pair-lifted design checks" (Phase 3) apply to DNABERT-2 only.

This is not a downgrade decided after seeing weak GENERator results — it is a basis-audit
finding, fixed before any E9 measurement of either model.

---

## What DNABERT-2's basis freeze enables next (Phase 2-4)

`n_D = 10` gives `C(10,2) = 45` pairs for the lifted design — small enough that Phase 3's
"exhaustive enumeration is trivial for small n" clause applies for pairs; mask-subset
sampling at densities `rho in {0.25, 0.5, 0.75}` (`~2-3, 5, 7-8` active components out of
10) is still used for the F0-F2 additive-family design matrix, per Phase 3, rather than only
ever intervening on 1 or 2 components at a time.

## What GENERator's scope limitation means for later phases

- Phase 5 (observer ladder): not run for GENERator — reported as a scope limitation.
- Phase 6 (adequacy decision): not applicable to GENERator — there is no multi-component
  model-complexity choice to adjudicate.
- Phase 8 (GENERator hypotheses): H1-H3 are evaluated as dose-response/steerability
  questions on the 1D curve, not as observer-family adequacy questions.
- Phase 12 (mechanical outcome): Branch A/B/C/D/E language that references "GENERator's
  simplest adequate observer" is understood as "GENERator's dose-response shape," not a
  fitted F0-F3 comparison.
