# UNMIGRATED.md — established claims left out of `results/keep/`, with reasons

Phase 0 §0.2 backfill, 2026-08-13. Migration is **established claims only**, and only where
provenance can be recovered **confidently**. Nothing here was reconstructed from prose.

## Migrated (3 batches, 5 claims)

| Batch | Claims | Class |
|---|---|---|
| `E1_nlp_retrospective/` | C-004, C-005 | derived |
| `E2_broadcast_locked/` | C-012, C-013, C-015 | raw |
| `E4_granularity/` | C-003 | derived |

All five were produced in this session's work, so script, checkpoint, seed, dtype, git
commit and environment are all known rather than inferred.

## Not migrated — evidence path is empty in the ledger

These are `established` but `CLAIMS_LEDGER.md` records **no evidence path** for them. The
`results/` tree holds plausible-looking candidates for several, but selecting one would be a
**guess at the mapping**, and this project has already been bitten once by exactly that:
N-009 showed a stored artifact that looked authoritative and does not reproduce.

| Claim | Subject | Why not migrated |
|---|---|---|
| C-002 | DNABERT-2 9/10 SW rows in top 3 | no evidence path recorded; candidate `sw_mechanistic_dnabert2.json` unverified |
| C-007 | EUK ablation +23,026% PPL | no evidence path; several ablation JSONs, mapping unconfirmed |
| C-008 | PROK ablation +25,975% | same |
| C-009 | Evo1 structurally concentrated, ΔPPL 0.0% | no evidence path; note N-001 corroborates the supporting *trace*, not this claim's own artifact |
| C-011 | Structural concentration ⇏ criticality | derived from C-007/8/9; has no artifact of its own and inherits their gap |
| C-019 | EUK ablation-KL vs activation r = +0.437 | no evidence path recorded |
| C-020 | PROK same on write magnitude r = +0.710 | no evidence path recorded |
| C-021 | EUK no shuffle sensitivity; PROK sensitive | no evidence path; `sw_shuffle_controls.json` unverified |
| C-023 | No canonical motif class enriched | no evidence path recorded |
| C-026 | GENERator 35-task bimodality | no evidence path; 35 per-task JSONs, aggregation unconfirmed |
| C-028 | Max single-row DNABERT-2 effect −1.45% | no evidence path; `gue_per_row_ablation.json` unverified |
| C-031 | INT4 SW-exemption below resolution | no evidence path; several quant JSONs, mapping unconfirmed |

**To migrate any of these**, trace it the way N-009 was traced — claim → result file → log →
script → config → checkpoint — and confirm the stored numbers reproduce. Do not shortcut by
filename resemblance.

## Not migrated — excluded by rule

| Item | Reason |
|---|---|
| **C-001** | **on hold** (N-009). Its PROK half does not reproduce from the current checkpoint at its own layer. Explicitly excluded. |
| **C-014 / X-006** | **retired.** Its value was the noise floor of a nondeterministic kernel. Never migrate. |
| C-016, C-032 | `supported`, not `established` — outside this backfill's scope. |
| C-017 | thesis slot, `pending`. |
| C-006, C-025 | `pending` — experiments not run. |
| C-029 | `supported` and carries an open metric issue (PHASE_0 §0.4). |
| C-010, C-027, C-030 | `supported`, not `established`. C-010 additionally needs eager re-verification (Phase 1b, not started). |

## Preserved but explicitly labelled, not migrated

- `results/sw_broadcast_impulse_PRE_D011_fixed_eps.json` — pre-D-011 baseline, superseded.
- `evo1_fixed_eps_SUPERSEDED` key inside the live E2 JSON — void, retained.
- `results/sw_residual_attribution_evo1_fp32.json` — **mislabelled**: written by a script
  that records `"dtype": "float32"` while running bf16 (N-008). Not migrated until relabelled.
