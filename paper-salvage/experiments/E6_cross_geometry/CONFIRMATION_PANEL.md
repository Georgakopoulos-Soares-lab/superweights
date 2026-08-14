# E6 — independent confirmation panel (frozen before any cross-term value is computed)

**This file was written by inspecting only:** git history up to and including E5's
preregistration lock (`PREREG_dimensionality_gate0.md`, locked 2026-08-13T18:13:24+00:00,
commit `2132e9b`), `docs/superweight_paper.txt` (Yu et al. 2024 Table 2, local copy), and
`results/super_weight_index.json`. **No `f_cross`, exact norm, cross-term sign, or any other
E6/E5 measurement was computed before this panel was frozen.** No candidate below was chosen,
kept, or dropped based on how "promising" it might look under cross-geometry analysis — the
selection rule is provenance-only (was this row on record before E5 existed, from a
non-cross-term source) and mechanical (include everything that qualifies, or exclude with a
stated mechanical reason).

## Selection rule (fixed before inspecting any candidate)

A row is **eligible** if all of the following hold:

1. It is on record in this repository's git history **strictly before** E5's prereg lock
   (2026-08-13T18:13:24+00:00) — verified by `git log` on the source artifact, not by file
   mtime alone (mtimes can be touched by checkouts; git history cannot).
2. Its selection came from a source **other than a cross-term/E5/E6 computation** — either
   Yu et al.'s own published table (NLP) or this repository's own pre-existing empirical
   detection sweep (`results/super_weight_index.json`, an activation-based max-|input|/
   max-|output| detector at `mlp.down_proj`, unrelated to `‖U_k‖_F` or any cross-term
   quantity).
3. It is **not** one of the six E5 primary rows.
4. Its checkpoint's required weight tensors are already locally cached (no new,
   previously-untouched-by-this-repo checkpoint download), so "loadable" is verified by
   existing cache inspection, not by a network fetch decision made now.
5. It carries no open contamination/reproducibility flag in the ledger (excludes the entire
   GENERator PROK family, both checkpoints, and Evo1 — see Mandatory Exclusions).

## NLP candidates

Source: `docs/superweight_paper.txt` (Yu et al. 2024, arXiv:2411.07191, Table 2 — local copy
added to this repository 2026-03-26, restructured without content change 2026-05-05, git
commits `7ba772f` / `6e789a3`, both ~4.5 months before E5's lock). E1/E5 used only the
**first-listed (lowest-layer)** coordinate per model as the primary; the table lists
additional coordinates for several models, which is exactly the independent, pre-existing,
non-cross-term-selected candidate pool this task asks for.

| Model | Checkpoint | Layer | Row *k* | Scalar *i* | Original evidence | Predates E5? | Weights cached? |
|---|---|---|---|---|---|---|---|
| OLMo-7B | `allenai/OLMo-7B-0724-hf` | 2 | 269 | 8275 | Yu et al. Table 2 | yes (commit `6e789a3`, 2026-05-05) | yes (checkpoint cached; new shard for L2 fetched this session, weight-only) |
| OLMo-7B | `allenai/OLMo-7B-0724-hf` | 7 | 269 | 453 | Yu et al. Table 2 | yes (commit `6e789a3`) | yes (new shard for L7 fetched this session, weight-only) |
| OLMo-7B | `allenai/OLMo-7B-0724-hf` | 24 | 269 | 2300 | Yu et al. Table 2 | yes (commit `6e789a3`) | yes (new shard for L24 fetched this session, weight-only) |

Note the recurring output row (269) across all four of OLMo-7B's listed layers (1, 2, 7, 24)
with a different scalar `i` each time — Yu et al.'s own table, not a coincidence introduced
here. Layer 1/row 269 is the E5 primary and is excluded from this panel; layers 2, 7, 24 are
independent candidates by this task's rule.

**Not included, with reasons (mechanical, not performance-based):**

| Model | Why not in this Stage-A panel |
|---|---|
| Llama-7B, Mistral-7B | Table 2 lists only one coordinate each; both are E5 primaries, no independent second row exists to select. |
| Llama-13B, Llama-30B, Llama2-7B, Llama2-13B, Phi-3-mini-4k-instruct | Table 2 lists additional coordinates for these, but none of these checkpoints has ever been downloaded by this repository (verified against the HF cache directory listing). Fetching a brand-new, previously-unused checkpoint (13B–30B parameters) is out of proportion to a bounded weight-only Stage A that already has a sufficient panel without them — excluded for scope, not for any property of the rows themselves. If a future session wants a larger NLP arm, these are the next candidates in line, already provenanced. |

## Genomic candidates

Source: `results/super_weight_index.json`, an activation-based detection sweep
(`hooks/activation_hooks.py`, max |input|/max |output| at each model's `down_proj`), added to
the repository in three commits, all months before E5: DNABERT-2 and GENERator EUK/PROK rows
in `ace31b6` (2026-05-05), NTv3's single row in `d7c57fc` (2026-05-15), Evo1's ten rows in
`3ffa97f` (2026-05-27). Full `git log --follow -p` confirms these exact (layer, row) pairs
have not changed since those commits — nothing was added or edited close to E5's lock.

### DNABERT-2 (9 additional rows; E5 primary was L5/r603)

`zhihan1996/DNABERT-2-117M` @ `7bce263b15377fc15361f52cfab88f8b586abda0` (same pinned revision
E5 already used and validated).

| Layer | Row | Detected col (empirical scalar) | out_max | Predates E5? |
|---|---|---|---|---|
| 3 | 86 | 2056 | 732.00 | yes (`ace31b6`) |
| 3 | 399 | 2056 | 703.01 | yes |
| 3 | 603 | 2056 | 618.13 | yes |
| 3 | 641 | 2056 | 413.52 | yes |
| 5 | 86 | 1062 | 137.55 | yes |
| 6 | 603 | 979 | 204.93 | yes |
| 7 | 603 | 2638 | 284.98 | yes |
| 9 | 264 | 1758 | 652.98 | yes |
| 9 | 294 | 1758 | 620.57 | yes |

These are exactly the 10-row set behind the already-established C-002/C-027/C-028 findings
(one of the ten, L5/r603, is the E5 primary and is excluded here). All ten rows are within
DNABERT-2's 12 layers; weights already validated loadable in E5 via
`AutoModelForMaskedLM` + `uk_frobenius.adapter_dnabert2`.

**Panel-size rule applied:** the task permits a deterministic reduction if the panel is too
large for practical exact computation. It is not — DNABERT-2's `d_ffn` is 3072 and E5 measured
its exact quadratic form in under 0.1s per row. **All 9 eligible rows are kept**, per the
stated preference for the full predetermined set over a convenient subset.

### GENERator EUK (1 additional row; E5 primary was L4/r2371)

`GenerTeam/GENERator-v2-eukaryote-3b-base` @ `7dc01bccce5b65e15141170538afdc2ff09d8dde` (same
revision E5 already used).

| Layer | Row | Detected col | out_max | Predates E5? |
|---|---|---|---|---|
| 4 | 1522 | 2536 | 143880.2 | yes (`ace31b6`) |

Same layer as the E5 primary, so the same already-loaded weight tensors apply; no new fetch
needed.

### NTv3 — no eligible independent candidate

`results/super_weight_index.json`'s `ntv3` entry contains exactly one row (L11/r1472), which
**is** the E5 primary. No second empirically-detected NTv3 row exists anywhere on record
before E5. A structural (non-activation) ranking exists (`results/ntv3_uk_per_layer.json`),
but using a structurally-selected row as the sole non-E5-derived "independent" NTv3 candidate
would blur exactly the detection-vs-structure distinction this task is designed to keep clean
(the whole point of E5/E6 is to interrogate the structural-decomposition machinery itself, so
a structurally-chosen confirmation row is not independent of it in the relevant sense).
**NTv3 contributes no additional candidate to this panel.** This does not block the panel —
the genomic side is not required to have per-model representation, only an aggregate minimum
of two, comfortably met by DNABERT-2 and GENERator EUK alone.

## Mandatory exclusions (checked and confirmed)

| Candidate | Why excluded |
|---|---|
| GENERator PROK 3B, L2/r1927 | C-001 on hold under N-009/N-015, unresolved. Would be the E5 primary anyway if included. |
| GENERator PROK 1.2B, L2/r1397 (`generator_prokaryote_1b`) | Same "prokaryote" family as the contested 3B checkpoint; no independent reproduction audit of this checkpoint exists in this repository, so its clean/contested status cannot be affirmatively established — excluded out of the same caution N-009/N-015 impose on its sibling checkpoint. **Independently, mechanically disqualifying on its own:** its checkpoint (`GenerTeam/GENERator-v2-prokaryote-1.2b-base`) is **not present in the local HF cache** — "current weights that can be unambiguously loaded" fails outright. |
| Evo1, all ten rows (L11) | SW status itself contested (C-009/C-011: structurally concentrated, ΔPPL = 0.0%). Reserved as a possible negative control only under a future explicit instruction, per E5's own exclusion rule, which E6 inherits unchanged. |

## Hard-stop check

- Independently predetermined NLP rows: **3** (OLMo-7B L2/L7/L24, all row 269) — meets the
  ≥2 minimum.
- Independently predetermined clean genomic rows: **10** (9 DNABERT-2 + 1 GENERator EUK) —
  meets the ≥2 minimum, comfortably.

**Both sides clear the minimum. E6 Stage A proceeds — `E6_BLOCKED.md` is not written.**

## Frozen panel for Stage A (13 candidate rows, 0 chosen by cross-term value)

| # | Group | Model | Layer | Row |
|---|---|---|---|---|
| 1 | NLP | OLMo-7B | 2 | 269 |
| 2 | NLP | OLMo-7B | 7 | 269 |
| 3 | NLP | OLMo-7B | 24 | 269 |
| 4 | Genomic | DNABERT-2 | 3 | 86 |
| 5 | Genomic | DNABERT-2 | 3 | 399 |
| 6 | Genomic | DNABERT-2 | 3 | 603 |
| 7 | Genomic | DNABERT-2 | 3 | 641 |
| 8 | Genomic | DNABERT-2 | 5 | 86 |
| 9 | Genomic | DNABERT-2 | 6 | 603 |
| 10 | Genomic | DNABERT-2 | 7 | 603 |
| 11 | Genomic | DNABERT-2 | 9 | 264 |
| 12 | Genomic | DNABERT-2 | 9 | 294 |
| 13 | Genomic | GENERator EUK | 4 | 1522 |

This table is copied verbatim into `docs/prereg/PREREG_cross_geometry_stageA.md` and is fixed
from this point forward — no row is added or dropped after the prereg locks.
