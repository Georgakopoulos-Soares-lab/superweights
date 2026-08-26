# Part 2 revision work — round 2 audit report

Follow-up to `audit/AUDIT_REPORT.md`, acting on `audit_2_prompt.md`. Written incrementally,
section by section, reported after each. GATED sections wait for explicit go-ahead.

---

## Section 0 — Repo hygiene

### 0a. Inventory of untracked root-level / one-level-down `.md`/`.txt`/`.tex`

**On `"source_of _truth.md"` vs `paper-salvage/docs/MANUSCRIPT_SOURCE_OF_TRUTH.md`**:
diffed — **not duplicates**. The root file (12.3KB, first line `# Claude Code Prompt —
Build the Manuscript Source-of-Truth Package`) is the *prompt* used to generate the
99.5KB doc under `paper-salvage/docs/`. Classified as SCRATCH, alongside the other
`*_prompt.md` files, not as a stale copy of the evidence doc.

| file | size | mtime | classification | reasoning |
|---|---|---|---|---|
| `check_prompt.md` | 16.3KB | 08-25 | SCRATCH | Prior audit's prompt; task complete, output lives in `audit/AUDIT_REPORT.md` |
| `audit_2_prompt.md` | 15.0KB | 08-26 | SCRATCH | This task's own prompt |
| `e10_prompt.md` | 23.3KB | 08-22 | SCRATCH | E10 task instructions; superseded by `PREREG_E10_*` + `DECISIONS.md` |
| `e10_prompt_correction.md` | 6.8KB | 08-22 | **UNCLEAR** | A real protocol correction ("decoder arm" redesign) — grepped `PROTOCOL_CORRECTION_01.md` for its content (0 matches). May carry decision history not captured elsewhere in tracked docs. |
| `e13_prompt.md` | 7.3KB | 08-25 | SCRATCH | E13 task instructions |
| `e_10b_prompt.md` | 10.4KB | 08-22 | SCRATCH | E10b task instructions |
| `e_11_e_12_prompt.md` | 17.7KB | 08-23 | SCRATCH | E11/E12 task instructions |
| `part_prompt.md` | 12.2KB | 08-24 | SCRATCH | Early draft of the E13 census task prompt (superseded by `part13b_prompt.md`) |
| `part13b_prompt.md` | 10.1KB | 08-25 | SCRATCH | Later draft of the same task prompt (superseded by `e13_prompt.md`) |
| `prompt.md` | 2.4KB | 08-23 | SCRATCH | One-off "update figures" instruction |
| `source_of _truth.md` | 12.3KB | 08-23 | SCRATCH | Confirmed via diff: the prompt that built `MANUSCRIPT_SOURCE_OF_TRUTH.md`, not a duplicate of it |
| `INVENTORY.md` | 11.3KB | 08-23 | **UNCLEAR** | Real technical content (function/line-number map of `spectral_lib.py` etc.) written as a one-off orientation note per `e_11_e_12_prompt.md §0`. Useful reference, but a root-level orphan relative to `paper-salvage/docs/`. |
| `manuscript.txt` | 38.6KB | 06-25 | EVIDENCE | The actual manuscript text |
| `previous_main.tex` | 43.2KB | 06-25 | **UNCLEAR** | A prior LaTeX version of the manuscript. Could be a deliberately retained historical baseline or superseded scratch — no signal either way in the file itself. |
| `PART1_STRUCTURAL_MANUSCRIPT_PACKET.md` | 11.1KB | 08-25 | EVIDENCE | Cited extensively by the prior audit (Section 2's Fig 1 verification) |
| `PART2_EVIDENCE_PACKET.md` | 18.5KB | 08-25 | EVIDENCE | Cited extensively by the prior audit (Section 2's whole-cohort verification) |
| `reports/single_neuron_pilot_dnabert2.md` | 29.9KB | 08-01 | EVIDENCE | Referenced by tracked `paper-salvage/docs/NEW_DIRECTION_EVIDENCE_AUDIT.md`; supporting scripts (`scripts/interpretability/neuron_pilot_common.py` etc.) still in repo — cited, live evidence, not an orphan |
| `paper-salvage/docs/MANUSCRIPT_SOURCE_OF_TRUTH.md` | 99.5KB | 08-23 | EVIDENCE | Authoritative claims doc |

**3 UNCLEAR items awaiting author decision**: `e10_prompt_correction.md`, `INVENTORY.md`,
`previous_main.tex`. Not acted on.

**STATUS: stopped here per instruction — awaiting confirmation before 0b (gitignore
patterns) and 0c (commits).**

### 0d. The no-op shuffle in `attention_sink_implicit_bias.json`

**Root cause found — not a simple code bug, a structural tautology.**
`scripts/mechanism/run_attention_sink.py` measures `pos0_real_mean` vs.
`pos0_shuffled_mean` by extracting channel activation at **sequence position 0** for both
the real and dinucleotide-shuffled sequence, on **GENERator, a causal decoder**
(`AutoModelForCausalLM`, confirmed `models/generator_wrapper.py:9-16`). Two independent,
compounding reasons this is guaranteed to produce ratio=1.0 exactly, not approximately:

1. `dinuc_shuffle()` (`run_attention_sink.py:61`, `out = [s[0]]`) **explicitly preserves
   the first character** — standard for Altschul-Erikson dinucleotide shuffles (needed to
   keep the shuffle graph Eulerian) — so position 0's token is *literally identical*
   between "real" and "shuf" by construction.
2. Even if it weren't: under **causal attention masking**, position 0's hidden state at
   any layer can only depend on position 0's own token — it is mathematically blind to
   every token that comes after it. So even a shuffle that changed the first character
   could not move `h[0]`.

**This test can never produce a non-trivial result for a causal-decoder model.** It would
be meaningful on a bidirectional encoder (e.g. DNABERT-2), where position 0/CLS does see
the whole sequence via bidirectional attention.

**Proposed minimal fix (not implemented)**: either (a) drop this implicit-bias sub-block
for GENERator entirely — it cannot be informative as designed — or (b) redefine it to
measure something causally reachable, e.g. how much *incoming attention to position 0 from
later positions* shifts under shuffled content (a variant of the `attention` block already
computed in the same script), rather than position 0's own hidden value.

**Manuscript impact**: none currently — the numbers the manuscript cites (37.9%, 33.0×,
78.8%) come from the separate `attention` block, not `implicit_bias`, and don't depend on
this. Worth fixing/removing before the JSON is shown at review, since as-is the exact 1.0
looks like a bug rather than an explainable non-result.

---

## Section 1 — GENERator c-grid re-analysis

### The narrow question, answered first

**GC fraction is measured at only two points: c=0.0 and c=0.0125.** Every other grid point
(0.025 through 8.0 — 12 more values) has NLL/damage only. Confirmed directly from
`results/E12/raw/gen_records.jsonl` and `dose_response_extended.csv` — note the label there
is `matched_random_direction`, not `random_direction` (that label is used only in the
damage-only artifacts, `damage_matching.csv`'s `grid_json`). **The re-analysis is not
free** — testing whether a large random direction reproduces the low-GC phenotype needs new
generation runs at larger c.

**One additional wrinkle found, not asked about but relevant**: there were **two runs** of
this measurement. `results/E12/run_e12_full.log` (2026-08-23/24) shows an early pass where
the matcher found `scale=0.0, reachable=True` — trivial, since damage(c=0) is defined to
equal the target exactly (c=0 replaces the row with the zero vector = full ablation, same
object as the target). The **current, authoritative** files
(`damage_matching.csv`/`dose_response_extended.csv`, mtime 2026-08-24 07:57) come from a
**later rerun**, `results/E12/run_e12_rerun_directionfix.log` (07:31–07:50), which fixed
something about the random-direction sign/computation and re-matched at `scale=0.0125,
reachable=False` — the number the manuscript currently cites. The c=0.0 GC records
(mean GC 0.3065, seeds 42/43/44) present in the current `dose_response_extended.csv` are
**not stale** — they survive from the earlier pass via checkpointing because c=0 always
produces the zero vector regardless of direction sign, so that one point is invariant to
whatever the fix changed. I did not chase what exactly the "directionfix" corrected — flag
if you want that traced (the log filename is the only description available;
`run_e12_rerun_signfix.log` also exists and I have not looked at it).

### `audit/round2/generator_random_direction_grid.csv` — emitted

14 rows, one per grid point. Baseline NLL (untouched row) = **6.38538052380085**, verified
`STORED` and byte-for-byte identical across all six alpha=1.0/value=1.0 entries in
`damage_evals.jsonl` (row2371 and all five control rows) — this is a fixed reference point,
*not* part of the c-grid itself (the grid never contains an "untouched" state; c=0 means
zero vector, not "no perturbation").

| c | nll | frac. of full-ablation damage | gc_fraction | obtained_via |
|---|---|---|---|---|
| 0.0 | 8.754320 | **1.0000** | 0.30650 | STORED |
| 0.0125 | 8.753589 | 0.9997 | 0.30687 | STORED |
| 0.025 | 8.752900 | 0.9994 | — | NOT FOUND (NLL only) |
| 0.05 | 8.751532 | 0.9988 | — | NOT FOUND (NLL only) |
| 0.1 | 8.748447 | 0.9975 | — | NOT FOUND (NLL only) |
| 0.25 | 8.742925 | 0.9952 | — | NOT FOUND (NLL only) |
| 0.5 | 8.733294 | 0.9911 | — | NOT FOUND (NLL only) |
| 0.75 | 8.722664 | 0.9866 | — | NOT FOUND (NLL only) |
| 1.0 | 8.711739 | 0.9820 | — | NOT FOUND (NLL only) |
| 1.5 | 8.684713 | 0.9706 | — | NOT FOUND (NLL only) |
| 2.0 | 8.663317 | 0.9616 | — | NOT FOUND (NLL only) |
| 3.0 | 8.608203 | 0.9383 | — | NOT FOUND (NLL only) |
| 5.0 | 8.495453 | 0.8907 | — | NOT FOUND (NLL only) |
| 8.0 | 8.375072 | **0.8399** | — | NOT FOUND (NLL only) |

**Substantive finding on the NLL side, RECOMPUTED**: even at the largest tested c (8.0 —
8× the candidate row's own norm, injected as a random direction), damage is still **84% of
the full-ablation level**. The random direction never comes close to neutral, even at 8×
scale — it decays slowly and monotonically, not sharply. This is consistent with (but does
not, on its own, prove) the claim you're trying to establish: a large random perturbation at
this location does substantial, non-trivial damage. **But this is NLL only** — whether the
GC/low-GC phenotype tracks this same slow decay, or does something qualitatively different
(e.g. threshold/step behavior), is exactly the part that's unmeasured.

### Plots — not produced

Per your instruction, plots were conditional on GC existing across the grid. It doesn't
(2/14 points), so no plot was made. If useful even with 2 points, say so and I'll produce
the GC-vs-damage scatter with the row-2371 α-sweep and 5 control rows overlaid, using what
exists.

### Cost estimate for extending GC to a few more c values — **projected, not run**

From `run_e12_rerun_directionfix.log`'s actual timing for the c=0.0125 point (3 seeds, 96
prompts each): **230–270s per (c, seed) combination** (mean ≈ 244s), plus a one-time model
load of **351.6s** if starting a fresh process (amortized across all c values if run in one
session).

**Recommended c values** (to span the damage-fraction range, not just hit the extreme):
**c=8.0** (frac=0.84, the largest already-swept damage point), **c=3.0** (frac=0.938),
**c=1.0** (frac=0.982) — 3 new points, chosen so the GC-vs-damage-fraction relationship can
be read off across a real range rather than interpolated between 2 points 0.0003 apart.

**Projected cost**: 3 c values × 3 seeds × ~244s ≈ **37 minutes GPU time**, plus one model
load (~6 min) ≈ **~43 minutes total**, reusing the exact existing 96-prompt/3-seed
protocol and harness unchanged. **Not run. Stopping here for your go-ahead** per your
instruction and the prompt's own gating rule.

### Control-row result, restated for Methods

Exact per-row numbers (all `STORED`, `damage_evals.jsonl`, baseline 6.38538052380085,
target 8.754319605827332, gap = 2.368939553...):

> Across the full α ∈ [0, 8] sweep, each of the five control rows moved damage by at most
> 0.0006–0.0016 NLL (row 2621: span 0.0016, range [6.3841, 6.3857]; row 456: span 0.0006,
> range [6.3853, 6.3860]; row 102: span 0.0002, range [6.3854, 6.3856]; row 3039: span
> 0.0011, range [6.3853, 6.3864]; row 1126: span 0.0016, range [6.3852, 6.3869]) — three
> orders of magnitude short of the 2.37 NLL gap to row 2371's own ablation damage (8.7543).
> None of the five approaches the target at any α tested; the shortfall does not shrink as α
> increases. These rows are causally insensitive at this location, not merely
> under-damaged by an insufficiently large sweep.

Deliverable: `audit/round2/generator_random_direction_grid.csv`,
`audit/round2/scripts/section1_random_direction_grid.py`.

---

## Section 2 — Tomography condition-level uncertainty

PENDING

## Section 3 — Detector reconciliation, 5 remaining models (GATED)

PENDING

## Section 4 — Top-K-by-norm controls (GATED)

PENDING

## Section 5 — Persist Section 3 re-analysis artifacts

PENDING
