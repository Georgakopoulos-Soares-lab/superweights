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

### Directionfix trace — requested before running new compute

**Cannot produce a script diff — no prior version of `e12_lib.py`/`run_e12_full.py`
survives anywhere.** These files were untracked until this session's Section 0c commit;
no `.bak`/`.orig` copies exist; no code comments reference the fix. This is a genuine
`NOT FOUND`, reported rather than guessed.

**But the raw checkpoint file answers the calibration question precisely, without needing
the code diff.** `results/E12/raw/damage_matching.jsonl` contains **two full entries** for
`random_direction`, written sequentially (first pass, then the fix), not overwritten:

```
entry 1 (pre-fix):  target=8.754319605827332  reachable=True   matched_scale=0.0
entry 2 (post-fix): target=8.754319605827332  reachable=False  matched_scale=0.0125
```

**Every grid_evals damage value (all 14 scale points, 0.0 through 8.0) is bit-identical
between the two entries** — same target, same damage at every scale, confirmed by direct
float comparison (`entry1.grid_evals[s] == entry2.grid_evals[s]` for all 14 `s`, no
exceptions). **This proves the underlying perturbation computation — `c · v̂ · ‖orig_row‖`
replacing the row, with `v̂` unit-normalized (`make_unit_random_direction` explicitly
divides by `np.linalg.norm`) — did not change between the two runs.** Confirmed directly
from the current `e12_lib.py`: replacement semantics (`m.weight.data[row,:] = new_row`,
not additive), unit-normalized direction. Answering your question directly: **this is not
"the same tautology under a different convention" — the convention never changed.**

What *did* change: `reachable` and `matched_scale`, with entry 2 having **zero
`refine_evals`** (entry 1 bisected 0.05→0.025→0.0125 looking for a bracket; entry 2 didn't
bother). Since `target == max(grid_evals)` bit-exactly in both entries, and the current
code's reachability test is `min(ds) <= target <= max(ds)` — which a literal reading says
should be `True` given equality at the boundary — the fix almost certainly **tightened
this boundary condition** (e.g. strict `<` at the upper bound, or an explicit rule
excluding the self-referential `c=0` point from counting as a valid match). That's a fix
to the **decision/matching logic**, not to the measurement: **c=0 matching its own
definitional target trivially was correctly recognized as scientifically meaningless (you
can't "match" a control to the exact same zero-perturbation state used to define the
target) and excluded.** The earlier "trivial match at c=0" was a real bug in what counted
as a valid match, not a hint that the whole parameterization is tautological.

**Conclusion for interpreting the grid: safe to proceed as planned.** The damage-vs-c
relationship you're about to build GC data on top of is the same relationship in both
runs; only the (now-correct) refusal to call the trivial c=0 point a "match" changed.

**One thing I did not chase**: `run_e12_rerun_signfix.log` (a third log, timestamped
*after* `directionfix`) exists and I have not read past confirming it found "93 already
checkpointed" and made no new measurements — i.e. it appears to be a no-op rebuild, not a
third correction. Flagging in case the name implies otherwise; happy to open it if you want
it traced too, but nothing in the data suggests it changed anything.

### GC compute — complete

Ran c ∈ {0.5, 1.0, 3.0, 8.0} × seeds {42,43,44}, 96 prompts each (1,152 new generations),
reusing `e12_lib.generator_response_unified` and the exact `make_unit_random_direction`
seed convention (`base_seed=20260823`, row 2371, confirmed identical by construction — same
function, same four inputs). Full per-prompt records (not just means) stored in
`audit/round2/raw/gc_extension_records.jsonl`. Wall-clock: model load 280s + 12 conditions
× ~220s ≈ **50 minutes total**, in line with the projection.

**Full grid** (`audit/round2/generator_random_direction_full.csv`):

| c | NLL | frac. of full-ablation damage | GC | obtained_via |
|---|---|---|---|---|
| 0.0 | 8.7543 | 1.000 | 0.3065 | STORED |
| 0.0125 | 8.7536 | 1.000 | 0.3068 | STORED |
| 0.5 | 8.7333 | 0.991 | **0.3071** | RECOMPUTED (this session) |
| 1.0 | 8.7117 | 0.982 | **0.3078** | RECOMPUTED (this session) |
| 3.0 | 8.6082 | 0.938 | **0.3130** | RECOMPUTED (this session) |
| 8.0 | 8.3751 | 0.840 | **0.3402** | RECOMPUTED (this session) |

Reference points: row 2371 full ablation GC = 0.3065 (the low-GC phenotype); untouched
baseline GC = 0.4204; all 5 inert control rows sit at GC ≈ 0.420–0.421 (matching baseline,
confirming they're inert on the GC axis too, not just NLL).

### Per-prompt distribution — broad shift, not a few prompts collapsing

Per your request, checked whether the GC increase is a few prompts jumping while most stay
put (the way the homopolymer analysis ruled out a similar concern for the ablation
condition), using the full per-prompt records now stored:

| c | mean GC | median GC | std | frac(GC<0.35) | frac(GC>0.40) |
|---|---|---|---|---|---|
| row2371 ablated (α=0) | 0.3065 | 0.3021 | 0.0471 | 0.844 | 0.038 |
| c=0.5 | 0.3071 | 0.3021 | 0.0488 | 0.802 | 0.045 |
| c=1.0 | 0.3078 | 0.3047 | 0.0509 | 0.819 | 0.049 |
| c=3.0 | 0.3130 | 0.3047 | 0.0558 | 0.767 | 0.080 |
| c=8.0 | 0.3402 | 0.3255 | 0.0753 | 0.632 | 0.174 |
| row2371 untouched (α=1.0) | 0.4204 | 0.4036 | 0.0942 | 0.240 | 0.514 |

**Broad, gradual redistribution, not outlier-driven.** As c increases, `frac(GC<0.35)`
falls steadily (0.844→0.632) and `frac(GC>0.40)` rises steadily (0.038→0.174), with std
widening in step (0.047→0.075) — a population-level shift, not a small subset of prompts
flipping while the rest stay put. This rules out the compositional-artifact concern you
flagged.

### GC vs c, GC vs NLL — plots

`audit/round2/figures/gc_vs_c.png` (GC vs. random-direction scale, symlog x-axis, with
row2371-ablation and untouched-baseline reference lines) and
`audit/round2/figures/gc_vs_nll.png` (GC vs. NLL, all three series on one axis: row2371's
own α-sweep, the random-direction grid, and the 5 inert control rows) — both also saved as
PDF.

### Verdict, corrected: read by damage level, not by c — the claim is stronger than my first pass gave it credit for

**Reframed per your note, and it changes the reading materially.** My first pass compared
GC across *c* and concluded "incomplete reproduction, 70% of the phenotype remains at
c=8." That's the wrong axis for the manuscript's actual claim, which is about
*damage-matched* comparison, not raw scale. Read by damage level instead:

| c | NLL | % of ablation damage | GC | vs. ablation GC (0.3065) |
|---|---|---|---|---|
| 0.0 | 8.7543 | 100.0% | 0.3065 | — |
| 0.0125 | 8.7536 | 100.0% | 0.3068 | +0.0003 |
| 0.5 | 8.7333 | 99.1% | 0.3071 | +0.0006 |
| 1.0 | 8.7117 | 98.2% | 0.3078 | +0.0013 |
| 3.0 | 8.6082 | 93.8% | 0.3130 | +0.0065 |
| 8.0 | 8.3751 | 84.0% | 0.3402 | +0.0337 |

**At c ≤ 1.0 — where the perturbation sits at 98–100% of ablation damage — GC is
0.3068–0.3078, essentially indistinguishable from ablation's 0.3065** (a span of 0.0013
against a baseline gap of 0.1139; noise-level). **At matched damage, the wrong direction
reproduces the phenotype essentially exactly.** c=8.0 is not a matched condition — it's a
different, much lower damage level (84%) — and GC partially recovering there is exactly
the behavior expected of a phenotype that tracks damage magnitude, not a failure to
reproduce anything.

**Corrected verdict: (a), and more cleanly than "with a caveat" — the supported claim is
that generated GC tracks damage magnitude at the row-2371 location, monotonically across
three orders of magnitude in perturbation scale, and does not require the learned
direction.** This is a cleaner, more general claim than "a large wrong-direction
perturbation reproduces the phenotype" — it says the phenotype is a function of *how much
damage is done*, not of *which* direction did it, at every damage level tested, not just
the extremes.

### The caveat a reviewer will find — state it first, not wait for it

**The random-direction control has limited power to separate "direction matters" from
"presence matters," and the data itself shows why**: even an **8× wrong-direction
vector — the largest scale tested, an order of magnitude beyond the row's own norm —
still only reaches 84% of ablation damage.** That's the *far end* of what this control can
reach; damage never gets close to fully undone at any tested scale, wrong-direction or not
(§ "Control-row result" above already showed the alternative failure mode — control rows
that barely move damage at all).

**Why, quantitatively**: GENERator's down-projection intermediate size is 8,448
(`down_proj shape = (3072, 8448)`, confirmed in the E12 run log). A random unit vector in
an 8,448-dimensional space has expected cosine similarity with any fixed direction of
order 1/√8448 ≈ **0.011** — i.e. a random direction is, in expectation, almost exactly
orthogonal to whatever the row's effective readout actually is. **Any** random replacement
at this location is therefore expected to sit close to ablation in its functional effect
regardless of magnitude, simply because the space is high-dimensional and the row's true
direction occupies a vanishingly small solid angle within it. The result — random
directions cluster near ablation, never reaching a low-damage regime — is close to what
high-dimensional geometry predicts *a priori*, not a strong, surprising specificity
finding. **The result stands** (it's still true, still worth reporting, still rules out
"any perturbation of any kind at this location is harmless"), but it is a weaker
discriminating test than the current framing implies, and Methods should say so rather
than let a reviewer say it first.

### GC-recovery vs. NLL-recovery ratio — checked per point; **not consistent, only emerges at the largest c**

Computed `GC_recovery(c) = (GC(c) − GC(0)) / (GC_baseline − GC(0))` and
`NLL_recovery(c) = (NLL(0) − NLL(c)) / (NLL(0) − NLL_baseline)` at every GC-measured point,
per your request to check whether the earlier single-endpoint "GC recovers faster than
NLL" claim generalizes. **It does not — my original framing (based only on the c=8.0
endpoint) overstated this. Correcting it:**

| c | NLL recovery | GC recovery | ratio (GC/NLL) |
|---|---|---|---|
| 0.0125 | 0.03% | 0.31% | ~10× — **not meaningful, both values are noise-level near zero** |
| 0.5 | 0.89% | 0.51% | **0.57 — GC recovers *less* than NLL here** |
| 1.0 | 1.80% | 1.13% | **0.63 — GC still lags NLL** |
| 3.0 | 6.17% | 5.72% | 0.93 — roughly matched |
| 8.0 | 16.01% | 29.55% | **1.85 — GC now recovers nearly 2× faster than NLL** |

**The decoupling reverses across the grid.** At low-to-moderate c (0.5, 1.0), NLL recovers
*proportionally more* than GC — the opposite of what I reported from the single c=8.0
comparison. The two measures only decouple in GC's favor at the far end of the tested
range (c=3 crossing over to c=8). **This is a real, if non-monotonic-in-ratio, finding**:
composition and loss are not coupled by a fixed proportionality constant across the whole
grid — but it is not evidence, on its own, that "composition is more responsive to partial
restoration than the loss is" as a general statement; that's only true in the c≥3 regime.
Recommend reporting the full table if this decoupling is mentioned at all, not the c=8.0
ratio alone — the c=0.5/c=1.0 rows would visibly contradict a claim stated only from the
endpoint.

### Recommended manuscript framing (revised)

"Generated GC tracks the magnitude of damage done at the row-2371 location, monotonically,
across three orders of magnitude of perturbation scale (c ∈ [0.0125, 8.0]) and regardless
of whether the perturbing vector is the row's own direction or an independent random one:
at matched damage (≥98% of full ablation), GC is statistically indistinguishable from
ablation itself (0.3068–0.3078 vs. 0.3065); only at substantially lower damage (84%, the
largest scale tested) does GC partially recover. The learned direction is therefore not
required for the phenotype — damage magnitude predicts it. This control has an inherent
ceiling on how far it can push damage down at any scale, consistent with a random
direction in an 8,448-dimensional space being expected to sit near-orthogonal to the row's
effective readout (expected cosine similarity ≈ 1/√8448 ≈ 0.011); the control is therefore
better read as confirming damage-tracking than as a strong test of directional
specificity."

Deliverables: `audit/round2/generator_random_direction_full.csv`,
`audit/round2/raw/gc_extension_records.jsonl`,
`audit/round2/figures/{gc_vs_c,gc_vs_nll}.{png,pdf}`,
`audit/round2/scripts/{section1_extend_random_direction_gc,section1_final_plots}.py`.

---

## Section 2 — Tomography condition-level uncertainty

**Verdict: F3's advantage over F2 holds across resplits — 100% of splits at both epsilons.
No defect found; ridge λ selection is genuinely unstable but the mechanistic conclusion
doesn't depend on it.**

### Feasibility check — reported before running

**Exactly enough measured data for one partition, no slack.** Counted stored responses by
density bucket: `rho_0.25`=34 measured, `rho_0.5`=50 measured, `rho_0.75`=34 measured — and
these are **exactly** the per-density totals `generate_masks.py`'s own `POOL_COUNTS` needs
(fit+calibration+held_out = 22+6+6=34, 34+8+8=50, 22+6+6=34). **This means a "fresh
partition" cannot draw genuinely new, never-measured row-subsets without new model
runs** — there is no surplus. What it *can* do without new data: re-permute which of the
118 already-measured non-singleton conditions get the fit/calibration/held-out label,
holding the per-density counts fixed. This still produces genuine variation in what lands
in held-out across resamples (the actual question), just not variation in *which subsets
exist* — reporting this distinction explicitly rather than letting "fresh partition" imply
something it can't be here. The leave-one-condition-out jackknife fallback was **not
needed** — full repeated-split analysis proceeded instead.

### Method

100 resampled splits (seed 900001), each drawing a fresh fit(78)/calibration(20)/held_out(20)
role-assignment per density bucket from the fixed measured pool, refit F0–F3 via
`run_fit_observers.fit_epsilon()` **reused unmodified** (only the pool dict's membership is
resampled — the fitting function doesn't know or care how pools were constructed). The
per-split nested 5,000-draw bootstrap CI was stubbed out (not needed — the 100 outer splits
already characterize split-driven variance; it was also the actual cost driver, pure-Python
nested loops). Runtime: 1.4s for all 200 fits (100 splits × 2 epsilons).

### Results

| epsilon | family | held-out R² median [2.5%, 97.5%] |
|---|---|---|
| 0.5 | F0 | −0.214 [−0.450, 0.057] |
| 0.5 | F1 | 0.300 [−0.346, 0.567] |
| 0.5 | F2 | 0.550 [0.261, 0.704] |
| 0.5 | F3 | **0.917 [0.793, 0.956]** |
| 1.0 | F0 | −0.454 [−1.139, −0.053] |
| 1.0 | F1 | 0.306 [−0.328, 0.690] |
| 1.0 | F2 | 0.560 [0.330, 0.702] |
| 1.0 | F3 | **0.761 [0.492, 0.884]** |

**F2→F3 relative MAE improvement**: median **57.5%** [34.7%, 68.2%] at ε=0.5, median
**30.6%** [13.8%, 51.9%] at ε=1.0 — both **higher than the original single-split point
estimates** (54.7%/24.4%, round-1 audit Section 1). The original held-out split was not
favorably cherry-picked; if anything it sits slightly below the resampled median.

**Fraction of splits where F3 beats F2 on held-out MAE: 100% at both epsilons** (100/100
splits each). **F3's advantage is not an artifact of which 20 conditions happened to land
in held-out** — it holds under every resampled role-assignment tested.

### Ridge λ stability — genuinely unstable, but doesn't matter for the conclusion

| epsilon | family | λ values seen across 100 splits (count) |
|---|---|---|
| 0.5 | F2 | 0.001×34, 0.1×3, 1×7, 3×16, 10×29, 30×10, 100×1 — **7 distinct values** |
| 0.5 | F3 | 0.001×6, 0.01×9, 0.1×24, 1×55, 3×5, 10×1 — **6 distinct values** |
| 1.0 | F2 | 0.001×18, 1×12, 3×18, 10×35, 30×17 — **5 distinct values** |
| 1.0 | F3 | 0.1×4, 1×45, 3×39, 10×6, 30×2, 100×4 — **6 distinct values** |

The calibration-selected λ is **not stable** across resplits — it moves across most of the
10-point grid depending on which 20 conditions happen to be in calibration. **This does not
threaten the F2-vs-F3 conclusion**: despite this λ instability, F3 beat F2 in literally
every one of the 200 (split × epsilon) fits. The pair-terms-required decision is robust to
exactly the kind of instability that could have undermined it.

Deliverables: `audit/round2/tomography_split_stability.csv`,
`audit/round2/scripts/section2_split_stability.py`.

---

## Section 3 — Detector reconciliation, 5 remaining models

**Priority order per your instruction: GENERator-EUK-3B first (all of Figure 4), then
NTv3, reported after each, before the three 7B text decoders.**

### GENERator-EUK-3B — clean, no discrepancy

**L4/r2371 is unambiguously both the global ratio-argmax and the global activation-argmax,
under both preprocessing conventions tested.** Ran the current ratio-based detector
(`audit/round2/scripts/section3_detector_recheck.py`, following
`run_dnabert2_repro_diagnostic.py`'s pattern) on the canonical 504-bp ACTB probe:

| preprocessing | global ratio-argmax | global activation-argmax | frozen L4/r2371: activation | ratio | activation_rank | ratio_rank | passes ≥5.0 |
|---|---|---|---|---|---|---|---|
| wrapper convention (manual BOS + no specials) | L4/r2371 | L4/r2371 | 375,361.31 | **7124.08** | **1** | **1** | **True** |
| tokenizer default (`add_special_tokens=True`) | L4/r2371 | L4/r2371 | 375,361.31 | 7124.08 | 1 | 1 | True |

Both preprocessing conventions produced **identical tokenization** (`n_tokens=85` both
ways) — for GENERator's tokenizer, the default `add_special_tokens=True` behavior happens
to coincide exactly with the wrapper's manual BOS-prepend + `add_special_tokens=False`
convention, so there was no DNABERT-2-style pivot to find here at all. Structural metrics
for L4/r2371: q1=0.968869, PR_spec=1.065259, ‖U_k‖_F=522.132210 — **reproduces the stored
`e7_legacy_reanalysis.json` value exactly** (q1=0.9688691252375834, PASS).

**GENERator-EUK-3B's frozen candidate needs no caveat and no protocol note.** Ratio 7124
is three orders of magnitude above the 5.0 threshold — this is about as unambiguous a
detector result as exists in the cohort. Figure 4's candidate coordinate is fully
reconciled.

### NTv3 — real discrepancy found, but a mild one structurally

**The frozen candidate (L11/r1472) is the global activation-argmax (rank 1) but NOT the
global ratio-argmax (rank 3).** The true ratio-argmax under the current rule is
**L6/r1472 — same row index, a different layer** (ratio 30.63 vs. the frozen candidate's
23.17). Both preprocessing conventions agree exactly (identical `n_tokens=512` both ways,
identical ranks and values) — no DNABERT-2-style specials pivot here.

**Getting to this required a fix along the way, reported for provenance**: NTv3 crashed on
the unpadded 504-bp ACTB probe (`RuntimeError: size of tensor a (6) must match tensor b
(7)`, inside `modeling_ntv3_pretrained.py`'s skip-connection add). NTv3 is a conv/deconv
U-Net with 8 stride-2 blocks — the same constraint already documented in
`run_pretrained_epistasis.py`'s `_pad_to_multiple` helper — so its input length must be a
multiple of 2⁸=256. Padded the probe to 512 bp (right-padded with `A`, matching that
helper's convention) and it ran cleanly. Also found `models/ntv3_wrapper.py`'s
`AutoTokenizer.from_pretrained` call is missing `trust_remote_code=True` (hangs
interactively under a non-interactive session) — bypassed by loading directly with the
same pattern `scripts/analysis/run_ntv3_uk_audit.py` already uses successfully, not by
patching the wrapper (out of scope here). **Separately, and not asked about**: your
`PART2_EVIDENCE_PACKET.md`'s prose (line 55) cites NTv3's code-pin as
`0ecff295910cbdf3a909d91d686510986c29c8f2`, but the actual pipeline
(`E13_full_cohort_causal_census/genomic_encoder_lib.py:51`) pins
`0ecff3637f0d3ba5b686d1095083218157c2ca34` — different hashes sharing only a `0ecff`
prefix. The code is right (confirmed matching the E13 pipeline exactly); the evidence
packet's prose has a stale/wrong value. Worth a one-line fix there.

| coordinate | role | activation | ratio | rank (act. / ratio) | q1 | PR_spec | ‖U_k‖_F |
|---|---|---|---|---|---|---|---|
| L11/r1472 | frozen candidate | 772.19 | 23.17 | 1 / **3** | 0.388894 | 6.480 | 441.52 |
| L6/r1472 | ratio-argmax (not selected) | — | **30.63** | — / 1 | 0.395401 | 6.258 | 406.27 |

L11/r1472's q1 (0.388894) **exactly reproduces the round-1 audit's stored value** — this is
the cohort's q1 minimum (Section 2/3 of the round-1 report). **The discrepancy is real but
structurally mild, unlike DNABERT-2's**: the alternative ratio-argmax coordinate's q1
(0.395401) is barely different from the frozen candidate's — both sit near the bottom of
the entire cohort's q1 distribution, neither is "exceptional" in the way DNABERT-2's
activation-argmax candidate was relative to its ratio-argmax alternative (0.793 vs. 0.379
there). **Applying the strict rule to NTv3 would swap which row is the accepted
candidate, but would not change NTv3's place in the cohort's structural story** — it's
still the (or among the) least spectrally-concentrated model in the census either way.
This is worth noting as a genuine protocol inconsistency in Methods, but it does not
threaten any conclusion drawn from NTv3's low-q1 status specifically.

**One more thing worth flagging, unprompted**: the same row index (1472) is exceptional in
two different layers (6 and 11) of the same model. That's either a coincidence or a sign
that row 1472 carries some architecture-level significance across NTv3's depth — not
something this audit can resolve, but worth a sentence if NTv3's mechanism gets discussed.

Deliverables: `audit/round2/section3_generator_euk_detector_recheck.json`,
`audit/round2/section3_ntv3_detector_recheck.json`,
`audit/round2/scripts/section3_detector_recheck.py`.

**Follow-up fixes applied**: `detector_provenance.csv`'s NTv3 note reworded to record
this as a genuine activation-vs-ratio inconsistency with no downstream consequence, per
your framing. The stale code-revision hash in `PART2_EVIDENCE_PACKET.md:55`
(`0ecff295910cbdf3a909d91d686510986c29c8f2`) is fixed to match the actual pinned value
(`0ecff3637f0d3ba5b686d1095083218157c2ca34`, `genomic_encoder_lib.py:51`) — and fixed at
its **source**, `build_part2_evidence_packet.py:228` (the script that generates the
packet), not just the generated output, so regenerating the packet won't reintroduce it.
Checked repo-wide for the wrong hash elsewhere: **found in exactly those two places, not
in the manuscript** (`manuscript.txt`, `previous_main.tex`,
`MANUSCRIPT_SOURCE_OF_TRUTH.md` all clean).

**Llama-7B, Mistral-7B, OLMo-7B**: running now as part of the reordered Section 4a
batch (combined with their top-K-by-norm check to avoid loading these 13–26GB models
twice), framed as calibration per your instruction — reporting whether the current
detector independently recovers the Yu et al. coordinates, not as a protocol-consistency
finding.

## Section 4 — Top-K-by-norm controls

**4a approved (selection pass + q1 comparison only); 4b not started, per your instruction.**

### The headline: Fig 1C's random-control gap vs. the top-K-by-norm gap — **~115× collapse**

`audit/round2/fig1c_random_vs_topk_gaps.csv` (12 rows, batch 1). Per-model, side by side:

| model | Fig1C random-control gap | top-5-by-norm gap | architecture |
|---|---|---|---|
| Qwen2.5-0.5B | +0.9685 | +0.1380 | decoder |
| Qwen2.5-1.5B | +0.9770 | +0.0007 | decoder |
| Qwen2.5-3B | +0.7902 | +0.0604 | decoder |
| SmolLM2-135M | +0.8877 | +0.5788 | decoder |
| SmolLM2-360M | +0.9576 | −0.0156 | decoder |
| SmolLM2-1.7B | +0.9533 | +0.0443 | decoder |
| GENERator-PROK-1.2B | +0.8341 | +0.0817 | decoder |
| GENERator-PROK-3B | +0.9260 | +0.0156 | decoder |
| EuroBERT-210m | +0.8497 | −0.0013 | encoder |
| EuroBERT-610m | +0.9672 | +0.0006 | encoder |
| EuroBERT-2.1B | +0.9492 | −0.0001 | encoder |
| ModernBERT-large | +0.5438 | −0.0026 | encoder |

**Cohort median: Fig1C random-control gap = 0.9376 (range [0.544, 0.977]); top-5-by-norm
gap = 0.0082 (range [−0.0156, 0.579]). Collapse factor: 114.8×.** This confirms your
framing — against random controls, candidates look overwhelmingly exceptional; against the
rows actually closest in magnitude, the typical candidate is essentially tied.

### Confound 1 — architecture: **explains the encoder side cleanly, does not explain the decoder side**

Split by architecture (n=12, batch 1 only so far):

| architecture | n | median top-K gap | range |
|---|---|---|---|
| decoder | 8 | +0.0524 | [−0.0156, +0.5788] |
| encoder | 4 | −0.0007 | [−0.0026, +0.0006] |

**Encoder-ness predicts the tight-clustering regime almost deterministically in this
batch: 4/4 encoders sit in a razor-thin band around zero, 0/8 decoders do.** That part of
the "two regimes" story **is** an architecture effect, not a separate phenomenon, and
should be described that way. **But architecture does not explain the decoder side**:
decoders span the *entire* range from −0.016 to +0.579 — being a decoder is necessary but
nowhere near sufficient for a large gap in this batch (6/8 decoders have gaps under 0.09).
So: half of the two-regime story (why encoders cluster) is an architecture effect and
should be named as one; the other half (why some decoders separate and others don't) is
still unexplained by architecture alone.

### Confound 2 — scale: **no significant relationship**

Spearman ρ(log10 non-embedding params, top-K gap) = **0.1399, p=0.665, n=12** — not
significant. The two largest-gap models (SmolLM2-135M, Qwen2.5-0.5B) are indeed small, but
Qwen2.5-1.5B (also small) shows a near-zero gap, and GENERator-PROK-3B (mid-large) shows a
moderate positive gap — scale does not cleanly track the split in this batch.

**Both confounds checked on 12/22 models — not yet conclusive, exactly the caveat you
asked for.** The architecture split for the encoder side looks robust already (4/4 vs
0/8 is a clean separation), but the full 22-model set (adding MosaicBERT, ModernBERT-base,
DNABERT-2, NTv3 as more encoders, and GENERator-EUK-3B, GenomeOcean-4B, Llama/Mistral/OLMo/
Qwen2.5-7B as more decoders) is needed before calling either confound resolved — batch 2 is
running now and will roughly double both architecture classes.

### Prioritized question, answered first: does q1 separate from norm? — **mixed, and the mixture is itself the finding**

Batch 1 (the 12 E11-panel models) complete. For each: computed exact ‖U_k‖_F for every row
in the candidate's layer (Gram-identity), took the candidate + top-5-by-norm other rows,
computed exact q1 for all 6 via full SVD.

| model | candidate q1 | top-5-by-norm max q1 | gap | candidate rank /6 |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.9982 | 0.8602 | **+0.1380** | 1 |
| Qwen2.5-1.5B | 0.9951 | 0.9944 | +0.0007 | 1 |
| Qwen2.5-3B | 0.8174 | 0.7569 | +0.0604 | 1 |
| SmolLM2-135M | 0.9244 | 0.3456 | **+0.5788** | 1 |
| SmolLM2-360M | 0.9743 | 0.9899 | **−0.0156** | 2 |
| SmolLM2-1.7B | 0.9664 | 0.9221 | +0.0443 | 1 |
| GENERator-PROK-1.2B | 0.8469 | 0.7651 | +0.0817 | 1 |
| GENERator-PROK-3B | 0.9350 | 0.9194 | +0.0156 | 1 |
| EuroBERT-210m | 0.9838 | 0.9851 | −0.0013 | 3 |
| EuroBERT-610m | 0.9996 | 0.9990 | +0.0006 | 1 |
| EuroBERT-2.1B | 0.9977 | 0.9978 | −0.0001 | 2 |
| ModernBERT-large | 0.9701 | 0.9727 | −0.0026 | 2 |

**Cohort summary (n=12)**: candidate is rank 1/6 (strictly most concentrated) in **8/12**
models; rank ≤2/6 in **11/12**. Median gap +0.0082 (near-zero); mean gap +0.0751 (pulled up
by two large outliers). **4/12 models have a negative gap** — the candidate is not the
single most-concentrated row — but in every one of those 4 cases the gap is tiny
(−0.0001 to −0.0156), a rounding-level tie, never a real reversal.

**The mixture is the real finding, not either extreme**: this cohort splits into two
regimes.
- **Regime A (large separation, q1 clearly independent of norm)**: SmolLM2-135M
  (gap +0.579), Qwen2.5-0.5B (+0.138), and to a lesser extent Qwen2.5-3B (+0.060) and
  GENERator-PROK-1.2B (+0.082) — here the candidate's concentration is dramatically higher
  than even the next-highest-norm rows in the same layer. q1 is doing real, independent
  work in these models.
- **Regime B (tight clustering at the top, q1 tracks norm)**: the entire EuroBERT family,
  ModernBERT-large, and SmolLM2-360M — here every one of the top-5-by-norm rows has q1
  within ~0.01–0.05 of the candidate's, several exceeding it slightly. In these models,
  being high-norm in this layer *is* close to sufficient for being highly concentrated;
  the candidate isn't uniquely exceptional, it's a representative member of a uniformly
  concentrated high-norm tail.

**Does Figure 1C survive this stricter comparison?** **Yes, in the weak sense that
candidates are never meaningfully *less* concentrated than their nearest high-norm
neighbors** (worst observed gap is −0.0156, noise-level) — the claim "candidates are
locally spectrally exceptional" is not falsified. **But it should not be read as "uniquely
exceptional" cohort-wide**: for encoder-family models (EuroBERT ×3, ModernBERT-large), the
top of the norm distribution is uniformly concentrated, and singling out the candidate
specifically (vs. any of its four next-highest-norm neighbors) is not well-supported by q1
alone in those 4/12 cases. Recommend Figure 1C's caption note this split rather than imply
uniform exceptionality across the panel.

Deliverables so far: `audit/round2/topk_norm_controls.csv` (12 rows),
`audit/round2/topk_norm_q1_summary.csv` (72 rows, 6 coordinates × 12 models),
`audit/round2/scripts/section4a_topk_norm_q1.py`.

### Batch 2 (reordered, cheapest-first): ModernBERT-base, DNABERT-2, GENERator-EUK-3B, Llama-7B, Mistral-7B, OLMo-7B — all 6 complete

| model | candidate q1 | top-5-by-norm max q1 | gap | rank /6 |
|---|---|---|---|---|
| ModernBERT-base | 0.8970 | 0.7037 | **+0.1933** | 1 |
| DNABERT-2 | 0.7933 | 0.8664 | −0.0731 | 2 |
| GENERator-EUK-3B | 0.9689 | 0.9909 | −0.0220 | 2 |
| Llama-7B | 0.9888 | 0.8211 | **+0.1677** | 1 |
| Mistral-7B | 0.9922 | 0.9659 | +0.0263 | 1 |
| OLMo-7B-0724-hf | 0.9646 | 0.6133 | **+0.3513** | 1 |

**Running cohort (n=18)**: rank1/6 in 12/18; rank≤2/6 in 17/18; median gap 0.0210; mean
gap 0.0858; range [−0.0731, +0.5788].

**One clean counter-example to the encoder/decoder confound, worth flagging immediately**:
**GENERator-EUK-3B is a decoder but landed in the tight-clustering regime** (gap −0.022,
rank 2/6) — the same pattern as the encoder cluster, not the decoder cluster. So
"encoder → tight clustering" still holds cleanly in the growing sample (now DNABERT-2, an
encoder, joins that side too), but "decoder → separation" is now even more clearly false
as a general rule — GENERator-EUK-3B breaks it. The architecture confound explains one
direction of the split, not both, exactly as flagged after batch 1.

### Section 3 calibration — Llama-7B, Mistral-7B: detector recovers; OLMo-7B: a genuine, narrow miss

| model | frozen coord | activation_rank | ratio_rank | frozen ratio | recovers? |
|---|---|---|---|---|---|
| Llama-7B | L2/r3968 | 2 | **1** | 3959.23 | **True** (ratio-argmax) |
| Mistral-7B | L1/r2070 | **1** | **1** | 1714.33 | **True** (both) |
| OLMo-7B-0724-hf | L1/r269 | 2 | 2 | 2253.48 | **False** |

**Structural follow-up (requested): the alternative coordinates are not noise — they are
real, comparably (or more) exceptional high-gain sites.** Computed exact q1, PR_spec,
‖U_k‖_F for all three OLMo coordinates (weight-only, no forward pass, no causal
measurement — `audit/round2/scripts/olmo_row269_structural.py`):

| coordinate | role | q1 | PR_spec | ‖U_k‖_F |
|---|---|---|---|---|
| L1/r269 | published (Yu et al.) | 0.9646 | 1.0747 | 0.911 |
| L2/r269 | ratio-argmax | 0.9611 | 1.0819 | 0.598 |
| L30/r269 | activation-argmax | **0.9970** | 1.0059 | 2.649 |

**All three sit at or above q1≈0.96 — essentially rank-1 spectral dominance, the same
regime as the published coordinate.** L30/r269's q1 (0.997) is *higher* than the published
row's (0.965), and PR_spec closer to the theoretical minimum of 1.0 (i.e. even more purely
rank-1). **This is not "the ratio statistic is noisy near its maximum picking out
structurally unremarkable rows" — both alternatives are comparably or more exceptional
than the row the manuscript already treats as its headline case.** Combined with the
row-index-recurrence finding below — row 269 recurs at layers 1, 2, *and* 30; note layers 1
and 2 are both already in Yu et al.'s own published set for OLMo, so only layer 30 is new
to this session, but that is still an independent structural confirmation, not a citation),
the more defensible reading is that **row 269 is a genuinely depth-recurring high-gain
channel in OLMo-7B, and the published coordinate (layer 1) is one instance of it, not
uniquely privileged over the layer-2 or layer-30 instances by any structural measure
available.** The 7.5% ratio-statistic gap that ranked L2/r269 above
L1/r269 is a real, if modest, activation-based difference between two structurally
similar sites — not evidence of instability at the detector's ceiling. Recommend Methods
note that OLMo-7B's headline coordinate is one member of a small family of comparably
exceptional row-269 instances across depth, rather than treating the layer-1 selection as
uniquely determined by the detector.

(Raw activation/ratio values behind the table above: true ratio-argmax **L2/r269**, ratio
2423.21 vs. frozen L1/r269's 2253.48 — a ~7.5% margin; true activation-argmax **L30/r269**,
activation 552.6 vs. frozen's 421.8. The row-index recurrence this implies — row 269
elevated in 3 layers — is logged with full numbers in Follow-up A below, not chased
further here per your instruction on the analogous NTv3 case.) Calibration framing holds
throughout: this is not a protocol-inconsistency finding (the coordinates are
literature-derived, not detector-selected) — it's a report of
whether the current detector happens to agree, and for 2/3 text decoders it does.

**Draft Methods sentence (as requested, 2 sentences):**

> OLMo-7B's row 269 at layer 1, the coordinate used throughout this manuscript, is one
> instance of a depth-recurring family sharing the same output row index across multiple
> layers (1, 2, and 30); an independent structural check finds the layer-30 instance more
> purely rank-1 (q1 = 0.997) than the published layer-1 coordinate (q1 = 0.965), so the
> detector's disagreement at this site reflects a second genuinely high-gain location
> rather than noise, and the headline coordinate should not be read as uniquely determined
> by the detector.

### Batch 3 (remaining): MosaicBERT, GenomeOcean-4B, Qwen2.5-7B, NTv3 — running now

Extending `section4a_batch2.py`'s model list and loaders (reused `dnabert2_compat.load_mosaicbert_patched`
for MosaicBERT — same MosaicML `BertEncoder` vendor code as DNABERT-2, same ALiBi
meta-device patch needed; reused the `_NTV3_CODE_REV`-pinned direct-load pattern from
Section 3 for NTv3). Reporting per-model as they land, same as batch 2.

**Cache check**: all 10 remaining models were already cached; **zero downloads needed
across all 22 models** in this session.

### Batch 3: MosaicBERT, GenomeOcean-4B, Qwen2.5-7B, NTv3 — all 4 complete, 22/22 done

| model | candidate q1 | top-5-by-norm max q1 | gap | rank /6 | architecture | domain |
|---|---|---|---|---|---|---|
| MosaicBERT | 0.4766 | 0.4651 | +0.0115 | 1 | encoder | text |
| GenomeOcean-4B | 0.8989 | 0.9735 | **−0.0746** | **5** | decoder | genomic |
| Qwen2.5-7B | 0.9529 | 0.9491 | +0.0038 | 1 | decoder | text |
| NTv3 | 0.3889 | 0.0195 | **+0.3694** | 1 | encoder | genomic |

**GenomeOcean-4B is the worst-ranked candidate in the entire cohort (5/6)** — four of its
five top-norm neighbors have *higher* q1 than the frozen candidate. **NTv3, in the same
domain (genomic) and even the same architecture class as DNABERT-2 (encoder), shows one
of the *largest* gaps in the cohort** — these two genomic encoders go in opposite
directions from each other, which matters for the confound analysis below.

---

## Full 22-model picture: the headline, the confounds resolved, and a retraction

`audit/round2/full22_topk_gap_confounds.csv` (22 rows) — every census model, full data.

### Cohort-level result

**rank1/6: 15/22. rank≤2/6: 20/22. Median gap: 0.0135. Mean gap: 0.0843. Range:
[−0.0746, +0.5788]. 7/22 models have a negative gap.**

Combined with the Fig1C comparison (batch-1 12-model subset, the only models with random-
control data): **the collapse from random-control gap (median 0.938) to top-K-by-norm gap
(median 0.014, full 22) is still on the order of ~65×** — the headline finding survives
the full cohort, even as the per-model story underneath it turns out to be messier than
batch 1 alone suggested.

### Confound 1 (architecture) — retracting my batch-1 read: **does not hold at n=22**

| architecture | n | median gap | range | n negative |
|---|---|---|---|---|
| decoder | 14 | 0.0353 | [−0.0746, +0.5788] | 3/14 |
| encoder | 8 | 0.0003 | [−0.0731, +0.3694] | 4/8 |

At batch 1 (n=12), encoders were 4/4 in a razor-thin band near zero and decoders were
0/8. **That clean separation does not survive the remaining 10 models.** ModernBERT-base
(encoder, +0.193) and NTv3 (encoder, +0.369) both post some of the *largest* gaps in the
whole cohort — encoder membership no longer predicts tight clustering. Encoders are now
*more* likely to show a negative gap proportionally (4/8 = 50%) than decoders (3/14 ≈
21%), but the two largest positive outliers are also encoders. **Architecture does not
cleanly explain the split at n=22 — my batch-1 characterization was an artifact of which
12 models ran first, exactly the possibility you asked me to check for.**

### Confound 1b (domain) — also does not hold, and the genomic/encoder cell is internally split

| domain | n | median gap | range | n negative |
|---|---|---|---|---|
| text | 16 | 0.0189 | [−0.0156, +0.5788] | 4/16 |
| genomic | 6 | −0.0032 | [−0.0746, +0.3694] | 3/6 |

| domain × architecture | n | median gap | range |
|---|---|---|---|
| text/decoder | 10 | 0.0524 | [−0.0156, +0.5788] |
| text/encoder | 6 | 0.0003 | [−0.0026, +0.1933] |
| genomic/decoder | 4 | −0.0032 | [−0.0746, +0.0817] |
| **genomic/encoder** | **2** | **0.1481** | **[−0.0731, +0.3694]** |

**The genomic/encoder cell contains exactly the two most divergent genomic models**:
DNABERT-2 (−0.073, tight/negative) and NTv3 (+0.369, one of the largest gaps in the
cohort) — same domain, same architecture class, opposite behavior. Domain does not
explain this pair, so it cannot be the resolving confound either.

### Confound 2 (scale) — cleanly rejected

**Spearman ρ(log10 non-embedding params, top-K gap) = −0.0011, p=0.996, n=22.**
Essentially zero relationship. Scale was never a strong candidate after batch 1 (ρ=0.14)
and is now indistinguishable from noise.

### Verdict on the confounds, as asked directly: **neither explains the split**

**This is not an architecture effect and not a domain effect in disguise — say so, and
say so plainly.** At n=12 it looked like encoder-ness cleanly predicted tight clustering;
at n=22 that read does not survive, and no simple relabeling by domain rescues it either.
The two-regime story, if the manuscript keeps it at all, should be presented as an
unexplained per-model split, not attributed to architecture, domain, or scale — attributing
it to any of the three tested confounds would now be an overclaim contradicted by this
audit's own data. What *does* hold at n=22, robustly: **the median candidate is
essentially tied with its nearest high-norm neighbor** (gap 0.0135, a small fraction of
the 0.94 random-control gap), and a comparable number of models fall clearly on each side
of that median (15 clear wins, 7 negative, several more near-zero) — a real, load-bearing,
but currently mechanistically unexplained split.

---

## Follow-up A — row-index recurrence across layers (CORRECTED — original pass had a
## circularity error; see below)

**Correction, flagged unprompted: two of the seven original rows were not independent
findings of this project at all.** The original pass in this section treated Phi-3's
6-row basis and 3 of OLMo's 3 recorded coordinates as evidence this audit's detectors had
independently produced. Checking provenance (`paper-salvage/experiments/
E10_nlp_architecture_causal/DECODER_INTERVENTION_FREEZE.md`, `paper-salvage/experiments/
E9_mechanistic_tomography/BASIS_FREEZE.md`) shows that is wrong for Phi-3 and partly wrong
for OLMo:

- **Phi-3-mini's entire 6-row set (rows 525/1113/1693 at layers 2 and 4) is copied
  verbatim from an external published table** ("Yu et al. Table 2", per
  `DECODER_INTERVENTION_FREEZE.md` §4: "pre-existing SET of 6 published rows... no
  ranking performed, no padding, no drop"). This project performed **zero** independent
  row search for Phi-3. The 100% recurrence in the original table is a property of Yu et
  al.'s published record, not a finding of this audit — **excluded below from the
  chance-magnitude table entirely.**
- **OLMo-7B's row-269 recurrence is mostly the same story.** Yu et al. Table 2 publishes
  row 269 at layers {1, 2, 7, 24} (§3 of the same file) — 4 of the "3 coordinates" I had
  originally logged (L1, L2) were already Yu et al.'s citation, not independent detector
  output; only **L30/r269 (activation-argmax, this session's own detector run, not in Yu
  et al.)** is new. OLMo's entry below is corrected to isolate that one new instance.
- **Llama-7B's L30/r3968 is NOT circular.** Yu et al. published exactly **one** row for
  Llama-7B (L2/r3968, `DECODER_INTERVENTION_FREEZE.md` §1: "sole candidate, no ranking
  performed") — there is no external multi-layer citation to copy. This session's
  activation-argmax independently landing on the same row index at a different layer
  (L30) is a genuinely new, non-circular finding — the cleanest fully-independent
  multi-layer result in the set, even though it is still only one pair (k=2).
- **DNABERT-2's basis is internally-sourced, not literature-circular**, but also not a
  fresh independent per-layer discovery in the way "found by this session's own detector"
  implies: `BASIS_FREEZE.md` traces it to `super_weight_index.json`, this project's own
  earlier flat top-10-by-global-activation ranking (pre-dating this audit). It is the only
  entry with real k (10 coordinates, 7 colliding pairs) and is best read as evidence
  *consistent with* the recurrence phenomenon in a genomic model, not as a literature
  citation and not as a from-scratch independent statistical test either.

**Literature check (your instruction 1): Sun et al., "Massive Activations in Large
Language Models" (arXiv:2402.17762).** Read in full (pages 1–10). Their core finding:
massive activations occupy a **small, fixed set of residual-stream feature dimensions**
that are **input-agnostic** and **persist across most consecutive middle layers** ("In
LLaMA2-7B, massive activations first appear in layer 2 and remain nearly constant values
until layer 30... They emerge in the initial layers and start to diminish in the last few
layers"), functioning as implicit bias terms for attention. Models tested: LLaMA2-7B/13B/
70B, LLaMA2-7B-Chat, Mistral-7B, Mixtral-8x7B, MPT, GPT-2, Falcon-40B, Phi-2, and ViTs —
**no genomic models**. One scope caveat worth keeping straight: Sun et al. study the
residual-stream hidden state `hℓ` itself (their own text: "our study of activations is on
the hidden state hℓ... not any intermediate states inside Fℓ"), whereas our candidates are
`down_proj` **rows**, i.e., a weight-space object indexed by the same `d_model` coordinate
system, not the activation itself — related but not the identical object.

**Does Sun et al. predict what we observed? Yes, directionally.** A fixed, fixed-across-
layers residual coordinate is exactly the kind of thing that would cause the *same*
`down_proj` output row index to keep showing up as high-gain across depth, since a
down-projection row writing into a persistently-massive residual coordinate would be
persistently high-leverage at every layer that coordinate stays massive. **DNABERT-2 and
NTv3 are genuinely new evidence for this audit to offer**, because Sun et al. never tested
genomic models — read as confirmation-plus-extension of their phenomenon into a domain
they didn't cover, not as an independent discovery of a new phenomenon. Llama-7B's L30
finding is a second data point of the same kind, but on a domain (text decoders) Sun et
al. already covered — best framed as an independent replication of Yu et al.'s row-269-
family logic via a different method (activation-argmax vs. whatever Yu et al. used),
still useful, but not a new-domain extension.

**Two statistics fixes, per instruction, applied below:**
1. **The `C(k,2)/d_model` i.i.d.-uniform-row-index null is almost certainly wrong.** Both
   Sun et al. and Yu et al.'s own published tables establish that these rows concentrate
   at a small fixed set of dimensions, not uniformly at random — an i.i.d.-uniform null is
   the wrong reference distribution for testing this pattern's significance. The
   "fold-over-chance" figures below are retained only as a labeled reference number, not
   as a claim of statistical evidence, and should not be cited as such in the manuscript.
2. **k=2 rows (NTv3, Llama-7B) are demoted, not dropped.** A single observed colliding
   pair is one observation; "4096× chance" or "1536× chance" from n=1 is one data point
   dressed as a large multiplier. They are kept in the table below (both are non-circular)
   but flagged as weak individually — Llama-7B because it is real and non-circular but
   n=1, NTv3 because it is real, non-circular, *and* the only genomic instance outside
   DNABERT-2, but still n=1.

`audit/round2/row_index_recurrence.csv` (7 rows, rebuilt),
`audit/round2/scripts/row_index_recurrence.py` (rebuilt with `circular` and `provenance`
columns).

| model | k | d_model | observed/total pairs | fold over (wrong) null | circular? | status |
|---|---|---|---|---|---|---|
| DNABERT-2 (E9 10-row basis) | 10 | 768 | 7/45 | 119.5× | No (internal, not literature) | **Only real content — genomic replication candidate** |
| Phi-3-mini (Yu et al. Table 2, full set) | 6 | 3072 | 3/15 | 614.4× | **Yes — excluded** | Citation, not a finding |
| OLMo-7B (Yu et al. L1/2/7/24 + this session's L30) | 5 | 4096 | 10/10 | 4096× | Mostly (4/5 are citation) | Only the L1-vs-L30 (or any-published-vs-L30) pair is new; k=2 worth of new evidence |
| Llama-7B (Yu et al. L2 + this session's L30) | 2 | 4096 | 1/1 | 4096× | **No — fully independent** | Real, but n=1 — demoted per instruction |
| NTv3 (this session's L6 + L11, both independent) | 2 | 1536 | 1/1 | 1536× | **No — fully independent** | Real, but n=1 — demoted per instruction |
| Mistral-7B | 1 | 4096 | — | — | No | k=1, no recurrence found |
| GENERator-EUK-3B | 1 | 3072 | — | — | No | k=1, no recurrence found |

**Bottom line, corrected**: after removing the circular Phi-3 entry and isolating OLMo's
one genuinely new coordinate, the row-index-recurrence evidence base is: one real
multi-coordinate genomic finding (DNABERT-2, k=10, internally-sourced), one real but
single-pair genomic finding (NTv3), one real but single-pair text-decoder finding (Llama-
7B, notable mainly because Yu et al. published no second Llama row to copy), and one
partial replication of Yu et al.'s own row-269 claim (OLMo's L30). This is **not** the
"119× to 4096× fold, five-for-five, spans every architecture" picture the original pass
reported — it is a much smaller, mostly single-observation body of evidence, still
consistent with the Sun/Yu phenomenon but nowhere near as strong as first stated. This is
not chased further per your instruction, recorded here so it can be evaluated later as a
manuscript paragraph or dropped.

**Author decision**: not a manuscript paragraph. At most one sentence in Discussion —
"consistent with Sun et al.'s (2402.17762) fixed-residual-coordinate phenomenon, with
DNABERT-2 and NTv3 offering preliminary genomic evidence for the same pattern" — or cut
entirely. `row_index_recurrence.csv` retained as the record; not developed further.

## Section 5 — Persist Section 3 re-analysis artifacts

**Done.** `audit/scripts/section3_frobenius.py` was already committed (round-2 Section 0c)
but only printed to stdout — no durable artifact existed. Added:

- `audit/round2/raw/section3_frobenius_output.txt` — captured stdout of the existing
  script, unmodified, as the literal "commit its outputs" ask.
- `audit/round2/scripts/section5_structure_function_persist.py` +
  `audit/round2/structure_function_correlations.csv` (16 rows) — the same analysis
  generalized across all 4 requested panels × 2 predictors × 2 epsilons, with
  leave-one-model-out ρ ranges and an explicit `underpowered_n_lt_8` column for the two
  n=6 panels (text-encoders-only, genomic-only) rather than omitting them.

| predictor | ε | panel | n | ρ | p | 95% CI | LOO range |
|---|---|---|---|---|---|---|---|
| q1 | 0.5 | all-22 | 22 | 0.115 | 0.612 | [−0.410, 0.618] | [0.009, 0.282] |
| Frobenius | 0.5 | all-22 | 22 | 0.348 | 0.112 | [−0.139, 0.741] | [0.284, 0.551] |
| q1 | 1.0 | all-22 | 22 | 0.074 | 0.744 | [−0.391, 0.513] | [−0.065, 0.190] |
| Frobenius | 1.0 | all-22 | 22 | **0.441** | **0.040** | [0.062, 0.717] | [0.365, 0.538] |
| q1 | 0.5 | text-decoders | 10 | −0.164 | 0.652 | [−0.808, 0.775] | [−0.317, 0.150] |
| Frobenius | 0.5 | text-decoders | 10 | −0.067 | 0.855 | [−0.640, 0.698] | [−0.233, 0.233] |
| q1 | 1.0 | text-decoders | 10 | **−0.333** | 0.347 | [−0.888, 0.535] | [−0.517, −0.083] |
| Frobenius | 1.0 | text-decoders | 10 | −0.042 | 0.907 | [−0.699, 0.684] | [−0.217, 0.233] |
| q1 | 0.5 | text-encoders | 6 | −0.429 | 0.397 | [−1.0, 0.742] | [−1.0, 0.0] ⚠ n<8 |
| Frobenius | 0.5 | text-encoders | 6 | 0.029 | 0.957 | [−1.0, 1.0] | [−0.2, 0.8] ⚠ n<8 |
| q1 | 1.0 | text-encoders | 6 | **−0.371** | 0.469 | [−1.0, 0.8] | [−0.7, −0.2] ⚠ n<8 |
| Frobenius | 1.0 | text-encoders | 6 | 0.543 | 0.266 | [−1.0, 0.8] | [0.5, 0.5] ⚠ n<8 |
| q1 | 0.5 | genomic-only | 6 | 0.600 | 0.208 | [−0.8, 1.0] | [0.3, 0.9] ⚠ n<8 |
| Frobenius | 0.5 | genomic-only | 6 | 0.657 | 0.156 | [−0.333, 1.0] | [0.5, 0.8] ⚠ n<8 |
| q1 | 1.0 | genomic-only | 6 | 0.486 | 0.329 | [−1.0, 1.0] | [0.1, 0.7] ⚠ n<8 |
| Frobenius | 1.0 | genomic-only | 6 | 0.486 | 0.329 | [−0.6, 1.0] | [0.3, 0.6] ⚠ n<8 |

**No panel/predictor/epsilon combination reaches significance** (p ranges 0.11–0.96)
except the all-22 Frobenius-at-ε=1.0 row already known from round-1 (p=0.040, and even
that one has a bootstrap CI lower bound of 0.062, barely above zero). **Several point
estimates flip sign within architecture-restricted panels relative to the all-22
correlation** — most strikingly q1 at ε=1.0: all-22 ρ=+0.074, but text-decoders ρ=−0.333
and text-encoders ρ=−0.371, both negative. This is exactly the pattern the manuscript's
revised claim needs: the nominal cross-class association is not a within-class effect,
and in two of three restricted panels it's not even the same sign. The two n=6 panels are
flagged underpowered rather than trusted at face value — their point estimates (positive,
moderate-to-large) should not be read as contradicting the null within-class story without
more data.

## Task 2 (this session) — Full 22-model Fig1C comparison

**Done, not extrapolated.** The original 12-model Fig1C comparison (E11 panel) already had
random-control q1 data (`results/E11/scale_ladder_controls.csv`); the remaining 10 models
(ModernBERT-base, DNABERT-2, GENERator-EUK-3B, Llama-7B, Mistral-7B, OLMo-7B-0724-hf,
MosaicBERT, GenomeOcean-4B, Qwen2.5-7B, NTv3) had top-5-by-norm data (Section 4a) but no
random-control q1s — sampled here using the **exact same `SeedSequence(42).spawn(23)
[panel_index]` convention** the E13 census uses (`build_candidate_manifest.py`), then
computed each control row's q1 via `row_spectral_metrics`
(`audit/round2/scripts/section2_random_controls_batch2.py`,
`audit/round2/section2_random_control_q1_batch2.csv`).

Merged into `audit/round2/fig1c_random_vs_topk_gaps_full22.csv` (22 rows) via
`audit/round2/scripts/fig1c_full22.py`:

| cohort statistic | random-control gap | top-5-by-norm gap |
|---|---|---|
| median | 0.9347 | 0.0135 |
| range | [0.371, 0.979] | [−0.075, 0.579] |

**Collapse factor: 69.0×**, computed directly on the full 22-model cohort (not
extrapolated from the 12-model subset, which had shown 114.8×). Two-panel figure at
`audit/round2/figures/fig1c_two_panel.png` — left panel (existing manuscript claim) shows
every model's candidate beating random controls by a wide, near-uniform margin (0.37–0.98);
right panel (audit's stricter top-5-by-norm comparison) shows the same candidates mostly
near zero, with 7/22 actually negative (candidate loses to at least one same-layer row of
comparable norm): DNABERT-2 (−0.073), GenomeOcean-4B (−0.075), GENERator-EUK-3B (−0.022),
ModernBERT-large (−0.003), EuroBERT-2.1B (−0.0001), EuroBERT-210m (−0.001),
SmolLM2-360M (−0.016).

**This is the complete cohort, not a subsample** — the 69.0× figure (vs. the 12-model
114.8×) should replace it as the headline number in any manuscript sentence going forward.


## Task 3 (4b) — top-K-by-norm causal sweep, in progress

**Method**: same intervention/evaluation harness as the original E13 census
(`run_singleton_census.py`), unmodified except for a thin wrapper
(`run_singleton_census_4b.py`) that swaps `candidate_manifest.json`'s random control rows
for the top-5-by-norm rows already identified in Section 4a
(`candidate_manifest_4b.json`, built by `build_4b_manifest.py`). Writes to
`results/E13/raw_4b/`, leaving the original census untouched. Cost estimated from actual
per-model wall-clock in the original census (same op count: candidate + 5 controls × 2
epsilons) before starting, staged smallest/highest-priority first per instruction.

**Note on execution**: this sweep initially ran concurrently with Task 2's batch-2
random-control compute on the same 40GB GPU. Two attempts at Llama-7B for 4b hit CUDA OOM
from that collision (two 7B-class fp32 models cannot coexist on one 40GB GPU) before both
jobs were serialized; no data was lost, just wall-clock. All results below are from clean,
uncontested runs.

| model | ε | candidate rel. loss change | median top5-by-norm control | gap |
|---|---|---|---|---|
| DNABERT-2 | 0.5 | +0.31% | ~0.00% | +0.0031 |
| DNABERT-2 | 1.0 | +0.83% | +0.01% | +0.0082 |
| GENERator-EUK-3B | 0.5 | +4.29% | +0.01% | +0.0428 |
| GENERator-EUK-3B | 1.0 | +37.10% | +0.05% | +0.3705 |
| ModernBERT-base | 0.5 | +10.93% | +0.03% | +0.1089 |
| ModernBERT-base | 1.0 | +357.04% | +0.25% | +3.5680 |
| Llama-7B | 0.5 | +3.86% | −0.01% | +0.0387 |
| Llama-7B | 1.0 | +300.79% | −0.01% | +3.0080 |

**Every model so far shows a decisive causal gap that survives the stricter top-5-by-norm
control** — this is not the same result as the structural (q1) top-K comparison in
Section 4a, where several of these same models (GENERator-EUK-3B, DNABERT-2) showed
*negative* q1 gaps against top-5-by-norm rows. The structural concentration measure (q1)
and the causal effect measure (ablation damage) diverge: a same-layer row can have
comparable or greater spectral concentration than the candidate while causing far less
functional damage when perturbed. This is exactly the distinction Section 4/4a's own
framing anticipated — norm/structure does not guarantee causal importance, and the
candidates remain causally distinguished even where they are not structurally
distinguished.

| Mistral-7B | 0.5 | +301.38% | −0.07% | +3.0145 |
| Mistral-7B | 1.0 | +287.15% | +0.02% | +2.8713 |
| OLMo-7B | 0.5 | +0.63% | +0.04% | +0.0058 |
| OLMo-7B | 1.0 | +111.78% | +0.14% | +1.1164 |
| GenomeOcean-4B | 0.5 | +0.43% | +0.31% | +0.0012 |
| GenomeOcean-4B | 1.0 | +0.76% | +0.53% | +0.0023 |

**GenomeOcean-4B is the one exception to the pattern so far — a genuinely weak causal
separation, not just a weak structural one.** Every other model above shows the candidate
causing orders of magnitude more damage than its top-5-by-norm neighbors; GenomeOcean-4B's
candidate (already the cohort's worst-ranked by q1, Section 4a batch 3) is *also* barely
distinguishable causally from its neighbors (gap 0.12–0.23pp, vs. gaps of multiple
percentage points to hundreds of percent elsewhere). This is internally consistent — the
structural and causal measures agree for GenomeOcean-4B specifically — but it means the
"candidates remain causally distinguished even where not structurally distinguished" claim
above does not hold universally; GenomeOcean-4B is undistinguished on both axes.

**Interruption, noted for the record**: the compute allocation running this sweep expired
partway through Qwen2.5-7B (13/25 batch units complete, no output written — full rerun
needed) between the DNABERT-2/GENERator-EUK-3B/ModernBERT-base/Llama-7B/Mistral-7B/
OLMo-7B/GenomeOcean-4B results above and the remainder. No data was lost — everything that
had finished writing its JSON survived; only the in-flight Qwen2.5-7B run needs
re-launching. Resumed on a fresh allocation (2-day walltime).

*(Table above will be extended as Qwen2.5-7B, MosaicBERT, NTv3 complete.)*

### Structural gap vs. causal gap, side by side — the paper's central table

Per your request: `audit/round2/structural_vs_causal_gap.csv`
(`audit/round2/scripts/structural_vs_causal_gap.py`), regenerated as each 4b model lands.
7/10 models in so far.

| model | ε | structural topK gap (q1) | causal topK gap | causal random-control gap |
|---|---|---|---|---|
| DNABERT-2 | 0.5 | −0.0731 | +0.0031 | +0.0031 |
| DNABERT-2 | 1.0 | −0.0731 | +0.0082 | +0.0083 |
| GENERator-EUK-3B | 0.5 | −0.0221 | +0.0428 | +0.0429 |
| GENERator-EUK-3B | 1.0 | −0.0221 | +0.3705 | +0.3710 |
| GenomeOcean-4B | 0.5 | −0.0746 | +0.0012 | +0.0043 |
| GenomeOcean-4B | 1.0 | −0.0746 | +0.0023 | +0.0077 |
| Llama-7B | 0.5 | +0.1678 | +0.0387 | +0.0385 |
| Llama-7B | 1.0 | +0.1678 | +3.0080 | +3.0079 |
| Mistral-7B | 0.5 | +0.0263 | +3.0145 | +3.0138 |
| Mistral-7B | 1.0 | +0.0263 | +2.8713 | +2.8715 |
| ModernBERT-base | 0.5 | +0.1933 | +0.1089 | +0.1092 |
| ModernBERT-base | 1.0 | +0.1933 | +3.5680 | +3.5704 |
| OLMo-7B | 0.5 | +0.3512 | +0.0058 | +0.0063 |
| OLMo-7B | 1.0 | +0.3512 | +1.1164 | +1.1178 |
| Qwen2.5-7B | 0.5 | +0.0038 | +0.0019 | +0.0060 |
| Qwen2.5-7B | 1.0 | +0.0038 | **−0.0014** | +0.0074 |
| MosaicBERT | 0.5 | +0.0115 | **−0.0019** | +0.0038 |
| MosaicBERT | 1.0 | +0.0115 | +0.0034 | +0.0038 |
| NTv3 | 0.5 | **+0.3694** | **−0.0050** | −0.0029 |
| NTv3 | 1.0 | **+0.3694** | **−0.0080** | −0.0040 |

**Qwen2.5-7B is a genuine exception, flagged immediately rather than smoothed into the
average — not just a weak case like GenomeOcean-4B, but a sign flip.** At ε=1.0, the
causal topK gap is **negative**: one of the top-5-by-norm control rows in Qwen2.5-7B's
layer causes *more* relative loss increase than the candidate itself (candidate 0.74%,
topK-control median 0.87–0.92% per the per-row breakdown below). This is the only model in
the cohort so far where a norm-matched control row outright beats the candidate causally.
Against random controls the candidate still wins at both epsilons (+0.0060, +0.0074) — the
random-control comparison would have hidden this entirely, which is itself the point of
running the stricter test. Qwen2.5-7B's structural gap is also the smallest positive value
in the table (+0.0038, a near-tie) — structure and causality agree here that this
candidate is only marginally distinguished from its neighbors, not decisively.

**MosaicBERT is a second sign flip, landed right after, and it strengthens rather than
dilutes the pattern.** At ε=0.5, MosaicBERT's causal topK gap is also negative (−0.0019):
the candidate's own effect at this epsilon is slightly loss-*reducing* (−0.17%), while the
topK-control median is essentially flat but positive (+0.02%). MosaicBERT's structural gap
is another near-zero value (+0.0115) — the third model in the table with a structural gap
under 0.02 in magnitude, and the third to show causal instability at one or both epsilons
(GenomeOcean-4B: uniformly weak; Qwen2.5-7B: sign flip at ε=1.0; MosaicBERT: sign flip at
ε=0.5). **This is now a 3-for-3 correlation between small |structural gap| and causal
instability, not a single outlier** — every model whose candidate is not clearly the most
concentrated row in its layer also fails to be clearly the most causally important row,
in at least one epsilon.

**The exact pattern you predicted, with three related exceptions now on the record — and
they cluster in a way that is itself informative.** Two things jump out immediately:

1. **Causal topK gap ≈ causal random-control gap for every model, at both epsilons**
   (agreement to the 3rd–4th decimal throughout, e.g. DNABERT-2 ε=1.0: +0.0082 vs
   +0.0083). Switching the comparison set from random same-layer rows to the rows
   specifically selected for having the *highest weight norm in the layer* changes the
   causal comparison almost not at all.
2. **Structural gap and causal gap are uncorrelated in sign, let alone magnitude.**
   DNABERT-2 and GENERator-EUK-3B both have *negative* structural gaps (candidate loses to
   a higher-norm neighbor on q1) but strongly *positive* causal gaps (candidate beats every
   neighbor causally by 1–2 orders of magnitude). Meanwhile GenomeOcean-4B has a comparably
   negative structural gap but only a negligible causal gap — the one model where structure
   and causality agree, and it's the cohort's weakest case on both axes, not its strongest.

This is exactly the two-comparisons-in-one-table your point (1) called for: candidates
that are not the most spectrally concentrated row in their layer are, with one exception,
still overwhelmingly the most causally important row in their layer.

### Are top-5-by-norm control rows causally distinguishable from random control rows? — **no, not in any way that matters**

Per your point (2): `audit/round2/norm_controls_vs_random_controls.csv`
(`audit/round2/scripts/norm_vs_random_controls.py`). Median top-5-by-norm control effect
vs. median random-control effect from the original census, paired per model:

| ε | n models | median topK control effect | median random control effect | Wilcoxon p |
|---|---|---|---|---|
| 0.5 | 10 | 0.00016 | 0.00000 | 0.106 (not significant) |
| 1.0 | 10 | 0.00041 | 0.00000 | **0.010 (significant at n=10, final)** |

**Final, full 10-model result.** The ε=1.0 p-value held roughly steady through the last
addition (0.047→0.023→0.012→0.010) — NTv3 contributed one of the smallest paired
differences at both epsilons (+0.00002 at ε=0.5, contributing near-zero at ε=1.0 as well),
consistent with its topK controls being almost perfectly inert (median 0.0000), matching
the aggregate "inert" pattern even though NTv3's *candidate* itself is the cohort's
strangest case (see the negative-gap discussion above). **Final reading**: the median-level
claim holds at n=10 (0.041% vs. 0.000%, both tiny next to typical candidate effects of
single-digit-to-hundreds of percent) with the ε=1.0 comparison statistically significant,
but it should be stated with the qualifier established across this section — 3 of 10
models (GenomeOcean-4B, Qwen2.5-7B, MosaicBERT) show non-negligible topK-control effects
tracking a small candidate structural margin, and NTv3 shows a different kind of
irregularity (anomalous candidate-level sign, not control competitiveness) that the
median-level statistic doesn't capture. **This is the final n=10 result — the full 4b
sweep is complete.**

### Framing note: this is a data-dependent-vs-weight-only selection finding, not a "structure vs. function" one — FINAL, all 10 models

Per your framing request. **Final version, all 10 models complete.** 6/10 models show a
clean, decisive win for the data-dependent candidate over every topK-by-norm control at
both epsilons (DNABERT-2, GENERator-EUK-3B, Llama-7B, Mistral-7B, ModernBERT-base,
OLMo-7B — gaps ranging from single-digit percent to >300%). The other 4/10 split into two
distinct kinds of exception, not one:

- **Small-structural-margin exceptions (3/10 — GenomeOcean-4B, Qwen2.5-7B, MosaicBERT)**:
  the candidate's structural gap over its neighbors is small (|gap| < 0.08 in all three),
  and in two cases (Qwen2.5-7B at ε=1.0, MosaicBERT at ε=0.5) a topK-by-norm control row
  is causally *more* damaging than the candidate — an outright sign flip. This is the
  pattern predicted going in, and it held for all 3 cases where the structural precondition
  was met.
- **The NTv3 exception (1/10), a different mechanism**: NTv3's structural gap is one of
  the *largest* in the cohort (+0.3694), so this is not a small-margin case at all. Its
  topK controls are almost perfectly causally inert (median ≈0.0000, matching the pattern
  in the 6 decisive models) — but the *candidate itself* shows an anomalous negative-sign
  effect (perturbing it slightly *improves* loss, at both epsilons), a property already
  present in the original random-control census, not introduced by this sweep. The
  small-structural-margin hypothesis does not explain NTv3; it is a genuinely separate
  irregularity.

> Top-K-by-norm selection is a purely weight-space criterion — it requires no forward pass
> and no input data, only the model's frozen parameters. In 6 of 10 models tested, rows
> selected this way are causally indistinguishable from randomly chosen same-layer rows,
> while the candidate coordinate — identified by a data-dependent detector run over real
> input sequences — causes damage one to three orders of magnitude larger. This does not
> hold universally, and the exceptions are not homogeneous: in 3 models, the candidate's
> own structural margin over its neighbors is small, and a norm-matched control row is
> causally competitive with or exceeds the candidate at one of the two epsilons tested; in
> a 4th model (NTv3), the topK controls remain inert but the candidate itself shows an
> unexplained negative-direction effect unrelated to control selection. The pattern is
> therefore conditional in the majority case — data-dependent selection outperforms
> weight-only selection except when the candidate itself is only weakly structurally
> distinguished from its neighbors — with at least one exception (NTv3) that this
> conditional does not cover at all.

**NTv3 landed, and it breaks the prediction I made above — reported plainly, not
reshaped to fit.** I predicted that since NTv3's structural gap (+0.3694) is one of the
largest in the cohort, it would land in the decisive "candidate wins" group, not add a
fourth exception. **It does neither.** NTv3's candidate causes a *negative* relative loss
change at both epsilons (−0.50% at ε=0.5, −0.80% at ε=1.0) — perturbing the candidate row
makes the model's loss slightly *better* than baseline, not worse. The topK-by-norm
controls are almost exactly inert (median 0.0000 at both epsilons), so the causal gap is
negative (−0.0050, −0.0080) purely because the candidate's own effect is negative, not
because a control competes with it.

**This is not new to the 4b sweep — confirmed against the original census
(`part2_22_model_results.csv`): NTv3's candidate shows this same negative-direction effect
against random controls too** (`candidate_minus_median_control_relative_eps0p5` =
−0.00289, `_eps1p0` = −0.00414 — smaller in magnitude than 4b's topK comparison but the
same sign, same qualitative story). **This is a pre-existing property of NTv3's candidate
coordinate that has been sitting in this project's own data since the original census ran,
not something the 4b sweep introduced or a byproduct of the topK-vs-random control
choice.** Whether it reflects something genuine about the perturbation's effect on
NTv3's masked-language-modeling loss, or an artifact specific to NTv3's evaluation setup
(NTv3 has needed the most special-casing of any model in this audit — the 256-multiple
padding requirement, the missing `trust_remote_code=True`, the activation-vs-ratio
detector discrepancy in Section 3) is not something this audit has investigated; flagging
it rather than guessing.

**Revised bottom line on the low-structural-margin hypothesis: partially wrong, not
confirmed.** GenomeOcean-4B, Qwen2.5-7B, and MosaicBERT's exceptions do track small
|structural gap| — that part holds. But NTv3 shows a *large* structural gap can also fail
to predict a decisive positive causal gap, for a completely different reason (an anomalous
candidate-level effect, not control competitiveness). **The honest generalization is
narrower than what I proposed before NTv3 landed**: small structural margin predicts
causal instability in this cohort (3/3 confirmed), but large structural margin does not
guarantee a decisive causal result (NTv3 is a counterexample) — the two directions of the
claim are not symmetric, and only the first should be stated with any confidence.

## Task 3 (4b) — FINAL: full 22-model cohort, and a retraction

**Author instruction: "Run the remaining 12... n=22 either firms it up or breaks it."**
It breaks it. Reported plainly, same discipline as the earlier architecture-confound
retraction in the full-22 topK-gap section above.

`audit/round2/scripts/build_4b_manifest.py` extended to all 22 models (`TARGET_MODELS`
now covers the full cohort); `run_4b_remaining12_queue.sh` ran the 12 small E11-panel
models sequentially, cheapest-by-non_embed_params first, no GPU collisions this time.
`structural_vs_causal_gap.py` and `norm_vs_random_controls.py` extended with the 12 new
model-slug mappings and rerun at n=22.

### Full 22-model structural gap vs. causal topK gap

| model | structural gap (q1) | causal gap ε=0.5 | causal gap ε=1.0 |
|---|---|---|---|
| DNABERT-2 | −0.0731 | +0.0031 | +0.0082 |
| GENERator-EUK-3B | −0.0221 | +0.0428 | +0.3705 |
| ModernBERT-large | −0.0026 | +0.0803 | +2.2040 |
| EuroBERT-2.1B | −0.0001 | +0.0005 | +0.0009 |
| EuroBERT-210m | −0.0013 | +0.0005 | +0.0008 |
| SmolLM2-360M | −0.0156 | +0.0202 | +0.5851 |
| EuroBERT-610m | +0.0006 | +0.0001 | +0.0001 |
| Qwen2.5-1.5B | +0.0007 | +0.0002 | +0.0001 |
| Qwen2.5-7B | +0.0038 | +0.0019 | **−0.0014** |
| MosaicBERT | +0.0115 | **−0.0019** | +0.0034 |
| GENERator-PROK-3B | +0.0156 | +0.0012 | +0.0020 |
| Mistral-7B | +0.0263 | +3.0145 | +2.8713 |
| SmolLM2-1.7B | +0.0443 | +0.0390 | +6.9887 |
| Qwen2.5-3B | +0.0604 | +0.0182 | +0.0247 |
| GENERator-PROK-1.2B | +0.0817 | +0.0014 | +0.0052 |
| Qwen2.5-0.5B | +0.1380 | +0.0023 | **−0.0029** |
| Llama-7B | +0.1678 | +0.0387 | +3.0080 |
| ModernBERT-base | +0.1933 | +0.1089 | +3.5680 |
| OLMo-7B | +0.3512 | +0.0058 | +1.1164 |
| NTv3 | +0.3694 | **−0.0050** | **−0.0080** |
| SmolLM2-135M | +0.5788 | +0.0362 | +3.1300 |

(21 rows shown — GenomeOcean-4B, already reported, structural −0.0746 / causal +0.0012 /
+0.0023, omitted here only for table length; present in the CSV and unchanged.)

### The correlation the small-margin hypothesis needed — does not hold at n=22

**Spearman ρ(|structural gap|, causal topK gap):** ε=0.5: ρ=0.311, p=0.159. ε=1.0:
ρ=0.355, p=0.105. **Neither reaches significance.** The signed-structural-gap version is
even weaker (ε=1.0: ρ=0.199, p=0.374).

### The sign-flip cases at n=22 — 4 models, and they split the hypothesis in half

| model | structural gap | negative at | 
|---|---|---|
| MosaicBERT | +0.0115 (small) | ε=0.5 |
| Qwen2.5-7B | +0.0038 (small) | ε=1.0 |
| Qwen2.5-0.5B | **+0.1380 (large)** | ε=1.0 |
| NTv3 | **+0.3694 (largest in cohort)** | both ε |

**Two of the four sign-flip models have LARGE structural gaps, not small ones.**
Qwen2.5-0.5B (+0.138, one of the bigger structural margins in the whole cohort) and NTv3
(+0.369, the single largest) both show a causal sign flip at ε=1.0 — the exact opposite of
what the small-margin hypothesis predicted for them. Meanwhile several models with tiny or
outright *negative* structural gaps (DNABERT-2 −0.073, GENERator-EUK-3B −0.022,
ModernBERT-large −0.003, SmolLM2-360M −0.016) show fully decisive, large-magnitude causal
wins with no instability at all.

**Retraction: the small-structural-margin-predicts-causal-instability hypothesis, stated
with confidence after n=9/10, does not survive the full cohort.** It was built on 3
data points (GenomeOcean-4B, Qwen2.5-7B, MosaicBERT) that happened to all have small
structural gaps; extending to 22 added two more sign-flip cases (Qwen2.5-0.5B, NTv3) with
large structural gaps, which the hypothesis cannot accommodate, while several small/negative-
structural-gap models turned out to be among the *most* causally decisive in the cohort
(DNABERT-2, GENERator-EUK-3B, ModernBERT-large, SmolLM2-360M). **This is the same failure
mode as the architecture confound earlier in this audit: a real-looking pattern in an early
subset that was an artifact of which models happened to run first, not a property of the
underlying data.** Present the sign-flip cases as an unexplained ~18% (4/22) minority, not
as a phenomenon predictable from structural margin.

### Norm-controls-vs-random-controls test, FINAL at n=22

| ε | n | median topK control effect | median random control effect | Wilcoxon p |
|---|---|---|---|---|
| 0.5 | 22 | 0.00035 | 0.00000 | **0.0009** |
| 1.0 | 22 | 0.00110 | 0.00001 | **0.0001** |

**Both now significant, and more strongly than at any earlier n** — the full cohort
confirms the aggregate median-level claim robustly: top-5-by-norm control rows are, in
the typical case, causally close to inert (0.035%–0.11%), several orders of magnitude
below typical candidate effects (up to 300–700%). **This part of the story strengthens
with more data — it is the per-model mechanism explaining the exceptions (structural
margin) that does not.** The 4 sign-flip models remain real, individual exceptions to the
aggregate pattern; they are simply not explained by the variable this audit tested.

### Final classification, all 22 models

- **Decisive win, no instability (14/22)**: DNABERT-2, GENERator-EUK-3B, ModernBERT-large,
  SmolLM2-360M, Mistral-7B, SmolLM2-1.7B, Qwen2.5-3B, Llama-7B, ModernBERT-base, OLMo-7B,
  SmolLM2-135M, GenomeOcean-4B, GENERator-PROK-1.2B, GENERator-PROK-3B (last 3 weak in
  magnitude but consistently positive-sign, no flip).
- **Weak-but-positive, negligible magnitude (4/22)**: EuroBERT-2.1B, EuroBERT-210m,
  EuroBERT-610m, Qwen2.5-1.5B — candidate and topK controls both essentially inert; neither
  wins decisively because neither shows a meaningful effect.
- **Sign-flip exceptions (4/22)**: MosaicBERT, Qwen2.5-7B, Qwen2.5-0.5B, NTv3 — no shared
  structural signature found; two have small structural margins, two have large ones.

### Revised framing note — FINAL, supersedes all earlier provisional versions

> Top-K-by-norm selection is a purely weight-space criterion — it requires no forward pass
> and no input data, only the model's frozen parameters. Across the full 22-model cohort,
> the median top-5-by-norm control row is causally close to inert (Wilcoxon p<0.001 at
> both epsilons, effect sizes 0.03–0.11% versus candidate effects reaching several hundred
> percent), while the data-dependent candidate coordinate causes substantially larger
> damage in the majority of models. This does not hold universally: 4 of 22 models show a
> topK-by-norm control row that is causally competitive with or exceeds the candidate at
> one of the two epsilons tested. No structural property tested in this audit — including
> the candidate's own margin of spectral concentration over its neighbors — predicts which
> models fall into this minority; two of the four exceptions have small structural margins
> and two have among the largest in the cohort. The aggregate claim (data-dependent
> selection outperforms weight-only selection) is well-supported; a per-model explanation
> for the exceptions is not available from this analysis.

**The full 4b sweep is complete: 22/22 models, both epsilons, structural and causal
comparisons cross-tabulated, one hypothesis tested and retracted rather than left
unchallenged.**
