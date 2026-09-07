# E6 Stage A RESULTS — independent confirmation of exact cross-pathway interaction geometry

**STATUS: FINAL. Mechanical branch: C (no independent replication under the locked
criterion).** Read the nuance section below before treating "Branch C" as a clean negative —
this was a **near-miss on effect-size magnitude, not on direction**. Per the governing
instruction for a Branch C/D outcome, **E6 stops here**: no alternative metric was invented,
no threshold was adjusted, no row was excluded to rescue the decision, no causal experiment
was run, and no model was added.

**Prereg:** `docs/prereg/PREREG_cross_geometry_stageA.md`, sha256
`4cbe4416719b33a7a6eac256dbba49d40ef5df822cf15d72167374e58fc51bb5`, locked
2026-08-14T12:08:49+00:00, commit `2b9d9e4`, verified immediately before the confirmatory run
and again automatically at the start of `run_stageA.py`.
**Producing script:** `experiments/E6_cross_geometry/run_stageA.py`
**Shared library:** `experiments/E6_cross_geometry/cross_geometry_lib.py` (7/7 synthetic
tests green pre-lock; regression check reproduced all six of E5's published `f_cross` values
to ~1e-14 relative error pre-lock — see `test_cross_geometry_lib.py`,
`validate_against_e5.py`)
**Raw output:** `results/e6_stageA.json`

## Deviation from the locked prereg text (mechanical, not scientific)

The prereg does not specify network access for weight fetching. OLMo-7B's layer-2 shard
happened to already be cached (it shares a shard file with E5's layer-1 primary), but layers
7 and 24 needed new shards. These were fetched via the same single-shard-only method E1/E5
already established (`fetch_layer_tensors`), with `HF_HUB_OFFLINE` unset only for that fetch.
This changes zero panel members, zero formulas, and zero thresholds — it is an infrastructure
step needed to load an **already-selected, already-published** (Yu et al. Table 2) candidate,
not a new candidate-selection decision.

## Measurements (numbers only — interpretation follows in later sections)

### Panel-level summary

| Group | n | `f_cross` range |
|---|---|---|
| NLP (OLMo-7B, 3 layers, same row 269) | 3 | −0.0041 to +0.0758 |
| Genomic (DNABERT-2 9 rows + GENERator EUK 1 row) | 10 | +0.1134 to +0.8301 |

### Full per-candidate table

| Model / layer / row | Group | `f_cross` | `x_pos_norm` | `x_neg_norm` | `kappa` | Layer pct. | Exact rank |
|---|---|---:|---:|---:|---:|---:|---:|
| OLMo-7B L2/r269 | NLP | +0.0595 | 0.0713 | −0.0119 | 0.7149 | 99.9 | 1/4096 |
| OLMo-7B L7/r269 | NLP | +0.0758 | 0.0863 | −0.0106 | 0.7821 | 99.9 | 1/4096 |
| OLMo-7B L24/r269 | NLP | **−0.0041** | 0.0013 | −0.0054 | **−0.6022** | **0.1** | 1/4096 |
| DNABERT-2 L3/r86 | Genomic | 0.5589 | 1.5638 | −1.0049 | 0.2176 | 99.5 | 4/768 |
| DNABERT-2 L3/r399 | Genomic | 0.5454 | 2.1359 | −1.5905 | 0.1464 | 99.3 | 3/768 |
| DNABERT-2 L3/r603 | Genomic | 0.5889 | 1.3554 | −0.7665 | 0.2775 | 99.9 | 1/768 |
| DNABERT-2 L3/r641 | Genomic | 0.5685 | 3.6460 | −3.0775 | 0.0846 | 99.7 | 2/768 |
| DNABERT-2 L5/r86 | Genomic | 0.5203 | 0.5660 | −0.0457 | 0.8505 | 99.9 | 2/768 |
| DNABERT-2 L6/r603 | Genomic | 0.3104 | 0.3939 | −0.0835 | 0.6501 | 99.9 | 2/768 |
| **DNABERT-2 L7/r603** | Genomic | **0.1134** | 0.5623 | −0.4489 | 0.1121 | 100.0 | **686/768** |
| DNABERT-2 L9/r264 | Genomic | 0.4274 | 0.5070 | −0.0796 | 0.7287 | 100.0 | 2/768 |
| DNABERT-2 L9/r294 | Genomic | 0.2970 | 0.6290 | −0.3320 | 0.3090 | 99.9 | 1/768 |
| GENERator EUK L4/r1522 | Genomic | 0.8301 | 0.8388 | −0.0086 | 0.9796 | 100.0 | 2/3072 |

`x_pos_norm + |x_neg_norm|` values above 1.0 (e.g. DNABERT-2 L3/r641: 3.646 + 3.078 = 6.72)
mean gross cross-pathway activity substantially exceeds the net exact squared norm itself —
large partially-cancelling positive and negative interactions, not a contradiction (`f_cross`
is still bounded near the net difference; `x_pos_norm`/`x_neg_norm` are each individually
unbounded by construction).

## Preregistered decision, applied mechanically

**Step 1 (Q1 — replication):**

- `NLP_max = 0.0758` (OLMo-7B L7), `GEN_min = 0.1134` (DNABERT-2 L7/r603).
- **Complete separation: TRUE.** Every genomic candidate's `f_cross` exceeds every NLP
  candidate's. Exact rank-sum test (non-asymptotic, `C(13,3)=286` arrangements, genomic vs.
  NLP): **p = 0.0035**, the single most extreme of the 286 possible arrangements.
- **Effect-size margin (1.5× relative, since `NLP_max > 0`): FAIL.**
  `1.5 * 0.0758 = 0.1137`; `GEN_min = 0.1134`. **The genomic minimum falls short of the
  required margin by 0.0003 — a 0.28% relative shortfall.** This is a near-miss, not a
  wide failure.
- **Q1 result: FAIL** (both sub-conditions are required; the margin sub-condition did not
  hold). Per the locked decision tree, this assigns **Branch C** and stops the tree — Steps 2
  and 3 are **not evaluated as part of the decision**.

**Branches 2/3 were never reached, but were computed anyway as measurements** (Task 4 asks
for these per-candidate regardless of the eventual branch) and are reported here descriptively,
**not as part of the decision**:

- Median genomic layer percentile: **99.9** — would have cleared the Step 2 bar (≥75.0) by a
  wide margin had Q1 passed.
- Median genomic `kappa`: **0.293** (midpoint of 0.2775 and 0.3090, the two central values of
  the ten) — would **also** have narrowly missed the Step 3 bar (≥0.3) had Q1 passed.

## Why the near-miss, specifically

The margin failure is driven almost entirely by one row: **DNABERT-2 L7/r603**, `f_cross`
0.1134 — the smallest of all ten genomic candidates by a wide margin (the next-smallest is
0.297). This is not a row this study picked without history: it is **already flagged in this
repository's own ledger** (C-010: "DNABERT-2 L7 r603 predictor rank 706/768 reflects a
propagator, not a source") as structurally atypical relative to the rest of the same 10-row
ensemble — its earlier diagonal-form rank was already known to be poor (706/768), and this
experiment's **independent, exact-form** computation gives an essentially identical
diagnosis: **rank 686/768**, the only genomic candidate outside the top 4 of its layer. Every
other DNABERT-2 row in this panel (all from layers other than 7, or from a different position
within layer 3/9) ranks 1–4 of 768 under the exact form, matching the pattern already seen in
E5's own six rows.

**This is reported as a factual observation, not used to change the decision.** Per the
locked prereg's own discipline and the governing instruction's explicit prohibition on
post-hoc exclusions, **L7/r603 is not dropped and the decision is not recomputed without
it.** If a future session wants to test whether excluding a row already independently flagged
as atypical changes the outcome, that is a new, separately preregistered analysis — not a
retroactive edit to this one.

## What this establishes

- **Row-level exact-form ranking robustly generalizes beyond E5's six discovery rows.** 12 of
  13 independent confirmation-panel candidates rank in the top 4 of their layer under the
  exact (cross-term-complete) quadratic form — including all three OLMo-7B rows at rank
  exactly 1/4096. The one exception (DNABERT-2 L7/r603) was already flagged as atypical in
  this repository's ledger before E6 existed, from an independent (diagonal-form) computation.
- **Direction of E5's discovery-panel split independently replicates with a strong exact
  significance level (p = 0.0035, non-asymptotic).** Genomic `f_cross` uniformly exceeds NLP
  `f_cross` across all 13 candidates.
- **This study's own pre-committed magnitude bar for calling that a *confirmed* effect is not
  met** — by a margin (0.28%) small enough that this is fairly read as inconclusive-but-close
  rather than a clean refutation.

## What this is merely consistent with

- That the true group-level gap in cross-pathway dependence is real but this panel's
  particular composition (one atypical DNABERT-2 row, and an NLP arm that is one model's
  recurring channel across layers rather than three independent models) was not powered to
  clear a deliberately strict, pre-committed bar.
- That genomic candidates are, descriptively, extremely unusual within their own layers
  (median percentile 99.9) — consistent with Q2's specificity question, though this was never
  reached as part of the binding decision.
- That where cross mass is large, it leans constructive more often than not (8 of 10 genomic
  `kappa` values exceed 0.2; two — L7/r603 and L3/r641 — are closer to 0.1), but the group's
  central tendency (median 0.293) sits right at, not clearly above, this study's own
  pre-committed coherence bar.

## What this does NOT establish

- That the group-level gap is a confirmed, replicated effect — the locked criterion says
  otherwise, and this document does not talk it into a pass.
- Anything about why the gap exists, whether genomic sequence data causes it, whether cross
  terms cause functional criticality, whether any specific interacting pair or group is
  causal, or the collaborator-reported DNABERT-2 "redundant pair" mechanism (still without
  any artifact in this repository).
- Anything new about Evo1 or GENERator PROK — both remained excluded, as preregistered.

## Recommendation for existing claims (not enacted — no `CLAIMS_LEDGER.md` row is edited by
this document)

### C-002 / C-003 (row-ranking claims)

**Recommend: RETAIN, with a Methods clarification.** Both are primarily row-ranking claims.
Row-level ranking under the exact (cross-term-complete) form is now confirmed not just on
E5's six discovery rows but on 12 of 13 independently-selected confirmation rows too. The one
exception is a row already independently flagged as atypical for an unrelated reason (C-010).
Methods should state plainly that the underlying `‖U_k‖_F` computation is a diagonal
approximation and that row-level (not scalar-level) ranking is what has been directly
verified against the exact form — consistent with this repository's existing wording
discipline (`CLAUDE.md`'s standing rule to distinguish row recovery from scalar recovery).

### C-032 (granularity / participation-ratio interpretation)

**Recommend: HOLD.** This is the claim E5 flagged as most at risk, and the flag is not
resolved by this experiment either way. The direction of the underlying cross-term gap
independently replicates at a strong exact significance level: this is **not** grounds to
retire C-032 as though the gap were shown not to exist. But this study's own pre-committed,
disclosed-before-data bar for calling the gap *confirmed* — chosen specifically to demand more
than "genomic `f_cross` happens to be numerically larger" — was not met, by a narrow but real
margin. That is **not** grounds to treat C-032 as vindicated either. **Do not silently
redefine C-032 as a cross-term-aware granularity metric** — no such metric was computed or
validated in this pass, and any future replacement gets its own new claim ID, per instruction.
A future session wanting to resolve this should either (a) preregister a fresh, larger,
genuinely-independent-across-models NLP arm (this study's is disclosed as a single recurring
channel in one model), or (b) design and separately preregister a principled cross-term-aware
replacement for participation ratio — neither of which this pass attempts.

## Provenance

| | |
|---|---|
| Producing script | `experiments/E6_cross_geometry/run_stageA.py` |
| Shared library | `experiments/E6_cross_geometry/cross_geometry_lib.py` |
| Pre-lock validation | `test_cross_geometry_lib.py` (7/7 green), `validate_against_e5.py` (6/6 exact reproductions) |
| Raw output | `results/e6_stageA.json` |
| Checkpoints | `allenai/OLMo-7B-0724-hf` (no pinned revision, same gap already disclosed for E5); `zhihan1996/DNABERT-2-117M` @ `7bce263b15377fc15361f52cfab88f8b586abda0`; `GenerTeam/GENERator-v2-eukaryote-3b-base` @ `7dc01bccce5b65e15141170538afdc2ff09d8dde` |
| Dtype | float64 throughout |
| Device | CUDA, A100-PCIE-40GB (idle, confirmed available); `pair_matrix` ran on whatever device the loaded weights were on (CPU in this run — an efficiency note, not a correctness one: OLMo's and GENERator EUK's `K`-matrix builds took 51s and 22s respectively on CPU where a GPU-resident build would likely be under a few seconds, per E5's own timings for the same operation. Numerically identical either way; not rerun for speed since correctness, not runtime, is what this study is scored on.) |
| Seed | n/a — deterministic weight algebra throughout |
| Forward passes | zero |
| Ablations | zero |

## Mandatory stop (per the governing instruction for a Branch C/D outcome)

**E6 stops here.** No alternative geometry hypothesis is proposed. No causal experiment is
run (Task 6 is conditional on Branch A only and is not written). No model is added to the
panel. The result — including its near-miss character and the specific row responsible for
it — is preserved exactly as measured.
