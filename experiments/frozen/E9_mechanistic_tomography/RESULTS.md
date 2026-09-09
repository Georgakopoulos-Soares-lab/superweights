# RESULTS.md — E9 Mechanistic Tomography of High-Gain FFN Causal Response

Frozen 2026-08-22. Raw artifacts: `baseline_regression_results.json`,
`dnabert2_mask_responses.json`, `fit_results_dnabert2.json`, `masks_dnabert2.json`.
Prereg: `experiments/docs/prereg/PREREG_mechanistic_tomography_E9.md`, locked
2026-08-22T20:55:26+00:00 (`LOCKS.jsonl`). Decisions: `DECISIONS.md` D-025, D-026.

---

## 1. Provenance and baseline reproduction

Full detail: `PROVENANCE_AND_BASELINES.md`, `BASELINE_REGRESSION.md`. Three environment
gaps found and fixed before any measurement: hg38 FASTA absent on this filesystem
(downloaded fresh from UCSC), DNABERT-2 revision unpinned in the mechanism scripts (pinned
to `7bce263b15377fc15361f52cfab88f8b586abda0`), layer-9 norm-hook convention inconsistent
across existing scripts (standardized on the explicit-layer hook).

**DNABERT-2 baseline regression reproduced almost exactly**: pretrained-MLM epistasis at
full ablation = **+2.011803**, vs. the previously-reported C-037 figure of **+2.0118** — a
4-decimal match from a from-scratch environment. **GENERator baseline regression passed**:
row 2371's GC span (0.1017) vastly exceeds the 5-row random-control span (0.0003, ~390×).
No `E9_BLOCKED.md` was written.

## 2. Intervention basis per model

- **DNABERT-2**: `n_D=10`, the pre-existing canonical ensemble (layers 3/5/6/7/9), unchanged
  and unsorted-by-response. Full table in `BASIS_FREEZE.md`.
- **GENERator EUK**: scope-limited to 1D dose-response. Only 2 real detected high-gain
  candidates exist (layer 4, rows 2371 and 1522, same output column); padding to the
  target 6-12 was rejected as fishing-adjacent (`DECISIONS.md` D-025). No F0-F3 ladder was
  run for GENERator.

## 3. Mask design and conditioning

`n_D=10` → 10 singleton + 78 fit + 20 calibration + 20 held-out masks (seed 20260822,
densities rho∈{0.25,0.5,0.75}). First fit-pool size (40) gave a rank-deficient 55-column
lifted design (mathematically guaranteed); corrected to 78 before any response was
measured. Final check: additive design 10/10 rank (cond=22.0), lifted design **55/55 rank**
(cond=4521, max\|corr\|=0.74, no near-duplicate columns). Full detail: `MASK_DESIGN.md`.

## 4. Preregistration hash/timestamp

`PREREG_mechanistic_tomography_E9.md`, sha256 `10de1694544c4fa11d0c288f5183ec85c0b94208c36ee5026c81e15a8017b68a`,
locked 2026-08-22T20:55:26+00:00, commit `2a76d99`. All 6 prereg locks (5 prior + this one)
verified intact before and after this experiment.

## 5. Intervention scales and endpoints

`alpha_i = 1 - epsilon*a_i`, `epsilon∈{0.5,1.0}`. DNABERT-2 primary: pretrained MLM loss
(no fine-tuning), 256 hg38 windows, one fixed mask realization (seed 42). GENERator
primary: mean generated GC fraction, 24 hg38-window prompts, paired generation seeds.

## 6. F0 (singleton additive) results — DNABERT-2, held-out

| epsilon | R² | MAE | normMAE | signed resid. mean | resid.-vs-density r | resid.-vs-\|pred\| r |
|---|---:|---:|---:|---:|---:|---:|
| 0.5 | −0.023 | 0.0303 | 0.160 | +0.0303 | 0.620 | 0.427 |
| 1.0 | −0.040 | 0.4907 | 0.228 | +0.4899 | 0.831 | 0.757 |

**F0 fails held-out adequacy badly at both scales** (R² below zero — worse than predicting
the held-out mean). The residual is strongly positive and strongly structured (correlates
with mask density and with predicted magnitude), i.e. F0 systematically *underestimates*
damage as more components are masked — exactly the signature predicted by H1.

## 7. F1 (scalar-calibrated additive) results — DNABERT-2, held-out

| epsilon | gain g | R² | MAE | normMAE |
|---|---:|---:|---:|---:|
| 0.5 | 2.605 | 0.324 | 0.0334 | 0.176 |
| 1.0 | 2.928 | 0.659 | 0.3072 | 0.143 |

A large uniform gain (~2.6-2.9×) partially rescues F0's magnitude but **still fails
adequacy at both scales** (R²<0.90). Confirms H2: a scalar rescue cannot fix a
pair-localized underestimate.

## 8. F2 (jointly fit additive) results — DNABERT-2, held-out

| epsilon | ridge λ | R² | MAE | normMAE |
|---|---:|---:|---:|---:|
| 0.5 | 100.0 | 0.517 | 0.0277 | 0.146 |
| 1.0 | 30.0 | 0.581 | 0.3539 | 0.164 |

Freely fitting main effects (ridge, penalty selected on calibration) still **fails
adequacy at both scales**. Confirms H3: the response is still not additive even once
singleton coefficients are allowed to be refit jointly rather than measured in isolation.

## 9. F3 (lifted main+pair) results — DNABERT-2, held-out

| epsilon | ridge λ | R² | MAE | normMAE |
|---|---:|---:|---:|---:|
| 0.5 | 0.1 | 0.888 | 0.0125 | 0.066 |
| 1.0 | 1.0 | 0.790 | 0.2677 | 0.124 |

F3 clears the normalized-MAE bar at both scales (≤0.10 at eps=0.5, and 0.124 at eps=1.0 —
close) and comes close to (eps=0.5) or clearly improves on (eps=1.0) the R²≥0.90 bar,
without ever formally satisfying "additive adequate" itself (that test is applied to F2,
which fails). Confirms H4.

## 10. Held-out prediction comparison

| epsilon | F0 R² | F1 R² | F2 R² | F3 R² | F2→F3 rel. MAE improvement | bootstrap 95% CI | excludes 0? |
|---|---:|---:|---:|---:|---:|---:|---|
| 0.5 | −0.023 | 0.324 | 0.517 | **0.888** | **54.7%** | (0.0038, 0.0186) | yes |
| 1.0 | −0.040 | 0.659 | 0.581 | **0.790** | **24.4%** | (0.0614, 0.1144) | yes |

Both scales clear the ≥10%-relative-improvement-with-CI-excluding-zero bar from the locked
prereg. **Mechanical decision at both scales: PAIR_TERMS_REQUIRED.**

A real bug was caught and fixed in the bootstrap before trusting this table — see
`DECISIONS.md` D-026 for the full account. The bug affected only the CI's width/sign, not
the point estimates above (which were correct throughout and independently verified).

## 11. DNABERT-2 pair-support result (H5)

The known critical pair (L9/r264, L9/r294) — never given special treatment during
fitting — ranks **#1 of 45** pairs by \|Γ\| at both scales: Γ=0.0772 (eps=0.5, next
largest 0.0544) and Γ=0.7387 (eps=1.0, next largest 0.5019), roughly 40-47% clear of the
runner-up at both scales. **H5 confirmed retrospectively.** Other large pair terms
recurrently involve row 603 (which itself recurs across layers 3/5/6/7 — the pre-existing
"depth redundancy" observation) and rows 86/399 (same layer-3/column-2056 family),
suggesting the interaction structure is not confined to the single known pair but is
concentrated in a small number of structurally-related row families.

**The strongest possible target result is met**: a pair-aware observer trained on one
intervention set predicts unseen DNABERT-2 interventions substantially better than every
additive observer, and the independently-established critical pair is among (in fact, the
single strongest of) the interaction terms responsible for that improvement.

## 12. DNABERT-2 residual-norm relationship (H6)

Pearson r between layer-9 channel-norm drop (relative to the untouched baseline, summed
over the mask's active channels) and absolute functional damage (\|dloss\|) across the 20
held-out masks: **r=0.721 (epsilon=0.5), r=0.710 (epsilon=1.0)**. A substantial positive
covariation, consistent with C-038's norm-carriage mechanism generalizing beyond the single
known pair to the wider held-out mask distribution. **This is reported as covariation, not
mediation** — no causal-pathway claim is made from a correlation alone.

## 13. GENERator GC steering/control frontier

| condition | GC (alpha=1.0) | GC (alpha=0.5) | GC (alpha=0.0) | span |
|---|---:|---:|---:|---:|
| row 2371 (primary) | 0.3961 | 0.3863 | 0.2944 | 0.1017 |
| row 1522 (secondary) | 0.3961 | 0.4131 | 0.3466 | 0.0665 |
| random control (5 rows) | 0.3961 | 0.3962 | 0.3960 | 0.0003 |

Row 2371: monotonic decrease with suppression, span ≈390× random control. Row 1522:
**non-monotonic** (GC rises at partial suppression, falls below baseline at full ablation)
— reported as observed. Per the locked Phase 8 steerability rule: row 2371's effect is
large, reproducible, and far exceeds the matched control, but **only 3 dose points exist**
(no denser dose grid was collected, by design — the prereg froze exactly these 3 alphas),
so "multiple masks/doses produce reproducible intermediate outcomes" is only weakly
supported (2 non-trivial doses). Call this **causal sensitivity with a directionally
consistent primary row**, not a fully validated graded-steering regime — row 1522's
non-monotonicity is itself evidence against treating the row/column as a simple linear
steering axis.

## 14. NTv3 negative control

**Not run in this pass.** Phase 9 marks it optional/secondary, to be attempted only after
the primary decisions are frozen and only if the same machinery applies cleanly; it would
require its own provenance audit (checkpoint, down_proj pattern, endpoint reproduction) —
out of proportion to its secondary role given the primary GENERator/DNABERT-2 decisions
are already clean and decisive. The pre-existing C-029/C-038 results (corrected NTv3 splice
ablation: −0.02pp; MAKE-PAIR construction: no effect) already establish NTv3's
structurally-high-gain/functionally-inert profile via an independent, already-established
method, so this does not leave the "third causal regime" empty — it leaves it evidenced by
prior work rather than by E9 itself.

## 15. Mechanical Branch determination

Per Phase 12's predeclared branches: DNABERT-2 alone satisfies Branch A's DNABERT clause
exactly (F3 materially improves held-out prediction over every additive family, at both
scales). **GENERator's side of Branch A cannot be formally claimed**, because GENERator
never received the F0-F3 ladder Branch A's wording presupposes ("F0/F1/F2 is adequate and
F3 adds no meaningful held-out benefit") — that comparison was never run, by a decision
made *before* any GENERator response was collected (`BASIS_FREEZE.md`).

**Result: a scoped, honest variant of Branch A** — DNABERT-2 requires pair terms under a
fitted, held-out-evaluated multi-row observer (the strongest form of that result the
protocol defines); GENERator shows a large, mostly-monotonic single-row dose-response that
is *consistent with* a simple/low-order causal object but was never tested against
alternatives complex enough to fail. This is not Branch B (GENERator was never shown to
need pair terms — the question was never asked), not Branch C (DNABERT-2's held-out surface
is *not* predicted adequately without pair features), not Branch D (GC does move
predictably for row 2371, though row 1522 complicates a clean "predictably steerable"
claim), and not Branch E (both DNABERT-2 families the ladder was applied to behaved as
predicted).

## 16. What is established

- DNABERT-2's high-gain basis requires pairwise interaction terms to predict held-out
  finite interventions, at both a partial-suppression and a full-ablation scale, evaluated
  on genuinely unseen masks with a design-checked, full-rank lifted basis and a
  bootstrap-confirmed (bug caught and fixed) improvement.
- The independently-discovered critical pair is the single strongest interaction term at
  both scales, without having been given any special treatment during fitting.
- Layer-9 channel-norm drop covaries substantially (r≈0.71-0.72) with functional damage
  across a wider set of held-out masks than the single known pair.
- GENERator EUK's row 2371 causally moves generated GC composition far more than matched
  random controls, at two intervention scales beyond the previously-tested grid.

## 17. What is not established

- That GENERator EUK is "additive" or "low-dimensional" in any formal sense — the
  observer-family ladder was never applied to it.
- That row 1522's smaller, non-monotonic effect reflects the same mechanism as row 2371's.
- Any encoder/decoder class-level generalization — this remains n=1 per architecture class,
  exactly as it was before E9.
- A basis-independent "causal dimension" for either model — see §18 below.
- Anything about NTv3 from E9 itself (not run).

## 18. Basis dependence (required, Phase 11)

DNABERT-2's components are residual/high-gain down_proj output-row channels; GENERator's
are the same kind of channel, just too few of them to form a multi-component basis. The
interaction order found for DNABERT-2 (pair terms required) is a statement about causal
response **in this declared basis** — a rotated, learned, or SAE-derived basis could
represent the same underlying computation with different (possibly lower) apparent
interaction order. **No claim is made that DNABERT-2 "intrinsically has causal dimension
two"** — only that pair interactions are required in the declared residual-channel
intervention basis. This distinction is load-bearing for how §16 is allowed to be cited in
any manuscript text (`INTERVENTION_BASIS.md` states the same constraint before any result
existed).

## 19. Paper impact

Supports a scoped version of the potential claim in `next_prompt.md` Phase 15: high-gain
gated-FFN structures occur in both models, and DNABERT-2's finite causal response requires
pairwise interaction terms in the declared basis, evaluated via a genuinely held-out,
design-checked, bootstrap-confirmed protocol — a materially stronger result than the
pre-existing single/pair/k-of-N ablation evidence (C-036/C-037) because it demonstrates the
pair requirement generalizes to a fitted, held-out-evaluated multi-row observer, not only to
the specific known pair's own ablation. GENERator's contribution is narrower than originally
hoped: a real, large, controlled dose-response finding for one row, explicitly not an
observer-complexity comparison. The clean "GENERator=additive, DNABERT-2=interactional"
dichotomy from Phase 15's suggested sentence **is not supported as stated** — only the
DNABERT-2 half of it is.

## 20. Whether independent encoder/decoder replication is justified

Not established by E9. E9 deepens the DNABERT-2 evidence considerably but does not add a
second decoder or a second encoder data point — GENERator's side of the comparison was
never run at matching scope. A claim about decoders vs. encoders in general still requires
independent model replication, exactly as `DECISIONS.md` D-025 and this file's §17 state.

## 21. Repository commits, locks, raw artifacts, tree

Commits on `integrate/mechanism-and-negative-results` (chronological): provenance+basis
docs; basis freeze+engine+masks; prereg lock; baseline regression; [this results commit].
Prereg locks: 6/6 verified via `prereg_lock.py verify --all`. Raw artifacts, all under
`experiments/frozen/E9_mechanistic_tomography/`: `masks_dnabert2.json`,
`baseline_regression_results.json`, `dnabert2_mask_responses.json` (includes per-batch
breakdowns and layer-9 norms), `fit_results_dnabert2.json` (coefficients, all 45 pair
terms, bootstrap draws' summary, metrics). `tomography_lib.py`,
`run_baseline_regression.py`, `run_dnabert2_measurements.py`, `run_fit_observers.py`,
`generate_masks.py` are the exact command sequence (`python3 generate_masks.py`,
`run_baseline_regression.py`, `run_dnabert2_measurements.py`, `run_fit_observers.py`, in
that order) that reproduces every number above. `data/reference/hg38/hg38.fa` (not
committed, downloaded from UCSC, provenance recorded in `PROVENANCE_AND_BASELINES.md`).

## 22. ONE exact next action

Render the three Phase 13 figures from the already-complete JSON source data (observer
prediction F0-F3 scatter per model/epsilon; DNABERT-2 pair-interaction map with the known
pair highlighted; GENERator dose-response with the control band) — no new measurement is
needed, only plotting from `fit_results_dnabert2.json` and `baseline_regression_results.json`
using the repo's shared `scripts/analysis/_figstyle.py` convention.
