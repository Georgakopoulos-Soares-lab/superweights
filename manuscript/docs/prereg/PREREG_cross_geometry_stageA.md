# PREREG — E6 Stage A: independent confirmation of exact cross-pathway interaction geometry

**Lock before generating any confirmatory number from a real confirmation-panel weight:**
`python3 src/prereg_lock.py lock docs/prereg/PREREG_cross_geometry_stageA.md`

**Verify the lock before running the confirmatory script:**
`python3 src/prereg_lock.py verify docs/prereg/PREREG_cross_geometry_stageA.md`

Synthetic-tensor tests (`experiments/E6_cross_geometry/test_cross_geometry_lib.py`) and a
regression check reproducing E5's own six already-published `f_cross` values
(`experiments/E6_cross_geometry/validate_against_e5.py`) may be run before this file is
locked — neither uses a confirmation-panel candidate row. No tensor from any panel candidate
listed below may be passed through this experiment's code before this file is locked.

---

## Disclosure (read before anything else)

**This hypothesis was generated after observing E5's six-row cross-term split.** E5 found
`f_cross` (fraction of the exact squared row norm the diagonal `c_{k,i}` decomposition misses)
at 0.067–0.193 for its three NLP rows and 0.207–0.829 for its three genomic rows — see
`experiments/E5_dimensionality/GATE0_RESULTS.md`. **Those six rows are discovery examples.
They are excluded from E6's confirmatory primary endpoint below** (though, per the governing
instruction, they are reused separately as an implementation regression check — reproducing
already-published numbers, not new evidence — in `validate_against_e5.py`, which is not part
of the locked confirmatory run).

## Premise

E5 established, and this prereg does not re-test: the diagonal `c_{k,i}` decomposition
underlying this repository's existing granularity claims (C-002/C-003/C-032) is a
substantially worse approximation of the true exact bilinear amplifier for the genomic side of
E5's panel than for the NLP side. E5 correctly stopped before asking *why*, or whether that
gap holds outside its own six discovery rows. This is what E6 Stage A tests.

## Candidate hypothesis (not established)

> Canonical NLP super-weight rows are approximately coordinate-separable: most exact
> gated-FFN amplifier strength is explained by individual self-terms. Genomic high-gain rows
> can instead derive substantial strength from coherent cross-pathway interactions among
> intermediate FFN coordinates.

## Primary scientific questions

- **Q1 (independent group replication).** Do independently predetermined NLP high-gain rows
  show systematically lower cross-term dependence (`f_cross`) than independently
  predetermined clean genomic high-gain rows?
- **Q2 (candidate specificity).** Within a target layer, is the predetermined high-gain row
  unusually cross-interaction-heavy compared with the layer's other output rows, or is high
  `f_cross` an ordinary property of that layer generally?
- **Q3 (nature of the cross mass).** Where cross terms are large, are they predominantly
  constructive/coherent, or large positive/negative terms that mostly cancel?

## Confirmation panel (fixed; copied verbatim from
`experiments/E6_cross_geometry/CONFIRMATION_PANEL.md`, frozen before any cross-term value was
computed — see that file for full provenance, exclusion reasoning, and the hard-stop check)

| # | Group | Model | Checkpoint | Layer | Row *k* |
|---|---|---|---|---|---|
| 1 | NLP | OLMo-7B | `allenai/OLMo-7B-0724-hf` | 2 | 269 |
| 2 | NLP | OLMo-7B | `allenai/OLMo-7B-0724-hf` | 7 | 269 |
| 3 | NLP | OLMo-7B | `allenai/OLMo-7B-0724-hf` | 24 | 269 |
| 4 | Genomic | DNABERT-2 | `zhihan1996/DNABERT-2-117M` @ `7bce263b15377fc15361f52cfab88f8b586abda0` | 3 | 86 |
| 5 | Genomic | DNABERT-2 | (same) | 3 | 399 |
| 6 | Genomic | DNABERT-2 | (same) | 3 | 603 |
| 7 | Genomic | DNABERT-2 | (same) | 3 | 641 |
| 8 | Genomic | DNABERT-2 | (same) | 5 | 86 |
| 9 | Genomic | DNABERT-2 | (same) | 6 | 603 |
| 10 | Genomic | DNABERT-2 | (same) | 7 | 603 |
| 11 | Genomic | DNABERT-2 | (same) | 9 | 264 |
| 12 | Genomic | DNABERT-2 | (same) | 9 | 294 |
| 13 | Genomic | GENERator EUK | `GenerTeam/GENERator-v2-eukaryote-3b-base` @ `7dc01bccce5b65e15141170538afdc2ff09d8dde` | 4 | 1522 |

**No row is added to or dropped from this table after this file is locked.**

### A disclosed asymmetry in this panel — stated now, not after seeing results

The three NLP candidates are **not three independent models**: they are the same model
(OLMo-7B), the same output row (269), at three different layers — the same recurring
super-weight *channel* Yu et al.'s own table records across layers 1, 2, 7, 24 (layer 1 is the
E5 primary, excluded). This is weaker independence than three different NLP architectures
would give, and it is disclosed here, before any measurement, precisely so it cannot be
raised only if the result is unwelcome. The genomic side has two models (DNABERT-2, GENERator
EUK) and substantially more within-model layer diversity (DNABERT-2 spans layers 3, 5, 6, 7,
9). Q1's decision rule below is deliberately stringent (a multiplicative margin, not just
direction) partly to compensate for the small, non-independent NLP arm — see "Why these
specific thresholds" below.

## Exact formulas (memory-safe; see `cross_geometry_lib.py` for the tested implementation)

Pair-index convention: **ordered pairs**, every unordered `{i,j}` counted twice (once as
`(i,j)`, once as `(j,i)`), so that `sum_{i,j} X[i,j]` (all pairs including `i=j`) equals the
exact quadratic form with no separate factor-of-2 bookkeeping anywhere downstream.

```
K[i,j]  = (W_gate[i,:] . W_gate[j,:]) * (W_up[i,:] . W_up[j,:])          [d_ffn, d_ffn]
X[i,j]  = W_down[k,i] * W_down[k,j] * K[i,j]                              (for fixed row k)

S_exact^2 = sum_{i,j} X[i,j]              S_diag^2 = sum_i X[i,i]
f_cross   = (S_exact^2 - S_diag^2) / S_exact^2                    -- SIGNED, not clipped

X_+ = sum_{i != j} max(X[i,j], 0)          X_- = sum_{i != j} min(X[i,j], 0)     (X_- <= 0)
x_pos_norm = X_+ / S_exact^2               x_neg_norm = X_- / S_exact^2
gross_cross_norm = x_pos_norm - x_neg_norm       (= |x_pos_norm| + |x_neg_norm|)

kappa = (X_+ + X_-) / (X_+ + |X_-|)        -- the ONE preregistered coherence statistic.
        Range [-1, +1]. +1 = all cross mass constructive, -1 = all destructive,
        0 = exact cancellation of gross mass, regardless of gross magnitude.
        Undefined ("undefined (no cross mass)") when X_+ = X_- = 0 exactly.
```

**Identity checked at runtime, not just in tests:** `x_pos + x_neg` must equal
`S_exact^2 - S_diag^2` to float64 precision; `cross_geometry_lib.row_cross_geometry` asserts
this on every call.

**Layer-wide reference distribution** (Task 2F): `f_cross` computed for every output row of
the candidate's own layer, reusing E5's already-validated `exact_uk_all_rows` (weight-only,
same cost class already proven fast enough for every model in this panel) and
`uk_frobenius.uk_frobenius` for the diagonal. Candidate position within that distribution is
reported as (a) percentile, (b) a robust MAD-based z-like score, using the identical
convention E5's `AE_*` statistics already use, for methodological consistency across the two
experiments — not a new invented metric.

## Primary endpoints (exactly three; no more are added after locking)

1. **`f_cross`** for each of the 13 candidate rows.
2. **Candidate's percentile of `f_cross` within its own layer's full row distribution.**
3. **`kappa`** (the one preregistered coherence statistic) for each of the 13 candidate rows.

Row rank under the exact form is computed (it is a near-free byproduct of the layer-wide
distribution already needed for endpoint 2) and reported, but is **secondary** — not part of
the decision rule.

## Group-level decision rule (mechanical; stated in full before any confirmation-panel weight
is loaded)

Applied as an ordered, three-step decision tree. Each step's failure assigns a branch and
**stops** the tree — later steps are not evaluated once a branch is assigned.

### Step 1 — Q1: replication (direction + effect size, not just direction)

Let `NLP_max = max(f_cross)` over the 3 NLP candidates, `GEN_min = min(f_cross)` over the 10
genomic candidates.

**PASS** iff both:
- **Complete separation:** every genomic candidate's `f_cross` exceeds every NLP candidate's
  `f_cross` (`GEN_min > NLP_max`).
- **Effect-size margin:** if `NLP_max > 0`: `GEN_min >= 1.5 * NLP_max` (the weaker group's
  best case must fall at least 50% short, in relative terms, of the stronger group's worst
  case). If `NLP_max <= 0` (fallback for a non-positive ceiling, where a relative multiplier
  is undefined): `GEN_min >= NLP_max + 0.10` (an absolute 0.10 margin on the `f_cross` scale).

**On FAIL → Branch C** (no independent replication). Stop. Do not proceed to Step 2.

The exact rank-sum p-value (via the same enumerated, non-asymptotic test E5 used, generalized
to unequal group sizes n=3 and n=10, `C(13,3) = 286` total arrangements) is computed and
reported alongside this decision for transparency, but the **mechanical decision is the
margin rule above, not the p-value** — consistent with the governing instruction not to rely
on asymptotic (or even exact-but-marginal) significance alone for a result this small.

### Step 2 — Q2: candidate specificity (only if Step 1 passes)

**PASS** iff the **median** percentile (endpoint 2) across the 10 genomic candidates is
**>= 75.0** — i.e., the typical genomic candidate sits in the top quartile of its own layer's
`f_cross` distribution, not merely high in absolute terms because its whole layer runs hot.

**On FAIL → Branch B** (layer-wide architectural property, not row-specific). Stop. Do not
proceed to Step 3.

### Step 3 — Q3: coherence (only if Steps 1 and 2 both pass)

**PASS** iff the **median** `kappa` across the 10 genomic candidates is **>= +0.3**
(constructive cross mass outweighs destructive cross mass by roughly 1.86:1 or more at that
threshold — a meaningful constructive skew, not merely `kappa > 0`).

**On FAIL → Branch D** (cancellation-dominated). Stop. Do not call this "cooperative
amplification."

**On PASS → Branch A** (clean confirmation).

### Why these specific thresholds (stated now, not fit to any confirmatory result)

- **1.5x / +0.10 margin (Step 1):** a "large, interpretable separation" appropriate to a
  tiny, partly non-independent sample, per the governing instruction. A 50% relative gap (or
  a 0.10 absolute one) is substantially more demanding than requiring only that group medians
  differ.
- **75th percentile (Step 2):** "meaningfully unusual within its own layer" — the top
  quartile is a standard, legible bar for "notably elevated" that does not require an extreme
  outlier (e.g., 95th percentile), which architectural/layer-size variance across two very
  different genomic model families (a SwiGLU decoder and a GLU BERT encoder) does not
  obviously warrant demanding.
- **kappa >= 0.3 (Step 3):** on the `[-1, +1]` scale, `kappa = 0.3` implies a constructive-to-
  destructive mass ratio of `(1+0.3)/(1-0.3) ~= 1.86`, a genuine skew rather than a
  coin-flip-close balance.

None of these three numbers was chosen by looking at any confirmation-panel `f_cross`,
percentile, or `kappa` value — Task 1 and Task 2's synthetic validation were both completed,
and this file was drafted, before any panel weight was loaded.

## Branch interpretations (preregistered; stated before the run)

- **Branch A — clean confirmation.** Independent rows reproduce substantially stronger,
  layer-specific, and predominantly constructive cross-pathway dependence in genomic
  high-gain rows. Permitted interpretation: *"coordinate-separability differs between these
  tested canonical NLP and genomic high-gain rows."* No causal claim. Triggers Task 6 (a
  design-only causal-feasibility audit — no causal experiment is run in this session even on
  Branch A).
- **Branch B — layer-wide architectural property.** High `f_cross` is ordinary for the
  candidate's own layer. Permitted interpretation: *"the cross-term difference reflects
  broader FFN geometry rather than a high-gain-row-specific mechanism."* Does not explain
  causal granularity specifically. Stop.
- **Branch C — no independent replication.** The held-out panel does not reproduce E5's
  discovery-panel split. Permitted interpretation: *"E5's apparent group separation does not
  generalize."* Stop.
- **Branch D — cancellation.** Net `f_cross` differs as predicted, but the underlying mass is
  dominated by large, mostly-cancelling positive and negative terms. Do not call this
  cooperative amplification. Stop.

## What even a Branch-A pass does NOT establish (stated now)

Why training produced this geometry; that genomic sequence data causally produces it; that
cross terms cause functional criticality; any specific interacting pair/group is causal; or
any external collaborator-reported mechanism (e.g. a DNABERT-2 "redundant pair" — no artifact
for that exists in this repository, per `CLAUDE.md` and `NEW_DIRECTION_EVIDENCE_AUDIT.md`,
and E6 does not change that).

## No post-hoc changes

No metric, threshold, or panel row is added, dropped, or altered after this file is locked or
after any confirmation-panel weight is loaded. A correction is a new, separately-locked file
that discloses both, per this project's standing discipline.
