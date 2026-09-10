# Integrated report — okstillnotlast.md (Priorities 1–3)

Manuscript **not** modified. No claims added to `CLAIMS_LEDGER.md` / `DECISIONS.md` — that is
a follow-up step if/when the author accepts any of this into the manuscript.

Raw artifacts:
- Priority 1: `results/E13/priority1_robustness/priority1_robustness_results.json`
- Priority 2: `results/E13_candidate_stability/*.json` (20 files)
- Priority 3: `results/E_BOS_MEDIATION/{provenance_check,smoke_tests,bos_mediation_results}.json`

---

## 1. Priority-1 robustness results

**Primary cohort definition (grounded in artifact provenance, not asserted):** of the 22-model
causal cohort (Phi-3 excluded, matching `part1_22_structure_function_correlations.json`), 16
models' E13 candidate coordinate was originally found by the activation-based detector
(`candidate.ratio >= 5.0` in the E7/E8/E11 per-model detection JSONs). The other 6 — Llama-7B,
Mistral-7B, OLMo-7B-0724-hf, NTv3, DNABERT-2, GENERator-EUK-3B — are sourced from
`results/E7/e7_legacy_reanalysis.json`, which carries published/structural coordinates and has
**no activation-ratio field at all**. This is why a 22-model activation-ratio correlation
cannot be built; the 22-model number reported below is a correlation on the adjacent
*structural* ratio, already computed before this session, used here only as sensitivity
context.

**Main analysis — activation ratio vs. signed full-ablation relative-loss effect, ε=1.0, n=16:**
ρ = 0.685, asymptotic p = 0.0034, bootstrap 95% CI [0.265, 0.924].

| Check | Result |
|---|---|
| Leave-one-model-out (16 runs) | ρ range **0.618–0.782**; every single-model removal stays positive and nominally significant. Most influential: **ModernBERT-base** (Δρ = +0.097 on removal), i.e. removing it strengthens the correlation slightly — no model is propping it up. |
| Leave-one-family-out | Qwen2.5 (n=4 removed): ρ=0.657, p=0.020. SmolLM2 (n=3 removed): ρ=0.544, p=0.055 (**crosses the conventional 0.05 line**). EuroBERT (n=3 removed): ρ=0.742, p=0.0037. GENERator-prok (n=2 removed): ρ=0.618, p=0.019. ModernBERT (n=2 removed): ρ=0.776, p=0.0011. Two singleton removals (MosaicBERT, GenomeOcean-4B): ρ=0.654/0.686, both p<0.01. |
| vs. candidate-minus-**random**-control gap, ε=1.0 | ρ = 0.685, p = 0.0034 (identical to main — random-position control effects are ~1e-6, negligible relative to candidate effects, so this doesn't add information beyond the main analysis). |
| vs. candidate-minus-**top-norm**-control gap, ε=1.0 | ρ = 0.294, p = 0.269, CI crosses 0 [-0.254, 0.706]. **Not significant.** |
| Repeat at ε=0.5 | ρ = 0.409, p = 0.116, CI crosses 0 [-0.176, 0.814]. **Not significant.** |
| Sensitivity only — abs(effect) magnitude, ε=1.0 | ρ = 0.685 (identical to signed; all 16 signed effects are already positive or negligibly negative, so sign flips don't reorder). Reported for completeness only, per instruction — not used as headline. |
| 22-model structural-ratio sensitivity (pre-existing) | ρ = 0.441, p = 0.040, CI [0.062, 0.717], n=22 — weaker than the 16-model activation-ratio result, still nominally significant. |

### 2. Does the association survive exclusions?

**Partially.** It is robust to *any single model or family* removal at ε=1.0 against a
random-position control (stays positive, p<0.06 in every case, borderline only on the
SmolLM2 family). It is **not** robust to two other reasonable perturbations of the same
question: (a) swapping the comparator from a random-position control to a same-layer
**high-norm** control, and (b) halving the intervention strength to ε=0.5. Both of those
knock the correlation below conventional significance. This is a real fragility, not a
rounding issue — the CI in both cases crosses zero comfortably.

**Verdict: SUPPORTING / SUPPLEMENTARY, not a strong headline claim.** Recommend describing it
(if used at all) as "activation ratio is associated with full-ablation severity at the higher
perturbation strength relative to random-position controls, but this association is not
robust to a same-layer high-norm-control comparison or to a lower perturbation strength" —
never "predicts."

---

## 3. Priority-2 candidate-stability results (multi-input)

All 20 feasible models, n=24 independent inputs each, frozen candidate coordinate never
reselected:

| Result | Value |
|---|---|
| frac_rank1 across all 20 models | **1.000** (candidate is the single most-extreme-activation row on every one of 24 inputs, every model) |
| frac_top1pct across all 20 models | **1.000** |
| percentile_median across all 20 models | **100.0** |

**Caveat worth flagging explicitly:** for 5/20 models — DNABERT-2, GENERator-EUK-3B,
GENERator-prok-1.2b, GENERator-prok-3b, MosaicBERT — the activation ratio is **bit-identical
across all 24 inputs** (IQR = [x, x], min = max). Cross-referencing `max_position_at_0_fraction`
confirms this is because these models' extreme activation fires at a fixed structural position
(token 0 / BOS) regardless of sequence content — "stability" for these 5 is a structural
invariant, not evidence that the signal tracks varying content. The remaining 15 models show
genuine input-to-input variation in the ratio's magnitude (real IQR) while still landing at
rank-1/top-1% on every input — that is the stronger, more informative form of the result.

**Verdict: STRONG MAIN-TEXT RESULT.** Activation extremeness is not a single-probe artifact for
any of the 20 models tested — with the explicit caveat that "input-stable" means two different
things across the cohort (content-invariant-by-construction for 5 models vs. genuinely
content-robust for 15), and the manuscript should say so rather than presenting one
undifferentiated number.

---

## 4/5. Priority-3 validation checks and exact NLL/GC/rescue results (GENERator-EUK-3B, L4/r2371)

**Provenance gate: PASSED.** Measured intact NLL 6.38538 vs. historical 6.385381; measured
ablated NLL 8.75432 vs. historical 8.75432 (rtol=0.001). Priority 3 was cleared to proceed.

**All 6 required smoke/invariance tests: PASSED** (no-op≡intact; hook-ablation≡weight-ablation;
restore-everywhere≡intact; BOS-only intervention touches only position 0; matched non-BOS
intervention touches only its target position; prefill vs. KV-cache decode both handled
correctly with zero positions touched during decode steps).

**Primary endpoint — native NLL** (bootstrap point estimates, n=100 damage windows):

| Condition | Mean NLL | vs. intact (A) | Interpretation |
|---|---|---|---|
| A. Intact | 6.385 | — | baseline |
| B. Full row ablation | 8.754 | +2.336 (CI [2.20, 2.49]) | full damage |
| C. Ablate row contribution at BOS only | 8.722 | ≈ B (rescue fraction 0.014, CI crosses 0) | **BOS-only loss reproduces essentially all of the full-ablation damage** |
| D. Ablate everywhere except BOS (preserve BOS only) | 6.386 | ≈ A (rescue fraction 0.9996, CI [0.998, 1.002]) | **preserving only BOS fully prevents damage** |
| E. Full ablation + restore intact BOS contribution | 6.386 | ≈ A (rescue fraction 0.9996, CI [0.998, 1.002]) | **restoring only BOS fully rescues native NLL** |
| F. Full ablation + restore at matched non-BOS position | 8.754 | ≈ B (rescue fraction 0.0003, CI crosses 0) | **no rescue — the effect is position-specific, not "any large injected activation"** |

Central contrasts, answered directly:
- **C vs. intact** — yes, BOS-only loss alone is sufficient to produce essentially the full
  damage magnitude.
- **D vs. full ablation/intact** — yes, preserving only BOS is sufficient to prevent
  essentially all damage.
- **E vs. full ablation** — yes, restoring only the intact BOS contribution after global
  ablation rescues native NLL almost completely (~100%, tight CI).
- **F vs. E** — confirmed: the rescue is BOS-specific. An equally large restoration at a
  matched non-BOS position gives ~0% rescue.

**Verdict: STRONG MAIN-TEXT RESULT**, scoped explicitly to this one model/coordinate
(GENERator-EUK-3B, L4/r2371) — do not generalize to "genomic decoders" broadly (n=1 model).

**Secondary endpoint — generated GC fraction** (n=24 prompts):

| Condition | Mean GC | Rescue fraction vs. B | 
|---|---|---|
| A. Intact | 0.419 | — |
| B. Full ablation | 0.309 | — |
| C. Ablate at BOS only | 0.292 | **-0.155** (CI [-0.306, -0.041], entirely below 0) |
| D. Preserve BOS only | 0.412 | 0.934 (CI [0.807, 1.093]) |
| E. Restore at BOS | 0.412 | 0.934 (CI [0.805, 1.097]) |
| F. Restore at matched non-BOS | 0.303 | -0.058 (CI crosses 0) |

D/E/F qualitatively match the NLL story (BOS-specific rescue, no non-BOS rescue). **Condition C
is an honest anomaly worth reporting, not smoothing over**: BOS-only ablation doesn't just fail
to rescue GC content, it pushes GC *below* the full-ablation level, with a CI that excludes
zero. NLL and GC dissociate at condition C specifically.

**Verdict: SUPPORTING / SUPPLEMENTARY** (secondary endpoint by design; directionally consistent
with the NLL mediation story on D/E/F; the C-condition GC anomaly should be reported as a
genuine open question, not resolved here).

**Attention-sink strength:** not recorded in this run (the instruction allowed skipping it
rather than delaying the core experiment) — **NULL / not attempted**, not claimed. No causal
manipulation of attention itself was performed, so no attention-sink causal claim is made,
consistent with the instruction.

---

## 6. Final recommendation for manuscript changes

1. **Priority 2 (input-stability) is ready to strengthen the manuscript now.** It directly
   supports the existing "structurally fired" language with a clean, well-powered (20 models ×
   24 inputs) result. Include the 5-model input-invariance caveat explicitly rather than
   presenting one undifferentiated "100% stable" number.
2. **Priority 3 (BOS mediation) is ready to enter as a new, scoped causal result** for
   GENERator-EUK-3B specifically — the cleanest mediation evidence produced in this audit
   (provenance-gated, 6/6 smoke tests passed, tight bootstrap CIs, position-specific rescue
   confirmed by the F-vs-E contrast). Frame it as single-model/single-coordinate evidence, and
   report the GC condition-C anomaly rather than omitting it.
3. **Priority 1 (activation-ratio ~ causal-effect association) should NOT be presented as a
   standalone headline claim.** It's a real, positive, leave-one-out/family-out-robust
   correlation at ε=1.0 against a random-position control (ρ=0.685, p=0.003, n=16) — but it
   collapses to non-significance both against a same-layer high-norm control and at ε=0.5. If
   used at all, present it as supplementary robustness-audit material with both failure modes
   stated alongside the positive result, using "associated with"/"tracks," never "predicts."
4. No manuscript files were touched. No `CLAIMS_LEDGER.md`/`DECISIONS.md` rows were added —
   add them in the same commit as whichever of the above the author decides to write into the
   manuscript, per this project's working discipline.
5. Priority 4 was not run, per instruction.
