# E10 Phase 5 — architecture synthesis

Written only after all seven models' primary results were frozen (last measurement: Qwen2.5,
2026-08-22T22:00 UTC). Governing protocol: `PREREG_E10_nlp_architecture_causal_v2.md`
(locked `3e4b991d71b38cfe15550a57c10c73323cb3b751ba3c768d6c95fc4989833a8a`), decoder arm per
`e10_prompt_correction.md`. No decoder tomography has been run — none is authorized under
this lock, and the author's launch instruction explicitly said to stop after this synthesis.

---

## Model table

| Model | Domain | Architecture | q1 (pre-E10, annotation) | High-gain intervention type | Primary causal result | Pair tomography run? | Simplest supported causal description |
|---|---|---|---:|---|---|---|---|
| Llama-7B | text | decoder, SwiGLU | 0.9888 | single row (K=1) | dNLL +300.8% at full ablation; controls null (~4e-5); non-monotonic vs. α=0.5 (+3.9%) | N/A (K=1) | **SINGLE_COMPONENT_DOMINANT** |
| Mistral-7B | text | decoder, SwiGLU | 0.9922 | single row (K=1) | dNLL +287.1%; controls null (~2e-5); α=0.5 (+301.4%) *exceeds* full ablation | N/A (K=1) | **SINGLE_COMPONENT_DOMINANT**, non-monotonic |
| OLMo-7B | text | decoder, SwiGLU | 0.9646 (canonical L1); 0.9990 (L24, structural top) | 4-row set (K=4) | C1=0.608; dominant row is L1 (+111.8%), **not** the structurally-largest L24 (+1.4%, weakest of the 4) | N/A (flagged eligible, not run — see below) | **SINGLE_COMPONENT_DOMINANT**, but dominance ≠ structural rank |
| Phi-3-mini | text | decoder, packed gate_up | 0.9028 (median) | 6-row set (K=6) | C1=0.333; top-2 rows nearly tied (+4.62%, +4.41%); all 6 clear controls by 69x-1045x | Flagged **MULTI_COMPONENT_CANDIDATE**; not run (see below) | Distributed causal contribution among structurally high-gain rows |
| Qwen2.5-7B | text | decoder, SwiGLU | 0.9529 | single row (K=1), project's own detector | dNLL +0.74%; ~360x control median; α=0.5 (+0.60%) directionally intermediate | N/A (K=1) | **SINGLE_COMPONENT_DOMINANT**, weak absolute magnitude |
| MosaicBERT | text | encoder, gated-FFN (DNABERT-2-class) | 0.4766 (canonical row) | top-10 structural basis | F2→F3 MAE improves 84.2% (ε=0.5) / 54.5% (ε=1.0); bootstrap CI excludes zero both scales | **Yes** | **PAIR_TERMS_REQUIRED** |
| ModernBERT | text | encoder, gated-FFN (independent codebase) | 0.8970 (canonical row, "thin margin" per E8) | top-10 structural basis | F2→F3 MAE improves 58.8% (ε=0.5) / 46.4% (ε=1.0); bootstrap CI excludes zero both scales | **Yes** | **PAIR_TERMS_REQUIRED** |

### Genomic anchors — descriptive only, not new E10 evidence

| Model | Domain | q1 | Prior causal characterization |
|---|---|---:|---|
| GENERator EUK | genomic | ~0.9689 | decoder exemplar, strong single-row causal composition sensitivity (C-039/C-040) |
| DNABERT-2 | genomic | ~0.7933 | encoder exemplar, pair-interactional causal response (C-036/C-037/C-038); PAIR_TERMS_REQUIRED at both scales (E9) |
| NTv3 | genomic | ~0.3889 | encoder negative case: structurally high-gain, causally null (C-029, corrected: −0.02pp / no effect) |

E10's new statistical evidence (the 7-model table above) is kept separate from this prior
genomic evidence throughout, per instruction.

---

## Per-decoder Step D4 decisions and whether tomography is justified

| Model | K | Decision | Tomography justified? | Run? |
|---|---:|---|---|---|
| Llama | 1 | SINGLE_COMPONENT_DOMINANT | No (K=1 edge case; STOP per rule) | No |
| Mistral | 1 | SINGLE_COMPONENT_DOMINANT | No | No |
| OLMo | 4 | SINGLE_COMPONENT_DOMINANT | No — rule says STOP on this decision regardless of *which* row dominates | No |
| Phi-3 | 6 | **MULTI_COMPONENT_CANDIDATE** | **Yes, flagged eligible** | **No** — prereg requires "a fresh, separate lock" before running F0-F3 on any decoder; the author's launch instruction was to stop after this synthesis, not add further experiments |
| Qwen2.5 | 1 | SINGLE_COMPONENT_DOMINANT | No | No |

Per the locked decision rule, OLMo's outcome is mechanically SINGLE_COMPONENT_DOMINANT
(C1=0.608 > 0.5, dominant row's control-normalized effect ~3.4x10^5 >> 3.0) — the rule does
not ask "does the causally-dominant row match the structurally-dominant row," so the
mechanical STOP applies even though the underlying finding (dominance sits on a
different row than the structural ranking predicts) is itself worth carrying forward
descriptively. This is not a rescue or a re-run of the decision; it is disclosure of what the
decision does and does not certify.

---

## Retrospective pair-structure inspection (encoders, AFTER the primary decision — descriptive only)

Per `e10_prompt.md`: run only after F0-F3 is frozen; never used to change the basis or refit.

**MosaicBERT** — no independently known critical pair exists pre-E10 (unlike DNABERT-2); none
is manufactured here. The single largest `|Gamma|` pair at ε=1.0 is **(L0/r287, L1/r287)**
(`Gamma=-3.22`) — both members share output row 287, one of the two structurally-recurring
rows in this basis (see `ENCODER_BASIS_FREEZE.md`). The canonical E8 detector row's best pair
rank is 9/45 (ε=1.0) — not the top pair, but well inside the top quintile.

**ModernBERT** — largest `|Gamma|` pairs cluster around output row 67 (`(L0/r67, L1/r67)`,
`Gamma=+3.18` at ε=1.0) and row 251/67 cross-pairs. The canonical row's best pair rank is
**3/45** (ε=1.0) — closer to the top than MosaicBERT's. At ε=1.0, the 3rd-strongest pair,
`(L15/r251, L0/r67)`, directly involves the canonical row.

Neither model's strongest pairs are dominated by same-layer combinations
(MosaicBERT: 20-40% same-layer among top pairs vs. 33% baseline rate; ModernBERT: 40-50% vs.
64% baseline) — if anything, the top pairs are *slightly less* same-layer-concentrated than
the full pair population, opposite of what a naive "adjacent-layer redundancy" story would
predict. This is reported descriptively; no mechanism claim is made from it.

---

## Answering the core (corrected) E10 question

> Does the existing structural encoder/decoder distinction correspond to a difference in the
> natural causal unit — single-component causal concentration in decoders versus
> interaction-dependent multi-component response in encoders?

**In this tested panel: partially yes, with one clean exception and one important internal
caveat.**

- **4 of 5 decoders** (Llama, Mistral, OLMo, Qwen2.5) mechanically resolve to
  SINGLE_COMPONENT_DOMINANT.
- **1 of 5 decoders** (Phi-3) does not — its causal effect is spread across at least 2-3 rows
  of comparable order (C1=0.333, well under the 0.5 threshold), and is flagged, not resolved,
  as a tomography candidate.
- **2 of 2 encoders** (MosaicBERT, ModernBERT) resolve to PAIR_TERMS_REQUIRED, at both
  intervention scales, replicating DNABERT-2's interactional phenotype independently in two
  architecturally distinct text encoders.
- **OLMo's internal result is a caveat that applies regardless of the aggregate pattern**: the
  row that dominates its causal response is not the row with the largest exact structural
  operator (‖U_k‖_F). Structural magnitude ranking and causal effect ranking dissociate
  *within a single model's own frozen row set*. This is the single clearest piece of evidence
  in this experiment that structural high-gain ranking and causal importance ranking are
  measuring different things, even when both point to "this model has a small single dominant
  causal pathway."

The supported paper-level statement, matching the corrected framing's own predeclared
language:

> In the tested panel, decoder high-gain systems tend to exhibit single-component causal
> concentration (4/5 decoders), whereas encoder high-gain systems require interactions among
> multiple structurally selected components (2/2 encoders) — but this is neither universal
> (Phi-3 is a counterexample on the decoder side) nor does structural ranking reliably predict
> *which* component will be causally dominant when concentration does occur (OLMo).

---

## What this does NOT establish

Per both `e10_prompt.md` and `e10_prompt_correction.md`, none of the following may be written,
and none is claimed here:

- "All decoders are single-component" — false on this panel's own data (Phi-3).
- "All encoders are interactional" — n=2, both agree, but this is not universal.
- "q1 causes causal concentration" — no such claim is testable from this design, and Phi-3
  (lower median q1 than Llama/Mistral) still shows *weaker* concentration, if anything the
  wrong direction for a naive q1-causes-concentration story, though n is far too small to
  treat this as evidence either way.
- "Architecture determines mechanism universally."
- "The decoder arm demonstrates additivity" — it was never designed to test pairwise
  interactions; a null multi-component finding on the decoder side would not by itself
  establish additivity, and no decoder in this panel produced a clean null.
- Any claim from the OLMo dissociation beyond what was measured — it does not imply the
  structural criterion is wrong (it correctly identifies operator *magnitude*, a different
  quantity from causal effect *on this endpoint*), nor does it generalize to other models.

---

## Statistics discipline note

Per the prereg: the primary unit for architecture-level synthesis is the MODEL. With 5
decoders and 2 encoders, the counts above (4/5, 1/5, 2/2) are reported as descriptive
model-level counts, not as a powered statistical test, and no p-value is computed across them.
Within-model bootstrap CIs (5000 resamples, paired over context batches) are reported per
condition in the raw per-model result files and are not re-aggregated into a cross-model
statistic here.
