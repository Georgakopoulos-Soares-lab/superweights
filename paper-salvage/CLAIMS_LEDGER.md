# CLAIMS_LEDGER.md

Every claim that appears in the manuscript gets a row. If you add a claim to a draft
section, add the row in the same commit. If you cannot fill the evidence column, the claim
is not ready to be written.

**Strength values:** `established` (direct evidence + controls) · `supported` (direct
evidence, controls partial) · `consistent-with` (association, no causal arrow) ·
`hypothesis` (stated as such in text) · `pending` (experiment not done) ·
`pending-rerun` (evidence exists but was produced under a protocol that has since been
superseded; the claim may well survive, but the number that will appear in the manuscript
is not yet measured)

---

| ID | Section | Claim (as it will appear) | Evidence path | Controls | Strength | Status |
|---|---|---|---|---|---|---|
| C-001 | R1 | ‖U_k‖_F ranks the empirical SW row 1/3,072 at the step-up layer in GENERator EUK and PROK | | layer median, 12× / 18× ratio | established | migrate |
| C-002 | R1 | Predictor places 9/10 DNABERT-2 SW rows in the top 3 of their layer (median rank 1.4 / 768) | | | established | migrate |
| C-003 | R1 | Predictor ranks the NTv3 SW row 1/1,536 at L11 | | | established | migrate |
| C-004 | R1 | Predictor recovers published NLP super-weight **output rows** in Llama/Mistral/OLMo from cold weights | | random rows | pending | E1 |
| C-005 | R1 | Published NLP **scalar** index dominates within the recovered row | | participation ratio | pending | E1 |
| C-006 | R1 | Prospective cold-weight prediction on an unseen NLP model matches the forward-pass sweep | | timestamped lock | pending | E1 |
| C-007 | R2 | GENERator EUK SW ablation: PPL 3.40 → 785.5 (+23,026%); 10 matched random rows < 0.01% | | matched random | established | migrate |
| C-008 | R2 | GENERator PROK SW ablation: +25,975%; random controls within 0.03% | | matched random | established | migrate |
| C-009 | R2 | Evo1 is structurally concentrated (top 1.2–4.1%) but ablation gives ΔPPL = 0.0% | | random rows ≈ 0 | established | migrate |
| C-010 | R2 | DNABERT-2 L7 r603 predictor rank 706/768 reflects a propagator, not a source (residual carry-in −10.96 vs. local MLP −5.80) | | | supported | migrate |
| C-011 | R2 | Structural concentration does not imply functional criticality | C-009 + C-007/8 | | established | migrate |
| C-012 | R3 | GENERator EUK: C = 1.0 through all downstream layers | `sw_broadcast_impulse.json` (fixed-ε, superseded) | random-coord, neighbour-row, random-dense | pending-rerun | E2 re-run, D-011 |
| C-013 | R3 | GENERator PROK: C ≈ 0.92 → ~0 by L14 while total gain grows toward output | as above | as above | pending-rerun | E2 re-run, D-011 |
| C-014 | R3 | DNABERT-2: C ≈ 0 immediately, largest impulse KL (≈0.31) | as above | as above | pending-rerun | E2 re-run, D-011 |
| C-015 | R3 | C describes routing geometry, not criticality | C-012–014 | | established | migrate |
| C-016 | R3 | Evo1 routing regime | | full control set | pending | E2 |
| C-017 | R3 | **[THESIS SLOT]** relation of T to criticality | | | pending | E2 |
| C-018 | R3 | Broadcast observability H = (L−ℓ−1)/L is a methodological covariate; NTv3 H ≈ 0.08 | | | consistent-with | write |
| C-019 | R4 | EUK ablation-KL scales with SW activation, r = +0.437 (p = 4.46e−7) | | | established | migrate |
| C-020 | R4 | PROK same relationship on write magnitude, r = +0.710; negative sign is a write-direction convention | | | established | migrate |
| C-021 | R4 | EUK shows no shuffle sensitivity at any order (all p > 0.31, n = 90); PROK is sensitive at all four orders | | 4 shuffle types | established | migrate |
| C-022 | R4 | EUK and PROK differ **jointly** in input-context sensitivity and downstream routing | C-012/13 + C-021 | | consistent-with | write |
| C-023 | R4 | No canonical regulatory motif class is enriched among SW-dependent k-mers (only EUK homopolymers, OR 1.85, q = 0.022) | | Fisher + BH | established | demote-to-supp, 1 sentence main |
| C-024 | R4 | Corpus-prior interpretation | | | hypothesis | conditional, D-009 |
| C-025 | R5 | A single amplifier pathway bidirectionally steers generated composition without comparable degradation | | random rows, entropy, PPL, complexity | pending | E3 |
| C-026 | R6 | GENERator SW ablation is bimodal: fungal species −82.5% acc / −85.3% MCC vs. splice −0.05% | | 35/35 random controls within ±0.1 pp | established | migrate |
| C-027 | R6 | DNABERT-2 SW-ensemble ablation costs −25.5 ± 0.7 pp on splice (p = 0.0004, n = 3) | | random < 0.05 pp | supported | migrate |
| C-028 | R6 | Max single-row DNABERT-2 effect is −1.45%; full effect requires all ten rows | | random-1-row mean −0.03% | established | migrate |
| C-029 | R6 | NTv3 splice replicates: ΔMCC = −0.119 ± 0.054, 5/5 seeds negative, p = 0.008 | | random \|Δ\| < 3e−4 | supported | **metric issue — PHASE_0 §0.4** |
| C-030 | R6 | Structurally related amplifiers are recruited for different functions across models | C-026 + C-027 | | supported | write |
| C-031 | R7 | The NLP SW-preservation heuristic does not transfer; INT4 with vs. without SW exemption differs below resolution | | | established | migrate, compress to 1 para |
| C-032 | R2/R3/R6 | Granularity of the causal object (scalar / row / ensemble) differs across models | | | pending | E4 |

## Ledger notes

**N-001 — independent corroboration of C-009's supporting residual trace.**
The Evo1 impulse run that motivated D-011 measured, as a side effect, the magnitude of the
Evo1 residual stream at row 3776: median |h| across the 4,096 dims reaches ~2.8e5 at layer
11 and settles at ~4.2e6–3e7 from layer 13 onward (max ~1.29e9). This independently
reproduces the manuscript's existing "residual freezes at ≈3×10⁷" claim for row 3776, from
a different script (`scripts/interpretability/run_sw_broadcast_impulse.py`) than the one
that produced the original trace (`scripts/analysis/run_evo1_residual_attribution_fp32.py`).

The impulse *measurement* through that protocol is void (D-011: fixed ε = 1.0 has no
dynamic range at that scale). The *magnitude observation* is not — it is a property of the
forward pass, not of the injection, and it is reproduced across two independent code paths.
**Keep it** as corroboration of the supporting trace behind C-009. It is unaffected by the
ε amendment and does not need to be re-derived from the re-run.

Caveat: this is corroboration of a magnitude, conditional on the STEP-2 input-sensitivity
gate in `PREREG_evo1_broadcast.md` §Precondition passing. If the gate fails, this note is
void along with everything else measured through the Evo1 trace.

## Retired claims

| ID | Claim | Why retired |
|---|---|---|
| X-001 | "SW neighbourhood is extremely tolerant to pruning" / shadow redundancy | Contradicted by our own pruning sweep: near-SW −1.60 pp vs. random −0.82 pp at 20% |
| X-002 | "We scanned eight genomic language models" (as a uniform benchmark) | Coverage is asymmetric; replaced by the coverage table |
| X-003 | (‖U_k‖_F, C) jointly predict criticality | Falsified by DNABERT-2 (C ≈ 0, largest KL) |
| X-004 | DNABERT-2 ensemble behaviour is a *consequence* of C ≈ 0 | Association only; no causal evidence. May return as `consistent-with` after E4 |
| X-005 | Broadcast headroom bounds broadcast | One extreme case; demoted to methodological covariate (C-018) |
