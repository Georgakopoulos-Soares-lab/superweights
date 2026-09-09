# CLAIMS_LEDGER.md

Every claim that appears in the manuscript gets a row. If you add a claim to a draft
section, add the row in the same commit. If you cannot fill the evidence column, the claim
is not ready to be written.

**Strength values:** `established` (direct evidence + controls) · `supported` (direct
evidence, controls partial) · `consistent-with` (association, no causal arrow) ·
`hypothesis` (stated as such in text) · `pending` (experiment not done) ·
`pending-rerun` (evidence exists but was produced under a protocol that has since been
superseded; the claim may well survive, but the number that will appear in the manuscript
is not yet measured) · `contested` (added 2026-08-13: the claim has real, controlled evidence
in this repo, but a later instruction/review calls it into question and the reason offered
for doing so has **no artifact of its own** in this repo — neither "still established" nor
"retired" is warranted without author adjudication; see N-012–N-015)

---

## New-direction reconciliation pass, 2026-08-13

A downstream review (outside this repository) asserted a substantially revised scientific
state — a redundant-pair mechanism in DNABERT-2, a GENERator attention-sink phenotype, causal
GC steering, an Evo1 numerical-saturation diagnosis, an NTv3 training-truncation bug, and a
PROK eukaryotic-probe contamination diagnosis, among others. Every one of these was traced
`claim → result → artifact → script` against the actual repository state (commit `6bf1601`
plus untracked files; nothing postdates it — verified by `find -newermt`). Full trace:
`docs/NEW_DIRECTION_EVIDENCE_AUDIT.md`. **None of the six items above has any supporting
artifact in this repository**, and three are in direct numeric or logical conflict with
artifacts that already exist. No claim below is added, upgraded, or silently deleted on the
strength of that review alone — every change below either (a) reflects work this repository
had already completed and locked before this pass (e.g. the KL≈0.31 retirement), (b) is a
directly code-verifiable mathematical fact requiring no new run (the quantization
scale-endpoint claim), or (c) downgrades a claim to `contested` where the review's guardrails
forbid presenting it as supported going forward but no artifact justifies retiring it either
— flagged for the author, not resolved by this pass. See `DECISIONS.md` for the accompanying
superseding decisions and `docs/PAPER_READINESS.md` for the main-text-sufficiency call.

---

| ID | Section | Claim (as it will appear) | Evidence path | Controls | Strength | Status |
|---|---|---|---|---|---|---|
| C-001 | R1 | ‖U_k‖_F ranks the empirical SW row 1/3,072 at the step-up layer in GENERator EUK (L4/r2371, 12× ratio) and, at the **corrected** PROK location (L8/r260, content-invariant activation spike out_max=30,167.07), also rank 1/3,072 | `results/mechanism/prok_contamination_audit.md`, `results/mechanism/D1_PROK_RERUN_REPORT.md` | layer median, EUK 12×; PROK ratio not separately recorded in the source report | established — **resolved 2026-08-17, see D-024/N-009 (closed)** | The old PROK location (L2/r1927) was detected with a mismatched eukaryotic probe (`--probe human_promoter` against the prokaryote model); retired as **X-008**. migrate at the corrected location. |
| C-002 | R1 | Predictor places 9/10 DNABERT-2 SW rows in the top 3 of their layer (median rank 1.4 / 768) | | | established | migrate |
| C-003 | R1 | Predictor ranks the NTv3 SW row 1/1,536 at L11 | `e4_granularity.json` | max/median 3.6× | established | **confirmed by E4** |
| C-004 | R1 | Predictor recovers published NLP super-weight **output rows** in Llama/Mistral/OLMo from cold weights: rank **1/4,096 in all three** | `e1_nlp_retrospective.json` | layer median (max/median 26.8–37.8×) | established | E1 done |
| C-005 | R1 | Published NLP **scalar** index dominates within the recovered row: rank **1** in all three, top1_share 0.89–0.99 | `e1_nlp_retrospective.json` | participation ratio 1.02–1.24 | established | E1 done |
| C-006 | R1 | Prospective cold-weight prediction on an unseen NLP model matches the forward-pass sweep | | timestamped lock | pending | E1 |
| C-007 | R2 | GENERator EUK SW ablation: PPL 3.40 → 785.5 (+23,026%); 10 matched random rows < 0.01% | | matched random | established | migrate |
| C-008 | R2 | GENERator PROK SW ablation: +25,975%; random controls within 0.03% | | matched random | established | migrate |
| C-009 | R2 | Evo1 is structurally concentrated (top 1.2–4.1%) but ablation gives ΔPPL = 0.0% | | random rows ≈ 0 | established | migrate |
| C-010 | R2 | DNABERT-2 L7 r603 predictor rank 706/768 reflects a propagator, not a source (residual carry-in −10.96 vs. local MLP −5.80) | | | supported | migrate |
| C-011 | R2 | Structural concentration does not imply functional criticality | C-009 + C-007/8 | | established | migrate |
| C-012 | R3 | GENERator EUK: C = 1.0 through all downstream layers (+1.0000 → +0.9911, L5–29) | `sw_broadcast_impulse.json` key `generator`, lock `3d7515d0b788…` | random-coord, neighbour-row, random-dense; **matched-norm arm empty (0/3072 rows in band)** | established | **CONFIRMED under locked protocol** |
| C-013 | R3 | GENERator PROK: C ≈ 0.92 → ~0 while total gain grows toward output (C +0.9177 → +0.0056; T 1.47e-2 → 6.18, L3–29) | `sw_broadcast_impulse.json` key `generator_prokaryote` | all four arms incl. matched-norm (n=5) | established | **CONFIRMED under locked protocol** |
| C-014 | R3 | _(retired — see X-006)_ | | | **RETIRED** | D-014 |
| C-015 | R3 | C describes routing geometry, not criticality | C-012/13 + re-measured DNABERT-2 C | | **restored: established** | **N-006 rule fired — see N-011** |
| C-016 | R3 | Evo1 routing regime: C = +0.9516 (L12) → +0.7044 sustained to L31; T 6.76e-03 → 1.20e-01 | `sw_broadcast_impulse.json` key `evo1`, lock `3d7515d0b788…` | all four arms; peak T SW 1.20e-01 vs controls 1.10–1.17e-01 | supported | **MEASURED — headroom 26×, noise floor 0.0** |
| C-017 | R3 | **[THESIS SLOT]** relation of T to criticality | | | pending | **E2 done; branch A vs B NOT assigned — author's call. Per the 2026-08-13 reconciliation (D-016), this is no longer eligible to be assigned as a main-text thesis in this pass — broadcast/T-C work is supplementary unless a specific new-headline claim needs it locally. Do not restore C-017 as a headline thesis.** |
| C-018 | R3 | Broadcast observability H = (L−ℓ−1)/L is a methodological covariate; NTv3 H ≈ 0.08 | | | consistent-with | write |
| C-019 | R4 | EUK ablation-KL scales with SW activation, r = +0.437 (p = 4.46e−7) | | | established | migrate |
| C-020 | R4 | _(retired 2026-08-17 — see X-008; based on the contaminated L2/r1927 channel. Corrected finding: C-041)_ | | | **RETIRED** | superseded by C-041 |
| C-021 | R4 | EUK shows no shuffle sensitivity at any order (all p > 0.31, n = 90); PROK half _(retired 2026-08-17 — see X-008; was keyed to the contaminated L2/r1927 channel)_ | | 4 shuffle types | EUK half established; PROK half **RETIRED** | EUK half migrates; PROK half superseded — see C-041/C-042 |
| C-022 | R4 | EUK and PROK differ **jointly** in input-context sensitivity and downstream routing | C-012/13 + C-021 | | consistent-with | write |
| C-023 | R4 | No canonical regulatory motif class is enriched among SW-dependent k-mers (only EUK homopolymers, OR 1.85, q = 0.022) | | Fisher + BH | established | demote-to-supp, 1 sentence main |
| C-024 | R4 | Corpus-prior interpretation | | | hypothesis | conditional, D-009 |
| C-025 | R5 | A single amplifier pathway bidirectionally steers generated composition without comparable degradation | | random rows, entropy, PPL, complexity | pending | E3 not started; prereg exists (`prereg/PREREG_steering.md`) but is unlocked. No "38.6×" or any steering number exists anywhere in this repo — see N-012. Do not write a steering number into the manuscript. |
| C-026 | R6 | GENERator SW ablation is bimodal: fungal species −82.5% acc / −85.3% MCC vs. splice −0.05% | | 35/35 random controls within ±0.1 pp | established | migrate |
| C-027 | R6 | DNABERT-2 SW-ensemble ablation costs −25.5 ± 0.7 pp on splice (p = 0.0004, n = 3) | | random < 0.05 pp | supported | migrate |
| C-028 | R6 | Max single-row DNABERT-2 effect is −1.45%; full effect requires all ten rows | | random-1-row mean −0.03% | established | migrate |
| C-029 | R6 | ~~NTv3 splice replicates: ΔMCC = −0.119 ± 0.054, 5/5 seeds negative, p = 0.008~~ **retired — truncation-bug artifact (see X-009)**. **Corrected (2026-08-17, D-024):** refit at the correct `max_length=400` (nucleotide-level tokenizer) reaches MCC 0.86–0.91; SW ablation effect is **−0.02 pp — no functional effect**. NTv3 does not replicate the SW-ensemble functional phenotype; DNABERT-2 remains the sole functional replication (n=1) | `results/mechanism/HANDOFF_ANALYTICAL_SUMMARY.md` §1 item 1 | random \|Δ\| < 3e−4 (old); corrected run's own controls per source report | established — **resolved 2026-08-17, see D-024/N-014 (closed)** | migrate as a null/negative result: NTv3 structurally replicates (C-003) but does not functionally replicate the ensemble effect |
| C-030 | R6 | Structurally related amplifiers are recruited for different functions across models | C-026 + C-027 | | supported | write |
| C-031 | R7 | The NLP SW-preservation heuristic does not transfer; INT4 with vs. without SW exemption differs below resolution | | | established | migrate, compress to 1 para. See C-033 for the mathematical mechanism now available to explain this null. |
| C-032 | R2/R3/R6 | _(retired 2026-08-14 — see X-007; superseded in part by C-034)_ | **`experiments/E4_granularity/CANONICAL_TABLE.md`** | shape-verified per model; PROK row marked CONTESTED (C-001 on hold) | **RETIRED** | E4 canonical table preserved historically; do not migrate the old PR numbers as a domain-general contrast |
| C-033 | R6 | Under the per-row RTN quantization rule already implemented in this repo (`scale = max\|row\| / maxval`), the element defining a row's scale is mapped exactly to the quantizer endpoint and is therefore preserved by construction — an SW that sets its own row's scale is trivially exempt from further precision loss regardless of whether it is separately protected | `scripts/compression/run_quantization_ablation.py` (docstring, lines 12–14), `scripts/compression/run_whole_model_quantization.py:300-303` (`amax = rows.abs().amax(dim=1, keepdim=True)`) | none needed — this is a mathematical property of the code as written, not a measured effect | established | **new 2026-08-13, added from direct code inspection, no run performed.** Scoped strictly to the per-row rule as implemented; group-wise quantization was not found anywhere in this repo and this claim must not be generalized to it. "Consistent with" C-031's empirical null, not proven to be its cause — C-031 itself has no recorded evidence path (see `results/keep/UNMIGRATED.md`), so the link is disclosed as an inference, not a demonstration. **2026-08-14 addendum:** an independently-authored implementation on a since-integrated branch (`scripts/compression/run_pair_aware_compression.py`, `run_per_tensor_sw_exemption.py`) uses the identical `scale = max\|w\|/qmax` formula and reports a directly-measured 0.000e+00 element-level quantization error for two SW rows under this rule — corroborating evidence from a second, independent implementation. The corresponding empirical artifacts are not committed in this repo; this addendum does not change C-033's status, which was already `established` from code alone. |
| C-034 | R2/R6/R7 | Exact bilinear-operator dimensionality (`q1`, `PR_spec`) of gated-FFN high-gain rows tracks encoder/decoder organization more closely than text-vs-genomic domain across the nine models tested (7 decoders, `q1` range 0.90–0.99; 4 encoders — 2 new, 2 discovery — `q1` range 0.39–0.90): one genomic model (NTv3) is substantially more distributed than every other model measured, one (DNABERT-2) is intermediate, and two (GENERator EUK, GenomeOcean-4B) are statistically indistinguishable from the NLP decoder cluster. The two newly, prospectively confirmed text encoders (MosaicBERT, ModernBERT) both cleared the pre-registered decoder-floor criterion, though ModernBERT's margin was narrow (0.21% relative on `q1`) | `experiments/E7_exact_dimensionality/RESULTS.md`, `experiments/E8_encoder_decoder/RESULTS.md`, `experiments/E7_exact_dimensionality/spectral_lib.py` | pre-registered decoder floor/ceiling fixed from a held-out reference table before either E8 model was measured; same-layer control rows (5 seeded, per model) | supported | **new 2026-08-14.** Folds E7's proposed claim and E8's confirmatory narrowing into one row per E8's own instruction not to duplicate. Explicitly **supersedes** C-032's specific numeric contrast (C-032's own diagonal-PR numbers are preserved historically, not mutated) — do not present this as a clean, universal domain split; it is descriptive and heterogeneous (NTv3 outlier, DNABERT-2 intermediate, ModernBERT thin margin). **Narrowed further 2026-08-23 (D-027):** with E10/E10b's causal-response-complexity result (C-046/C-047) now the paper's primary architecture-level finding, this claim is confirmed secondary/calibration material only — any scale-confound check on it (E11) tests calibration material, not the manuscript's headline. |
| C-035 | R2/R6 | On the two models where E8 measured same-layer control rows (MosaicBERT, ModernBERT), the detected high-gain row is far more concentrated than ordinary rows at the same layer: MosaicBERT candidate `q1`=0.477 vs. 5 control rows 0.03–0.05 (`PR_spec` 4.1 vs. 140–170); ModernBERT candidate `q1`=0.897 vs. controls 0.02–0.05 (`PR_spec` 1.2 vs. 90–120) | `experiments/E8_encoder_decoder/RESULTS.md` §"Same-layer control rows" | 5 seeded random rows per model, fixed `numpy.random.SeedSequence(42)` | established | **new 2026-08-14.** Secondary, descriptive observation from E8 — dramatic, unambiguous effect requiring no threshold judgment (unlike the primary Branch A/B decoder-floor decision behind C-034). Not part of E8's preregistered decision rule. |
| C-036 | R2/R6 | DNABERT-2's functional unit is a **redundant pair**, not a single row: 10-row ensemble ablation −26.84 ± 2.56 pp (n=5 seeds) vs. sum-of-individual-parts −3.51 ± 2.15 pp; critical pair (L9/r264, L9/r294) separate effects −0.02/−0.11 pp but joint −33.76 pp; random-pair control +0.006 to +0.007 pp; top-7 epistatic pairs 7/7 structurally related across 3 tasks (splice p=0.00014, promoter p=0.0035, histone p=0.0035, n=5 seeds) | `results/mechanism/SUPERADDITIVITY_AND_COMPOSITION_REPORT.md` §1 | random-pair floor; sum-of-parts control | established | **new 2026-08-17, adopted per D-024.** Adjacent to, not a restatement of, the pre-existing 10-row ensemble result (C-027/C-028) — this identifies the specific pair that drives it. |
| C-037 | R2/R6 | The critical pair's epistasis is **intrinsic to pretraining** (MLM loss, no task head, held-out hg38): epistasis +2.0118; top pair = 136,521× the random-pair floor sd (0.0000147); enrichment top-3 3/3 (p=0.032), top-5 5/5 (p=0.0025), top-7 6/7 (p=0.0035); pretrained-vs-finetuned rank correlation ρ=+0.316, p=0.034 | `results/mechanism/CHECKPOINT_1_PRETRAINED_EPISTASIS.md` | random-pair floor distribution | established | **new 2026-08-17, adopted per D-024.** |
| C-038 | R2/R6 | Mechanism = **joint norm carriage**, DNABERT-2-specific, does not generalize: the critical pair carries 58% of layer-9 residual norm (17.20→7.14); a controlled co-dominance intervention (constant total pair norm, redistributed between channels) moves epistasis −33.63→−0.66 while single-channel effect rises −0.02→−33.03 (E2 BREAK-PAIR). Imposing the identical co-dominance ratio onto **NTv3** (E2 MAKE-PAIR) produces **no effect at all** — neither joint nor single criticality. NTv3's SW channel is *more* norm-dominant than DNABERT-2's (29.4× gap vs. ~4.5×) yet ablating it costs −0.02 pp vs. DNABERT-2's −17.72 pp for the matched test — norm dominance does not predict functional criticality outside DNABERT-2 | `results/mechanism/E1_E2_CODOMINANCE_AND_CONFOUND.md`, `results/mechanism/FALSIFICATION_NORM_VS_SUPERWEIGHT.md` | co-dominance ratio sweep (5 ratios); NTv3 cross-check | established, scoped as DNABERT-2-specific | **new 2026-08-17, adopted per D-024.** The NTv3 null is part of the fact, not a caveat to smooth away — do not generalize this mechanism beyond DNABERT-2. |
| C-039 | R4 | GENERator decoder shows a **BOS attention-sink phenotype** DNABERT-2 (encoder) does not: EUK 37.96% of attention mass at position 0 (33.0× uniform), 78.3% of heads argmax there, SW activation at pos 0 is 45,585× other positions, argmax at token 0 in 60/60 probes; PROK replicates directionally (28.72%, 25.0×); DNABERT-2 contrast: argmax at token 0 in 0/40 windows | `results/mechanism/HANDOFF_ANALYTICAL_SUMMARY.md` §1.4, `scripts/mechanism/run_attention_sink.py` | DNABERT-2 cross-architecture contrast; shuffle controls | established | **new 2026-08-17, adopted per D-024.** MosaicBERT's side of the dissociation rests on activation data only (no attention-map export in this pipeline) per the source report's own caveat. |
| C-040 | R5 | GENERator EUK: scaling the SW row's write causally steers generated GC composition, non-monotonically: {0.0, 0.5, 1.0, 2.0, 5.0}× gives GC 0.2944→0.3961→0.3549 (saturates by 2.0×, reverses at 5.0×); random-row control flat 0.396–0.399; span ratio 38.59×; PROK replicates directionally, weaker (17.87×) | `results/mechanism/HANDOFF_ANALYTICAL_SUMMARY.md` §1.5, `scripts/mechanism/run_sw_steering.py` | random-row scaling control | established, with the source report's own caveat retained | **new 2026-08-17, adopted per D-024.** Do not present as a linear/monotone knob — the source report is explicit that the effect saturates then reverses; this is part of the fact. |
| C-041 | R4 | **Corrected** PROK hexamer causal test, at the corrected L8/r260 location: Spearman ρ = +0.0007 (p = 0.96) — **no relationship** between SW activation and ablation KL, unlike EUK's r=+0.437. Corrected causal-tracing ΔPPL = +1.2459 ± 0.5379 (random control +0.0008 ± 0.0007, t=21.84) — smaller than the old, contaminated-channel figure (+9.47) but still real and significant vs. random | `results/mechanism/D1_PROK_RERUN_REPORT.md` | random hexamer/row controls | established | **new 2026-08-17, adopted per D-024.** Replaces C-020 (retired, X-008). The old r=−0.710/+0.710 sign-convention story (also reflected in `README.md`'s pre-existing "Pre-submission status" section, commit `4ba1686`) is superseded, not merely disputed — the corrected channel shows no sign relationship of either direction to resolve. |
| C-042 | R4 | Corrected PROK: a *different* kingdom contrast survives at the corrected location — GC-dependence of SW-ablation *cost* (not activation): PROK r = −0.661 (up from the old channel's −0.520), EUK r = −0.001 | `results/mechanism/D1_PROK_RERUN_REPORT.md` | cross-kingdom contrast | established | **new 2026-08-17, adopted per D-024.** Replaces C-021's PROK half (retired, X-008); EUK half of C-021 (shuffle sensitivity) is unaffected and remains established under its own ID. |
| C-043 | R7 | Empirical confirmation of C-033's mathematical claim (Q2): per-row RTN quantization error for the SW element is exactly 0.000e+00 at INT8/4/3/2 for 2 rows tested (L9/r264, L9/r294); 5 of DNABERT-2's 10 SW rows are additionally the global tensor max (row_max/tensor_max=1.000). Destructive-regime/Q4: per-tensor SW exemption gives mean benefit +0.048pp, median −0.035pp, t≈+0.15 over 14 cells — **no reliable benefit at any granularity or precision tested** | `results/mechanism/SUPERADDITIVITY_AND_COMPOSITION_REPORT.md` §9-11 | 14-cell exemption-benefit sweep (3 tasks × 2 granularities × 3 precisions) | established | **new 2026-08-17, adopted per D-024.** Empirical companion to C-033 (which remains established from code alone, independent of this row) and to C-031's existing null. |
| C-044 | R2/R6 | **E9 mechanistic tomography**: in the declared 10-row high-gain basis, DNABERT-2's finite causal response to held-out multi-row masks **requires pairwise interaction terms** at both tested scales — a lifted ridge model (F3) improves held-out MAE over the best jointly-fit additive model (F2) by 54.7% (epsilon=0.5) and 24.4% (epsilon=1.0), bootstrap 95% CI excluding zero at both ((0.0038,0.0186) and (0.0614,0.1144), batch-resampling, n=5000). Singleton-additive (F0) and scalar-calibrated (F1) fail held-out adequacy at both scales (R²<0.90 or worse). The independently-established critical pair L9/r264+r294 ranks **#1 of 45** pairs by fitted \|Γ\| at both scales, without being given special treatment during fitting — confirming H5 retrospectively | `experiments/frozen/E9_mechanistic_tomography/{RESULTS.md,fit_results_dnabert2.json,dnabert2_mask_responses.json}` | fit/calibration/held-out mask split (disjoint, seed 20260822); ridge penalty selected on calibration only; design-matrix rank/conditioning checked before fitting | established | **new 2026-08-22.** This is a distinct, stronger claim than C-036/C-037 (which used only single/pair/k-of-N ablation): C-044 shows the pair requirement holds for a fitted, held-out-evaluated multi-row observer across two intervention scales, not only for the specific known pair's own ablation. |
| C-045 | R5 | **E9 mechanistic tomography**: GENERator EUK's causal-response basis is too thin (2 pre-existing detected rows, not the target 6-12) to support the same F0-F3 tomography given to DNABERT-2; padding it with structurally-arbitrary rows was rejected as fishing-adjacent. Scoped instead to a 1D dose-response: row 2371 (primary) shows a large, mostly-monotonic GC decrease under suppression (GC 0.3961→0.3863→0.2944 at alpha=1.0/0.5/0.0), span ≈390× the 5-row random-control span (0.1017 vs 0.0003); row 1522 (secondary) shows a smaller, **non-monotonic** effect (0.3961→0.4131→0.3466) | `experiments/frozen/E9_mechanistic_tomography/{RESULTS.md,BASELINE_REGRESSION.md,baseline_regression_results.json}` | 5-row random-control, seed 42; paired generation seeds across conditions | established, explicitly scope-limited | **new 2026-08-22.** Does not establish "GENERator is additive" — no multi-component observer-family ladder was run for GENERator at all (see `BASIS_FREEZE.md`, D-025). Row 1522's non-monotonicity means this does not clear Phase 8's "graded steering" bar either; call it causal sensitivity, not controlled steering, pending a wider basis. |
| C-046 | R1 (outline section TBD — see D-027) | **E10 architecture-causal-complexity synthesis**: across 5 decoders + 2 encoders (text), 4/5 decoders (Llama, Mistral, OLMo, Qwen2.5) mechanically resolve SINGLE_COMPONENT_DOMINANT while 2/2 encoders (MosaicBERT, ModernBERT) resolve PAIR_TERMS_REQUIRED at both intervention scales (ε=0.5, 1.0); Phi-3 is a decoder-side counterexample (MULTI_COMPONENT_CANDIDATE, C1=0.333, flagged not resolved). OLMo additionally shows structural-rank/causal-rank dissociation: its causally dominant row (L1, +111.8% dNLL) is not its structurally largest row (L24, `‖U_k‖_F` rank 1, only +1.4% dNLL). Explicitly does not establish "q1 causes causal concentration" — Phi-3's lower median q1 (0.9028) than Llama/Mistral (0.99+) with weaker, not stronger, concentration runs the wrong direction for that story | `experiments/E10_nlp_architecture_causal/{ARCHITECTURE_SYNTHESIS.md,run_decoder_spectrum.py,fit_encoder_observers.py}`, `results/e10_*.json` | 5-row same-layer seeded controls per model (`SeedSequence(42)`); fit/calibration/held-out mask split for encoders | established, model-level descriptive counts (4/5, 1/5, 2/2), not a powered test | **new 2026-08-23, reconciled per D-027** (measured 2026-08-22, previously unlogged). Statistics discipline: unit of analysis is the model; no cross-model p-value computed. Supersedes no prior claim; narrows C-034's role (see D-027, that row's own note). |
| C-047 | R1 (outline section TBD — see D-027) | **E10b Phi-3 tomography**: the one decoder E10 flagged as multi-component splits by intervention strength — at ε=0.5, explicit pairwise interaction terms are required (F2→F3 held-out MAE improves 36.4%, bootstrap CI (0.0057,0.0109) excluding zero); at ε=1.0, no finite second-order model (additive or pairwise) adequately describes the held-out response (all of F0-F3 have negative held-out R²), driven by a specific three-way redundancy break among the layer-2 triplet (L2/r525, r1693, r1113) — damage is catastrophic only when all three are removed together, not predictable from any pairwise combination | `experiments/E10b_phi3_tomography/{E10B_SYNTHESIS.md,run_phi3_tomography.py,fit_phi3_observers.py}`, `results/e10b_phi3_{tomography_responses,fit_results}.json` | fit/calibration/held-out mask split; bootstrap 5000 resamples, paired over context batches | established, explicitly scope-limited (n=1 decoder tested for multi-row interaction) | **new 2026-08-23, reconciled per D-027** (measured 2026-08-22, previously unlogged). Does not establish decoders in general require pair/higher-order terms — Llama/Mistral/OLMo/Qwen2.5 were never tested for multi-row interaction. No triple-interaction term was fit at ε=1.0 (forbidden by hard constraints); the MIXED_OR_UNRESOLVED result is reported as a limit of the licensed F0-F3 ladder, not rescued with a higher-order model. |

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

Caveat resolved 2026-08-12: the STEP 2 gate **passed**
(`experiments/E2_evo1_broadcast/GATE_RESULTS.md`), so this note stands.

**N-002 — "bit-for-bit identical from L13" was a statistic, not the hidden state.**
The Evo1 residual does **not** freeze. No layer is bitwise identical to any other; the
appearance came from a layer-to-layer max |Δ| statistic that saturates on a few very large
coordinates. Any manuscript wording implying a frozen residual must be corrected to describe
the magnitude plateau instead. Evidence: `results/evo1_layer_freeze_check.json`.

**N-003 — relative-change collapse (Phase 1b candidate, do not chase).**
After the L12→L13 magnitude explosion (max |h| → 1.29e9), consecutive-layer change falls to
~1e-6 of residual magnitude and blocks 21–31 alter < 0.2% of coordinates. This is an
observation about how little the second half of Evo1 moves the residual, not a claim about
what it computes, and it is **not** the AC/DC decomposition Branch C would require. Logged
as a Phase 1b candidate under the binding Phase 1 stop rule. No ledger row until it is
promoted. Evidence: `experiments/E2_evo1_broadcast/GATE_RESULTS.md`.

**N-004 — DNABERT-2's impulse assay is noise-dominated. BLOCKER.**
Two *identical* DNABERT-2 forward passes, no injection, differ by KL = **0.305** and
‖Δh‖ = 4.50. The SW injection gives KL = 0.277 and ‖Δh‖ = 4.50. Per-layer SNR is
0.930–1.000: the injection changes the downstream residual by the same amount as changing
nothing. C-014's headline "largest impulse KL ≈ 0.31" and the noise floor agree to two
significant figures.

GENERator EUK, GENERator PROK and NTv3 have an **exactly zero** noise floor on the same
harness, device and fp32 path, so this is specific to DNABERT-2 (likely its Triton
flash-attention kernels; it is in `eval()` and has no dropout). Not diagnosed further and
no fix attempted — that decision is pending.

Consequences: C-014 contested; C-015 downgraded from `established` because its DNABERT-2
leg supplies the C ≈ 0; D-004's stated rationale and the X-003 retirement rest on the same
measurement; the `CLAUDE.md` hard constraint "DNABERT-2 has C ≈ 0 and the strongest encoder
phenotype" is affected in its first clause only. **C-027 and C-028 are unaffected** — the
ablation phenotype is a different experiment. The conclusion "C is not necessary for
criticality" may survive on the ablation evidence, but not on the impulse C value.

Evidence: `results/impulse_determinism_dnabert2.json`,
`results/impulse_determinism_{generator,generator_prokaryote,ntv3}.json`,
`experiments/E2_evo1_broadcast/INSTRUMENT_VALIDATION.md`.

**N-005 — T is ε-invariant where powered, so the AC-ε re-run stays comparable.**
On GENERator PROK (deterministic), T is flat to three significant figures across
ε = 1e-2 … 10, and T at the AC-relative ε reproduces T at ε = 1.0 to 1.5%. Below ε ≈ 1e-3 T
inflates — the fp32 accumulation floor D-013's headroom column is designed to flag. Note
that α = 0.01 does **not** land near ε = 1.0: ε spans 2,500× across the four non-Evo1 models
(EUK 9.37, PROK 1.68e-2, DNABERT-2 3.71e-3, NTv3 0.60). The prereg's sanity expectation is
false as written and should be corrected before locking; the comparability it was meant to
protect is delivered by ε-invariance instead.
Evidence: `results/impulse_linearity_generator_prokaryote.json`.

**N-007 — DNABERT-2 determinism is fixable, but the fix fails the perplexity guard. OPEN.**
Attempt 1 of the STEP 3a ladder succeeded on determinism: forcing the eager PyTorch
attention path (module-global `flash_attn_qkvpacked_func = None`) gives an **exactly zero**
noise floor. Attempts 2 and 3 were not needed.

The signal that emerges is nothing like what was published. Through the Triton path the
"signal" was ‖Δh‖ = 4.50 and KL = 0.277 — all of it noise. Through the eager path the real
signal is **‖Δh‖ = 2.4e-3 and KL = 1.8e-8**, roughly 1,800× smaller. C-014's "largest
impulse KL ≈ 0.31" does not survive in any form.

**But the guard fails.** Masked-LM perplexity, same weights, same inputs, same process,
kernel the only difference:

| Attention path | PPL | repeats | spread |
|---|---|---|---|
| Triton kernel, fp16 attention (original) | 687.9 | 720.6 / 681.1 / 662.2 | **8.48%** |
| eager PyTorch, fp32 attention (fixed) | **176.9** | 176.9 / 176.9 / 176.9 | **0.000%** |

Δ = **−74.3%** against a ±1% tolerance. Per the standing rule this is reported, not adopted.

Two facts constrain the interpretation. The Triton path's perplexity is itself
nondeterministic by 8.5%, so it is not a fixed quantity to be matched against. And the
Triton path silently casts qkv to **fp16** (the kernel takes only fp16/bf16), so it was
never running the fp32 that D-013 mandates. The eager path is 3.9× better on perplexity and
bit-reproducible. That points to the Triton path being the defective one rather than the fix
having damaged the model — but adjudicating which path *is* DNABERT-2 is a substantive call,
not a harness detail, and it is not made here.

Scope: the loaded config has `attention_probs_dropout_prob = 0.0`, so `p_dropout` is falsy
and the guard `if self.p_dropout or flash_attn_qkvpacked_func is None` selects **Triton by
default**. Every inference-time DNABERT-2 analysis in the manuscript therefore went through
the Triton/fp16 path. Fine-tuning runs with dropout > 0 would have taken the eager path.
Which of C-002, C-010, C-027, C-028 that touches has **not** been audited.

Evidence: `results/dnabert2_kernel_guard.json`, `results/impulse_determinism_dnabert2.json`,
`logs/dnabert2_guard.log`.

**N-008 — the fp32 residual-attribution figures were bf16 under an fp32 label.**
`scripts/analysis/run_evo1_residual_attribution_fp32.py` is named `_fp32`, writes
`"dtype": "float32"` into `results/sw_residual_attribution_evo1_fp32.json`, and runs
**bf16** — it casts every parameter except poles/residues to bfloat16 and says so in its own
comment. The manuscript's Evo1 residual-attribution figures are therefore bf16 results
carrying an fp32 label.

The numbers are probably sound: the STEP 2 gate measured both paths and they agree closely
(L13 median |h| 4.09e6 fp32 vs 4.23e6 bf16; max 1.2953e9 vs 1.2918e9; std AC 5.64e7 vs
5.53e7). This is a **labelling** defect, not necessarily a numerical one. Methods must state
the actual dtype, and the stored JSON's `dtype` field should be corrected or annotated.

**N-009 — GENERator PROK: claim layer resolved to L2; stored artifact does not reproduce.
C-001 stays on hold.** _(updated 2026-08-12; the layer-mismatch hypothesis below was WRONG)_

Provenance traced backward claim -> result -> log -> script -> config. The claim's layer is
**2**, row 1927 — the same layer E4 used — and the original log records the **same checkpoint
name** now in the config (`GenerTeam/GENERator-v2-prokaryote-3b-base`, run 29 May 2026).
C-001's "18×" is that artifact's 17.90; the "12×" is EUK's 12.24.

Recomputed at that exact layer on current weights:

| | value | rank | max/median |
|---|---|---|---|
| stored artifact | 2648.4773 | 1/3072 | 17.90 |
| exact ‖U_k‖_F (original formula) | 5.5106 | **1277**/3072 | 7.18 |
| decomposition (`uk_frobenius`) | 5.4710 | **1289**/3072 | 6.13 |

The two current implementations **agree with each other** and both disagree with the stored
artifact, so this is not a formula difference. **EUK reproduces exactly** (max/median 12.24
both, rank 1); PROK does not. Cause not determined — that needs a decision, not a trace.

C-001 is **not** restored and **not** retired; its PROK half currently has no reproducible
support and it must not be migrated. Evidence:
`experiments/E4_granularity/N009_RESOLUTION.md`, `results/n009_prok_layer_resolution.json`.

**N-010 — `uk_frobenius` ADAPTERS registry is wrong for NTv3.**
`ADAPTERS["ntv3"] = adapter_llama_swiglu` (flagged "verify") would raise: NTv3 has no
`model.model.layers` and no `mlp` module. Its `SelfAttentionBlock` carries the FFN inline as
`fc1` (12288, 1536) and `fc2` (1536, 6144) with SiLU — `fc1` is **packed** gate+up, the
DNABERT-2 layout. **RESOLVED 2026-08-12.** `adapter_ntv3` added to `src/uk_frobenius.py` with shape guards
that raise rather than transpose; `ADAPTERS["ntv3"]` repointed. Seven tests added in
`src/test_ntv3_adapter.py`, all of which fail under the old Llama-style entry (including an
explicit assertion that `adapter_llama_swiglu` raises on an NTv3 model). Full suite green:
7 passed, plus the E1 self-test. Recomputing NTv3's E4 row through the shared implementation
reproduces the overnight value **exactly on every field** (rank 1/1536, max/median
3.612997884175097, top-1 index 1713, top-1 share 0.17540364719820148, PR 22.71939144675925),
so no NTv3 result changes. E1's NLP results were not touched — no test demonstrated an error
there. Evidence: `results/e4_ntv3_shared_adapter.json`.

Gate/up ordering within the packed half is unresolvable from shapes and immaterial: c_{k,i}
is symmetric under swapping them.

**N-011 — N-006's rule fired: DNABERT-2 C is still ≈ 0 on the clean kernel.**
N-006 pre-committed: *"Re-measured DNABERT-2 C ≈ 0 → X-003 stays retired, C-015 restored."*
The locked-protocol re-measurement on eager attention gives DNABERT-2 **C = +0.0331 at L6,
peak +0.0625, −0.0221 at L11** — against EUK's +1.0000 and PROK's +0.9177 at their first
downstream layers. That is ≈ 0. The rule fires mechanically:

- **X-003 stays retired.** Its falsification rested on DNABERT-2 having C ≈ 0, which
  survives; it never depended on the retracted "largest KL ≈ 0.31" (X-006).
- **C-015 restored to `established`.**
- `CLAUDE.md`'s "C is NOT necessary for criticality" constraint may drop PROVISIONAL **only**
  in respect of the C value. Its second clause — "and the strongest encoder phenotype" —
  rests on C-027/C-028, which are untouched and unaudited for the Triton/eager question.

Applied as a mechanical status change under a pre-existing rule. **No new interpretation**,
and nothing here says what T does or does not predict. Evidence:
`experiments/E2_evo1_broadcast/RESULTS.md` §B.

**N-012 — six new-direction headline claims have no artifact anywhere in this repository.
Do not write any of them into the manuscript.** _(2026-08-13)_

Full trace in `docs/NEW_DIRECTION_EVIDENCE_AUDIT.md`. Summary: (1) a DNABERT-2 "redundant
pair" ablation (−33pp) — the only −33pp-adjacent number belongs to the existing 10-row
ensemble (C-027/C-028), a single-seed instance, not a 2-row pair; (2) that pair's pretrained/
base-model superadditivity; (3) its layer-9 joint residual-norm carriage; (4) a GENERator
BOS/attention-sink phenotype (25–33× uniform, ~100% argmax) — no attention-pattern analysis
of any kind exists for any GENERator model; (5) GENERator EUK causal GC steering at "38.6×" —
E3, the preregistered experiment for exactly this, was never locked or run (`C-025` stays
`pending`); (6) its degradation/quality controls. None of these six has a script, log, JSON,
or even a partial run backing it. Two adjacent-but-different real results exist and must not
be conflated with the missing claims: the existing 10-row DNABERT-2 ensemble ablation
(C-027/C-028, real, established, −25.5pp mean) and the existing GC↔AT counterfactual-swap
experiment (`results/sw_counterfactual_swap_generator.json`, real, demoted to supplement per
`CUT_LIST.md`, measuring KL under activation interpolation — not free-generation GC% under
row scaling).

**CLOSED 2026-08-17, see D-024.** The colleague branch this note anticipated (`mechanism-and-
negative-results`, commit `5b0220c`) was located, audited, integrated, and — per explicit
author decision D-024 — its reports are now adopted as results facts. Claims 1–3 → **C-036,
C-037, C-038**. Claim 4 → **C-039**. Claim 5 → **C-040** (steering; C-025 stays `pending` as
its own, separately-preregistered claim — C-040 is the colleague's independent measurement,
not a retroactive confirmation of C-025). Claim 6 → part of C-040's own reported caveats.

**N-013 — PROK SAE: the fp16 clamp mechanism is real and code-verified; the specific
"98% variance destroyed / pathological features" description is not.** _(2026-08-13)_

`sae/collect.py:176-179` clamps every channel to ±60,000 before an fp16 cast, with an
in-code comment citing exactly this rationale ("avoid Inf from super-weight overflow"). The
PROK SW's own detected `out_max` is 506,014 (C-008) — roughly 8–10× the clamp ceiling — so
the mechanism for corrupting the SW channel's dynamic range specifically is real and worth
disclosing. However, the manuscript's own recorded SAE health diagnostics
(`manuscript.txt:389-393`: reconstruction MSE 4.78 vs. random baseline 7,238; 97.4% of
dictionary features active) describe a healthy fit, which is not what "pathological/
single-active features" would predict, and no artifact anywhere computes a variance-destroyed
percentage. There is no `CLAIMS_LEDGER.md` row for the PROK SAE result at all (it lives only
in manuscript prose and is already marked for demotion to supplement in `CUT_LIST.md`) — this
note exists so that whatever replaces it discloses the clamp as a real methods caveat without
asserting the unmeasured 98% figure or "pathological" characterization.

**PARTIALLY CLOSED 2026-08-17, see D-024 — one residual conflict flagged, not silently
resolved either way.** D-024 adopts the colleague's mechanism-session reports as facts, which
would ordinarily also cover the PROK SAE retraction (their `HANDOFF_ANALYTICAL_SUMMARY.md`
item 6: "PROK SAE withdrawn — layer-2 contamination, an fp16 clamp destroying 98% of
SW-channel variance..."). The fp16-clamp mechanism itself is uncontroversial and stays
adopted. But the specific "98% variance destroyed / pathological features" figure is not
merely *unsupported by a missing artifact* (D-024's usual situation) — it directly
**contradicts** an artifact that already exists in this repo (`manuscript.txt:389-393`'s
recorded SAE diagnostics: MSE 4.78, 97.4% of dictionary features active, describing a healthy
fit). D-024's rationale is trusting a colleague's report where this repo has no competing
measurement of its own; that does not obviously extend to overriding a measurement this repo
already has. **Left open for explicit author attention** — this is the one sub-item this pass
does not fold into the D-024 sweep by default.

**N-014 — C-029 (NTv3 splice replication) is `contested`, not retired, not confirmed.**
_(2026-08-13)_

A later review asserts the old NTv3 checkpoints were trained under a silent max-length/
truncation bug (only ~20% of the intended context window preserved) and that retrained
checkpoints show no SW effect (p≈0.48). `models/ntv3_wrapper.py:21` and
`scripts/evaluation/run_gue_multiseed.py:269` do hardcode `max_length=512` — a real,
inspectable fact — but nothing in the repository calls it a bug, states an intended larger
window, or reports any retrained checkpoint or a p≈0.48 value under any metric. The only
NTv3 splice p-value that exists is C-029's own p=0.008 (`results/gue_multiseed_ntv3_splice.json`,
5/5 seeds negative, real random-row controls). The claim and the artifact directly disagree.
Per this review's own guardrails, NTv3 must not be presented as a confirmed positive splice
replication going forward regardless of the mechanism's provenance — so C-029 does not
migrate and is not written as `supported`. But nothing in this repo justifies actively
retiring an established-looking, controlled 5-seed result either. **Flagged for the author.**
If a truncation-bug fix and retrain genuinely happened, the artifacts (config diff, retrained
checkpoint, new eval JSON) need to be located or reproduced in a future session before this
note can be closed either way — this pass does not retrain anything.

**UPDATE 2026-08-14 — bug independently confirmed by direct code trace; C-029 moved from
`contested` to `pending-rerun`.** A colleague branch (`mechanism-and-negative-results`,
integrated `5b0220c`) was located and audited (`experiments/docs/COLLEAGUE_BRANCH_AUDIT.md`
§4.3). Independently of that branch's own prose, this trace was performed directly against
this repo's own code, before and confirmed after integration: the max-length figure this note
originally cited (`ntv3_wrapper.py:21`, `run_gue_multiseed.py:269`, hardcoded 512) is not what
actually governs the splice task. `run_gue_multiseed.py:269` resolves
`max_length = args.max_length or _MAX_LEN.get(tkey, 512)`; for `tkey="reconstructed"` (the
splice task), `scripts/evaluation/run_gue_ablation.py`'s `_MAX_LEN["reconstructed"] = 80`
fires instead of the 512 fallback. `scripts/evaluation/submit_ntv3_splice_multiseed.sh` — the
exact launch script that produced `results/gue_multiseed_ntv3_splice.json`, C-029's own
backing artifact — never passes `--max_length`, so this resolves to 80 for that exact run.
NTv3 tokenizes at nucleotide level (1bp = 1 token), so 80 tokens truncates every splice window
to its first 80bp, before the splice junction the task is about. **This is a real, confirmed
bug in C-029's own production run**, found by tracing this repo's code, not by taking the
colleague's word for it. What remains missing: any corrected retrained checkpoint or
evaluation JSON — the colleague's reports describe a refit (MCC 0.86–0.91, SW ablation effect
reduced to −0.02pp) but no artifact for it exists anywhere in this repository
(`experiments/docs/MISSING_COLLEAGUE_ARTIFACTS.md` §G). Per this project's standing
discipline, the old number is retired but no replacement number is written in until it has its
own artifact — hence `pending-rerun`, not `established` and not silently dropped.

**CLOSED 2026-08-17, see D-024.** Author decision: adopt the colleague's reported corrected
result (MCC 0.86–0.91, SW ablation effect −0.02pp) as fact, notwithstanding the continued
absence of a retrained-checkpoint/eval-JSON artifact. Old result retired as **X-009**; C-029
updated in place to the corrected result.

**N-015 — C-020/C-021(PROK half): contested on the strength of an existing, repo-native
reason, not the unaudited "wrong-probe" mechanism offered downstream.** _(2026-08-13)_

A later review asserts the old GENERator PROK super-weight (layer 2, row 1,927) was
identified using a eukaryotic probe on the prokaryotic model, invalidating the EUK/PROK sign
relationship (r=±0.710, C-020), the PROK shuffle-sensitivity finding (C-021 PROK half), the
PROK SAE, and the kingdom-composition story, and that the real PROK SW is elsewhere
(reported as ≈L8/r260). No file in this repository names a probe-species mismatch, layer 8,
or row 260 in a PROK context — this diagnosis has no local artifact. What the repository
**does already have**, independently, is N-009: the same layer-2/row-1927 artifact's stored
‖U_k‖_F value (2648.48, rank 1/3,072) does not reproduce against current weights at that same
layer (recomputed 5.51/5.47, rank 1277/1289) — C-001 has been `on hold` since 2026-08-12 for
this reason, which N-009 explicitly states is *not* a probe or formula issue ("cause not
determined"). Because C-020/C-021's PROK halves are keyed to the identical contested
artifact (same layer, same row), they inherit C-001's hold rather than remaining
`established` — this is the audited, repo-native path to the same practical outcome the
review's guardrails require (do not present the old PROK composition/kingdom story as
supported), without asserting the specific unaudited contamination mechanism or the L8/r260
replacement, neither of which this pass can verify. **Flagged for the author**: if the
eukaryotic-probe diagnosis and the L8/r260 relocation are real findings from work done outside
this repository, the artifacts (probe-provenance check, L8/r260 detection sweep output) need
to be added to the repo before either can be written into the manuscript.

**UPDATE 2026-08-14 — the external work this note asked for has been located; its numeric
artifacts have not.** A colleague branch (`mechanism-and-negative-results`, integrated
`5b0220c`) was audited (`experiments/docs/COLLEAGUE_BRANCH_AUDIT.md` §4.2). It contains a
detailed contamination trace (`results/mechanism/prok_contamination_audit.md`,
`D1_PROK_RERUN_REPORT.md`): old detection used `--probe human_promoter` (a human ACTB
sequence) against the prokaryote-specialized model; a corrected sweep with the
already-present `pseudomonadota` probe (`probes/dna_probes.py`, not a new resource) finds
L8/r260. **One specific, credible corroboration**: the colleague's independently-computed
`‖U_k‖_F` rank for the contaminated L2/r1927 channel is **1289/3072** — matching this repo's
own N-009 recomputation (`uk_frobenius` decomposition: rank 1289/3072) **exactly**, via a
different computational method. This does not resolve N-009's "cause not determined," but it
is a specific, independently-motivated candidate explanation for the same gap, not a
generic assertion. What is still missing, confirmed absent from the pushed branch and this
filesystem (`experiments/docs/MISSING_COLLEAGUE_ARTIFACTS.md` §H): the corrected
`super_weight_index.json` entry (the colleague's own reports claim this file was updated to
L8/r260 with the old entry archived — **the pushed file is byte-identical to this repo's
pre-existing, uncorrected version**, still showing L2/r1927), the archived
`_ORIGINAL_SUSPECT.json`, and all D1 rerun outputs (corrected hexamer causal test, now
reported as ρ=+0.0007/p=0.96 rather than the old r=−0.710, though that number too has no
committed artifact). **C-001, C-020, C-021 remain unchanged — still on hold / contested** —
this update only narrows what "flagged for the author" is now waiting on.

**CLOSED 2026-08-17, see D-024.** Author decision: adopt the eukaryotic-probe contamination
diagnosis and the L8/r260 relocation as fact, notwithstanding the continued absence of the
corrected `super_weight_index.json` edit and D1 rerun output artifacts. Old channel (L2/r1927)
and everything keyed to it retired as **X-008**; C-001 updated in place to the corrected
location; C-020 retired, replaced by **C-041**; C-021's PROK half retired, replaced by
**C-042**.

**N-016 — Evo1 "2^24" (colleague claim): the colleague's own later work already retracted the
literal version; the corrected value does not contradict N-001/N-002. No headline claim
change.** _(2026-08-14)_

`CLAUDE.md` §B instructs: do not claim Evo1's SW value was pinned at 2^24, since no artifact
in this repo names it. That constraint is unaffected by this note — no artifact for the
corrected value exists here either (`experiments/docs/MISSING_COLLEAGUE_ARTIFACTS.md` §F).
What changed is narrower: a colleague branch (`mechanism-and-negative-results`, integrated
`5b0220c`) was audited, and its own internal timeline shows the "exactly 2^24" claim was the
colleague's own early (2026-08-03), less careful observation, which their own later
(2026-08-07) fp64 adjudication explicitly re-tested and rejected: *"Frozen at exactly 2²⁴? No.
Plateau = 29,360,128 = 1.75 × 2²⁴; no block hits 2²⁴ exactly."* That corrected value
(2.94e7) falls inside this repo's own independently-measured N-001 settling range
(~4.2e6–3e7, measured via a different code path, `run_sw_broadcast_impulse.py`, months
earlier). **This is convergent evidence via a different instrument, not a contradiction to
resolve** — the colleague's corrected account is directionally consistent with, and adds
mechanistic detail (an MLP-then-mixer contribution decomposition) to, C-009/C-011's existing
"structurally concentrated but functionally null" account. No headline claim (C-009, C-011)
changes as a result of this note; the fp64-adjudication artifact itself remains unrecovered
and is not cited as evidence for anything beyond this note.

**CLOSED 2026-08-17, see D-024.** Author decision extends to this finding too: the fp64
adjudication (real plateau 1.75×2²⁴, MLP-then-mixer decomposition, causal-null under both
natural ablation and forced-survival rescue) is now accepted supplementary detail for
C-009/C-011, citable alongside them. Headline status of C-009/C-011 still does not change —
this was corroboration, not a competing claim, from the start.

**N-017 — C-040 vs. C-045: two GENERator row-2371 GC dose-response measurements disagree on
grid range, span ratio, and shape; unreconciled, pending E12.** _(2026-08-23)_

C-040 (colleague-adopted per D-024, `scripts/mechanism/run_sw_steering.py`, 5-point grid
{0,0.5,1,2,5}×) reports GC saturating by 2× and *reversing* at 5× (span ratio 38.59×). C-045
(E9's own locked measurement, `run_baseline_regression.py`/`tomography_lib.py`, 3-point grid
restricted to α∈[0,1]) reports a span ratio of ~390× and describes row 2371 as
"mostly-monotonic" — but C-045 was never measured at amplification (α>1), so it cannot confirm
or contradict C-040's reversal, and the two span ratios are not computed over the same range
so are not directly comparable. Neither claim is retracted by this note. Left open pending
E12, which extends row 2371's own alpha grid into the amplification range under E9's tighter
protocol (disjoint held-out damage windows, paired generation seeds) specifically to test
whether C-040's reversal replicates. Do not cite a single span-ratio number for row 2371
without specifying which grid it was measured over until this is resolved.

## Retired claims

| ID | Claim | Why retired |
|---|---|---|
| X-001 | "SW neighbourhood is extremely tolerant to pruning" / shadow redundancy | Contradicted by our own pruning sweep: near-SW −1.60 pp vs. random −0.82 pp at 20% |
| X-002 | "We scanned eight genomic language models" (as a uniform benchmark) | Coverage is asymmetric; replaced by the coverage table |
| X-003 | (‖U_k‖_F, C) jointly predict criticality | Falsified by DNABERT-2 C ≈ 0. **STAYS RETIRED** — re-measured on the clean eager kernel, C peaks at +0.0625 and ends −0.0221, i.e. still ≈ 0. The falsification never depended on the retracted KL value. See N-011. |
| X-006 | "DNABERT-2 has C ≈ 0 immediately and the largest impulse KL (≈0.31)" (was C-014) | The 0.31 was the **noise floor of a nondeterministic Triton kernel**, not a measurement. Two identical passes differed by KL = 0.305. On the deterministic eager path the primary-dose value is **1.8e-8**. Retired, not revised — the number was never a measurement of anything. See D-014, N-007. |
| X-007 | Granularity of the causal object differs across models: PR 3.64 (DNABERT-2) / 4.56 (EUK) / 22.7 (NTv3) / 122 (Evo1) / 2192 (PROK, contested) vs 1.02–1.24 for published NLP SWs, presented as a general NLP-vs-genomic granularity contrast (was C-032) | E7's exact spectral metric (`q1`, `PR_spec`, not the diagonal approximation) does not reproduce this as a domain-general distinction: GENERator EUK's exact `PR_spec` (1.0653) sits squarely inside the published-NLP range (1.02–1.24), not the genomic pole the old claim implies. Every genomic model's diagonal PR overstates its exact `PR_spec` by a large, consistently-directional factor (EUK 4.3×, DNABERT-2 2.4×, NTv3 3.5×) — a systematic bias in the diagonal approximation itself, matching E5's independent `f_cross` finding. Not a colleague finding — decided from this repo's own E5–E7 work alone. Old PR values and their evidence path (`experiments/E4_granularity/CANONICAL_TABLE.md`) preserved historically, not mutated. Superseded in part by C-034. See `experiments/E7_exact_dimensionality/C032_RETIREMENT_RECOMMENDATION.md`. |
| X-008 | GENERator PROK super-weight at layer 2 / row 1,927 (out_max=506,014), and everything keyed to it: the r=±0.710 write-direction-convention sign story (was C-020), the PROK shuffle-sensitivity finding (was C-021's PROK half), the PROK SAE, and the kingdom-composition/sign-reversal narrative | Detected using `--probe human_promoter` — a eukaryotic (human ACTB) sequence — fed into the prokaryote-specialized model. Corrected re-detection with a matching prokaryotic probe (`pseudomonadota`, already present in `probes/dna_probes.py`) finds the real PROK super weight at layer 8 / row 260 (rank 1/3,072, content-invariant across all 4 canonical probes). The old channel's stored `‖U_k‖_F` value independently failed to reproduce at its own layer (N-009, 2026-08-12) — the eukaryotic-probe diagnosis is the credible explanation for that gap, and its own independently-computed rank for the contaminated channel (1289/3072) matches N-009's own recomputation exactly. Retired 2026-08-17 per **D-024** (author decision to adopt colleague mechanism-session results). Corrected findings: **C-001** (location), **C-041** (hexamer causal test), **C-042** (GC-cost kingdom contrast). |
| X-009 | NTv3 5-seed splice replication: ΔMCC = −0.119 ± 0.054, 5/5 seeds negative, p = 0.008 (was C-029) | Produced under a silent truncation bug: `run_gue_multiseed.py`'s task-key fallback resolves `_MAX_LEN["reconstructed"]=80` (tuned for DNABERT-2's BPE tokenizer) whenever `--max_length` isn't passed; the exact launch script behind this result (`submit_ntv3_splice_multiseed.sh`) never passed it. NTv3 tokenizes at nucleotide level, so this truncated every splice window to its first 80bp, before the splice junction. **Independently confirmed by direct code trace, not colleague prose** (see the 2026-08-14 update to N-014). Refit at the correct `max_length=400` reaches MCC 0.86–0.91 with SW ablation effect −0.02pp (no effect). Retired 2026-08-17 per **D-024**. Corrected finding: **C-029** (same ID, updated in place — NTv3 does not functionally replicate the SW-ensemble effect). |
| X-004 | DNABERT-2 ensemble behaviour is a *consequence* of C ≈ 0 | Association only; no causal evidence. May return as `consistent-with` after E4 |
| X-005 | Broadcast headroom bounds broadcast | One extreme case; demoted to methodological covariate (C-018) |

**N-006 — X-003 may be un-retired.**
X-003 was retired on exactly one piece of evidence: DNABERT-2 having C ≈ 0 alongside the
largest impulse KL. N-004 shows that measurement was taken through a noise-dominated
instrument. The retirement is therefore **conditional on the E2 re-measurement**:

- Re-measured DNABERT-2 C ≈ 0 → X-003 stays retired, C-015 restored, `CLAUDE.md`'s
  constraint becomes final.
- Re-measured DNABERT-2 C high → **X-003 returns to the live ledger**, D-004's rationale
  fails, and `CLAUDE.md`'s "C is not necessary for criticality" is wrong as written.

Both outcomes are reportable and neither is preferred. The existing framing is not to be
protected. Same applies to X-004, which depends on the same C ≈ 0 value.

Unaffected either way: C-027, C-028. The ablation phenotype is independent evidence.
