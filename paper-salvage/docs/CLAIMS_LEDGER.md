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
| C-004 | R1 | Predictor recovers published NLP super-weight **output rows** in Llama/Mistral/OLMo from cold weights: rank **1/4,096 in all three** | `e1_nlp_retrospective.json` | layer median (max/median 26.8–37.8×) | established | E1 done |
| C-005 | R1 | Published NLP **scalar** index dominates within the recovered row: rank **1** in all three, top1_share 0.89–0.99 | `e1_nlp_retrospective.json` | participation ratio 1.02–1.24 | established | E1 done |
| C-006 | R1 | Prospective cold-weight prediction on an unseen NLP model matches the forward-pass sweep | | timestamped lock | pending | E1 |
| C-007 | R2 | GENERator EUK SW ablation: PPL 3.40 → 785.5 (+23,026%); 10 matched random rows < 0.01% | | matched random | established | migrate |
| C-008 | R2 | GENERator PROK SW ablation: +25,975%; random controls within 0.03% | | matched random | established | migrate |
| C-009 | R2 | Evo1 is structurally concentrated (top 1.2–4.1%) but ablation gives ΔPPL = 0.0% | | random rows ≈ 0 | established | migrate |
| C-010 | R2 | DNABERT-2 L7 r603 predictor rank 706/768 reflects a propagator, not a source (residual carry-in −10.96 vs. local MLP −5.80) | | | supported | migrate |
| C-011 | R2 | Structural concentration does not imply functional criticality | C-009 + C-007/8 | | established | migrate |
| C-012 | R3 | GENERator EUK: C = 1.0 through all downstream layers | `sw_broadcast_impulse.json` (fixed-ε, superseded) | random-coord, neighbour-row, random-dense | pending-rerun | E2 re-run, D-011 |
| C-013 | R3 | GENERator PROK: C ≈ 0.92 → ~0 by L14 while total gain grows toward output | as above | as above | pending-rerun | E2 re-run, D-011 |
| C-014 | R3 | _(retired — see X-006)_ | | | **RETIRED** | D-014 |
| C-015 | R3 | C describes routing geometry, not criticality | C-012/13 only; C-014 retired | | **unsupported pending re-run** | **hold — X-003 is live, see N-006** |
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

## Retired claims

| ID | Claim | Why retired |
|---|---|---|
| X-001 | "SW neighbourhood is extremely tolerant to pruning" / shadow redundancy | Contradicted by our own pruning sweep: near-SW −1.60 pp vs. random −0.82 pp at 20% |
| X-002 | "We scanned eight genomic language models" (as a uniform benchmark) | Coverage is asymmetric; replaced by the coverage table |
| X-003 | (‖U_k‖_F, C) jointly predict criticality | Falsified by DNABERT-2 (C ≈ 0, largest KL) — **⚠ MAY BE UN-RETIRED, see note below** |
| X-006 | "DNABERT-2 has C ≈ 0 immediately and the largest impulse KL (≈0.31)" (was C-014) | The 0.31 was the **noise floor of a nondeterministic Triton kernel**, not a measurement. Two identical passes differed by KL = 0.305. On the deterministic eager path the primary-dose value is **1.8e-8**. Retired, not revised — the number was never a measurement of anything. See D-014, N-007. |
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
