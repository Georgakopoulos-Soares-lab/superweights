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
| C-001 | R1 | ‖U_k‖_F ranks the empirical SW row 1/3,072 at the step-up layer in GENERator EUK and PROK | | layer median, 12× / 18× ratio | **hold — see N-009** | **do not migrate until the PROK layer is pinned** |
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
| C-017 | R3 | **[THESIS SLOT]** relation of T to criticality | | | pending | **E2 done; branch A vs B NOT assigned — author's call** |
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
| C-032 | R2/R3/R6 | Granularity of the causal object differs across models: PR 3.64 (DNABERT-2) / 4.56 (EUK) / 22.7 (NTv3) / 122 (Evo1) / **2192 (PROK, contested)** vs 1.02–1.24 for published NLP SWs | **`experiments/E4_granularity/CANONICAL_TABLE.md`** | shape-verified per model; PROK row marked CONTESTED (C-001 on hold) | supported | E4 canonical table done — association only |

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

## Retired claims

| ID | Claim | Why retired |
|---|---|---|
| X-001 | "SW neighbourhood is extremely tolerant to pruning" / shadow redundancy | Contradicted by our own pruning sweep: near-SW −1.60 pp vs. random −0.82 pp at 20% |
| X-002 | "We scanned eight genomic language models" (as a uniform benchmark) | Coverage is asymmetric; replaced by the coverage table |
| X-003 | (‖U_k‖_F, C) jointly predict criticality | Falsified by DNABERT-2 C ≈ 0. **STAYS RETIRED** — re-measured on the clean eager kernel, C peaks at +0.0625 and ends −0.0221, i.e. still ≈ 0. The falsification never depended on the retracted KL value. See N-011. |
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
