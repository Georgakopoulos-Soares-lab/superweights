# PHASE 1 — Blocking experiments (Tier 0)

**Goal:** four experiments. Then stop and re-assess. Nothing else starts until these are
done or explicitly abandoned with a DECISIONS entry.

**This stop rule is binding.** The project failed once by scattering.

---

## E1 — NLP cold-weight validation, two-level (+ prospective lock)

**Why blocking:** highest value-per-hour experiment available. Weights only, no forward
pass, no GPU beyond loading. Converts the paper from a genomics observation into a
general method with external validation on a domain we did not tune on.

**Design**

1. Models with published super-weights (Yu et al., arXiv:2411.07191): Llama-7B,
   Mistral-7B, OLMo-7B. Extract the published `(layer, k, i)` coordinates.
2. **Level 1 — row recovery.** Compute ‖U_k‖_F at the published layer. Report the rank of
   the published output row *k* among all d_model coordinates, plus max/median ratio.
3. **Level 2 — scalar recovery.** Within row *k*, compute c_{k,i} and report the rank of
   the published scalar index *i* among all d_ffn contributors, plus the top-1 share and
   participation ratio.
4. **Prospective (D-006).** Pick one NLP model Yu et al. did **not** cover (Qwen3, Gemma,
   Phi — whatever loads trivially). Compute the predicted `(layer, k, i)` cold, lock it
   with `src/prereg_lock.py`, *then* run the forward-pass detection sweep and compare.

**Tooling:** `src/uk_frobenius.py` (already written, adapter registry covers
SwiGLU / packed-GLU / Evo1 l1-l2-l3 naming).

**Reporting rule:** "recovers the published super-weight output rows" if only Level 1
holds. Scalar recovery only if Level 2 also holds.

**Prereg:** `prereg/PREREG_nlp_prospective.md` (lock before step 4's sweep).

---

## E2 — Evo1 standardized broadcast

**Why blocking:** decides R3's thesis sentence, and nothing else.

**Context.** Across the three measured positives, criticality tracks **T**; **C** varies
wildly (EUK 1.0, PROK 0.92→0, DNABERT-2 ≈0). Evo1 is structurally concentrated
(‖U_k‖_F ranks 48–168 / 4,096, top 1.2–4.1%), its MLP writes ~165 into the candidate row,
the residual deposit is faithful (block_delta / mlp_out ≈ 1.0), ~44% survives into the
next block, and ablation is ΔPPL = 0.0%. The impulse run was prepared but never completed.

**Design.** _(As originally planned. **Superseded** — see D-011, D-013, D-015 and the
locked prereg. ε = 1.0 was replaced by AC-relative ε = α·std(h−mean h) with α = 0.01
primary / 1.0 secondary; the run is fp32 with a per-layer headroom column. Also note the
"same four control arms" was inaccurate when written: the harness implemented three, and
the fourth — matched-norm — was only added for the locked run.)_ Identical protocol to the
other four models — ε = 1.0 into the residual stream at the SW row/token at its source
layer, tracked across all 32 StripedHyena blocks, with the same four control arms:
random-coordinate, neighbouring-row, random-dense-direction, and matched-norm.

**Preregister the prediction before running.** See `prereg/PREREG_evo1_broadcast.md`.

**The two thesis-sentence variants (R3):**

- **If Evo1 T is low:**
  > Total broadcast gain T, not coordinate preservation C, predicts criticality; C
  > describes routing style.
  → Single-variable criticality predictor. Two-axis taxonomy. R3 becomes mechanistic.

- **If Evo1 T is high and ablation is still null:**
  > Neither T nor C alone is sufficient; criticality additionally requires that the routed
  > signal reach a readout the task depends on.
  → More work (needs a logit-attribution follow-up) but a more interesting paper if true.
  Log the follow-up as a new phase item, do not start it inside Phase 1.

---

## E3 — Bidirectional steering with degradation controls

**Why blocking:** every causal claim in v15 is destructive ("remove X, things break").
A positive intervention is the strongest single result available, and it is cheap.

**Design**

- Models: GENERator EUK (row 2371, L4) and PROK (row 1927, L2). PROK should **mirror**
  EUK because its write is negative.
- Interventions: activation steering of the SW channel **and** row scaling.
- Scale factors: at least 4, spanning both directions (e.g. 0.25, 0.5, 2, 4) plus baseline.
- Controls: matched random rows at each scale factor; ≥10 random rows.
- Sequences: enough per condition for a stable composition estimate (≥1,000 suggested).

**Measure, per condition:**

| Metric | Purpose |
|---|---|
| GC% of generated sequence | primary effect |
| k-mer distribution (1/2/6-mer) | is the effect compositional or k-mer specific |
| sequence complexity / repeat fraction | degeneration check |
| generation entropy | **the key discriminator** |
| perplexity / loss vs. baseline | degeneration check |
| KL vs. baseline output distribution | magnitude of intervention |

**The discriminator.** If GC shifts monotonically while entropy stays flat → steering. If
entropy climbs alongside the GC shift → the model is degrading and composition is being
read off degraded output. Monotonic + bidirectional + entropy-flat is very hard to explain
as damage.

**Prereg:** `prereg/PREREG_steering.md`, including pre-committed null language (D-008).

---

## E4 — c_{k,i} granularity decomposition across all genomic gated FFNs

**Why blocking:** answers "at what granularity is the amplifier concentrated?" — question 2
of the five. Free alongside E1 (same code path, weights only).

**Design.** For every gated-FFN genomic model (GENERator EUK/PROK, DNABERT-2, NTv3, Evo1),
at the SW layer(s), compute for the empirical SW row:

- top-1 share: max_i c_{k,i} / Σ_i c_{k,i}
- participation ratio: (Σ c)² / Σ c² — the effective number of contributing hidden units
- rank profile of contributors

**What would be a result.** If the NLP published scalar is the dominant *i* but GENERator's
row is spread across many *i*, that is a genuine cross-domain statement about what the
causal object is in each domain — not bookkeeping. Likewise if GENERator and DNABERT-2
differ from each other, that is the structural correlate of single-row vs. ensemble
behaviour.

**Do not** overclaim that a low participation ratio *causes* ensemble behaviour. Report the
association.

---

## Stop rule

When E1–E4 are complete: update `PROJECT_STATUS.md`, re-read `PAPER_OUTLINE.md`, and
decide what (if anything) Phase 1b contains. Candidates already identified, **none
pre-approved**:

- corpus-prior fork (gated on E3's outcome, D-009)
- logit attribution (only if E2 returns high-T-null-ablation)
- DNABERT-2 seeds 3 → 10
- a second non-attention mixer, to give the routing hypothesis a second architecture transition
- EUK SAE

If a Tier 0 experiment fails, its replacement comes from this list — one item, chosen
deliberately, logged in DECISIONS.md.
