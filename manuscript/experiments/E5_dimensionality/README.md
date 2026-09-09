# E5 — structural/realized/causal dimensionality of gated-FFN high-gain rows

**Status:** Gate 0 only, pending. This experiment has its own directory and its own
preregistration ledger entries; it does not touch `manuscript/results/`,
`manuscript/docs/CLAIMS_LEDGER.md`, `PAPER_OUTLINE.md`, or `CLAUDE.md`. Nothing in the
existing feasibility audit (the S/A/R artifact published earlier this session) is modified.

## Premise correction (read before anything else in this experiment)

The scientific question motivating E5 is **not** "Frobenius/U_k recovers the published
super-weight extremely cleanly in NLP and fails in genomic language models." That framing was
the premise of an earlier request; the feasibility audit that preceded this experiment showed
it is not the right description of what the repository's own artifacts contain, and this
experiment does not inherit it.

**What the existing artifacts actually show** (all pre-existing, re-derived in the feasibility
audit, not re-measured here):

- **Row-level recovery is strong in both groups.** `‖U_k‖_F` ranks the empirically/publicly
  identified super-weight row **1/N** in six of eight models across both groups: Llama-7B,
  Mistral-7B, OLMo-7B (`results/e1_nlp_retrospective.json`), and GENERator EUK, DNABERT-2,
  NTv3 (`results/e4_granularity.json`, `results/e4_ntv3_shared_adapter.json`). Evo1 is
  48/4096 (98.85th percentile) — concentrated, not a miss. The one outright row-level failure
  is GENERator PROK (rank 1289/3072), and that failure is an **unresolved artifact-
  reproduction problem** (N-009: the stored value does not reproduce on current weights at its
  own claimed layer, cause undetermined), not a measured property of prokaryotic genomic
  language models. PROK is excluded from this experiment's primary panel for exactly that
  reason (see Exclusions below), not used as evidence of "genomic failure."
- **The striking existing contrast is within-row concentration, not row-level rank.**
  Top-1 share is 0.89–0.99 (PR 1.02–1.24) in the three NLP models vs. 0.18–0.39 (PR 3.6–22.7)
  in the three clean genomic models (`experiments/E4_granularity/CANONICAL_TABLE.md`). That
  granularity gap — not a recovery failure — is what this experiment investigates.
- **Scalar-level ground truth is asymmetric between the groups, and this is a design fact, not
  a result.** Yu et al. (2024) publish an independently validated scalar coordinate `i` for
  each NLP super-weight; no equivalent published scalar exists for any genomic model. The
  `CANONICAL_TABLE.md` "scalar rank of `i`" column is `n/s` for every genomic row **by
  definition**, not by omission. Any within-row comparison in this experiment is therefore
  read against each model's own empirically-detected winner index, never against a "true"
  scalar ground truth that only the NLP arm has.
- **The DNABERT-2 L7 706/768 result is a separate, local finding and is not used here.** It
  concerns a different layer (L7) than this experiment's DNABERT-2 target row (L5, row 603,
  rank 1/768) and was already resolved in the existing ledger as a propagator-not-source
  effect at that other layer (C-010). It is not evidence that the L5 canonical row is missed,
  and this experiment does not cite it as such.

This correction is recorded here, in this experiment's own directory, per instruction. It is
**not** written into `CLAUDE.md`, `PAPER_OUTLINE.md`, or `CLAIMS_LEDGER.md` in this pass — the
manuscript is not rewritten on the strength of a feasibility audit or a not-yet-run
experiment.

## Scientific question

Why do canonical NLP high-gain rows collapse onto essentially one scalar pathway, whereas
genomic high-gain rows remain distributed across several intermediate FFN pathways? Does this
structural dimensionality predict the dimensionality of the causal object?

The contribution is not "NLP vs. DNA because different architectures." The genomic side of
the primary panel already spans a SwiGLU decoder (GENERator EUK) and two encoder families
(DNABERT-2's packed-GLU BERT variant, NTv3's packed-GLU bidirectional transformer).

## Working hypothesis, decomposed into three separable, sequentially-tested claims

1. **Structural origin (Gate 0).** NLP scalar dominance is produced by unusually strong
   multiplicative alignment between the down-projection term `D_i = W_down[k,i]^2` and the
   gated-amplifier term `A_i = ‖W_gate[i,:]‖^2 ‖W_up[i,:]‖^2`, not merely by either factor
   having a more extreme marginal distribution.
2. **Realized dimensionality (Gate 1, gated on Gate 0).** The structural concentration
   difference survives contact with real activations on native-domain inputs.
3. **Causal dimensionality (Gate 2, gated on Gates 0 and 1).** Structural/realized
   participation ratio predicts how many ranked scalar pathways must be ablated to recover the
   full-row causal phenotype.

Each gate has a preregistered, mechanical stop rule. Failure at any gate stops the sequence;
no replacement hypothesis is generated in the same pass.

## Primary panel (fixed for all three gates)

| Group | Model | Checkpoint | Layer | Row *k* | d_model | d_ffn |
|---|---|---|---|---|---|---|
| NLP | Llama-7B | `huggyllama/llama-7b` | 2 | 3968 | 4096 | 11008 |
| NLP | Mistral-7B | `mistralai/Mistral-7B-v0.1` | 1 | 2070 | 4096 | 14336 |
| NLP | OLMo-7B | `allenai/OLMo-7B-0724-hf` | 1 | 269 | 4096 | 11008 |
| Genomic | GENERator EUK 3B | `GenerTeam/GENERator-v2-eukaryote-3b-base` | 4 | 2371 | 3072 | 8448 |
| Genomic | DNABERT-2 117M | `zhihan1996/DNABERT-2-117M` @ `7bce263b15377fc15361f52cfab88f8b586abda0` | 5 | 603 | 768 | 3072 |
| Genomic | NTv3 650M | `InstaDeepAI/NTv3_650M_pre` @ code rev. `0ecff3637f0d3ba5b686d1095083218157c2ca34` | 11 | 1472 | 1536 | 6144 |

Source: `results/e1_nlp_retrospective.json`, `results/e4_granularity.json`,
`results/e4_ntv3_shared_adapter.json`, cross-checked against
`experiments/E4_granularity/CANONICAL_TABLE.md`.

## Exclusions (fixed for all three gates — no post-hoc exclusion beyond these two)

- **GENERator PROK.** C-001 on hold (N-009, N-015). Its stored rank-1 artifact does not
  reproduce at its own claimed layer on current weights, and the cause is undetermined. No
  alternative layer is searched for this experiment, consistent with the standing rule that
  produced N-009 in the first place.
- **Evo1.** Its super-weight interpretation is itself contested (row 48/4096, structurally
  concentrated but ΔPPL = 0.0% under ablation — C-009/C-011). It may serve as a negative or
  pathological control under a future, explicit instruction, but is not part of primary
  inference here.

No model may be added to or dropped from the primary six after Gate 0 is locked, beyond these
two pre-specified exclusions.
