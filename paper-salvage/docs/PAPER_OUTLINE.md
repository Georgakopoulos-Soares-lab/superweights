# PAPER_OUTLINE.md — frozen skeleton (v2, 2026-08-13)

Status: **FROZEN** as of the 2026-08-13 reconciliation pass. Changes require a DECISIONS.md
entry. This supersedes the v1 outline (R1–R7, U_k-led) per `DECISIONS.md` D-016. v1's content
is not deleted; `docs/MANUSCRIPT_MIGRATION_MAP.md` records where each old section's material
goes under this skeleton.

## Thesis

**What survives — and what fails — when the NLP "single super-weight" concept is transferred
to genomic foundation models?**

Working headline message (refine wording as evidence allows; do not lose the substance):
super-weight-like activation concentration transfers to genomic language models, but the
causal object is not universal — it can appear as a canonical decoder super-activation, a
distributed row-ensemble in an encoder, or a false-feeling structural candidate that turns out
non-load-bearing. These differences expose limits of single-weight detection and
precision-preservation heuristics carried over from NLP.

Secondary thesis sentence for the Discussion (retained from v1, still true under the new
framing): **Structural origin generalizes; biological/functional recruitment does not.**

## Positioning

Mechanistic-interpretability spine, computational-biology venue. Targets unchanged: Genome
Biology, Nature Methods, NAR Genomics & Bioinformatics, Patterns.

### "Why genomics" paragraph (intro, paragraph 2) — retained verbatim from v1

> Genomic LMs provide unusually informative tests of whether super-weight mechanisms
> generalize beyond conventional NLP transformers, because they span markedly different
> sequence-mixing architectures and training objectives while also offering biologically
> interpretable sequence statistics and matched model contrasts. In particular,
> GENERator EUK and PROK provide the same architecture and training framework applied to
> distinct genomic corpora, allowing structural effects to be separated from
> corpus-dependent functional recruitment.

**Do not** argue that NLP lacks non-attention mixers. It doesn't (Falcon-Mamba, Jamba, Zamba,
StripedHyena itself). The matched-corpus contrast is the part with no NLP equivalent — lead
on that, but see D-020/N-015: the PROK half of that contrast currently rests on a contested
artifact (C-001 on hold) and must not be presented as settled until resolved.

---

## Results sections

### R1. Where NLP super-weight criteria transfer — and where they fail

- Anchor against Yu et al. (arXiv:2411.07191). Retrospective NLP recovery (C-004, C-005:
  rank 1/4,096 row and scalar, Llama-7B/Mistral-7B/OLMo-7B, top-1 scalar share 0.89–0.99,
  participation ratio 1.02–1.24) appears here as **calibration/bridging evidence**, not the
  paper's main contribution.
- `‖U_k‖_F`/Frobenius is introduced as a **known/confirmatory structural signature**
  (related to prior amplifier/massive-activation work — Sun et al., citation unresolved, see
  `docs/REFERENCE_AUDIT.md`), not a novel headline predictor. State plainly that it is
  leaky/non-universal: DNABERT-2 itself contains a row ranked 706/768 (C-010), resolved as a
  propagator-not-source layer, which is presented here as evidence that structural detection
  alone is insufficient, not buried as a footnote.
- Genomic cold-weight ranks for GENERator EUK/PROK\*, DNABERT-2, NTv3 (C-002, C-003; PROK
  marked contested, see below) under identical analysis.
- Evo1 may appear here briefly as a case where the structural signature fires
  (`‖U_k‖_F` concentrates) but the candidate is not load-bearing (C-009) — framed as "the
  detector alone does not establish criticality," not as a numerical-saturation false
  positive (that specific diagnosis has no artifact — see `NEW_DIRECTION_EVIDENCE_AUDIT.md`
  item 10, D-019).
- \*PROK's own C-001 is on hold (N-009); do not present the PROK cold-weight rank as settled
  without flagging the hold.

### R2. DNABERT-2 contains a redundant functional pair — **BLOCKED, no artifact**

This section cannot currently be written. The claimed mechanism (individual-row ablations
null, a specific 2-row pair collapsing splice by ≈−33pp, superadditivity) has no supporting
script, log, or JSON anywhere in this repository (`NEW_DIRECTION_EVIDENCE_AUDIT.md` items
1–4). The real, adjacent, and already-established result — the 10-row super-row ensemble
(C-027, C-028: −25.5±0.7pp mean, max single-row −1.45%, all 10 required for the full
collapse) — remains available and may anchor a section on **distributed/ensemble causal
granularity in DNABERT-2** if the author chooses, but that is a different claim from a
redundant pair and must be labeled as such. Do not draft R2 as "the redundant pair" until the
pair-ablation artifacts exist.

### R3. The pair is pretrained and acts through joint norm carriage — **BLOCKED, no artifact**

Same status as R2, of which it is a refinement (superadditivity, base-vs-fine-tuned
comparison, layer-9 joint residual-norm carriage, top-2-residual-norm-carrier limitation) —
none of it exists in this repository. R2/R3 may be merged once (if) the underlying
experiments exist; until then neither is drafted.

### R4. GENERator shows a canonical decoder super-activation / BOS attention-sink phenotype
— **BLOCKED, no artifact**

No script in this repository computes an attention weight, an argmax over attention, or a
uniform-expectation ratio for any GENERator model (`NEW_DIRECTION_EVIDENCE_AUDIT.md` item 5).
The one adjacent, already-measured fact — SW peak **activation** (not attention) sits at
sequence position ∈ {0, 1} in the existing counterfactual-swap data
(`results/sw_counterfactual_swap_generator.json`) — is real but does not establish an
attention-sink phenotype and should not be written as though it does.

### R5. The GENERator EUK channel causally steers generated composition — **BLOCKED, no
artifact**

E3 (`prereg/PREREG_steering.md`) specifies almost exactly this experiment — row-scaling dose
response, ≥10 matched random-row controls, ≥1,000 generated sequences, GC% primary readout,
entropy/PPL/complexity as degeneration controls, PROK expected to mirror EUK — but it was
never locked and never run (C-025 stays `pending`). No "38.6×" or any steering effect size
exists anywhere in this repository. This section cannot be drafted until E3 actually runs; it
is not this pass's job to run it.

### R6. Quantization granularity determines whether explicit SW protection is meaningful

- **This section can be drafted now**, on real evidence. Derivation: under the per-row RTN
  rule already implemented in this repo (`scale = max|row| / maxval`, verified directly in
  `scripts/compression/run_quantization_ablation.py` and
  `scripts/compression/run_whole_model_quantization.py`), the row's own max-magnitude element
  maps exactly to the quantizer endpoint and is therefore preserved by construction — new
  claim C-033, established from code, no run required.
- State this **precisely scoped to the per-row rule as implemented**. Group-wise
  quantization is not implemented anywhere in this repo and must not be discussed as though
  it were tested.
- Empirical support: C-031 (established) — INT4 with vs. without explicit SW exemption
  differs below resolution. Word the connection to C-033 as "consistent with," not
  "explained by" or "proven by" — C-031 itself has no recorded evidence path (`UNMIGRATED.md`).
- Do **not** claim a "deliberately destructive regime" test exists — the only more-aggressive
  script (INT2, `run_int4_multimodel_benchmark.py`) has no output artifact.
- Replaces the old R7 "compression, one paragraph" pruning/shadow-redundancy story (retired,
  X-001) entirely — see D-022.

---

### Discussion + explicit Limitations block

Keep the v1 Limitations block — still one of the better-written parts of the draft — and
update it against R1–R6: name each BLOCKED section explicitly as a limitation of *this
manuscript's current evidence*, not as a permanent architectural claim. State the R2/R3 top-
2-residual-norm-carrier ambiguity if/when R2/R3 are ever written (they are not yet).

### Broadcast/impulse (T/C/KL) — supplementary only

The E2 results (C-012, C-013, C-015, C-016 — all real, locked-protocol measurements) do not
get a main-text section under this outline (`DECISIONS.md` D-018). They may appear as:
supplementary characterization; descriptive evidence for a specific model where a main-text
claim needs it locally (none currently does); or methodology/provenance material. Do not
create a new main-text section around T/C, do not adjudicate Evo1 Branch A vs. B, and do not
restore C-017 as a headline thesis in this or any pass without a new instruction to do so.

---

## Figures

Figure numbering is not re-derived in this pass (no manuscript rewrite performed) — see
`docs/MANUSCRIPT_MIGRATION_MAP.md` for how existing figure panels map (KEEP / REWRITE / MOVE
TO SUPPLEMENT / DELETE / BLOCKED BY PROVENANCE) onto R1–R6.

---

## Model coverage table (goes in R1, main text) — retained from v1, still accurate

Fill in and keep honest. `✓` = done, `—` = not applicable/not run, `✗` = blocked.

| Model | Arch class | Cold-weight ‖U_k‖_F | Detection sweep | Ablation ΔPPL | Fine-tune + task ablation | Impulse T/C/KL (supplementary only) | Notes |
|---|---|---|---|---|---|---|---|
| GENERator EUK 3B | SwiGLU decoder | ✓ | ✓ | ✓ | ✓ (35 tasks) | ✓ | |
| GENERator PROK 3B | SwiGLU decoder | ✓ (contested, C-001 on hold) | ✓ | ✓ | — | ✓ | see N-009, N-015 |
| DNABERT-2 117M | GLU encoder | ✓ | ✓ | ✓ | ✓ (3 GUE tasks) | ✓ | strongest uncontested functional encoder example |
| NTv3 650M | gated-FFN encoder | ✓ | ✓ | ✓ | ✓ (splice, 5 seeds) | partial | splice result `contested`, see N-014 |
| Evo1 7B | StripedHyena hybrid | ✓ | ✓ | ✓ | — | measured, not thesis-bearing | structural/systemic account stands (D-019) |
| HybridNA 7B | Transformer-Mamba2 | — | — | ✓ | — | — | ablation only |
| MegaDNA 145M | EMA | — | — | ✓ | — | — | ablation only |
| Evo2 7B | StripedHyena | — | — | ✗ | — | — | needs Singularity container |
| Caduceus-PS | bidirectional SSM | ✓ (static) | ✗ | ✗ | — | — | CUDA-extension incompatibility |
