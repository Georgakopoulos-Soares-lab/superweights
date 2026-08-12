# PAPER_OUTLINE.md — frozen skeleton

Status: **FROZEN** as of Phase 0. Changes require a DECISIONS.md entry.

## Thesis

Gated MLPs create structurally predictable high-gain channels, but architectures differ
fundamentally in how these signals are routed and recruited for biological computation.

Secondary thesis sentence for the Discussion:
**Structural origin generalizes; biological recruitment does not.**

## Positioning

Mechanistic-interpretability spine, computational-biology venue.
Targets: Genome Biology, Nature Methods, NAR Genomics & Bioinformatics, Patterns.

### "Why genomics" paragraph (intro, paragraph 2)

> Genomic LMs provide unusually informative tests of whether super-weight mechanisms
> generalize beyond conventional NLP transformers, because they span markedly different
> sequence-mixing architectures and training objectives while also offering biologically
> interpretable sequence statistics and matched model contrasts. In particular,
> GENERator EUK and PROK provide the same architecture and training framework applied to
> distinct genomic corpora, allowing structural effects to be separated from
> corpus-dependent functional recruitment.

**Do not** argue that NLP lacks non-attention mixers. It doesn't (Falcon-Mamba, Jamba,
Zamba, StripedHyena itself). The matched-corpus contrast is the part with no NLP
equivalent — lead on that.

---

## Results sections

### R1. A closed-form, forward-pass-free predictor of high-gain gated-FFN channels

- Derivation of ‖U_k‖_F from the rank-1 outer-product bound.
- Per-*i* decomposition c_{k,i} = W_down[k,i]² · ‖W_gate[i,:]‖² · ‖W_up[i,:]‖².
- **NLP external validation, two-level**: published SW output row *k* rank, and whether
  the published scalar *i* dominates within that row. Llama-7B / Mistral-7B / OLMo-7B
  against Yu et al. (arXiv:2411.07191).
- **Prospective lock**: one NLP model Yu et al. did not cover, predicted cold, then confirmed.
- Genomic models under identical analysis.

Wording: "recovers the published super-weight output rows" unless scalar recovery holds.

### R2. High-gain channels exhibit divergent functional criticality across genomic architectures

- Model coverage table up front (below). No "eight models" framing.
- Detect-then-ablate, ranks, ΔPPL, random-row controls.
- DNABERT-2 L7/r603 outlier and its residual-attribution resolution — **main text**. It
  clarifies what the mathematical object measures: predictor succeeds at source layers,
  need not succeed at propagator layers.

### R3. Impulse response separates routing regimes

Metric definitions, stated crisply:
- **T** — total perturbation transmission / amplification downstream of injection.
- **C** — retention of the original residual coordinate.
- **KL** — functional change in the output distribution.

Key conceptual point: **C describes routing geometry, not criticality.**

Thesis sentence is a SLOT — see `PHASE_1_BLOCKING.md` for the two drafted variants,
resolved by the Evo1 run.

Broadcast observability covariate: H = (L − ℓ_source − 1) / L. NTv3 (ℓ=11, L=12) has
H ≈ 0.08. This is a **methodological covariate**, not a mechanistic law. One extreme
case does not establish that headroom bounds broadcast.

### R4. What the amplifier computes

- Hexamer scan; activation → ablation-KL relationship (r = +0.437 EUK, +0.710 PROK on
  write magnitude). Sign convention gets one sentence and a footnote, not three paragraphs.
- **Compact shuffle experiment** — kept, because it pairs with R3:
  - EUK: no detectable shuffle sensitivity at any order; C = 1.0 end-to-end.
  - PROK: shuffle-sensitive at every order; coordinate diffusion by ~L14.
  - Phrasing: "the two models differ jointly in input-context sensitivity and downstream
    routing." No causal arrow.
- Corpus-prior fork (conditional, see PHASE_1): raw marginal k-mer frequency vs.
  deviation from a lower-order Markov expectation.

### R5. Bidirectional steering

Headline if it works:
> A single amplifier pathway bidirectionally steers generated sequence composition
> without comparably degrading generation quality.

Not a headline if it fails, but **still reported** in two sentences. See
`docs/prereg/PREREG_steering.md` for pre-committed language.

### R6. Functional recruitment differs by model

- GENERator 35-task dissociation. Main text shows: fungal species, one representative
  chromatin/composition task, splice, one local promoter/TF task, plus a compact
  heatmap/distribution across all 35. Full table → supplement.
- DNABERT-2 splice −25.5% + per-row ablation showing ensemble behaviour.
- NTv3 five-seed replication. **Metric issue must be resolved** — see PHASE_1 item 5.
- Apple-style single-neuron results slot in here and in R2/R3. They do **not** get their
  own section.

Conclusion: structurally related amplifiers are recruited for very different functions.

### R7. Compression — one paragraph

> The NLP heuristic of simply preserving SWs at high precision did not transfer:
> including versus exempting the genomic SW rows during whole-model INT4 produced
> differences below the current resolution.

Everything else → supplement. No "shadow redundancy."

### Discussion + explicit Limitations block

Keep the Limitations block from v15 — it is one of the better-written parts of the draft.
Update it against the new structure.

---

## Figures

| # | Content |
|---|---------|
| 1 | Structural predictor: derivation schematic + NLP external validation + genomic c_{k,i} concentration |
| 2 | Structural concentration vs. functional criticality: genomic panel, ablation effects, source/propagator distinction |
| 3 | Routing regimes: T/C/KL trajectories + controls across architectures |
| 4 | What it computes and causal steering: EUK/PROK hexamers, shuffles, steering dose-response |
| 5 | Functional recruitment: GENERator task spectrum + DNABERT-2/NTv3 splice effects |

---

## Model coverage table (goes in R2, main text)

Fill in and keep honest. `✓` = done, `—` = not applicable/not run, `✗` = blocked.

| Model | Arch class | Cold-weight ‖U_k‖_F | Detection sweep | Ablation ΔPPL | Fine-tune + task ablation | Impulse T/C/KL | Notes |
|---|---|---|---|---|---|---|---|
| GENERator EUK 3B | SwiGLU decoder | ✓ | ✓ | ✓ | ✓ (35 tasks) | ✓ | |
| GENERator PROK 3B | SwiGLU decoder | ✓ | ✓ | ✓ | — | ✓ | |
| DNABERT-2 117M | GLU encoder | ✓ | ✓ | ✓ | ✓ (3 GUE tasks) | ✓ | |
| NTv3 650M | gated-FFN encoder | ✓ | ✓ | ✓ | ✓ (splice, 5 seeds) | partial | H ≈ 0.08 |
| Evo1 7B | StripedHyena hybrid | ✓ | ✓ | ✓ | — | **PENDING** | blocking |
| HybridNA 7B | Transformer-Mamba2 | — | — | ✓ | — | — | ablation only |
| MegaDNA 145M | EMA | — | — | ✓ | — | — | ablation only |
| Evo2 7B | StripedHyena | — | — | ✗ | — | — | needs Singularity container |
| Caduceus-PS | bidirectional SSM | ✓ (static) | ✗ | ✗ | — | — | CUDA-extension incompatibility |
