# CHECKPOINT 1 — Pretrained vs fine-tuned epistasis

**Verdict: INTRINSIC — PASS (kill condition not met).**
**Paper branch: "genomic encoders natively build redundant super-weight ensembles during
pretraining; fine-tuning reads them."** (the strong, architectural claim)

Session date 2026-08-10. Outputs: `epistasis_pretrained_vs_finetuned.{json,csv}`,
`epistasis_pretrained_vs_finetuned_matrix.png`.

---

## Kill condition, stated first

> If structurally-related pairs were NOT superadditive above the random floor in the
> pretrained model, the pair structure would be MANUFACTURED by fine-tuning and the
> central claim would narrow to "fine-tuning recruits a redundant pair."

**Not met.** The structure is present before any fine-tuning, and by a very large margin.

---

## Method

| | |
|---|---|
| model | pretrained DNABERT-2 117M, `AutoModelForMaskedLM`, **no task head, no fine-tuning** |
| readout | masked-LM loss (the pretraining objective) — Δloss **positive = worse** |
| data | 256 held-out hg38 windows (600 bp) from `random_262kb.bed`, 36,676 masked tokens |
| ablation | zero `down_proj` rows — identical convention to `run_sw_pairwise_epistasis.py` |
| pairs | all C(10,2)=45 SW pairs + k-of-N curve + 10 random non-SW pairs |
| seed | 42 throughout |

**Fixed-mask control.** One mask realisation (seed 42) is generated once and reused for
every condition, so all deltas are paired and mask sampling contributes zero variance.

**Compatibility note (documented, not a method change).** DNABERT-2 ships a Triton
flash-attention kernel incompatible with the installed Triton (`dot() got an unexpected
keyword argument 'trans_b'`). The fine-tuned classification runs never hit it because
nonzero attention dropout routes them to the PyTorch implementation
(`bert_layers.py:161`); the MLM config has zero attention dropout and reaches the broken
kernel. Setting `flash_attn_qkvpacked_func = None` forces the **same PyTorch attention
path every prior result used**. Recorded as `triton_attention_patched` in the JSON.

**Sign convention.** Fine-tuned analyses used Δaccuracy (negative = worse); this uses
ΔMLM-loss (positive = worse). Superadditivity here means **epistasis > 0**. The
fine-tuned matrix is sign-flipped in all comparisons below.

---

## Result 1 — the same critical pair, in the base model

Top-7 pairs by epistasis, pretrained:

| pair | epistasis (ΔMLM loss) | relation |
|---|---|---|
| **L9r264 + L9r294** | **+2.0118** | **same layer** |
| L5r603 + L6r603 | +0.3321 | same row |
| L6r603 + L7r603 | +0.1785 | same row |
| L3r603 + L3r641 | +0.1679 | same layer |
| L5r86 + L5r603 | +0.1230 | same layer |
| L5r603 + L7r603 | +0.1038 | same row |
| L6r603 + L5r86 | +0.0780 | unrelated |

Structural-relatedness enrichment: **top-3 3/3 (p = 0.032), top-5 5/5 (p = 0.0025),
top-7 6/7 (p = 0.00345)** — matching the fine-tuned pattern.

**Random-pair floor: mean −0.0000015, sd 0.0000147.** The top pair is **136,521× the
random sd**. This is not a marginal effect.

## Result 2 — the same k=5 cliff

```
k:       1       2       3       4        5        6        7        8       9      10
ΔLoss: +0.039  +0.045  +0.081  +0.112  +2.059   +2.454   +2.483   +2.422  +2.786  +2.617
                                        ↑ 18.5x jump when L9r294 joins L9r264
```
all-10 **+2.617** vs sum-of-parts **+0.727** → **+1.89 emergent**. The cliff sits at
exactly the same k as the fine-tuned splice curve, and for the same reason.

## Result 3 — pretrained and fine-tuned epistasis agree

Spearman ρ across all 45 pairs = **+0.316, p = 0.034** (pretrained Δloss vs sign-flipped
fine-tuned Δacc). Modest but significant — the pretrained model's interaction structure
predicts the fine-tuned one.

| pair | relation | PRETRAINED ΔLoss | FINE-TUNED Δacc (pp) |
|---|---|---|---|
| L9r264+L9r294 | same layer | **+2.0118** | **−11.56** |
| L5r603+L6r603 | same row | +0.3321 | −0.94 |
| L6r603+L7r603 | same row | +0.1785 | −0.29 |
| L3r603+L3r641 | same layer | +0.1679 | −5.20 |
| L5r86+L5r603 | same layer | +0.1230 | +0.17 |

---

## Interpretation

The redundant-pair structure — including the *specific* critical pair (L9r264+L9r294),
the k=5 cliff, and the enrichment of same-layer / same-row pairs — is **already present in
the pretrained model**, measured on the pretraining objective itself with no task head.

This resolves the seed-dependence that motivated the task. Fine-tuning does not *create*
redundancy; it **selects which of several pre-existing redundancies becomes task-critical**
(splice seeds 0/2/4 → L9 pair, seed 1 → L3r603+L3r641, seed 3 → L3r86+L3r399 — all three
of those are among the pretrained model's superadditive pairs). The seed-dependence is
therefore evidence *for* a latent ensemble, not against it.

**Both relation types are intrinsic** (3 same-layer and 3 same-row in the pretrained
top-6), so this is not the MIXED branch — the persistent-channel (row 603) redundancy and
the within-layer redundancy are both pretrained properties.

---

## Caveats

1. **Single model.** This establishes intrinsic-ness for DNABERT-2 only. Whether it is a
   property of *genomic gated-FFN encoders* is exactly what TASK 2 must decide.
2. **Readout differs by necessity.** MLM loss vs task accuracy are different scales; the
   cross-comparison is rank-based (Spearman) for that reason. The within-pretrained
   statistics (enrichment, random floor, k-of-N) are internally consistent and do not
   depend on the cross-comparison.
3. ρ = +0.316 is a *modest* correlation. The top pair agrees strongly; mid-ranked pairs
   reorder. The claim supported is "the critical redundancies pre-exist", not "the full
   interaction matrix is preserved through fine-tuning."

---

## Paper status after Checkpoint 1

```
T1 ─ INTRINSIC ─► "encoders natively build redundant ensembles during pretraining"
                   Strong, architectural. Seed-dependence becomes supporting evidence.
                   ↓
                  Still one model. T2 decides class property vs single-model finding.
```
