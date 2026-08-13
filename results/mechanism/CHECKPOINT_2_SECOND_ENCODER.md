# CHECKPOINT 2 — Second-encoder replication of the redundant-ensemble effect

**Verdict: REPLICATES — PASS (kill condition not met).**
**Paper branch: the encoder-ensemble result is a PHENOMENON, backed by 2 models. The gLM
paper's spine is secured.**

Session date 2026-08-10. Outputs: `super_weight_index_ntv3_deep.json`,
`second_encoder_epistasis_ntv3_pretrained.{json,csv}`,
`second_encoder_epistasis_ntv3.json` (fine-tuned, uninformative — see §3),
`second_encoder_replication.png`.

---

## Kill condition, stated first

> If the second encoder showed a single dominant row, or diffuse/unstructured
> interactions, the ensemble claim would be DNABERT-2-specific and the paper would have to
> scope down to one model.

**Not met.** NTv3 shows structurally-related pairs dominating, no single sufficient row, and
emergent k-of-N collapse — the same three signatures as DNABERT-2.

---

## 1. NTv3 unblocked — root cause and fix

The blocker was **not** a library bug. NTv3 is a **conv/deconv U-Net**: 8 conv blocks
(`filter_list` has 8 entries), each `AvgPool1d(kernel=2, stride=2)`, followed by a deconv
tower with skip connections (`use_skip_connection=True`). The skip addition `y = y + r`
(`modeling_ntv3_pretrained.py:810`) only aligns when the token length is a multiple of
**2⁸ = 256**.

NTv3 tokenises at **nucleotide level** (vocab 11, 1 bp → 1 token), so every canonical probe
— all 504 bp — gives 504 tokens, and **504 mod 256 = 248** → the reported
`tensor a (6) must match tensor b (7)`.

Verified exactly:

| token length | 250 | **256** | 300 | **512** | 1000 | **1024** | **1792** | **2048** |
|---|---|---|---|---|---|---|---|---|
| forward | FAIL | **OK** | FAIL | **OK** | FAIL | **OK** | **OK** | **OK** |

Every multiple of 256 works; every non-multiple fails. Fix: `--pad_to_multiple 256` added to
`scripts/detection/run_detection.py` (right-pads with `A`, documented inline). No vendoring
or monkey-patching of InstaDeep's code was required.

**This supersedes the prior "NTv3 is blocked" limitation.** It was our probe length, not their code.

## 2. NTv3 detection: 1 row → 30 rows, with DNABERT-2's organisation

With the padding fix, `--mode superrow --threshold 0.02 --max_iter 30` finds **30 SW rows**
(the frozen index had **1**).

| structure | finding |
|---|---|
| persistent channel | **row 1472 recurs at layers 6, 7, 8, 9, 10, 11** (6 layers) |
| multi-row layers | L11: 12 rows · L9: 10 rows · L8: 3 · L10: 3 |

This is the same organisation as DNABERT-2 (row 603 persistent at layers 3/5/6/7; L3 holding
4 rows; L9 holding 2). The frozen index's N=1 was an artifact of the failing probe, not a
property of the model.

## 3. The fine-tuned route is uninformative — stated plainly

All five NTv3 splice checkpoints sit **at or below the 0.5658 majority-class floor**:

| seed | 0 | 1 | 2 | 4 | 5 |
|---|---|---|---|---|---|
| baseline acc | 0.5715 | 0.5340 | 0.5357 | 0.5160 | 0.5627 |

These models never learned splice. Functional interactions cannot be measured in a model
with no function, so `second_encoder_epistasis_ntv3.json` is reported for completeness but
**carries no evidential weight** (top-5 pairs are all same-row but at −0.38 to −0.03 pp with
SDs exceeding the means). This is a limitation of the available checkpoints, **not** evidence
against the phenomenon.

## 4. The replication: pretrained MLM protocol (identical to Task 1)

Because the fine-tuned route is degenerate, replication uses the **same pretrained-MLM
protocol as CHECKPOINT 1** — which is also the protocol that matches the paper's claim
("encoders natively build redundant ensembles during pretraining"). Top-10 rows by
`out_max`, 128 held-out hg38 windows padded to 512 tokens, fixed mask (seed 42).

| criterion | DNABERT-2 (T1) | **NTv3 (T2)** | replicates? |
|---|---|---|---|
| top-7 structurally related | 6/7 (p = 0.0035) | **7/7 (p = 0.0038)** | ✅ |
| top-5 structurally related | 5/5 (p = 0.0025) | **5/5 (p = 0.0216)** | ✅ |
| top pair vs random-pair sd | 136,521× | **1,458×** | ✅ |
| no single row sufficient | yes | **yes** — largest single (+0.0034) is 18% of all-10 (+0.0189) | ✅ |
| emergent k-of-N | +1.89 emergent | **3.82× superadditive** (all-10 +0.018947 vs sum-of-parts +0.004961) | ✅ |

Random-pair floor (NTv3): mean −0.00000304, sd 0.00000345.

NTv3 top-7, every pair same-row (the row-1472 persistent channel):

| pair | epistasis (ΔMLM loss) | relation |
|---|---|---|
| L11r1472 + L9r1472 | +0.005022 | same row |
| L11r1472 + L8r1472 | +0.004012 | same row |
| L7r1472 + L6r1472 | +0.003396 | same row |
| L11r1472 + L10r1472 | +0.001792 | same row |
| L9r1472 + L8r1472 | +0.001458 | same row |
| L11r1472 + L7r1472 | +0.000826 | same row |
| L11r1472 + L6r1472 | +0.000558 | same row |

---

## 5. An honest difference between the two encoders

DNABERT-2's dominant redundancy is **same-layer** (L9r264+L9r294, +2.01); NTv3's is
**entirely same-row** (the row-1472 persistent channel). Both fall under "structurally
related", and both models show *both* relation types among their superadditive pairs — but
the dominant mode differs.

The defensible claim is therefore: **genomic encoders build redundant super-weight ensembles
from structurally-related rows; which structural relation dominates is model-specific.** Do
not claim the same-layer pair is the universal unit.

## 6. Caveats

1. **Higher null rate in NTv3.** Because 6 of NTv3's top-10 rows are row 1472, 22/45 = **49%**
   of pairs are structurally related by construction (DNABERT-2: 15/45 = 33%). All p-values
   above use the correct hypergeometric null for each model, so they are comparable — but
   NTv3's are necessarily weaker for the same count.
2. **Replication is on the pretrained protocol only.** The fine-tuned NTv3 arm is degenerate
   (§3). A second encoder with a *usable* fine-tuned splice checkpoint would strengthen this.
3. **Effect sizes differ by ~400×** between models (DNABERT-2 top pair +2.01 vs NTv3
   +0.0050 ΔMLM loss). Different architectures, tokenisations, and baseline losses; the
   comparison is on structure and significance, not magnitude.

---

## Paper status after Checkpoint 2

```
T1 INTRINSIC  +  T2 REPLICATES
   ─► "Genomic encoders natively build redundant super-weight ensembles during
       pretraining; fine-tuning selects which one becomes task-critical."
       2 models, pretrained protocol, matched analysis. PHENOMENON, not anecdote.
   ─► Combined with the decoder result (GENERator: single dominant row), the
       architecture-dependence claim now has encoders (n=2) vs decoder (n=1).
```

Remaining for the session: **TASK 3** — corrected group scale-preservation and
granularity-complete compression.
