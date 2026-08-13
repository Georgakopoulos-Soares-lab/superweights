# Analytical handoff — genomic super-weight project

**Audience:** an overseeing LLM proposing next steps. **Written:** 2026-08-10.
**Scope:** spec-v3 (generalization/compression) + spec-v4 (mechanism) sessions.
**Convention:** every claim is tagged **[MEASURED]**, **[INFERRED]**, or **[UNTESTED]**.
Effects are absolute percentage points (pp) unless noted. Seed 42 for all controls.

---

## 0. One-paragraph state

DNABERT-2 (encoder) has a pair of down-projection rows whose *joint* ablation collapses
splice accuracy by ~-33 pp while either alone costs ~0. The pair structure pre-exists
fine-tuning, survives its strongest available control, and its mechanism is joint carriage
of the layer-9 residual-stream norm. GENERator (decoder) instead has a single BOS-anchored
massive activation that is a textbook attention sink, causally steers generated GC, and is
content-independent — a clean decoder/encoder dissociation. **However**: the functional
collapse does *not* replicate in a second encoder (NTv3) even after we fixed the bug that
had made NTv3 untrainable, and the obvious mechanistic explanation for that failure
(norm dominance) is falsified. The encoder functional claim is therefore **n=1**.

---

## 1. Established results, with numbers

### 1.1 Redundant ensemble in DNABERT-2 [MEASURED]
| quantity | value |
|---|---|
| top-7 epistatic pairs structurally related (same-layer or same-row) | 7/7, **p = 0.00014** (n=5 seeds) |
| splice all-10 ablation | **-26.84 ± 2.56** vs sum-of-parts -3.51 |
| seed-0 critical pair, separate vs joint | -0.13 → **-33.76** |
| uniform-random pair control | +0.006 |
| replication | 3 tasks (splice, promoter, histone) |

### 1.2 Intrinsic to pretraining [MEASURED]
Pretrained DNABERT-2, MLM loss, no task head, held-out hg38, fixed mask (seed 42):
- same critical pair L9r264+L9r294, epistasis **+2.0118**
- enrichment top-3 3/3 (p=0.032), top-5 5/5 (p=0.0025), top-7 6/7 (p=0.0035)
- random-pair floor sd **0.0000147** → top pair is **136,521×** it
- k-of-N cliff at k=5 (+0.112 → **+2.059**, 18.5×); all-10 +2.617 vs sum-of-parts +0.727
- pretrained vs fine-tuned epistasis rank correlation ρ = **+0.316, p = 0.034**

**Interpretation [INFERRED]:** fine-tuning *selects* among pre-existing redundancies rather
than creating them. Supported by all three seed-specific critical pairs being superadditive
in the base model.

### 1.3 Mechanism: joint norm carriage [MEASURED, one pair, one seed]
Layer-9 output residual stream, splice test set:

| condition | ‖h‖ | ch264 | ch294 |
|---|---|---|---|
| baseline | 17.20 | 8.83 | 9.81 |
| ablate r264 | 13.64 (-21%) | 0.03 | 9.81 |
| ablate r294 | 13.66 (-21%) | 8.83 | 1.13 |
| **ablate both** | **7.14 (-58%)** | 0.03 | 1.13 |

Norm drop is itself superadditive (10.06 observed vs 7.10 additive) → downstream LayerNorm
rescaling. **Two competing geometries were tested and KILLED:**
- depth redundancy on the wire: ablating *all four* writes to channel 603 leaves **91.9%**
  of its readout value → later writes do not restore the wire
- symmetric head readout: head |w| on r294 at percentile 98 but **r264 at percentile 31**
  (below the random mean) → head does not read both
- additive circuit prediction fails by **5.15×** (logit shift 0.396 predicted vs 2.038 observed)

### 1.4 Decoder/encoder dissociation [MEASURED]
| | GENERator EUK | GENERator PROK | DNABERT-2 |
|---|---|---|---|
| attention mass at pos 0 | **37.96%** | **28.72%** | *unmeasurable* |
| vs uniform (1/L) | 33.0× | 25.0× | — |
| heads with argmax at pos 0 | 78.3% | 77% | — |
| SW activation at pos 0 | 375,361 | 30,167 | — |
| activation vs other positions | **45,585×** | — | 0.8–79× |
| argmax at token 0 | **60/60 (100%)** | — | **0/40 (0%)** |
| dinuc-shuffle ratio | 1.0000 | 1.0000 | ~1.0 |

**Encoder positive result [MEASURED]:** DNABERT-2's persistent channel 603 is a *composition
detector* — top-50 vs bottom-50 splice sequences by peak activation give GC **0.401 vs 0.585,
Cohen's d = -1.89** (L3) and **-1.73** (L5), with homopolymer Δ +2.20 / +0.88.

**Caveat [MEASURED-LIMIT]:** MosaicBERT does not expose attention maps
(`output_attentions` unsupported), so the encoder's *attention* half is unmeasured. The
dissociation rests on the activation half for the encoder side.
**Caveat [STRUCTURAL]:** in a causal decoder, position 0 attends only to itself, so the
shuffle ratio of exactly 1.0000 at pos 0 is guaranteed once BOS-anchoring is known. It
corroborates; it is not independent evidence.

### 1.5 Causal steering [MEASURED]
GENERator EUK, scaling the L4/r2371 write, 24 prompts, fixed sampling seed:

| scale | 0.0 | 0.5 | 1.0 | 2.0 | 5.0 |
|---|---|---|---|---|---|
| SW-row GC | **0.2944** | 0.3863 | 0.3961 | 0.4016 | 0.3549 |
| random-row GC | 0.3958 | 0.3958 | 0.3961 | 0.3967 | 0.3986 |

GC span **0.1072 vs 0.0028 → 38.59×**. PROK replicates directionally but far weaker
(span 0.0155, ratio 17.87×). **Monotone on the suppression side only** — amplification
saturates at 2.0 and reverses at 5.0. Do not present as a linear knob.

### 1.6 Compression [MEASURED]
- Per-row symmetric RTN preserves the row-max element with **0.000e+00** error at INT8/4/3/2
  (arithmetic: s = max|w|/qmax ⇒ max maps to qmax exactly). A super weight *is* that max.
- Group-wise g=64: the **top-4 outliers are all block maxima at every bit width** → group
  preserved exactly. M=16 at INT2: **0.0772** rel. error vs per-row's **0.6973** (9× better).
- Exemption benefit `exempt_sw − exempt_random`, 3 tasks × 4 granularities × 3 precisions:
  **mean +0.032 pp, median +0.004, 17 pos / 15 neg, t = 0.23** (32 informative cells).
- Destructive regime (damage +36.30 pp): SW-**group** protection gives **exactly +0.00** at
  M=1/4/16. Block-size axis: accuracy declines identically with/without protection
  (0.6155→0.5684 vs 0.6155→0.5686); max gain +0.79 pp at one non-monotonic point.
- Quality ranking (mean damage): group_64 **-5.46** < group_128 -5.90 < per_row -8.71 <
  per_tensor -25.03.

**Portable claim:** SW-aware exemption is a no-op wherever the SW defines its own
quantisation scale. The heuristic is *undefined* unless granularity **and** SW definition
(scalar vs group) are both stated.

### 1.7 Falsification test of the central claim [MEASURED, with an important limit]
| condition | epistasis |
|---|---|
| SW pair | **-17.72** |
| largest available non-SW pairs (n=24) | **-0.00 ± 0.03** (most extreme -0.11) |
| uniform random (n=24) | -0.01 |
| controls as extreme as SW | **0 / 24** |

**Limit [MEASURED]:** the controls were *not* actually norm-matched. SW channels rank **1 and
2 of 768** in every seed; the best available non-SW pair falls **5.2–6.3× short** of their
combined norm. The designed control condition does not exist in this model.
**Consequence [INFERRED]:** "super weight" and "top-2 norm carrier" are **not empirically
separable** in DNABERT-2 — they are the same two channels. State this rather than assume it.

---

## 2. Negative results (these constrain the paper most)

### 2.1 NTv3 functional replication FAILS [MEASURED] ← most important
After fixing the bug that made NTv3 untrainable (§3.1), NTv3 reaches MCC 0.86–0.91 and
baseline 0.9235–0.9430. On that working model:

| | DNABERT-2 | NTv3 (working) |
|---|---|---|
| top pair epistasis | **-11.56** | **-0.02** |
| all single-row effects | up to -5.4 | ≤ 0.09 |
| top-5 structurally related | 5/5, p=0.0025 | 3/5, **p=0.478** |

Structural replication is **n=2** (30 SW rows, persistent channel 1472 across L6–11,
pretrained superadditivity 7/7 top-7 p=0.0038). Functional replication is **n=1**.

Note: the *previously reported* NTv3 ablation effects (e.g. -24.73 pp single row) came from
the truncated-input model and are artifacts.

### 2.2 The obvious explanation for 2.1 is FALSIFIED [MEASURED]
Hypothesis: NTv3's SW is not norm-dominant. It is:

| | norm rank | value | next channel | gap |
|---|---|---|---|---|
| NTv3 L11 r1472 | **1 / 1536** | 2369.66 | 80.57 | **29.4×** |
| NTv3 L9 r1472 | **1 / 1536** | 175.90 | 17.69 | **9.9×** |
| DNABERT-2 L9 r294 | 1 / 768 | 9.81 | (r264 8.83) | rank-3 gap 4.5× |

NTv3's SW is *more* norm-dominant and still functionally inert.
**⇒ Norm dominance does not predict functional criticality. Joint norm carriage explains
DNABERT-2 and does not generalise.**

### 2.3 Circuit replication is mixed [MEASURED]
| pair | epistasis | logit obs/pred |
|---|---|---|
| L9r264+L9r294 | -33.76 | 5.15× |
| L3r603+L3r641 | **-8.13** | 1.93× |
| L3r86+L3r399 | **-0.22** | 1.17× |

Same-row wire test flat for all three (0.93–1.01 across surviving-write counts); head
weights asymmetric in all three. Both killed geometries are killed *consistently*.

### 2.4 Manuscript claims killed earlier this project
"histone intact" (same k=5 cliff in 2/5 seeds) · composition framing (R²=0.37, GC alone
0.035) · "no motifs enriched" (false under GC-matched null) · PROK cross-kingdom numbers
(contaminated probe) · **"shadow redundancy"** (layer-depth artifact: `layer_matched_random`
reproduces `prox_far` on all 3 tasks; `prox_far` zeroes all 768 rows of layer 0) ·
PROK SAE (3 defects + degenerate n_active≈1 correlations).

---

## 3. Infrastructure bugs found (several affect published numbers)

| bug | impact | status |
|---|---|---|
| **`_MAX_LEN["reconstructed"]=80` is tokenizer-specific** | NTv3 is nucleotide-level, so 400 bp → 400 tokens; 80 truncated every splice window to its **first 20%**. All manuscript NTv3 GUE numbers are from truncated-input models. | fixed via `--max_length`; **manuscript re-run owed** |
| `_NTv3Classifier` padded to a *minimum*, not a *multiple* of 256 | NTv3's conv/deconv U-Net skip connections require exact multiples; 400 tokens crashed | fixed |
| NTv3 detection probe length | all canonical probes 504 bp; 504 mod 256 = 248 → detection crashed inside InstaDeep code. Not their bug. | fixed via `--pad_to_multiple 256`; index went **1 → 30 rows** |
| `sae/collect.py` ±60,000 fp16 clamp | clips 0.195% of tokens but destroys **98%** of SW-channel variance | patched (`--store_dtype float32`) |
| `sae/train.py` no standardisation | SW channel carries **6,888×** median sd; dead features **74.9% → 0.5%** after fix | patched (`--standardize`) |
| `sae/analyze.py` raw-vs-scaled mismatch | manufactured the `n_active=1 → r=−0.9949` artifact **even with a healthy dictionary** | patched; `data_scale` now persisted |
| proximity metric row coordinate | meaningless under d_model permutation symmetry | documented |
| harness reports relative % labelled "pp" | −9.39 rel = −7.83 pp at baseline 83.46 | flagged, re-check owed |

**Standing methodological rule adopted:** never report a correlation without its `n_active`.

---

## 4. Self-corrections made (for calibration)

1. "Per-row RTN preserves the super weight bit-exactly" → true only of the **single
   scale-defining element**; a 16-element SW *group* carries 70% mean rel. error at INT2.
2. "SW spikes in 0.195% of tokens" (offered as a content-driven definition) → **1/512 =
   0.1953%**, i.e. one BOS token per chunk. A positional artifact.
3. sd imbalance "~2,000×" → **6,888×** against the channel median.
4. An early "standardization still collapsed the SAE" reading → my own preprocessing bug.
5. "same-layer pairs" → generalised to "structurally related" only after promoter
   contradicted it.

---

## 5. Open questions, ranked by value-per-GPU-hour

**Q1. Why is NTv3 functionally inert?** Two untested hypotheses [UNTESTED]:
 (a) DNABERT-2 has **two co-dominant** channels (9.81, 8.83) while NTv3 has **one** dominant
 singleton (2369 vs 80) — the collapse may require *co-dominance*, not dominance;
 (b) NTv3's U-Net deconv tower adds skip residuals from the conv tower and may route around
 single-layer damage. Test (b) by ablating skip connections; test (a) by searching NTv3 for
 co-dominant channel *pairs* and ablating those. **This decides whether DNABERT-2 is special
 or the first case of a pattern, i.e. whether the paper is a case study or a class result.**

**Q2. A third encoder with co-dominant pairs.** Direct test of hypothesis (a) and the only
route to functional n=2. Requires fine-tuning a third genomic gated-FFN encoder.

**Q3. Does the norm-carriage mechanism hold across seeds/pairs?** Currently one pair, one
seed. Cheap to extend; L3r603+L3r641 (-8.13) is the natural second case.

**Q4. Encoder attention mass.** Blocked by MosaicBERT. Would need a re-implementation or a
different encoder that exposes attention. Completes the dissociation symmetrically.

**Q5. Protein-LM arm.** Untouched, correctly out of scope. A "genomic and protein LMs" title
requires ≥1 real protein result; otherwise retitle to genomic LMs.

---

## 6. Recommended framing given current evidence

- **Safe:** decoder super-weights are BOS attention sinks (n=2 decoders, 25–33× uniform),
  causally steer composition (38.6×/17.9× vs random), and are content-independent.
- **Safe:** the compression theorem, scoped by granularity and SW definition.
- **Safe but must be scoped to one model:** the redundant-ensemble finding, its
  pretraining-intrinsic nature, and joint norm carriage.
- **Must appear in the abstract:** functional replication is n=1; NTv3 replicates structure
  but not function *and is more norm-dominant*, which breaks the natural generalisation.
- **Do not claim:** norm carriage as a general encoder mechanism; that "super weight" is
  distinguishable from "dominant norm carrier" in DNABERT-2.

---

## 7. Artifact index (all under `results/mechanism/`, 82 files, canonical index untouched)

`CHECKPOINT_1_PRETRAINED_EPISTASIS.md` · `CHECKPOINT_2_SECOND_ENCODER.md` ·
`CHECKPOINT_3_GROUP_SCALE_PRESERVATION.md` · `CHECKPOINT_1_MECHANISM.md` ·
`CHECKPOINT_2_TIER2.md` · `FALSIFICATION_NORM_VS_SUPERWEIGHT.md` ·
`SUPERADDITIVITY_AND_COMPOSITION_REPORT.md` (§1–11) ·
`epistasis_pretrained_vs_finetuned.{json,csv,png}` ·
`second_encoder_epistasis_ntv3_pretrained.{json,csv}` ·
`second_functional_encoder_ntv3.json` · `super_weight_index_ntv3_deep.json` ·
`compensation_circuit{,_L3pair,_L3_86_399}.json` · `norm_matched_control.{json,csv}` ·
`attention_sink_implicit_bias.json` · `sw_steering_generation{,_prok}.{json,csv}` ·
`scale_preservation_grouped.{json,csv}` · `exemption_all_granularities_{splice,promoter,histone}.json` ·
`compression_destructive_and_activation.{json,csv}` ·
`ensemble_encoding_{characterization,direct_dnabert2}.json` · `ntv3_splice_fixed/seed_{0..4}/`

**Known gap:** `compression_destructive_histone` failed on a trivial `num_labels` mismatch
(histone is 2-class, script hardcodes 3). Cosmetic; splice already answers T2.3.
