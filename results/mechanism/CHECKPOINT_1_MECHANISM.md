# CHECKPOINT 1 (spec v4) — Mechanism: what, how, and a second functional encoder

Session date 2026-08-10.

## Headline: two kills and a dissociation — reported first, per operating rules

1. **KILL — T1.2's two proposed compensation geometries are both falsified.** Neither
   "depth redundancy on the wire" nor "the head reads both coordinates" is the mechanism.
   The actual mechanism was found and is different (§T1.2).
2. **KILL — T1.1's premise that the SW ensemble encodes sequence content is falsified for
   GENERator.** The SAE code fires at **token position 0 (`<s>` BOS) in 100% of windows**;
   it encodes position, not sequence.
3. **DISSOCIATION (T2.1, answered early and decisively):** the decoder super-weight IS a
   BOS-anchored massive activation (the NLP attention-sink signature); the encoder
   super-weights are **not**. This is the "split by architecture" branch and it supplies the
   mechanistic correlate of the singleton-vs-ensemble taxonomy.
4. **Correction to a prior claim of ours** (§Correction) — the "0.195% spike rate" was a
   chunking artifact.

---

## T1.1 — What the ensemble encodes: **position, not content** (for the decoder)

Running the top-K activating-context battery on the 13 SW-correlated SAE features produced
two immediate red flags: **modal firing offset 0.0 with modal share 1.00 for every feature**,
and *seven features with byte-identical context statistics*. Both are explained by one fact,
which we then measured directly on GENERator EUK layer 4, channel 2371:

| | |
|---|---|
| argmax token position | **0 in 60/60 windows (100%)** |
| mean \|act\| at position 0 | **375,361** |
| mean \|act\| at position 1 | 54.97 |
| mean \|act\| at position 2 | 2.21 |
| **ratio pos0 / mean(pos1..79)** | **45,585×** |
| token at position 0 | `<s>` (BOS) |

The composition battery is correspondingly null: |GC Cohen's d| mean **0.139** (max 0.172),
homopolymer Δ −0.16, entropy Δ +0.015, CpG O/E Δ −0.028, and no motif enrichment above
+0.055. Contexts were indistinguishable from background because *all features select the same
window prefix*.

**Verdict:** the 13-feature code does not encode nucleotide composition or any motif. It
encodes **the BOS position**. This is the textbook massive-activation / attention-sink
signature reported for NLP LLMs, now measured in a genomic decoder.

### ⚠️ Correction to our own earlier claim

We previously reported a "quantitative definition of a super weight": median 169, 99.9th
percentile 426,422, **spiking in 0.195% of tokens**. The SAE shards use
`chunk_tokens = 512`, and **1/512 = 0.1953%**. The spike rate was therefore *one BOS token
per chunk* — a positional artifact, not a content-dependent spike distribution. The
percentile values stand; the interpretation ("rare content-driven spikes") does not, and is
replaced by "one massive activation per sequence, at BOS."

### The encoder does NOT do this — the dissociation

DNABERT-2 (encoder, `[CLS]` at position 0), 40 hg38 windows:

| SW channel | argmax at token 0 | \|act\| pos0 | \|act\| rest | ratio |
|---|---|---|---|---|
| L9 r264 | **0/40 (0%)** | 5.1 | 6.70 | **0.8×** |
| L9 r294 | **0/40 (0%)** | 35.5 | 7.89 | 4.5× |
| L3 r603 | 0/40 (0%) | 585.7 | 8.31 | 70.5× |
| L5 r603 | 0/40 (0%) | 818.6 | 10.31 | 79.4× |
| L3 r86 | 0/40 (0%) | 97.0 | 9.94 | 9.8× |

No encoder channel peaks at `[CLS]`. The persistent row-603 channel does carry an elevated
CLS component (70–79×) but its maximum is content-positioned; the critical L9 pair shows
little (4.5×) or none (**0.8×, below average**).

> **Decoder = BOS-anchored implicit bias. Encoder = not a sink.** This is a genuine
> cross-architecture dissociation, and it is the mechanistic correlate of the
> singleton-vs-ensemble taxonomy.

---

## T1.2 — How compensation works: both proposed circuits KILLED, actual mechanism found

**(A) Same-row / depth redundancy on the wire — FALSIFIED.** Ablating writes to channel 603
at layers 3/5/6/7 barely moves its value at readout:

| surviving writes | 4 | 3 | 2 | 1 | 0 |
|---|---|---|---|---|---|
| fraction of baseline | 1.000 | 1.002 | 1.010 | 0.990 | **0.919** |

Removing **all four** writes leaves 91.9% of the readout value. The channel's readout value
is not supplied by these down-projection writes, so "later writes restore the wire" is wrong.

**(B) Same-layer / head reads both — FALSIFIED.** Classifier-head weight magnitude:

| coordinate | \|head w\| | percentile vs 200 random coords |
|---|---|---|
| r294 | 0.1066 | **98** |
| r264 | 0.0407 | **31** (below the random mean of 0.0524) |

The head reads r294 strongly and r264 *less than a random coordinate*. Symmetric readout is
not the mechanism.

**(C) Predictive check — the additive circuit fails by 5.15×.** Logit shift A = 0.2367,
B = 0.1593, additive prediction 0.3960, **observed AB = 2.0376**.

### ★ The mechanism actually found: joint norm carriage

Layer-9 output residual stream, splice test set:

| condition | ‖h‖ | \|ch264\| | \|ch294\| |
|---|---|---|---|
| baseline | **17.20** | 8.83 | 9.81 |
| ablate r264 | 13.64 (−21%) | 0.03 | 9.81 |
| ablate r294 | 13.66 (−21%) | 8.83 | 1.13 |
| **ablate BOTH** | **7.14 (−58%)** | 0.03 | 1.13 |

The two channels **jointly carry 58% of the layer-9 residual-stream norm**. Removing one
costs 21% — the partner sustains the norm. Removing both collapses it by 58%, and the norm
drop is itself superadditive (10.06 observed vs 7.10 additive), which then rescales
everything downstream through LayerNorm.

**So the pair are not redundant *encoders of a feature*; they are redundant *norm carriers*.**
This is the implicit-bias / massive-activation mechanism from the NLP literature, but
distributed across a **pair** instead of concentrated in a singleton — precisely matching the
encoder-ensemble vs decoder-singleton taxonomy.

---

## T1.3 — Second functional encoder: **UNLOCKED**

**Root cause of NTv3's training failure found — and it is a bug affecting the manuscript's
existing NTv3 results.** `_MAX_LEN["reconstructed"] = 80` is a **tokenizer-specific** constant
tuned for DNABERT-2's BPE (400 bp → 86 tokens, so 80 covers ~372 bp = the full window). NTv3
is **nucleotide-level**: 400 bp → **400 tokens**, so `max_length = 80` truncated every splice
sequence to its **first 80 bp (20%)**. The splice junction sits mid-window — NTv3 never saw it.

A second, independent bug: `_NTv3Classifier` padded only *up to* a minimum length, but NTv3's
conv/deconv U-Net requires an exact **multiple** of 256; a 400-token input is above the
minimum yet 400 mod 256 = 144, raising `tensor a (24) vs b (25)`. Fixed to pad to the next
multiple.

With both fixed (`--max_length 400`), NTv3 splice fine-tuning goes from the majority-class
floor to a working model:

| seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| **MCC (fixed)** | **0.8831** | **0.8797** | **0.8679** | **0.8640** | **0.8741** |
| accuracy (fixed) | — | 0.9294 | — | — | — |
| baseline before | 0.5715 | 0.5340 | 0.5357 | 0.5160 | 0.5627 (floor = 0.5658) |

Ensemble-ablation baseline on the fixed checkpoints: **0.9235**, with all single-row effects
≤ 0.09 pp — the "no single row matters" signature. Full epistasis + k-of-N in progress
(`second_functional_encoder_ntv3.json`).

**Manuscript action:** every NTv3 GUE number in the paper was produced with 20%-truncated
inputs and must be re-run.

---

## Paper grade after Checkpoint 1

```
T1.1 ─ position-not-content, and a decoder/encoder DISSOCIATION  (T2.1 answered early)
T1.2 ─ both proposed circuits killed; JOINT NORM CARRIAGE found and quantified
T1.3 ─ NTv3 unlocked (MCC 0.86-0.88); functional replication now achievable
   ─► mechanism-grade: we can now say WHAT (position/norm, not composition),
      HOW (joint norm carriage -> LayerNorm rescaling), and engage the
      massive-activation / attention-sink literature with a real dissociation.
```

**Caveats.** (i) The norm-carriage mechanism is measured on one model, one layer, one seed —
it should be replicated on the other critical pairs and on NTv3. (ii) The BOS finding for
GENERator is 60 windows at one layer; robust and enormous (45,585×) but single-condition.
(iii) T1.1's composition battery is reported as null *for the decoder's SAE features*; the
DNABERT-2 direct-mode characterization was not run and is still owed.
