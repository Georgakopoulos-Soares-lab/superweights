# CHECKPOINT 2 (spec v4) — Tier 2: attention sink, steering, honest compression

Session date 2026-08-10. Outputs: `attention_sink_implicit_bias.json`,
`sw_steering_generation.{json,csv}`, `compression_destructive_and_activation.{json,csv}`,
`ensemble_encoding_direct_dnabert2.json`.

---

## T2.1 — Attention sink: **CONFIRMED in the decoder, ABSENT in the encoder** (dissociation)

### GENERator EUK (decoder, L4 ch2371) — a textbook attention sink

| diagnostic | value |
|---|---|
| incoming attention mass at position 0 | **0.3796** (37.96% of all mass) |
| mass at all other positions (mean) | 0.007213 |
| uniform expectation (1/L) | 0.0115 |
| **sink / uniform** | **33.0×** |
| **sink / other positions** | **52.6×** |
| heads whose argmax target is position 0 | **78.3%** |
| activation at pos 0 vs positions 1–79 | **45,585×** |
| argmax position across windows | **0 in 60/60 (100%)** |
| dinuc-shuffle ratio (content dependence) | **1.0000** |

Both halves of the attention-sink account hold: a massive activation *and* the attention
mass that defines a sink. The NLP mechanism (Sun et al. massive activations; Xiao et al.
attention sinks) **transfers to a genomic decoder**.

> **Caveat on the shuffle test.** In a causal decoder, position 0 attends only to itself, so
> a ratio of exactly 1.0000 at pos 0 is structurally guaranteed once the peak is known to be
> at BOS. It corroborates BOS-anchoring; it is *not* independent evidence of content
> independence. The max-over-positions ratio is also 1.0000, which is the informative form.

### DNABERT-2 (encoder) — not a sink

| | L9 r264 | L9 r294 | L3 r603 | L5 r603 | L3 r86 |
|---|---|---|---|---|---|
| argmax at token 0 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| pos0 / rest ratio | **0.8×** | 4.5× | 70.5× | 79.4× | 9.8× |

No encoder channel peaks at `[CLS]`; the critical L9r264 is *below average* there.
**Limitation:** DNABERT-2's MosaicBERT implementation does not expose attention maps
(`output_attentions` unsupported), so the encoder's *attention* half could not be measured —
only the activation half. That is a stated gap, not a null result.

### And the encoder super-weight IS a composition detector (T1.1 direct mode)

Splice test sequences, top-50 vs bottom-50 by peak SW activation:

| channel | GC (high act) | GC (low act) | **Cohen's d** | homopolymer Δ | activation range |
|---|---|---|---|---|---|
| **L3 r603** | 0.401 | 0.585 | **−1.89** | +2.20 | 586.7 → 891.3 |
| **L5 r603** | 0.422 | 0.601 | **−1.73** | +0.88 | 661.8 → 934.6 |
| L9 r294 | 0.425 | 0.412 | +0.16 | +1.42 | 353.3 → 664.7 |

The persistent row-603 channel fires on **AT-rich, homopolymer-rich** sequence with very
large effect sizes (|d| ≈ 1.7–1.9). This is content-dependent feature detection, the
opposite of the decoder's content-independent positional bias.

> ### ★ The dissociation, stated plainly
> **Decoder super-weight = BOS-anchored attention sink / implicit bias** (38% attention
> mass, 45,585× activation, content-independent).
> **Encoder super-weight ensemble = content-dependent composition detector** (AT-rich /
> homopolymer, |d| ≈ 1.8, never peaks at `[CLS]`).
>
> This is the *stronger, more novel* branch of the spec's decision rule, and it supplies the
> mechanistic correlate of the singleton-vs-ensemble taxonomy: singletons are biases,
> ensembles are detectors.

---

## T2.2 — Steering: the super weight **causally controls generated composition**

GENERator EUK, scaling the L4/r2371 write during generation (24 hg38 prompts, 64 new
tokens, fixed sampling seed so scaling is the only variable):

| scale | SW-row GC | ± sd | random-row GC |
|---|---|---|---|
| **0.0** | **0.2944** | 0.0421 | 0.3958 |
| 0.5 | 0.3863 | 0.0475 | 0.3958 |
| 1.0 (baseline) | 0.3961 | 0.0701 | 0.3961 |
| 2.0 | 0.4016 | 0.0776 | 0.3967 |
| 5.0 | 0.3549 | 0.0832 | 0.3986 |

**GC span: SW = 0.1072 vs random = 0.0028 → 38.59×.** Zeroing the super-weight row moves
generated GC by **−10.2 points**; five matched random rows move it by 0.3 points.

**PASS.** Suppression is strongly monotonic (0.0 → 1.0). Amplification saturates by 2.0 and
*reverses* at 5.0 (0.3549) — so the relationship is monotone only on the suppression side;
report it that way, not as a linear knob.

Note the interpretive point: the SW is a *content-independent BOS bias*, yet scaling it
steers the composition of everything generated downstream. That is exactly what an implicit
bias does — it sets a global prior rather than detecting a local feature.

---

## T2.3 — Honest compression: the no-op survives a destructive regime and the block axis

**Destructive regime.** All four regimes drove the model from 0.9288 to the 0.5658
majority-class floor (damage **+36.30 pp**). SW-**group** protection (top-M outlier
elements, not row exemption) versus count-matched random-group protection:

| regime | damage | M=1 | M=4 | M=16 |
|---|---|---|---|---|
| per_tensor INT4 | +36.30 | **+0.00** | **+0.00** | **+0.00** |
| per_tensor INT3 | +36.30 | **+0.00** | **+0.00** | **+0.00** |
| per_tensor INT2 | +36.30 | **+0.00** | **+0.00** | **+0.00** |
| per_row INT2 | +36.30 | **+0.00** | **+0.00** | **+0.00** |

*Caveat, stated:* these four cells all saturate at the majority floor, so they show
protection cannot *rescue* a destroyed model — they do not probe the intermediate regime.

**Block-size axis (Yu's actual mechanism), INT2 group-wise — this one has headroom:**

| block g | no protection | protect SW group (M=16) | gain |
|---|---|---|---|
| 32 | 0.6155 | 0.6155 | +0.00 |
| 64 | 0.6000 | 0.6078 | **+0.79** |
| 128 | 0.5921 | 0.5938 | +0.18 |
| 256 | 0.5684 | 0.5686 | +0.02 |

Accuracy declines monotonically with block size **whether or not the SW group is protected**
(0.6155 → 0.5684 unprotected; 0.6155 → 0.5686 protected). Protection does **not** extend the
usable block size, and its largest gain is +0.79 pp at a single non-monotonic point.

**Verdict: no crossover found.** The no-op conclusion is now tested where quantisation
genuinely bites and on Yu's own block-size axis, and it holds: *"Yu's heuristic does not
transfer to these models — weight exemption is vacuous by scale preservation, and the
activation/block-size benefit does not replicate here."*

---

## Paper grade after Checkpoint 2

```
T1.1 what   ─ decoder: position (BOS); encoder: composition (AT-rich, |d|~1.8)
T1.2 how    ─ joint norm carriage (pair carries 58% of layer-9 residual norm)
T1.3 n=2    ─ NTv3 unlocked (MCC 0.86-0.91) after fixing a tokenizer-length bug
T2.1 sink   ─ CONFIRMED decoder (38% attention mass, 33x uniform) / ABSENT encoder
T2.2 steer  ─ PASS, 38.6x random (suppression side monotone)
T2.3 compr  ─ no-op holds in destructive + block-size regimes
  ─► mechanism-grade mech-interp paper with a genuine cross-architecture dissociation:
     decoder super-weights are attention sinks; encoder super-weight ENSEMBLES are
     composition detectors that work by jointly carrying the residual norm.
```

### Still owed / in flight
- NTv3 functional epistasis on the fixed checkpoints (running; baseline 0.9430).
- Compensation-circuit replication on the other critical pairs (queued).
- PROK decoder sink + steering (queued) — would make the decoder side n=2.
- Encoder attention mass: **blocked** by MosaicBERT not exposing attention maps.
