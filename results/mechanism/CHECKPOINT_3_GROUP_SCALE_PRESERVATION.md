# CHECKPOINT 3 — Corrected, granularity-complete scale preservation

**Verdict: expected result confirmed; the prior claim is CORRECTED and now portable.**
**Kill condition (group-wise exemption confers reliable benefit) NOT triggered.**

Session date 2026-08-10. Outputs: `scale_preservation_grouped.{json,csv}`,
`exemption_all_granularities_{splice,promoter,histone}.json`.

---

## The correction, stated first

Prior claim: *"per-row RTN preserves the super weight bit-exactly."*
**True only of the single scale-defining element.** Per-row symmetric RTN sets
`s = max|w_row|/qmax`, so the max element maps to `round(qmax) = qmax` and returns exactly.
Every *other* element — including the other large outliers that "super weight" often denotes
— is quantised normally. The claim as previously stated was too strong.

---

## 3a — Per-row RTN: the max is free, the rest of the group is not

Mean **relative** error by rank within each SW row's top-|w| group (10 DNABERT-2 SW rows):

| bits | rank 1 | rank 2 | rank 3 | rank 4 | rank 5 |
|---|---|---|---|---|---|
| INT8 | **0.0000** | 0.0027 | 0.0056 | 0.0063 | 0.0109 |
| INT4 | **0.0000** | 0.0153 | 0.1093 | 0.1437 | 0.1855 |
| INT3 | **0.0000** | 0.0218 | 0.1365 | 0.2287 | 0.4444 |
| INT2 | **0.0000** | 0.0487 | 0.1696 | 0.3311 | **0.6396** |

Group mean relative error by group size M:

| bits | M=1 | M=4 | M=16 |
|---|---|---|---|
| INT8 | 0.0000 | 0.0037 | 0.0138 |
| INT4 | 0.0000 | 0.0671 | 0.2538 |
| INT3 | 0.0000 | 0.0967 | 0.5321 |
| INT2 | **0.0000** | 0.1374 | **0.6973** |

Rank-1 is exactly preserved at every precision, exactly as predicted. But a 16-element
super-weight *group* carries **70% mean relative error at INT2**. Under the group definition,
per-row RTN does **not** protect the super weight.

## 3b — ★ Group-wise RTN protects the whole group, nearly for free

Group-wise RTN (contiguous blocks of g, the scheme production kernels use). A SW element is
exact iff it is the max within its **own block**. Reported as `frac_block_max / mean_rel_err`:

| g=64 | M=1 | M=4 | M=16 |
|---|---|---|---|
| INT8 | 1.00 / 0.0000 | **1.00 / 0.0000** | 0.88 / 0.0012 |
| INT4 | 1.00 / 0.0000 | **1.00 / 0.0000** | 0.88 / 0.0241 |
| INT3 | 1.00 / 0.0000 | **1.00 / 0.0000** | 0.88 / 0.0413 |
| INT2 | 1.00 / 0.0000 | **1.00 / 0.0000** | 0.88 / **0.0772** |

| g=128 | M=1 | M=4 | M=16 |
|---|---|---|---|
| INT4 | 1.00 / 0.0000 | 0.93 / 0.0005 | 0.72 / 0.0468 |
| INT2 | 1.00 / 0.0000 | 0.93 / 0.0106 | 0.72 / **0.1694** |

**At g=64, all four of the top outliers are block maxima at every bit width — so the entire
top-4 SW group is preserved bit-exactly.** For M=16 at INT2, group-wise error is **0.0772 vs
per-row's 0.6973 — a 9× improvement.**

The mechanism: outliers are *spread* across a row, so each tends to dominate its own
64-element block and therefore defines its own scale. Group-wise quantisation protects
super-weight groups as a structural side-effect, with no SW-awareness at all.

## 3c — Exemption benefit across all four granularities

`exempt_sw − exempt_random` (5 count-matched random exemptions), 3 tasks × 4 granularities
× 3 precisions × 3 seeds. SW-specific gain in pp (`*` = saturated at the task's majority
floor, excluded from statistics):

| granularity | splice INT4/3/2 | promoter INT4/3/2 | histone INT4/3/2 |
|---|---|---|---|
| per_row | +0.13 / +2.21 / −0.02* | −0.02 / −0.00 / −1.14 | −0.05 / −0.48 / −2.13 |
| group_64 | +0.03 / +0.03 / +0.39 | +0.01 / −0.01 / −0.17 | +0.00 / +0.01 / −0.04 |
| group_128 | +0.02 / +0.03 / −0.12 | +0.01 / +0.01 / +0.08 | −0.01 / −0.10 / +0.15 |
| per_tensor | +0.00* / +0.00* / −0.02* | −0.52 / −0.08 / +2.51 | +0.12 / −0.70 / +0.84 |

**32 informative cells: mean +0.032 pp, median +0.004, 17 positive / 15 negative,
range [−2.13, +2.51], t = +0.23.** A coin flip.

By granularity: per_row **−0.187**, group_64 **+0.029**, group_128 **+0.008**,
per_tensor **+0.362**. The deployment-standard granularities (group-wise) show the
*smallest* effect of all — essentially exactly zero.

**Kill condition not triggered.** Group-wise exemption does not confer reliable benefit.

## Bonus — which granularity actually preserves quality

Mean damage across 3 tasks × 3 precisions (`quant_all`, pp):

| granularity | mean damage |
|---|---|
| **group_64** | **−5.46** |
| group_128 | −5.90 |
| per_row | −8.71 |
| per_tensor | −25.03 |

Group-wise is both the best-quality granularity *and* the one where SW exemption is most
useless — the two facts share a cause (§3b).

---

## Corrected, portable claim

> Super-weight-aware full-precision exemption is a **no-op wherever the super weight defines
> its own quantisation scale**. Under per-row RTN this holds for the single scale-defining
> element (0.000e+00 error at INT8–INT2); under group-wise RTN it extends to the whole
> outlier group (at g=64, the top-4 outliers are block maxima at every precision). Measured
> across 3 tasks × 4 granularities × 3 precisions, exempting super-weight rows confers
> **+0.03 pp** on average (17 positive / 15 negative, t = 0.23) — no benefit at any
> granularity, and least of all at the group-wise granularity production kernels use.
>
> The heuristic is **undefined unless two things are stated: the quantisation granularity
> and the super-weight definition (scalar vs group).** Under a *group* definition with
> **per-row** or **per-tensor** quantisation, super-weight outliers beyond the maximum are
> genuinely destroyed (up to 70% mean relative error at INT2, M=16) and *would* need
> explicit protection — but that regime is not what practitioners deploy.

---

## Caveats

1. Single model (DNABERT-2) for the weight-level analysis; the arithmetic argument is
   model-independent, but the `frac_block_max` numbers are empirical and specific.
2. Splice per-tensor cells and one per-row INT2 cell saturate at the 0.5658 majority floor
   and are excluded from statistics, not counted as evidence.
3. `exempt_random` uses 5 draws per cell; the ±2 pp outliers in both directions are within
   that noise, which is why the sign is a coin flip rather than a consistent effect.

---

## Paper status after Checkpoint 3

```
T1 INTRINSIC + T2 REPLICATES + T3 corrected & granularity-complete
  ─► Publishable gLM paper:
     (a) architecture-dependent SW organisation — encoder ensembles (DNABERT-2, NTv3;
         both pretrained, matched protocol) vs decoder singleton (GENERator)
     (b) SW-aware compression is a quantiser artifact, now stated portably with
         granularity AND super-weight definition made explicit.
```
