# Falsification test — is the collapse about SUPER WEIGHTS, or about NORM?

Session date 2026-08-10. Outputs: `norm_matched_control.{json,csv}`.
Run unprompted, applying the project's own falsification discipline to its strongest claim.

## Why this test

The compensation mechanism we identified (CHECKPOINT_1_MECHANISM §T1.2) is **joint norm
carriage**: DNABERT-2's L9 pair carries 58% of the layer-9 residual norm, and removing both
collapses it. That is a *magnitude* property, not a super-weight property. Our random-pair
control drew uniformly from all 768 rows — overwhelmingly low-norm channels — so it could
not distinguish "the SW pair is special" from "any two big-norm channels are". Same class of
confound as the shadow-redundancy artifact (which turned out to be layer depth).

## Result 1 — the effect is not reproduced by the largest available non-SW pairs

DNABERT-2 splice, 3 seeds, layer 9:

| condition | epistasis (pp) |
|---|---|
| **SW pair (r264+r294)** | **−17.72** |
| best-available "norm-matched" non-SW pairs (n=24) | **−0.00 ± 0.03** (most extreme −0.11) |
| uniform-random pairs (n=24) | −0.01 |
| controls at least as extreme as SW | **0 / 24** |

## Result 2 — ⚠️ but the controls were NOT actually norm-matched

The SW channels are ranked **1 and 2 of 768** by norm contribution in *every* seed, with a
large gap to rank 3:

| seed | r294 | r264 | rank-3 (ch 439) | SW pair norm | best non-SW pair | shortfall |
|---|---|---|---|---|---|---|
| 0 | 9.81 | 8.83 | 1.96 | 18.64 | 2.97 | **6.3×** |
| 1 | 7.47 | 6.37 | 1.72 | 13.84 | 2.47 | **5.6×** |
| 2 | 7.06 | 5.72 | 1.77 | 12.77 | 2.45 | **5.2×** |

**No pair of non-SW channels can match the SW pair's norm** — the best available is 5–6×
short. So the designed control condition *does not exist in this model*, and the test could
not fully discriminate the two accounts.

**What is established:** the effect is not produced by pairs at the highest *achievable*
non-SW norm (~2.5, i.e. 5–6× smaller), which are flat at −0.00 pp.

**What is NOT established:** whether "super weight" and "top-2 norm channel" are
distinguishable. In DNABERT-2 **they are literally the same two channels**. The SW detector
found the two dominant norm carriers. Choosing between the labels is a naming question here,
not an empirical one.

## Result 3 — ★ norm dominance does NOT predict functional criticality (my proposed
## unifying explanation is falsified)

The obvious next move was to explain NTv3's functional null (epistasis −0.02 pp despite
MCC ≈ 0.90) by saying its SW channel is not norm-dominant. **That is false.** Measured on the
working NTv3 checkpoint:

| model / layer | SW channel norm rank | value | next channel | gap |
|---|---|---|---|---|
| NTv3 L11 r1472 | **1 / 1536** | 2369.66 | 80.57 | **29.4×** |
| NTv3 L9 r1472 | **1 / 1536** | 175.90 | 17.69 | **9.9×** |
| DNABERT-2 L9 r294 | 1 / 768 | 9.81 | (r264 8.83) | rank-3 gap 4.5× |

NTv3's super-weight channel is **more** norm-dominant than DNABERT-2's (29× vs ~4.5× gap to
the next non-pair channel), yet ablating it costs **−0.02 pp** versus DNABERT-2's −17.72 pp.

> **Conclusion: norm dominance is necessary-looking but not sufficient.** The joint-norm-
> carriage mechanism explains DNABERT-2 and does **not** generalise to NTv3. The paper cannot
> claim norm carriage as a general mechanism for genomic encoders.

### Structural difference worth noting (hypothesis, not measurement)
DNABERT-2 has **two co-dominant** channels (9.81 and 8.83, comparable) that must both be
removed. NTv3 has **one** dominant channel (2369 vs 80) — a singleton, on the encoder side.
NTv3 is also a conv/deconv U-Net whose deconv tower adds skip residuals from the conv tower,
which could route around damage at any single layer. Both are plausible explanations for
NTv3's robustness; **neither is tested here.**

## Impact on the paper

1. The **DNABERT-2 ensemble result survives** its strongest available control (0/24 controls
   as extreme), and the mechanism (joint norm carriage) is quantified.
2. The claim must be **scoped to DNABERT-2**. "Genomic encoders build redundant super-weight
   ensembles that are functionally load-bearing" is supported for one model; NTv3 replicates
   the *structure* (30 rows, persistent channel, pretrained superadditivity) but not the
   *function*, and its SW is more norm-dominant while being functionally inert.
3. The framing "super weight" vs "dominant norm carrier" is **not empirically separable** in
   DNABERT-2 and should be stated that way rather than assumed.
4. Report NTv3's functional null prominently — it is the most informative negative in the
   project, because it breaks the obvious mechanistic generalisation.
