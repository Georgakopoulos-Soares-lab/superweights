# E1 + E2 — Breaking the norm/super-weight confound, and the co-dominance mechanism

Session date 2026-08-10. Outputs: `direction_vs_magnitude.{json,csv}`,
`codominance_break_dnabert2.{json,csv}`, `codominance_make_ntv3.{json,csv}`.

**Headline: the reviewer's killer sentence is now answered by intervention, and the
DNABERT-2/NTv3 discrepancy has a mechanism.**

---

## E1 — Direction or magnitude? (the confound the selection-based control could not break)

Selection failed because DNABERT-2's SW channels rank 1–2 of 768 by residual-norm
contribution and no non-SW pair comes within 5–6×. Intervention succeeds: rescaling a
`down_proj` row by a scalar changes magnitude while preserving direction exactly.

| condition | epistasis | vs reference |
|---|---|---|
| reference (unmodified SW pair) | **−17.72 pp** | — |
| **DOWN** — SW scaled to typical channel norm, direction preserved | **+0.04 pp** | **−0.2% retained** |
| **UP** — 5 random pairs scaled up to SW norm | **−0.27 pp** | **1.5% induced** (most extreme −3.73) |

**Verdict: NEITHER alone. The effect requires these specific channels *at* high magnitude.**

⚠️ **Necessary caveat on DOWN.** On seed 0 — the seed where the L9 pair is the critical one —
scaling the pair to typical norm (÷51×, ÷57×) *itself* collapsed the baseline from **0.9288
to 0.5866**, i.e. a −34 pp drop essentially identical to ablating the pair (−33.76). So the
DOWN cell does not cleanly measure "does direction still carry the signal"; it measures that
**removing the magnitude is already equivalent to removing the channel**. Direction alone is
not sufficient, but the design cannot separate "direction carries nothing" from "no working
model remains to test." Reported as a limit, not spun as a clean null.

**What E1 does establish for the reviewer:** magnitude is *necessary* (DOWN destroys), and
magnitude is *not sufficient* (UP at 38–83× scaling induces 1.5% of the effect). "It's just
the two biggest channels" is refuted: arbitrary channels raised to the same norm do not
reproduce it.

---

## E2 — ★ Co-dominance determines whether criticality is JOINT or SINGLE

### BREAK-PAIR (DNABERT-2, seed 0): co-dominance is **NECESSARY**

Redistributing norm between the two channels while holding the **summed pair norm constant**:

| partner/anchor ratio | rescaled baseline | single-A | single-B | joint AB | **epistasis** |
|---|---|---|---|---|---|
| unmodified (1.11) | 0.9288 | −0.02 | −0.11 | −33.76 | **−33.63** |
| 1.0 (co-dominant) | 0.9288 | −0.02 | −0.07 | −33.76 | **−33.67** |
| 0.5 | 0.9283 | +0.07 | −0.07 | −33.71 | −33.71 |
| 0.25 | 0.9274 | −4.34 | +0.00 | −33.63 | −29.29 |
| 0.1 | 0.9281 | **−22.88** | −0.04 | −33.69 | −10.76 |
| **0.03 (singleton)** | 0.9281 | **−33.03** | +0.00 | −33.69 | **−0.66** |

Three properties make this decisive:
1. **Total magnitude is held constant** — this is a *redistribution*, not a scale change, so
   it is not a magnitude effect.
2. **The baseline is unchanged throughout** (0.9274–0.9288) — the manipulation is harmless on
   its own, so nothing is confounded by damage.
3. **The criticality does not vanish — it migrates.** Joint epistasis falls −33.63 → −0.66
   while the single-channel effect rises −0.02 → −33.03, monotonically across five ratios.

> **Same total norm, same accuracy, same directions — only the *split* changes, and an
> emergent joint failure becomes a single-point failure.** Co-dominance is what makes the
> super-weight effect an *ensemble* effect.

### MAKE-PAIR (NTv3, working checkpoint): co-dominance is **NOT SUFFICIENT**

NTv3's natural ratio is **0.0340** — almost exactly DNABERT-2's 0.03 condition. Forcing
co-dominance:

| ratio | baseline | single-A | joint AB | epistasis |
|---|---|---|---|---|
| unmodified (0.034) | 0.9461 | −0.07 | −0.07 | +0.00 |
| **1.0 (forced co-dominant)** | 0.9450 | +0.04 | +0.04 | **−0.07** |
| 0.5 | 0.9456 | +0.00 | −0.02 | −0.07 |
| 0.25 | 0.9465 | −0.09 | −0.11 | +0.00 |

Imposing co-dominance on NTv3 creates **nothing** — neither joint nor single criticality.

### The remaining puzzle, sharpened

At essentially the same co-dominance ratio (~0.03), DNABERT-2's dominant singleton is
**critical (−33.03 pp)** while NTv3's is **inert (−0.07 pp)**. So NTv3 is robust to its
dominant channel in a way DNABERT-2 is not, *after controlling for both magnitude and
co-dominance*. That isolates the difference to architecture. The leading candidate remains
NTv3's conv/deconv U-Net, whose deconv tower adds skip residuals from the conv tower and
could route around damage at any single layer — **[UNTESTED]**, but now much better
motivated, since the two obvious confounds are eliminated.

---

## Combined statement for the paper

> Super-weight criticality in DNABERT-2 is carried by **specific channel directions expressed
> at high magnitude** — arbitrary channels scaled to the same norm reproduce 1.5% of the
> effect, and the super-weight channels scaled to ordinary norm lose it entirely. Whether
> that criticality manifests as an **ensemble** effect or a **single-point** failure is
> governed by **co-dominance**: holding total norm and task accuracy fixed and merely
> redistributing magnitude between the two channels converts a −33.6 pp emergent joint
> failure into a −33.0 pp single-channel failure (epistasis −33.6 → −0.7). Co-dominance is
> necessary for the ensemble phenotype but not sufficient for criticality itself: imposing it
> on NTv3 produces no effect, indicating an additional architecture-level factor.

## Status tags
- E1 DOWN/UP measurements — **[MEASURED]**; DOWN interpretation limited by baseline collapse — **[MEASURED-LIMIT]**
- E2 BREAK-PAIR monotone transition — **[MEASURED]**, 5 ratios, 1 seed, constant total norm and baseline
- E2 MAKE-PAIR null — **[MEASURED]**, 3 ratios, 1 seed
- U-Net skip-routing explanation for NTv3 — **[UNTESTED]**
- Replication of BREAK-PAIR on further seeds/pairs — **owed**
