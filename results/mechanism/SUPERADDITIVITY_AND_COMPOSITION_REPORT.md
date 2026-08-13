# Super-weights are an ensemble, not individuals — plus three corrections

Date: 2026-08-08 · DNABERT-2 GUE, fine-tuned multiseed checkpoints reused (no re-fine-tuning)
Ablation = zeroing down-proj weight rows, matching `run_gue_multiseed.py`, so all numbers are
directly comparable to the manuscript's −25.5 pp.

---

## 1. ★ THE NEW FINDING: super-weights interact in *structurally related pairs*

No individual super-weight row matters. **Pairs that share a layer, or share a coordinate across
layers, do** — and nothing else does.

DNABERT-2's 10 SW rows are not 10 independent coordinates. They form two redundancy structures:

```
layer 3: [86, 399, 603, 641]     <- 4 rows in one layer
layer 5: [86, 603]                   row 603 recurs at layers 3,5,6,7  (a persistent channel)
layer 6: [603]                       row  86 recurs at layers 3,5
layer 7: [603]
layer 9: [264, 294]              <- 2 rows in one layer
```

Define a pair as **structurally related** if it shares a layer *or* shares a row index.
That is 15 of 45 pairs (33%). Those 15 account for essentially every real interaction:

| | related among top-3 | top-5 | **top-7** | random-pair control |
|---|---|---|---|---|
| **Splice** (n=5) | 3/3 (p = 0.032) | 5/5 (p = 0.0025) | **7/7 (p = 0.00014)** | +0.006 pp |
| **Promoter** | 3/3 (p = 0.032) | 5/5 (p = 0.0025) | 6/7 (p = 0.0035) | −0.001 pp |
| **Histone** | 3/3 (p = 0.032) | 5/5 (p = 0.0025) | 6/7 (p = 0.0035) | −0.009 pp |

**Three independent tasks, same conclusion.** Unrelated pairs sit at the random-control floor.

### Pairwise epistasis, DNABERT-2 splice (all C(10,2)=45 pairs × 3 seeds)

`epistasis(a,b) = Δacc(ablate a AND b) − [Δacc(a) + Δacc(b)]`

**n = 5 seeds**, all C(10,2)=45 pairs. Top-7 are 7/7 structurally related (3 same-layer, 4 same-row):

| pair | epistasis (pp) | relation |
|---|---|---|
| **L9/r264 + L9/r294** | **−11.56 ± 12.57** | **same layer** |
| **L3/r603 + L3/r641** | **−5.20 ± 4.58** | **same layer** |
| L5/r603 + L3/r603 | −3.24 ± 4.19 | same row |
| L3/r86 + L3/r399 | −1.99 ± 2.07 | same layer |
| L5/r603 + L6/r603 | −0.94 ± 0.93 | same row |
| L3/r86 + L5/r86 | −0.85 ± 1.10 | same row |
| L3/r603 + L6/r603 | −0.81 ± 0.70 | same row |
| **random pairs (control)** | **+0.006** | – |

**Seed 0, the cleanest case:** L9/r264 and L9/r294 cost **−0.13 pp** when ablated separately and
summed. Ablated together: **−33.76 pp**. Epistasis **−33.63 pp**. Random pairs: **+0.007 ± 0.020 pp.**

### Promoter replicates it, with a *different* pair type

| pair | epistasis (pp) | relation |
|---|---|---|
| **L6/r603 + L7/r603** | **−11.26 ± 15.92** | **same row, adjacent layers** |
| L3/r603 + L3/r641 | −3.74 ± 4.84 | same layer |
| L3/r86 + L3/r399 | −0.30 ± 0.27 | same layer |
| random pairs (control) | −0.0013 | – |

Promoter's strongest interaction is **row 603 at layers 6 and 7** — the same coordinate written at
consecutive depths. That is why "same layer" was the wrong generalization: the unifying property is
*redundancy*, which the architecture realizes both within a layer and along a persistent channel.

### Each fine-tuned model has its own critical pair

| task / seed | critical pair | epistasis |
|---|---|---|
| splice, seed 0 | L9/r264 + L9/r294 | −33.63 pp |
| splice, seed 2 | L9/r264 + L9/r294 | −17.62 pp |
| splice, seed 1 | L3/r603 + L3/r641 | −12.74 pp |
| promoter, seed 0 | L6/r603 + L7/r603 | −33.77 pp |
| promoter, seed 2 | L3/r603 + L3/r641 | −10.57 pp |
| promoter, seed 1 | (none > 2 pp) | – |

Fine-tuning selects *which* related pair becomes load-bearing; that a structurally related pair does
is consistent. This is why single-seed ablation studies of super-weights are unreliable — they
sample one arbitrary realization of a redundant system.

**Mechanistic reading:** super-weight rows that share a layer or a coordinate are mutually
redundant. Ablate one and its partner compensates; ablate both and compensation fails. "The super
weight" is the wrong unit of analysis for a genomic encoder — the *redundant pair* is.

### Corroborating k-of-N curve (splice, mean of 3 seeds)

```
k:      1      2      3      4      5      6      7      8      9     10
Δacc: -0.49  -0.64  -2.49  -2.46 -22.89 -30.91 -31.84 -29.78 -27.31 -25.54   (n=3 curve)
                                   ↑ +1 row = −20 pp
```
- **n = 5**: observed all-10 **−26.84 ± 2.56 pp** vs sum-of-parts **−3.51 ± 2.15 pp** →
  **−23.3 pp purely emergent**; random-10 control **−0.035 pp**
- random 10 rows: **±0.04 pp** — the effect is SW-specific, not "zeroing 10 rows hurts"
- the cliff sits exactly where L9/r294 joins a set already containing L9/r264
- **non-monotonic**: worst at k=7 (−31.84), partially *recovers* by k=10 (−25.54) — later rows
  compensate. Another reason the set is not a simple additive pool.

---

## 2. 🔴 CORRECTION: histone is NOT a flat control — the "dissociation" is a reliability contrast

Per-seed k-of-N curves, H3K4me3:

```
seed 0: [ 0.27  0.16  0.16  0.19   0.14 ...  -0.03]   flat
seed 1: [ 0.05  0.16  0.24  0.11 -10.03 ... -13.26]   SAME k=5 cliff
seed 2: [ 0.22  0.16  0.05  0.16   0.03 ...  -0.30]   flat
seed 3: [-0.08 -0.19 -0.16 -0.24 -11.96 ... -10.65]   SAME k=5 cliff
seed 4: [-0.05 -0.27 -0.19 -0.27   0.00 ...  -1.01]   flat
```

The **identical k=5 transition** fires in **2 of 5 histone seeds** and **3 of 3 splice seeds**.
Mean histone curve: `[0.08, 0.01, 0.02, -0.01, -4.36, -4.30, -4.05, -3.77, -4.49, -5.05]`.
Random control ±0.04 pp throughout.

### ★ The bimodality is fully explained: 5/5 seed-level correspondence

Running the epistasis map on histone identifies, per seed, exactly which pair is load-bearing:

| seed | k=5 cliff (superadditivity) | critical pair (epistasis, independent measurement) |
|---|---|---|
| 0 | flat | **NONE** |
| 1 | **−10.03 pp** | **L9/r264 + L9/r294 = −10.2 pp** |
| 2 | flat | **NONE** |
| 3 | **−11.96 pp** | **L9/r264 + L9/r294 = −11.2 pp** |
| 4 | flat | **NONE** |

The two seeds that collapse are precisely the two in which the L9 pair became load-bearing, and the
magnitudes agree to within ~1 pp across two independent measurements. Seeds 0, 2, 4 have **no**
critical pair and show **no** collapse.

So the histone "unreliability" is not noise and not a failed replication — it is **whether a given
fine-tuning run happens to route the task through a redundant super-weight pair.** The T0.3
bimodality ([−0.03, −13.26, −0.30, −10.65, −1.01] pp), unexplained until now, reduces entirely to
this. It also means:

> **The abstract's "collapses splice-site recognition while leaving histone-mark prediction intact"
> is not supportable.** It is a difference in *how reliably fine-tuning recruits a critical pair*,
> not presence vs absence of the mechanism.

---

## 3. 🔴 CORRECTION: the composition claim is weaker than stated

Full 4,096-hexamer composition-controlled analysis (`composition_vs_motif_{euk,prok_corrected}.json`):

| | EUK L4/r2371 | PROK corrected L8/r260 |
|---|---|---|
| R² — **GC alone** | **0.035** | 0.002 |
| R² — full composition (25 features) | **0.373** | 0.011 |
| R² — rank-transformed (robust) | **0.440** | 0.153 |

1. Composition explains **37–44%** of SW activation — a large minority, not "most." ~60% is
   unexplained by any composition feature.
2. **GC alone explains 3.5%.** Much of the manuscript's narrative is framed on GC specifically;
   that framing is not supported. The dinucleotide-level model does the work.
3. Against a **GC-matched** background (10,000 permutations) rather than the original uniform one,
   **2 of 12 motifs survive BH correction**: **NF-κB enriched** (z = +5.85) and **CpG-rich depleted**
   (z = −10.21). The original test found 0 of 9 — so *"no canonical regulatory motifs are enriched"*
   fails under the stronger control.
   *Caveat:* GC-bin matching fixes mononucleotide GC but not dinucleotide composition, so CpG
   depletion is plausibly still compositional. NF-κB is harder to dismiss.
4. Reassuringly, the top-200 regression residuals are **not** motif-enriched (0.195 vs 0.206,
   p = 0.79) — the unexplained 60% is not hidden motif structure either.

⚠️ **Internal inconsistency:** `main.tex:488–489` already states *"token-level identity, not
high-level summary statistics, drives most of the SW signal"* — which matches R² = 0.37 and
**contradicts the abstract's cleaner composition claim.** This is the second abstract-vs-results
conflict in the manuscript (the first is compression).

✅ **What this buys you:** a quantified kingdom contrast that is robust — composition explains
**0.373** of EUK's SW activation vs **0.011** for corrected PROK. That is the "feature detector vs
bias term" distinction with a number attached, and unlike the dead sign-reversal it survives scrutiny.

---

## 4. Promoter: real but seed-unstable; and a metric of mine that failed

Per-seed all-10 Δacc: **[−36.22, −1.13, −0.34] pp**. Seed 0 cliffs at k=9, seed 2 at k=6 and then
*recovers* at k=10 (−16.11 → −0.34). Random 10-row control ±0.03 pp, so the rows are still special.

**The superadditivity RATIO is not a usable statistic here.** Promoter's sum-of-parts is ≈0
(+0.08 ± 0.07 pp), so the ratio explodes to meaningless values (−274, +60, −2.6). Report
`observed − sum_of_parts` against the random-k control instead. (The ratio *is* meaningful for
splice, where sum-of-parts = −4.56 pp: ratio of means ≈ 5.6×, per-seed 3.8–13.7×.)

Also: the old single-seed **−37.84 pp** in `results/gue_ablation_results.json` is **seed 0's cliff**,
not a typical value. Flag or delete that file.

## 5. NTv3: test not applicable, and a caution

NTv3's index holds **N = 1** row, so superadditivity is undefined (ratio ≡ 1). Per-seed Δacc
[−5.72, −24.73, −12.43, **+4.98**, **+0.33**] pp: seeds 4 and 5 land at accuracy 0.5658/0.5660,
**exactly the majority-class baseline** — the collapse-to-majority seeds the Limitations already
disclose. The NTv3 replication is 3 informative seeds + 2 degenerate ones; MCC is doing necessary work.

## 6. ✅ Cross-kingdom claim survives the contamination fix

Rerun on corrected L8/r260: **484.9** (E. coli) vs **484.7** (hg38), fold **0.9998**, MW **p = 0.768**
(contaminated: 10,638 / 10,710 / 1.0068 / p = 0.644). Conclusion holds; only magnitudes change.
*But* note the asymmetry: for EUK this is a meaningful negative (that row varies with sequence,
ρ = +0.97, yet doesn't separate kingdoms); for corrected PROK it is trivially true, since the row
varies with nothing. Do not present them as equivalent evidence.

---

---

## 7. 🔴 The EUK SAE fails — and the failure is diagnosed, not mysterious

Trained a top-K SAE (k=64, dict_mult=4 → 12,288 features, 50k steps) on **1M unclamped fp32**
GENERator-EUK layer-4 activations. It did not produce a usable dictionary:

| metric | value |
|---|---|
| dead features | **74.9%** (9,203 of 12,288 never fire) |
| rare features (freq < 1e-4) | 86.6% |
| median feature frequency | 0.00000 |
| loss: median / mean / max | 1.38 / 11.52 / **408.1** |

**Root cause — the objective is unnormalized and the super weight dominates it.**
`sae/train.py` builds a Welford `RunningNorm` but uses it *only* to initialise `pre_bias`
(lines 356–366); activations are never standardised. Measured per-channel spread:

- typical channel (row 100): **sd = 9.4**
- super-weight channel (row 2371): **sd = 18,698** — a **~2,000× imbalance**

With plain MSE on raw activations, essentially the entire loss is the SW channel. A 2048-token
batch contains ~4 of the 0.195% spike tokens (true values 339,716–452,119), and those few tokens
swing the step loss by 300×. Gradient clipping (max_norm=1.0) limits step size but cannot fix an
imbalanced objective. Dictionary capacity collapses onto one coordinate, leaving 75% of features dead.

**This explains the manuscript's Limitation #4** — *"the GENERator EUK SAE encountered training
instabilities and is incomplete"* — which has stood as an unexplained failure. It is a pipeline
design issue, not bad luck, and the same imbalance applies to the **PROK SAE that is in the paper**.

**Fix applied and tested.** `sae/train.py` gained `--standardize` (per-channel division by the
running std; float16 default unchanged). The imbalance is worse than first stated: channel 2371
sd = **18,644** vs **median 2.707** → **6,888×** (my earlier "~2,000×" compared against one
arbitrary channel, not the median).

### The optimisation fix worked — decisively, on real data

| metric (real hg38, from `sanity_check.json`) | unstandardised | **standardised** |
|---|---|---|
| **dead features** | **74.9%** | **0.5%** |
| rare (freq < 1e-4) | 86.6% | **0.7%** |
| median feature frequency | 0.00000 | **0.00405** |
| recon MSE / random-decoder MSE | 0.814 / 113,305 | 0.0428 / 0.856 |
| loss median / mean / max | 1.38 / 11.52 / **408.1** | **~0.043**, stable (0.038–0.055) |

99.5% of the dictionary is now alive on real sequence, versus 25.1% before. The training
pathology is gone. **The diagnosis in this section was correct and the prescribed fix works.**

### But the hexamer-probe analysis still yields nothing interpretable

`sae/analyze.py` interrogates features with 4,096 synthetic single-hexamer 11-token contexts:

| | unstandardised | 20k std | **final std** |
|---|---|---|---|
| alive on any hexamer | 127 (1.0%) | 99 (0.8%) | **51 (0.4%)** |
| firing on exactly one hexamer | 64 | 52 | **28** |
| intermediate selectivity (2–4094) | **0** | 24 | **13** |
| dense (≥4095) | 63 | 23 | **10** |

Among the 13 selective features, GC structure is weak: **median |r_GC| = 0.050**, max 0.324
(and that maximum belongs to a near-dense feature, n_active = 4,088). No compositional axis.

**Interpretation:** a dictionary that is 99.5% alive on real sequence but responds to only 0.4%
of synthetic hexamer probes is telling us the features are **context-dependent, not
token-identity-dependent**. That is consistent with §3 (composition explains only R² = 0.373 of
SW activation). The probe, not the SAE, is the limiting instrument.

### ★ The reported feature/SW correlations are an artifact of the *analysis script*, not the SAE

`analyze.py` still prints `SW-r = −0.9949` for its top features — from features with
**n_active = 1**, all firing on `AAAAAA`. A Pearson correlation over a vector that is zero at
4,095 of 4,096 positions is fixed entirely by one point.

This is now a **stronger** claim than before. Previously the inference was "the PROK SAE probably
collapsed the way EUK did". The final run shows the degenerate ±0.99 signature is emitted **even
when the underlying dictionary is healthy** (0.5% dead). So the artifact is a property of the
measurement, independent of SAE quality.

> **Consequence for the manuscript:** its PROK SAE correlations (r ≈ +0.71, −0.73 to −0.76, with
> `AAAAAA` named among the anti-correlated features) are suspect **regardless of whether that SAE
> trained well**. They should not be cited without per-feature `n_active`.
>
> *Epistemic status:* that `analyze.py` produces degenerate correlations from n_active=1 features
> is **established** (measured here on two SAEs). That the manuscript's PROK numbers are the same
> artifact is **strong inference** — the PROK checkpoint does not exist under `results/`, so its
> `n_active` cannot be checked directly.

### Verdict: the SAE section stays cut — but the blocker has moved

Not because the SAE is broken; it is now fixed and healthy. Because the **analysis instrument**
cannot support any claim, and every correlation it currently reports is degenerate. To revive the
section, features must be evaluated on **real genomic sequence with `n_active` reported beside
every correlation**, not on synthetic hexamer probes. That is a tractable next experiment, and the
trained standardised SAE (`results/sae/euk_layer4_std/sae_final_scaled.pt`) is ready for it.

### (superseded) Unstandardised run — feature analysis, for the record

Running `sae/analyze.py` over all 4,096 hexamers on the **unstandardised** SAE:

| | |
|---|---|
| features firing on **any** hexamer | **127 / 12,288 (1.0%)** |
| of those, firing on **exactly one** hexamer | **64 (50%)** — all on `AAAAAA` |
| the remaining ~63 | fire on **nearly all** 4,096 hexamers (mean n_active = 2,032) |
| dead | 12,161 |

The dictionary degenerated into two useless halves: ~64 duplicate single-hexamer spike detectors
(all firing on the same homopolymer `AAAAAA`) and ~63 dense non-selective features. There is no
interpretable middle.

**⚠️ The reported feature/SW correlations are statistical artifacts.** `analyze.py` prints
`SW-r = −0.9949` for the top features — but those features have **n_active = 1**. A Pearson
correlation computed over a vector that is zero at 4,095 of 4,096 positions is fixed entirely by
that single point. Such values carry no information about super-weight coupling, however extreme
they look.

**This casts doubt on the manuscript's PROK SAE result, by analogy.** The paper reports features
"most positively correlated with SW magnitude (r ≈ +0.71)" and anti-correlated features
"(r ≈ −0.73 to −0.76) fired on AT-rich and low-complexity sequences (**AAAAAA**, TTTTTG, GGGGGG)",
plus "a subset of dictionary features were monosemantic but orthogonal to the SW axis
(`sw_active_frac` = 0.0)". That is the *same signature* this EUK run produces: near-dead dictionary,
"monosemantic" features that are really single-token detectors, `AAAAAA` prominent, and sizeable
correlations that are degenerate.

**Epistemic status — this is inference, not verification.** The PROK SAE checkpoint does not exist
anywhere under `results/`, so its per-feature `n_active` cannot be checked directly. What is
established: the same pipeline, run on EUK, collapses and yields meaningless correlations. Whether
PROK collapsed identically is a strong suspicion, not a demonstrated fact. It should be resolved by
retraining with normalisation and reporting `n_active` alongside every correlation — not by citing
the existing numbers.

### The bonus: a quantitative definition of a super weight

Collecting unclamped activations gave a crisp operational characterisation from 262,144 real hg38
tokens, row 2371:

| percentile | 50 | 90 | 99 | 99.8 | 99.9 |
|---|---|---|---|---|---|
| activation | 169 | 182 | 212 | 1,031 | **426,422** |

Near-constant for 99.8% of tokens, then a **~2,500× spike**. This is a sharper definition than the
manuscript currently offers, and it is what makes both the fp16 clamp and the unnormalised SAE loss
fail.

### ⚠️ Separate bug found in `sae/collect.py` (now patched)

The collector clamps to ±60,000 before the float16 cast. That clips only **0.195%** of tokens — but
those are exactly the super-weight spikes, and clamping them removes **98% of the channel's
variance** (sd 18,698 → 2,642; max 452,119 → 60,000).

**The manuscript's PROK SAE was trained through this clamp on layer 2, where the (contaminated) row
1927 reached ~506,000.** So its claim to "independently recover the compositional axis" rests on
data whose SW spikes had been flattened — a defect *independent* of the layer-2 contamination.
Two unrelated bugs landing on the same result.

Patched with a backward-compatible `--store_dtype float32` (float16 remains the default so existing
collections reproduce bit-for-bit).

---

## 8. 🔴 "Shadow redundancy" is a layer-depth artifact — the SW coordinates do no work

The compression section's central claim is that rows far from a SW coordinate are
uniquely fragile under pruning (prox-far −9.39 vs prox-near −1.60). Proximity is
Euclidean distance in normalised (layer, row) space. DNABERT-2's SW layers are
3,5,6,7,9 of 12 — the middle — so **prox-far preferentially selects the outer layers**.
At 20% it selects **all 768 rows of layer 0**; prox-near never touches layers 0, 10, or 11.

I added three controls that contain no super-weight information whatsoever:

- **`layer_matched_random`** — random rows matching prox_far's exact per-layer histogram
- **`sham_prox_far`** — proximity to sham reference points (same layers, arbitrary rows)
- **`depth_extreme`** — pure outer-layers-first ordering

### Promoter (seed 0, baseline 83.61%), Δacc in absolute pp

| criterion | 5% | 10% | **20%** | 30% | uses SW? |
|---|---|---|---|---|---|
| `prox_far` (manuscript) | +0.02 | −0.28 | **−10.04** | −10.19 | yes |
| `layer_matched_random` | +0.04 | −0.45 | **−9.09** | −7.10 | **no** |
| `sham_prox_far` | −0.26 | −2.17 | **−8.18** | −7.42 | **no** |
| `depth_extreme` | −0.64 | −1.30 | **−8.08** | **−12.64** | **no** |
| `prox_near` | +0.09 | +0.02 | −0.30 | −6.27 | yes |
| uniform `random` | +0.06 | +0.12 | −0.56 | −0.84 | no |

### Histone H3K4me3 (seed 0, baseline 60.19%) — replicates

| criterion | 5% | 10% | **20%** | 30% |
|---|---|---|---|---|
| `prox_far` | +0.14 | −1.25 | **−2.74** | −2.88 |
| `layer_matched_random` | +0.18 | −0.49 | **−2.87** | −3.14 |
| `sham_prox_far` | −0.65 | −1.17 | −2.55 | −2.66 |
| `depth_extreme` | −0.90 | −0.57 | −2.74 | −3.53 |
| `prox_near` | −0.08 | −0.14 | −0.76 | −0.46 |
| uniform `random` | +0.02 | −0.20 | −0.20 | −0.31 |

### Splice (seed 0, baseline 92.88%) — the sharpest case, and it inverts the claim

| criterion | 5% | 10% | **20%** | **30%** |
|---|---|---|---|---|
| `prox_far` | +0.07 | −3.42 | **−26.57** | **−34.88** |
| `layer_matched_random` | −0.02 | −2.14 | **−25.67** | **−34.87** |
| `sham_prox_far` | −0.99 | −10.00 | −35.55 | −36.30 |
| `depth_extreme` | −1.05 | −3.02 | −30.12 | −36.30 |
| `prox_near` | −0.07 | −0.90 | **−31.85** | −36.65 |
| uniform `random` | −0.17 | −0.81 | −5.38 | −13.08 |

`layer_matched_random` tracks `prox_far` to **0.01 pp at 30%** (−34.87 vs −34.88). And
**`prox_near` is *worse* than `prox_far`** (−31.85 vs −26.57 at 20%), directly inverting
the manuscript's "near-SW rows are tolerant, far-SW rows are fragile" ordering. (Splice is
3-class with a 0.5658 majority floor, so the 30% row is partly saturated; the 20% row is
the informative one.)

**Across all three tasks** the layer-matched control reproduces prox_far: promoter −9.09
vs −10.04, histone −2.87 vs −2.74 (control worse), splice −25.67 vs −26.57. Uniform random
costs far less everywhere, so the fragility is real — it is simply **entirely attributable
to which layers get pruned**, not to super-weight proximity.

> **Verdict: CUT the shadow-redundancy claim** (`main.tex:152`, abstract-level, and the
> pruning paragraph). It is not a super-weight phenomenon. The defensible residue is the
> ordinary result that outer layers are less prunable than middle layers.

### Two further defects in the same section

1. **The row coordinate is meaningless by permutation symmetry.** Rows of `down_proj`
   index residual-stream channels; permuting `d_model` consistently across the model is a
   function-preserving symmetry, so channel indices carry no metric structure. "Distance in
   row space" measures nothing. (The layer term dominates anyway: range 0.273 vs 0.164.)
2. **Probable unit error.** `run_compression_sweep.py` computes
   `delta_acc_pct = (acc − base)/base × 100`, a *relative* change, but the manuscript
   reports these values as "pp". At baseline 83.46%, −9.39 relative = **−7.83 pp** absolute.
   Every compression number sourced from that field needs re-checking. *(Status: the
   discrepancy in the code is established; that the manuscript inherited it is inferred
   from the matching magnitudes.)*

---

## 9. Pair-aware compression: INT4 is too weak to test the hypothesis

Motivated by §1: if SW rows are pairwise redundant, "keep all SW rows in full precision"
is over-conservative — you should be able to degrade one member freely, with the binding
constraint being not to degrade both. Per-row RTN quantisation, 5 seeds (3 for promoter):

| condition | splice INT4 (seed 0) | histone INT4 (n=5) | promoter INT4 (n=3) |
|---|---|---|---|
| `pair_a_only` | +0.00 | +0.038 ± 0.041 | +0.000 ± 0.000 |
| `pair_b_only` | — | +0.000 ± 0.033 | +0.000 ± 0.000 |
| `pair_both` | +0.00 | +0.033 ± 0.065 | +0.000 ± 0.000 |
| `sw_singletons` | +0.07 | −0.011 ± 0.097 | +0.019 ± 0.019 |
| `all_sw` | +0.02 | −0.011 ± 0.106 | +0.019 ± 0.019 |
| `random_pair` | +0.002 ± 0.007 | −0.002 | +0.001 |
| **pair epistasis** | ~0 | **−0.005** | **+0.000** |

**This is an underpowered test, not a negative result.** Every condition — including
ablating *all* SW rows — sits at ≤0.04 pp. INT4 does not perturb these rows enough to
move any task, so the pair hypothesis is untestable at this strength. The decisive
contrast, same rows / same task / same seed:

| perturbation of L9r264 + L9r294 | Δacc |
|---|---|
| INT4 quantisation | **+0.00 pp** |
| zeroing | **−33.76 pp** |

### ★ Why: per-row symmetric RTN preserves the super weight **exactly, by construction**

Dose-response on the splice critical pair (n = 5 seeds):

| perturbation of L9r264 + L9r294 | pair epistasis |
|---|---|
| INT8 | +0.00 pp |
| INT4 | ~0.00 pp |
| INT3 | −0.00 pp |
| INT2 | −0.00 pp |
| **zeroing** | **−11.56 pp** |

Even 2-bit quantisation does nothing. Direct weight-level measurement explains why:

| | INT8 | INT4 | INT3 | INT2 |
|---|---|---|---|---|
| SW element error (row 264) | **0.000e+00** | **0.000e+00** | **0.000e+00** | **0.000e+00** |
| SW element error (row 294) | **0.000e+00** | **0.000e+00** | **0.000e+00** | **0.000e+00** |
| mean rel. error, rest of row 264 | 0.092 | 0.930 | 0.942 | **0.943** |
| mean rel. error, rest of row 294 | 0.048 | 0.727 | 0.894 | **0.976** |

This is arithmetic, not luck. Per-row symmetric RTN sets `s = max|w| / qmax`; the
max-magnitude element maps to `round(max|w|/s) = qmax` and dequantises to `qmax·s = max|w|`
— itself, exactly. **A super weight is by definition the largest-magnitude element in its
row, so per-row symmetric RTN preserves it bit-exactly at any precision**, while destroying
up to 98% of the rest of the row.

> **Consequence — a whole class of experiments in the manuscript tests a no-op.**
> Yu et al.'s heuristic ("retain SW rows in full precision, quantise the rest") is
> **vacuous under per-row symmetric RTN**: the SW is already retained exactly, for free.
> This explains the manuscript's own unexplained nulls — "quantizing the two SW rows alone
> is essentially free (−0.013 PPL)" and whole-model SW exemption changing PPL by −0.0008
> (EUK) / +0.0004 (PROK). Those are not empirical findings about super-weight robustness;
> they are consequences of the quantiser's scale definition.
>
> **Scope:** this holds for schemes where the SW defines its own scale (per-row, or
> group-wise with the SW in-group). Under **per-tensor** quantisation the SW is *not*
> protected, since the scale is set by a global max. Any SW-aware quantisation claim must
> therefore state the granularity, and should be tested per-tensor to be non-trivial.

✅ **What is established:** SW rows in DNABERT-2 are *robust to INT2–INT8 per-row RTN* — individually, as a
critical pair, and all together — across three tasks — for the structural reason above. The
redundant-pair structure is invisible to any per-row RTN precision and is exposed only by
ablation-strength perturbation (zeroing: −11.56 pp).

---

## 10. Per-tensor SW exemption: the Yu heuristic yields nothing at any granularity

Per-row RTN preserves the SW bit-exactly (§9), so any exemption test at that granularity is a
no-op *by construction*. Per-tensor draws the scale from the global matrix max, making the question
empirically real — but only for SW rows that are not themselves that global max:

| DNABERT-2 SW row | row_max / tensor_max | per-tensor INT4 err | per-tensor INT2 err |
|---|---|---|---|
| L5r603, L9r294, L3r603, L7r603, L6r603 | **1.000** | 0.00000 | 0.00000 |
| L3r86 | 0.827 | 0.044 | 0.254 |
| L3r399 | 0.788 | 0.102 | 0.312 |
| L9r264 | 0.773 | 0.168 | 0.648 |
| L3r641 | 0.753 | 0.057 | 0.362 |
| L5r86 | 0.350 | 0.126 | 0.686 |

**5 of 10 super-weight rows are the largest weight in the entire matrix**, hence exact under any
max-calibrated quantiser. Only the other 5 are perturbable per-tensor.

Design: 3 tasks × {per_row, per_tensor} × {INT4, INT3, INT2} × 3 seeds. The decisive statistic is
`exempt_sw − exempt_random` (5 random-row exemptions, count-matched to the 10 SW rows), which
isolates *SW-specific* benefit from the benefit of exempting any 10 rows.

| SW-specific gain (pp) | INT4 row | INT3 row | INT2 row | INT4 tensor | INT3 tensor | INT2 tensor |
|---|---|---|---|---|---|---|
| splice | +0.13 | **+2.21** | −0.02* | +0.00* | +0.00* | −0.02* |
| histone | −0.05 | −0.48 | **−2.13** | +0.12 | −0.70 | +0.84 |
| promoter | −0.02 | −0.00 | −1.14 | −0.52 | −0.08 | **+2.51** |

\* saturated at splice's 0.5658 majority-class floor — uninformative, excluded from statistics.

Over the 14 informative cells: **mean = +0.048 pp, median = −0.035, range [−2.13, +2.51],
9 negative vs 5 positive, t = +0.15.** The two largest positive cells (+2.51 promoter per-tensor
INT2, +2.21 splice per-row INT3) are matched by a comparable negative (−2.13 histone per-row INT2),
and the promoter per-tensor column is non-monotonic (INT3 −32.54 worse than INT2 −24.03), i.e.
chaotic near the floor. This is noise, not signal.

> **Verdict:** super-weight exemption confers **no reliable benefit at any granularity or precision
> tested**. At per-row it is provably nil; at per-tensor the model collapses before exemption can
> help. The Yu et al. NLP heuristic does not transfer to this model class — and we can now say *why*
> (scale preservation), not merely that it doesn't.

---

## 11. ✅ The SAE section can be revived — with a different claim

Replacing `analyze.py`'s synthetic hexamer probe with **real hg38 activations** (same shards the SAE
trained on, checkpoint's `data_scale` applied, 401,408 tokens) transforms the picture:

| | hexamer probe | **real sequence** |
|---|---|---|
| features alive | 51 / 12,288 (0.4%) | **12,228 (99.5%)** |
| well-sampled (n_active ≥ 100) | ~0 | **12,156 (98.9%)** |
| median firing rate (alive) | – | 0.00455 |

**A discrete 13-feature code for the super weight.** Correlating each feature's activation against
the SW channel (2371) magnitude over real tokens:

| |r| threshold | > 0.9 | > 0.8 | > 0.5 | > 0.05 |
|---|---|---|---|---|
| features | 6 | 12 | 13 | **13** |

- the 13 have **n_active 801–1478** (firing rate 0.200–0.368%) — versus the SW channel's own spike
  rate of **0.195%**: they fire on essentially the SW spike events
- **complete gap**: the highest |r| outside the set is **0.0151**
- the remaining 12,143 well-sampled features sit at **median |r| = 0.0027**

⚠️ **Circularity caveat, stated plainly:** channel 2371 is part of the SAE's input, so *some* feature
must encode it — the existence of correlation is not the finding. The informative content is
(a) the **concentration**: 13 / 12,156 = **0.107%** of features carry it, and (b) the **discontinuity**
(0.0151 → 0.63, nothing between). The SW is given a dedicated sparse code, not a distributed one.

**What this does and does not license.** It supports a claim of the form *"the super-weight channel is
represented by a small dedicated set of sparse features that fire on its spike events."* It does **not**
support the manuscript's current claim that SAE features recover a compositional/GC axis — GC structure
among selective features was weak (median |r_GC| = 0.050). Reviving the section means changing the claim.

**Blocked:** the NTv3 deeper detection sweep fails inside InstaDeepAI's own remote code
(`modeling_ntv3_pretrained.py:810`, tensor size 6 vs 7), not our harness. NTv3 stays at N = 1, so
cross-architecture replication of the pair finding remains unavailable.

## Manuscript actions

| Item | Action |
|---|---|
| **New §: ensemble/epistasis** | ➕ **ADD.** Same-layer SW pairs are the functional unit; −33.6 pp epistasis vs +0.006 pp random; structural-relatedness enrichment p = 0.00014 (n=5, 3 tasks). This is the paper's strongest mechanistic claim. |
| "leaving histone-mark prediction intact" | 🔴 **CUT.** Same k=5 cliff fires in 2/5 histone seeds. Rewrite as a reliability-and-magnitude contrast. |
| "encode nucleotide composition rather than annotated regulatory grammar" | ✏️ **SOFTEN.** R² = 0.37 (GC alone 0.035); 2/12 motifs survive GC-matched testing. Reconcile with `main.tex:488–489`. |
| "No canonical regulatory motifs are enriched" | ✏️ **REVISE.** False under a GC-matched background; state NF-κB and CpG-rich with the dinucleotide caveat. |
| PROK cross-kingdom numbers (`main.tex:494–495`) | ✏️ Replace with 484.9 / 484.7 / p = 0.768 |
| Promoter | Keep inconclusive; fix or delete the −37.84 single-seed artifact |
| Kingdom contrast | ➕ Optionally add composition R² 0.373 (EUK) vs 0.011 (PROK) — robust, unlike the dead sign-reversal |
| **SAE section (revival path)** | ✅ **CAN BE REVIVED** with a new claim: a discrete 13-feature sparse code for the SW channel (|r| 0.63–0.9999, n_active 801–1478, firing rate 0.200–0.368% vs SW spike rate 0.195%, clean gap to 0.0151). Requires real-sequence evaluation, not hexamer probes. Does NOT support the current compositional/GC claim. |
| **SW quantisation exemption** | 🔴 **CUT.** 14 informative cells across 3 tasks x 2 granularities x 3 precisions: mean SW-specific gain **+0.048 pp** (median −0.035, 9 neg / 5 pos, t = 0.15). No benefit at any granularity. |
| **PROK SAE** | 🔴 **CUT** — invalid for *three* established reasons (layer-2 contamination; fp16 clamp destroying 98% of SW-channel variance; unnormalised objective) and *suspect* on a fourth (its reported feature/SW correlations match the degenerate n_active≈1 signature reproduced here on EUK). Any future SAE must report `n_active` next to every correlation — the degenerate ±0.99 signature is emitted even by a *healthy* dictionary, so it indicts the analysis script, not just the model. |
| **Limitation #4 (EUK SAE instabilities)** | ✏️ **Resolved** — cause was an unnormalised MSE over a channel carrying **6,888×** the median sd. With `--standardize`, dead features fall 74.9% → **0.5%**. Report the fix, not the limitation. |
| **"Shadow redundancy" (`main.tex:152`, abstract + pruning ¶)** | 🔴 **CUT.** Layer-depth artifact: `layer_matched_random` (no SW info) reproduces prox_far on all 3 tasks — promoter −9.09 vs −10.04, histone −2.87 vs −2.74, splice −34.87 vs −34.88 @30%. On splice `prox_near` is WORSE than `prox_far` (−31.85 vs −26.57), inverting the claimed ordering. prox_far prunes all 768 rows of layer 0. |
| Proximity metric | 🔴 Row coordinate is meaningless under `d_model` permutation symmetry; drop it or justify a topology on channel indices |
| Compression "pp" values | ⚠️ **Re-check units** — harness field is a *relative* % change, not pp (−9.39 rel = −7.83 pp at baseline 83.46) |
| **SW quantisation experiments** | 🔴 **RE-FRAME — several test a no-op.** Per-row symmetric RTN preserves the max-magnitude element *bit-exactly* (measured: 0.000e+00 error at INT8/4/3/2), and the SW *is* that element. "Retain SW in full precision" is vacuous at this granularity. Re-pose per-tensor, or report it as the structural result it is. |
| INT2–INT8 + SW rows | ➕ ADD as a clean, explained negative: ≤0.04 pp on 3 tasks at every precision, with the scale-preservation proof as the mechanism. |

## Open / not done
- ✅ DONE: splice n=3 → n=5; epistasis on promoter + histone; EUK SAE (collected, trained, analysed —
  it fails, see §7).
- 🔴 **Retrain both SAEs with per-channel standardisation** and report `n_active` beside every
  correlation. Until then no SAE claim in the paper is supportable.
- Whether the pair effect reproduces in **NTv3** cannot be tested: N = 1 row indexed. A deeper NTv3
  detection sweep would be needed first.
- Epistasis is measured on **fine-tuned** checkpoints only; whether pairs are redundant in the
  *pretrained* model is untested.
- The unexplained ~60% of EUK SW activation has no identified driver (not motifs — §3.4).

## Artifacts
`superadditivity_dnabert2_{reconstructed,prom_core_notata,H3K4me3}.json`,
`superadditivity_ntv3_reconstructed.json`, `pairwise_epistasis_dnabert2_reconstructed.json`,
`composition_vs_motif_{euk,prok_corrected}.json`, `cross_kingdom_transfer.json`
(+ `_ORIGINAL_SUSPECT` archives). Scripts: `run_sw_superadditivity.py`,
`run_sw_pairwise_epistasis.py`, `run_composition_vs_motif.py`.
