# Phase D1 — full PROK biology rerun on the corrected super weight (L8/r260)

Date: 2026-08-07 · GPUs 4–7 · model `GENERator-v2-prokaryote-3b-base` · fp32 · seed 42
All 9 pipeline steps rerun. Contaminated originals archived as `results/prokaryote/*_ORIGINAL_SUSPECT.{json,png}` (17 files).

---

## 🔴 HEADLINE: the `r = −0.710` kingdom sign-reversal is **DEAD**

The claim was that PROK's super-weight activation correlates *negatively* with ablation cost
(`r = −0.710`), reversing EUK's positive relationship. On the correct row there is **no
relationship at all**.

| measure | contaminated L2/r1927 | **corrected L8/r260** | EUK L4 (reference) |
|---|---|---|---|
| Pearson r(SW_act, KL) | −0.7096 | **+0.9577** | +0.4368 |
| **Spearman ρ** (outlier-robust) | −0.8043 | **+0.0007  (p = 0.96)** | **+0.9719** (p ≈ 0) |
| ρ excluding outliers | −0.408 | **−0.0007  (p = 0.96)** | — |
| n hexamers > 10× median act | 1432 / 4096 | **2 / 4096** | 0 / 4096 |

### ⚠️ The Pearson +0.958 is a two-point artifact — do not report it as a sign flip

Taken naively, `r` moves from −0.710 to **+0.958** and the story becomes "the reversal reverses."
That is wrong. The corrected row's activation is flat at ~621 for **4094 of 4096** hexamers;
only `ATAGAT` (26,074) and `AAAAAA` (25,048) sit above it. Those two points carry the entire
correlation:

- Spearman ρ = **+0.0007**, p = 0.96
- excluding the 2 outliers: Pearson r = +0.091, Spearman ρ = −0.0007
- the script's own quartile test agrees: t = 1.772, **p = 0.077 (n.s.)**

The correct statement is **"no correlation,"** not "positive correlation." Reporting +0.958 would
substitute one artifact for another.

---

## What the corrected PROK super weight actually is

**Its activation is content-invariant.** Four independent measurements agree:

| test | result |
|---|---|
| Hexamer activation spread (n = 4096) | 1/25/50/75/99 pct = 619 / 621 / 621 / 622 / **625** |
| GC ↔ activation correlation | **r = −0.045** |
| Motif enrichment (Fisher + BH, 200 top k-mers) | **all q = 1.000** — TATA, Shine-Dalgarno, CpG-rich, E-box, … nothing |
| Real vs shuffled activation shift | −0.00002 ± 0.0001, **p = 0.21 (n.s.)** |

**Shuffle controls** (4 types × 5 shuffles × 90 sequences):

| shuffle | corrected mean shift | p | contaminated mean shift | p |
|---|---|---|---|---|
| dinuc | −0.000020 | 0.121 n.s. | +0.000389 | 0.009 ** |
| mono | −0.000141 | <1e-6 *** | +0.000495 | 0.002 ** |
| kmer_block | −0.000003 | 0.788 n.s. | +0.000428 | 0.010 ** |
| trinuc | +0.000004 | 0.750 n.s. | +0.001148 | <1e-6 *** |

Where the contaminated row responded to *all four* shuffles, the corrected row responds to
essentially none. The one "significant" result (mono) has an effect size of **4.7 × 10⁻⁹ relative
to the activation magnitude** — significant only because the variance is minuscule. It is not a
biological signal.

**But the super weight is functionally real.** Content-invariance is not irrelevance:

- SW ablation ΔPPL = **+1.2459 ± 0.5379**
- random-row ablation ΔPPL = +0.0008 ± 0.0007
- paired t = **21.84**, p ≈ 0

So PROK's super weight is an **input-independent bias/gain term** — always on, not a feature
detector — yet removing it measurably degrades the model. EUK's, by contrast, is a genuine
compositional feature detector (activation varies 267k–283k across hexamers; ρ(act, KL) = +0.97).

### ⚠️ The `+9.47` log-PPL figure becomes `+1.25`
The manuscript's PROK ablation magnitude was inflated **7.6×** by the contaminated row. The effect
is still overwhelmingly significant versus random rows — but the number must be corrected everywhere.

---

## ✅ A *different* kingdom contrast survives — and strengthened

The GC-dependence of **ablation cost** (not of activation) was not an artifact. Correcting the row
made it stronger:

| | EUK | PROK contaminated | **PROK corrected** |
|---|---|---|---|
| r(GC, ΔPPL) | **−0.001** | −0.520 | **−0.661** |
| r(CpG density, ΔPPL) | −0.021 | −0.496 | −0.524 |
| regression R² | 0.158 | 0.347 | **0.497** |
| mean ΔPPL | +2.92 | +9.47 | +1.25 |
| ΔPPL by context | prom +2.68 / enh +2.79 / rand +3.28 | — | prom +0.99 / enh +1.18 / rand +1.57 |

In PROK, ablating the SW hurts most on **AT-rich** sequence (r = −0.66, R² = 0.50); in EUK the cost
is essentially independent of GC (r = −0.001). That is a real, interpretable difference and it is
robust to the correction.

**Report it as hypothesis-level, not as a headline.** Two reasons: n = 2 models, and the kingdoms'
training corpora differ in baseline GC (E. coli K-12 ≈ 50.8%, hg38 ≈ 41%), so corpus composition is
an unresolved confound for a "kingdom" interpretation. It is *not* a rescue of the sign-reversal
claim — it is a different claim about a different quantity.

---

## 🔴 SAE: do not retrain — the premise is gone

Task Spec D1 called for retraining the SAE on layer-8 activations. **This is now pointless and
should be dropped, not rescheduled.** A sparse autoencoder decomposes *variance* in activations
into interpretable features; the corrected row has essentially none (621 ± 2 across all 4096
hexamers). There is nothing to decompose.

The existing PROK SAE result is invalid regardless — `main.tex:925` states it was trained on
layer 2, the contaminated layer, and no SAE checkpoint exists anywhere under `results/`.

**Action: cut the SAE section**, and with it the claim that the SAE "independently recovers the
compositional axis" — on the correct row there is no compositional axis in the activation to recover.

---

## Supporting: attribution methods now disagree

Gradient attribution vs token omission on the corrected row: mean Spearman ρ = **−0.12**
(median −0.15), **0.0%** of sequences above ρ > 0.3. Two attribution methods that should agree if
there were a real localized signal do not — consistent with a content-invariant activation with
nothing to attribute to specific positions.

---

## Manuscript actions

| Item | Action |
|---|---|
| `r = −0.710` sign reversal (`main.tex:95`, `:434`) | 🔴 **CUT.** Corrected value is ρ ≈ 0 (p = 0.96). Do not replace with +0.958 — that is a 2-point artifact. |
| PROK SAE section (`main.tex:447–465`, `:752–756`, `:925`) | 🔴 **CUT.** Trained on the contaminated layer; no checkpoint; and no activation variance to decompose on the correct row. |
| PROK ablation `+9.47` log-PPL (`main.tex:399–402`) | ✏️ Replace with **+1.25 ± 0.54** (vs random +0.0008, t = 21.8) |
| "comparable phenotype at layer 2, row 1,927" (`main.tex:177`) | ✏️ Replace with layer 8, row 260 |
| `‖U_k‖_F = 2,648` for PROK row 1927 (`main.tex:265`) | ✏️ Replace — wrong row; corrected row 260 is rank 1/3072 |
| "inverse GC preference / AT-rich activators" (README:781) | 🔴 **CUT.** Activation is GC-independent (r = −0.045) |
| Fig 1 C/D, Fig 2 B/D, Fig 3 PROK panels | ✏️ Regenerate — corrected data now in `results/prokaryote/` |
| Limitations | ➕ **ADD**: EUK-vs-PROK is an L4-vs-**L8** comparison; and PROK's SW is content-invariant, so the two kingdoms' super weights are not the same kind of object |
| *New, optional* | GC-dependence of ablation cost (PROK −0.66 vs EUK −0.001) — hypothesis-level, corpus-GC confound stated |

---

## Verdict against the Task Spec D1 decision rule

> *"if L8/r260 also encodes composition like EUK, the 'kingdom sign reversal' is likely dead (both
> kingdoms do the same thing). If it differs, you have a real (n=2) kingdom contrast."*

**Neither branch, and the sign-reversal is dead either way.** L8/r260 does not encode composition
*like* EUK — it does not encode composition *at all*. The kingdoms differ, but the difference is
**"compositional feature detector (EUK) vs content-invariant bias term (PROK),"** not a reversal of
sign in a shared relationship. A sign cannot reverse where there is no correlation.

The one contrast that does survive — GC-dependence of ablation *cost* — is real and strengthened,
but is a different quantity than the one claimed and carries a corpus-composition confound.

**Net effect on the paper: D1 confirms the cut rather than rescuing the claim**, which is the
outcome flagged as likely when this run was proposed. Paper B remains the defensible paper; the
kingdom and SAE material comes out.

---

## Artifacts

Rerun (corrected, L8/r260): `sw_kmer_scan`, `sw_token_omission`, `sw_shuffle_controls`,
`sw_kmer_motifs`, `sw_grad_attribution`, `attribution_comparison`, `sw_hexamer_causal`,
`sw_causal_tracing`, `sw_ablation_regression` — all `.json` + `.png` in `results/prokaryote/`.
Contaminated originals preserved alongside as `*_ORIGINAL_SUSPECT.*`.
