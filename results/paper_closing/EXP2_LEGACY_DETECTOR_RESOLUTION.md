# EXP2 — legacy detector provenance, resolved for all 6 models

Run 2026-09-08 on this machine (8× A100-80GB). Closes the gap recorded in
`audit/detector_provenance.csv`, where 6 of 22 census candidates were selected by the
**legacy** rule (global argmax of *absolute* activation) rather than the **current** rule
(global argmax of the *layer-relative ratio*), and 3 of those 6 had never been checked.

Scripts: `scripts/paper_closing/run_uniform_detector_text.py` (the 3 text decoders + a
positive control) and `scripts/paper_closing/run_uniform_detector.py` (the 2 genomic models).
Artifacts: `uniform_detector_text_{llama,mistral,olmo,smollm2-1.7b}.json`,
`uniform_detector_{ntv3,dnabert2}.json`, `exp2_olmo_exposure_sensitivity.json`.

Every text-decoder run passed **four** reproduction gates against `audit/census_master.csv`
before anything was interpreted — `baseline_loss`, `R_cand` at ε=1.0 and ε=0.5, and the 5
stored control rows redrawn from `SeedSequence(42).spawn(23)[panel_index]`. Observed relative
errors were 2.1e-09 – 3.1e-06, i.e. far inside the 2 % tolerance, and the control rows
reproduced exactly for all four models.

---

## 1. Result table

| model | frozen | ratio-argmax | agree? | robust to input / special tokens? |
|---|---|---|---|---|
| GENERator-EUK-3B | L4 / r2371 | L4 / r2371 | ✅ | yes (round-2, both conventions) |
| **Llama-7B** | **L2 / r3968** | **L2 / r3968** | ✅ | **yes** — rank **1/131072**, argmax stable in **100 %** of 24 inputs, identical under all 3 tokenizations |
| **Mistral-7B** | **L1 / r2070** | **L1 / r2070** | ✅ | **yes** — rank **1/131072**, **100 %** stable, identical under all 3 tokenizations |
| **OLMo-7B-0724-hf** | **L1 / r269** | **L2 / r269** | ❌ | **yes, robustly disagrees** — frozen is rank **2/131072**; argmax is L2/r269 under all 3 tokenizations, stable in 83 % of inputs |
| DNABERT-2 | L5 / r603 | L8 / r603 *(default)* / **L9 / r264** *(no special tokens)* | ❌ | **no** — the argmax moves to a different **row** when special tokens are dropped |
| NTv3 | L11 / r1472 | L11 / r1472 *(6 DNA probes)* / L6 / r1472 *(ACTB probe, round-2)* | ⚠️ | **no** — input-dependent |

**Two of the three previously-unverified text decoders confirm the frozen choice exactly**, at
global rank 1 of 131,072 coordinates, with perfect input stability and no sensitivity to
special-token handling. That is a stronger pass than "the rules happen to agree".

---

## 2. OLMo-7B: the ratio rule picks the causally *worse* coordinate

The disagreement is **same row, one layer deeper** (r269 at L2 instead of L1). Both
coordinates were ablated on the *same* eval pool, so this is paired:

| coordinate | activation ratio | rel. NLL increase, ε=1.0 | rel. NLL increase, ε=0.5 |
|---|---|---|---|
| **frozen L1 / r269** (census) | 1539.17 | **+1.117758** | +0.006288 |
| ratio-argmax L2 / r269 | 2346.25 | **+0.023745** | +0.003676 |
| 5 seeded controls, L2 | ~1 | median −2.53e-07 | — |

**The coordinate the ratio rule prefers is 47× less damaging than the one the census used.**
The uniform candidate is still real (G = +0.0237 against a control median of −2.5e-07, i.e.
~10⁵× its controls), but on this model the *legacy* absolute-activation rule selected the more
causally important coordinate.

This is the same dissociation as the SmolLM2-1.7B within-layer result (where the row ranked
4th by ratio is 130× more damaging than the row ranked 2nd), now across layers of one
channel — and it is the paper's thesis in miniature: **the argmax of a structural statistic
is not the most causally important coordinate.**

For contrast, DNABERT-2 goes the *other* way: ablating the ratio-argmax L8/r603 *raises* MLM
loss (+0.1072) while ablating the frozen L5/r603 slightly *lowers* it (−0.1322). So the ratio
rule is **not uniformly better or worse** than the legacy rule — it is simply a different
statistic, and neither reliably lands on the most causal coordinate.

### What the disagreements have in common — a tendency, not a rule

Where the two rules disagree, the ratio-argmax tends to land on the **same row at a different
layer**: OLMo L1→L2 (r269), NTv3 L11→L6 (r1472, under the ACTB probe), DNABERT-2 L5→L8
(r603, under default tokenization). That is consistent with a super row being a *residual-stream
channel that persists across depth*, with the two statistics disagreeing only about where along
it to sample.

⚠️ **But this is not universal, and it should not be written as a rule.** Under
`add_special_tokens=False` DNABERT-2's argmax moves to a different row entirely (L9/r264 —
which is, separately, the anchor of the co-dominant pair). The "same channel, different depth"
reading is a **tendency observed in 3 of 3 disagreements under their default conventions**,
and it is sensitive to the detector's input and special-token handling. [MEASURED] for the
individual coordinates; the channel-persistence interpretation is **[UNTESTED]**.

---

## 3. Bounded exposure to the headline correlation

The concern on record was: *"All three are text decoders — the cohort carrying the headline
ρ = +0.770 — so if any is not the ratio-argmax, 3 of those 10 points were selected by the
non-predictive statistic."* Only **one** of the three is not the ratio-argmax, and the
exposure is small (`exp2_olmo_exposure_sensitivity.json`, 10,000-sample percentile bootstrap):

| cohort | published | OLMo dropped | OLMo → ratio-argmax coordinate |
|---|---|---|---|
| all 22 | ρ=+0.766, p=3.2e-05 | ρ=+0.769, p=4.7e-05, n=21 | ρ=+0.755, p=4.9e-05 |
| text decoders | ρ=+0.770, p=9.2e-03 | ρ=+0.783, p=1.3e-02, n=9 | **ρ=+0.673, p=3.3e-02** |

The 22-model result is insensitive. The text-decoder subgroup moves from +0.770 to +0.673 in
the worst case — still significant, but its 95 % CI lower bound falls to +0.032, so that
**n=10 subgroup was already fragile and should not carry weight on its own** regardless of
this substitution.

⚠️ *Convention caveat on the substitution column.* The other 21 models' `activation_ratio`
values come from the census's detection input, while OLMo's substituted 2346.25 was measured
here over 24 pooled WikiText-2 windows. The two conventions differ (OLMo's frozen coordinate
reads 2334.56 in the census table vs 1539.17 pooled here), so the substitution column mixes
conventions. The **OLMo-dropped** column is convention-free and is the cleaner robustness
check; both point the same way.

---

## 4. Status of the uniform selection rule

Provenance is now **closed for all 22 census rows**: 16 selected by the current rule by
construction, 3 legacy models verified to agree (GENERator-EUK-3B, Llama-7B, Mistral-7B), and
3 legacy models with a characterised disagreement (OLMo-7B robustly; DNABERT-2 and NTv3
input-dependently), each with a paired causal evaluation of both coordinates.

**Recommended wording change.** The rule should be described as *the detector*, not as *the
definition of the super row*. "Global argmax of the layer-relative ratio, accept if ≥ 5" is a
reproducible way to **find** candidates and it is what the manuscript should specify. It should
**not** be presented as identifying the most causally important coordinate — OLMo-7B (47×) and
SmolLM2-1.7B L7 (130×) both falsify that, in different geometries.

No census file was modified. Machine-readable per-model resolution:
`audit/detector_provenance_exp2_resolution.csv`.
