# E10b Phase 8 / final synthesis — Phi-3 mechanistic tomography

Written only after the primary F0-F3 decision was frozen at both epsilons
(`results/E10/e10b_phi3_fit_results.json`). Governing lock:
`PREREG_E10b_phi3_tomography.md` (sha256 `098fd52cd398b2f574490fb64dee3842aad057b2a367921bc8ba38926338cda0`).
No E10 v2 result (commit `626cddd`) is touched by this experiment.

---

## Primary result: the decision splits by epsilon

| epsilon | F2 (additive) R² | F3 (pairwise) R² | F2→F3 rel. MAE improvement | Bootstrap CI | Decision |
|---|---:|---:|---:|---|---|
| 0.5 | 0.278 | **0.574** | **+36.4%** | (+0.0057, +0.0109), excludes zero (improving) | **PAIR_TERMS_REQUIRED** |
| 1.0 | −0.196 | −0.103 | **−2.5%** | (−0.0500, −0.0450), excludes zero (**worsening**) | **MIXED_OR_UNRESOLVED** |

At ε=1.0 every one of F0-F3 has *negative* held-out R² — worse than predicting the mean — and
F3 is not merely inconclusive relative to F2, it is **reliably worse** (the bootstrap CI on
the improvement excludes zero entirely on the negative side). This is not a borderline call
decided by a threshold; the pairwise model actively fails to generalize at full ablation.

## Diagnosis (raw data, not a refit): a specific three-way redundancy break

At ε=1.0, exactly the masks that fully ablate **all three layer-2 rows together**
(`L2/r525`, `L2/r1693`, `L2/r1113` — basis indices 0,1,2) show catastrophic damage:

| Mask (fit pool, top 4 by dloss) | `dloss` (ε=1.0) |
|---|---:|
| rows {0,1,2,5} | +10.24 |
| rows {0,1,2,4,5} | +10.10 |
| rows {0,1,2,3,4} | +9.25 |
| rows {0,1,2,3} | +9.25 |

Every other mask in fit/calibration/held-out — including every mask containing only **2 of
the 3** layer-2 rows — stays in the +0.02 to +1.1 range (`results/E10/e10b_phi3_tomography_responses.json`).
Two of the ten held-out masks land in the catastrophic regime (rows {0,1,2} and {0,1,2,4}),
which is what drives F0-F3's negative held-out R² — a linear or pairwise model fit
predominantly on the "normal" regime cannot represent an effect that only appears when a
*specific triple* is simultaneously removed.

**Retrospective pair inspection (run only after the primary decision, no refit) independently
corroborates this from a completely different angle** — the fitted `Gamma` coefficients, not
the raw responses:

| Rank | Pair | `Gamma` (ε=1.0) | Same layer? |
|---:|---|---:|---|
| 1 | `L2/r525`, `L2/r1693` | 0.1346 | yes |
| 2 | `L2/r525`, `L2/r1113` | 0.1334 | yes |
| 3 | `L2/r1693`, `L2/r1113` | 0.1293 | yes |
| 4 | `L2/r525`, `L4/r1693` | 0.0796 | no |

The **top 3 of 15 pairs by `|Gamma|` are exactly the 3 pairwise combinations of the layer-2
triplet**, at roughly 1.6-1.7x the magnitude of the next-ranked pair. The quadratic model
correctly *locates* the interaction (its largest coefficients land exactly on the triplet
raw data independently identifies) but a second-order functional form cannot fully represent
what looks like a genuine **three-way** interaction (damage only when all 3 are removed
together, not predictable from any pairwise combination of the three). At ε=0.5, no such
concentration exists — the top-5 pairs are smaller in magnitude and mixed same/cross-layer
(`frac_same_layer_top5=0.4`, same as the pair population at large), consistent with the
absence of any catastrophic regime at half-strength suppression.

**This is not a manufactured "critical pair" narrative** — no pair was hypothesized in
advance, and the finding here is not even a pair, it is a triple, visible independently in
(a) the raw per-mask responses and (b) the fitted pairwise coefficients pointing at exactly
the same three rows. Per the hard constraints, **no triple-interaction term is fit** — this is
reported as a limitation of the licensed F0-F3 ladder on this basis, not rescued with a
higher-order model.

---

## Answering the core question

> In the one decoder that E10 identified as genuinely multi-component, does finite causal
> response remain approximately additive, or are explicit pair interactions required?

**Both, at different intervention strengths — and the full-strength regime exposes a limit
of the pairwise model itself, not a clean answer either way.**

- At **ε=0.5** (partial, half-strength suppression of multiple rows): explicit pair
  interactions are required (F3 materially and reliably outperforms F2). This maps to
  **Branch B (pair-interactional Phi-3)**.
- At **ε=1.0** (full ablation of multiple rows): neither the additive nor the pairwise model
  adequately describes the held-out response, and the pairwise model is reliably *worse* than
  the additive one, not merely inconclusive. This maps to **Branch C (unresolved / strongly
  nonlinear)** — driven by an identifiable, specific three-way redundancy break among the
  three layer-2 rows, which is disclosed and not chased further.

## What this changes about the architecture-level paper claim

Before E10b, the only decoder-side evidence on pairwise interaction was structural (q1) and
the E10 v2 singleton spectrum (which never intervened on more than one row at a time). E10b
is the first decoder-side test of whether combined finite responses stay additive.

**Result: interaction-dependent causal organization is not encoder-specific.** At ε=0.5,
Phi-3 — a decoder — requires pair terms exactly as MosaicBERT and ModernBERT do. This
supports the corrected framing's **Branch B** interpretation over Branch A:

> Encoder/decoder organization predicts structural geometry more consistently than causal
> response complexity; interactional causal organization can also occur in decoders.

The ε=1.0 result sharpens rather than contradicts this: it is not that Phi-3 "returns to
additive" at full strength — it is that *no* finite second-order model, additive or pairwise,
survives the specific catastrophic regime created by removing all three layer-2 rows at once.
If anything this is a stronger version of the same point: causal response complexity here
exceeds what either family in this ladder can capture, on a decoder.

## What this does NOT establish

- Does not establish that Phi-3 is "interactional" in a single, scale-independent sense — the
  two epsilons gave different mechanical decisions, and both are reported, not averaged or
  resolved into one label.
- Does not establish a "critical triple" as a general phenomenon — this is one basis, one
  model, one corpus; the layer-2 triplet finding is descriptive of this experiment's own data,
  not claimed to generalize to other rows, other decoders, or other corpora.
- Does not establish that decoders in general require pair (or higher-order) terms — this
  remains n=1 among the 5 E10 decoders (the only one flagged eligible); Llama, Mistral, OLMo,
  and Qwen2.5 were never tested for multi-row interaction and none of E10b's evidence bears on
  them.
- Does not establish anything about why layer 2 specifically shows this pattern (e.g., no
  claim about attention sinks, residual-stream capacity, or redundancy origin) — this is a
  finite-intervention behavioral fact, not a mechanistic explanation.
- Does not license fitting a triple-interaction, SAE, PCA, or any learned-direction model to
  "resolve" the ε=1.0 result — explicitly forbidden by the hard constraints and not attempted.
- Does not change or reopen any E10 v2 result. Llama/Mistral/OLMo/Qwen2.5 remain
  SINGLE_COMPONENT_DOMINANT and MosaicBERT/ModernBERT remain PAIR_TERMS_REQUIRED, unmodified.

## Raw artifact inventory

| Artifact | Path |
|---|---|
| Basis audit | `experiments/frozen/E10b_phi3_tomography/PHI3_BASIS_AUDIT.md` |
| Locked prereg | `experiments/docs/prereg/PREREG_E10b_phi3_tomography.md` (sha256 `098fd52c...`) |
| Mask design + validity checks | `experiments/frozen/E10b_phi3_tomography/masks_phi3.json` |
| Raw per-mask, per-batch responses (both epsilons) | `results/E10/e10b_phi3_tomography_responses.json` |
| F0-F3 coefficients, metrics, bootstrap, retrospective pairs | `results/E10/e10b_phi3_fit_results.json` |
| Measurement script | `experiments/frozen/E10b_phi3_tomography/run_phi3_tomography.py` |
| Fitting script | `experiments/frozen/E10b_phi3_tomography/fit_phi3_observers.py` |
| Mask-generation library | `experiments/frozen/E10b_phi3_tomography/mask_lib_e10b.py` |

## Commit inventory

| Commit | Content |
|---|---|
| `1128fea` | Phase 0: basis audit |
| `507c5fd` | Phase 1-3: prereg lock, mask design + validation |
| `13d2fa7` | Phase 4: measurements frozen |
| `3374fb5` | Phase 5-7: F0-F3 fit, bootstrap, decision |
| (this commit) | Phase 8 + final synthesis |

## ONE exact next action

None authorized. E10b's stop rule (Phi-3 only, no other decoder tomography, no
triple-interaction rescue) has been reached exactly as specified; per the launch instruction,
this experiment stops here.
