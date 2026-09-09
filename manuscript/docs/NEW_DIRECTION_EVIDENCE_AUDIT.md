# NEW_DIRECTION_EVIDENCE_AUDIT.md

**Date:** 2026-08-13. **Scope:** backward provenance trace of every new headline claim
handed down for the post-overnight-queue reconciliation, against the actual repository at
commit `6bf1601` (+ untracked files) at `/work/11034/atzanakak/glm_super_weight/genomic-super-weights`.
**Method:** claim → result/report → raw artifact → script/config/checkpoint. No script was
run, no model was loaded, no number in this document was computed for this pass — every
number below is either quoted from an existing file or is a direct, static reading of
existing code (e.g. the quantization scale formula). Two rounds of independent search were
performed for every claim (my own greps/reads, plus a second independent `Explore`-agent
pass per cluster) before anything below was marked `MISSING`.

**Headline finding of this audit.** No file, log, JSON artifact, or code change anywhere in
this repository — committed or untracked, in `results/`, `logs/`, `reports/`, or
`manuscript/` — postdates the last commit (`6bf1601`, 2026-08-13T06:23:11-05:00; verified
by `find -newermt`). Of the twelve numbered items below, **ten are `MISSING`** (no
supporting artifact exists anywhere in the repository), **three of those ten are additionally
`CONTRADICTORY`** — not merely unsupported but in direct numeric or logical conflict with
artifacts that do exist (Evo1's 2^24 saturation claim vs. the documented fp16/max\|Δ\|
saturation story; NTv3's p≈0.48 truncation-null claim vs. the existing p=0.008 5-seed result;
PROK's eukaryotic-probe-contamination claim vs. the existing, `established` write-direction-
convention explanation for the same r=±0.710 numbers) — and **two are `PARTIAL`** (a real,
code-verifiable mathematical fact or mechanism exists, but the specific number or framing
claimed does not). Separately, four items from the *rejected-claims* list (U_k/DNABERT-2
706/768, the DNABERT-2 KL≈0.31 retirement, the shadow-redundancy retirement, and the
histone-dissociation softening) are **already `AUDITED`** — this repository completed and
locked that work in the prior session, before this pass began. This document does not
recreate any missing result. Per the governing instructions, missing and contradictory
provenance is reported here, not filled in or silently resolved in either direction.

---

## 1. DNABERT-2 redundant functional pair (−33 pp joint ablation)

**Claimed statement:** individual ablation of either of a specific pair of DNABERT-2 rows
has ~zero effect on splice classification; ablating the pair together produces ≈−33pp.

**Evidence found:** none. Two independent searches (mine + a dedicated `Explore` pass) found
no script, JSON, or report describing a 2-row joint ablation. The only −33pp-adjacent number
in the repo is `results/gue_per_row_ablation.json`, a **single-seed instance of the existing
10-row super-row ensemble** ablation (`all_rows.delta_acc = −33.34pp`), which is the same
object as ledger claims C-027/C-028 (mean −25.5±0.7pp across 3 seeds; individual-row max
−1.45pp) — not a 2-row pair. The untracked single-neuron pilot report
(`reports/single_neuron_pilot_dnabert2.md`, §"Sanity check") independently notes seed
variance in this same 10-row ensemble number ("seed 0 was −36.2%, seeds 1–2 were ~−1%"),
which is a plausible source of a memorable "−33ish pp" figure attaching itself to the wrong
object.

**Controls:** n/a — no experiment exists.
**Provenance status:** **MISSING.**
**Caveat:** the existing, real result (10-row ensemble collapses splice by −25.5pp; no single
row exceeds −1.45pp; C-027/C-028) is not the same claim as a 2-row redundant pair and must
not be conflated with it.
**Ready for main text:** No.

## 2. The DNABERT-2 pair is intrinsic to pretraining (base-model superadditivity)

**Evidence found:** none. The only base-vs-fine-tuned DNABERT-2 comparison in the repo is in
the untracked single-neuron pilot report (§8), and it concerns a single hidden **neuron**
(layer 5, neuron 616) and the single row it maps to (row 26) — not a 2-row pair, and no
superadditivity is computed anywhere (base-vs-fine-tuned cosine similarities are reported,
not joint-vs-individual effect sizes).

**Provenance status:** **MISSING** (depends on claim 1, which is itself missing).
**Ready for main text:** No.

## 3. Residual-norm carriage / joint collapse (layer-9 top-2 carrier pair)

**Evidence found:** none. No file mentions "layer-9 residual norm," "residual-norm carrier,"
or a joint-removal norm-collapse test for DNABERT-2. The only layer-9 DNABERT-2 datum in the
repo is unrelated: `paper/main.tex:382` and `manuscript.txt` describe the *argmax* of the MLP
output shifting to row 264 at layer 9, as part of the existing row-603 source/propagator
residual-attribution figure (C-010) — a single-row observation, not a two-channel norm-
carriage analysis, and it names a different row (264) than anything else in this cluster.

**Provenance status:** **MISSING.**
**Ready for main text:** No.

## 4. Fine-tuning "recruitment" of a pretrained pair / seed dependence

**Evidence found:** none specific to a pair. General seed-dependence *is* documented for the
existing 10-row ensemble (splice ΔMCC range across NTv3's 5 seeds is −0.038 to −0.172,
`paper/main.tex` Table 1; DNABERT-2 splice seed-to-seed range noted in the neuron-pilot
report), but nothing ties seed variance to a specific 2-row pair or to a "recruitment"
mechanism.

**Provenance status:** **MISSING.**
**Ready for main text:** No.

## 5. GENERator BOS / attention-sink phenotype

**Claimed statement:** dominant activation localizes to token 0; attention to token 0 is
~25–33× uniform; token 0 is attention argmax ~100% of the time; shuffle-insensitive.

**Evidence found:** no script anywhere in the repo computes an attention weight, an argmax
over attention, or a uniform-expectation ratio for any GENERator model. The one adjacent
result is **activation** (not attention) localization: `run_sw_counterfactual_swap.py` and
`run_sw_relay_heads.py` both compute `sw_pos = hs[:, sw_row].abs().argmax()`, and
`results/sw_counterfactual_swap_generator.json` shows this activation-argmax position is
`{0, 1}` across its 20 pairs — i.e., there is a real, already-measured hint that the SW's
*peak activation* sits near sequence position 0. This is not the same measurement as
attention-weight concentration on the BOS token, has no ratio/percentage/argmax-rate
attached to it, and no shuffle-insensitivity test of it exists.

**Provenance status:** **MISSING**, with one adjacent, weaker, already-measured fact (SW
activation argmax position ∈ {0,1} in the counterfactual-swap data) that a future session
could use as a starting point but that does not itself establish an attention-sink phenotype.
**Ready for main text:** No.

## 6. GENERator EUK causal GC steering (≈38.6× vs. matched/random controls)

**Evidence found:** none. `manuscript/docs/prereg/PREREG_steering.md` (E3) specifies
almost exactly this design — row scaling at multiple doses, ≥10 matched random-row controls,
≥1,000 generated sequences, GC% as the primary readout, entropy/PPL/complexity as degeneration
controls, PROK expected to mirror EUK — but the file was **never locked** (`_Locked: (filled
by prereg_lock.py)_` is still a placeholder), `CLAIMS_LEDGER.md` row C-025 is `pending`, and
`PROJECT_STATUS.md` states explicitly "E3 not started." No "38.6" appears anywhere in the
repository.

The closest *existing* result is a different experiment: `run_sw_counterfactual_swap.py`
interpolates the SW coordinate between GC-rich/AT-rich donor-target activation pairs and
measures next-token KL and a logit-space GC shift (not free generation + sampled GC% of
output). `results/sw_counterfactual_swap_generator.json`: EUK `sw_kl` at dose 1.0 mean
0.0010968 vs. `rand_row_kl` mean 5.71e-08 (~19,000× in **KL**, not GC% and not "38.6×").

**Provenance status:** **MISSING.** The experiment this claim describes is preregistered but
explicitly not yet run; the number "38.6×" has no source in this repository under any
metric.
**Ready for main text:** No — the E3 preregistration exists and is ready to run in a future
session, but this pass may not run it (out of scope) and must not report a number that isn't
there.

## 7. Steering degradation / quality controls (entropy, PPL, complexity, collapse)

**Evidence found:** none computed. `PREREG_steering.md` specifies these controls as part of
E3's design; none has been run because E3 has not been run (see item 6).

**Provenance status:** **MISSING** (inherits from item 6).
**Ready for main text:** No.

## 8. Quantization: per-row scale-preservation derivation

**Claimed statement:** under per-row quantization, the row's max-magnitude element defines
the row's scale and is therefore automatically mapped to the quantizer's endpoint — so
explicitly exempting an SW that already defines its own row's scale can be a mathematical
no-op.

**Evidence found:** this is the one item in the audit that is a **direct, verifiable property
of code already in the repository**, not a new claim. Both existing quantization scripts
implement per-row (per-output-channel) round-to-nearest quantization with the scale set from
the row's own max absolute value:

- `scripts/compression/run_quantization_ablation.py:12-14` (docstring): `scale = max(|row|) /
  127`, `quantize(row) = round(row/scale) * scale`.
- `scripts/compression/run_whole_model_quantization.py:300-303`: `amax =
  rows.abs().amax(dim=1, keepdim=True)`, `scale = amax / maxval`, `q = round(rows/scale)`
  — `dim=1` confirms the reduction (and therefore the scale) is per-row.

For the specific element that equals the row's own max, `value/scale = value/(max/127) =
127` exactly (the quantizer endpoint), and dequantizing recovers the original value up to
the rounding already built into the endpoint case (round(127)=127 exactly) — i.e. the
mathematical claim is true of the code as written, and can be stated and derived without
running anything.

**Adjacent empirical evidence:** ledger claim C-031 (`established`) — "the NLP SW-preservation
heuristic does not transfer; INT4 with vs. without SW exemption differs below resolution" —
is existing, already-established evidence that is *consistent with* (not proof of, since the
causal mechanism was never explicitly stated in those runs) the scale-preservation
explanation: if the per-row scheme already preserves the SW automatically, explicit exemption
would be expected to add nothing, which is exactly what C-031 found.

**Provenance status:** **PARTIAL.** The derivation is real and auditable directly from code
(not a new experiment — reading existing code). The connection between the derivation and
the *existing* empirical null (C-031) is a reasonable, disclosed inference, not itself
measured — no run in the repo explicitly tests "does the SW define its row's scale" as a
separate boolean per model.
**Ready for main text:** Yes, for the derivation itself, stated as a mathematical fact about
the implemented per-row rule and explicitly scoped to it (not generalized to group-wise
quantization, which was not tested — see below). The connection to C-031 should be worded as
"consistent with," not "explained by," pending a check that the SW row's own max is in fact
the value being rounded to the endpoint in the specific runs C-031 draws on (not verified in
this pass; C-031 itself has no recorded evidence path per `UNMIGRATED.md`).

## 9. Quantization: group-wise granularity + destructive-regime protection test

**Claimed statement:** at group-wise quantization, interpretation depends on group size;
even under a deliberately destructive quantization regime, explicit SW protection gave no
measurable benefit.

**Evidence found:** no group-wise (as opposed to per-row) quantization scheme exists anywhere
in the repo — every quantization script found reduces `amax` over `dim=1` (strictly per-row).
"Group-wise," "group_size" and "groupsize" return zero hits. The word "destructive" does not
appear attached to any quantization run. What *does* exist and is close in spirit is the
already-established C-031 (INT4, SW-inclusive vs. SW-exempt PPL differ by −0.0008/+0.0004,
below noise floor, "no measurable benefit" — `manuscript.txt:280-293`, `README.md:69-99`) —
but that is the **standard/default INT4 condition**, not a regime chosen to be deliberately
destructive. A more aggressive variant exists as code
(`scripts/compression/run_int4_multimodel_benchmark.py`, which targets INT2) but **has no
results JSON anywhere in the repo** — only a stray PNG — so it was apparently never run to
completion, or its output was never saved. It cannot support a claim either way.

**Provenance status:** **MISSING** for both the group-wise granularity discussion and the
specifically-destructive regime. The adjacent standard-INT4 null (C-031) is real but is not
the same regime the claim describes, and should not be relabeled as "destructive."
**Ready for main text:** No — the group-wise / destructive-regime material should not be
written into the manuscript. C-031's existing standard-INT4 null may be kept as-is (see item 8).

## 10. Evo1: numerical-saturation false positive (value pinned at 2^24) + failed rescue

**Claimed statement:** the Evo1 SW value was pinned exactly at 2^24; a rescue experiment
forcing it to survive to the output still produced no functional effect; therefore Evo1 is a
detection false positive, not a real SW regime.

**Evidence found:** **none, and the existing artifacts point the other way.** No occurrence
of `2**24`, `2^24`, `16777216`, or `8388608` exists anywhere in the repo. The repository's
own, already-documented saturation story is quantitatively different: the **fp16** trace
(`sw_residual_attribution_evo1.json`, described in `scripts/analysis/FIGURES_README.md:150-164`
and `EVO2_REPLICATION.md:104`) saturates at fp16's actual max (~65,504), not 2^24, starting
layer 11 and going NaN afterward; the corrected fp32/bf16-clean trace instead shows the row
climbing to ≈1.2×10⁶ (L11), ≈1.6×10⁷ (L12), and ≈3×10⁷+ from L13 (later pushed to 1.29×10⁹ by
`CLAIMS_LEDGER.md` N-001) — none of these values is 2^24 (16,777,216), and the trajectory is
a continued climb, not a plateau pinned at a fixed ceiling. Moreover `CLAIMS_LEDGER.md` N-002
already establishes that the apparent "frozen residual" was **a layer-to-layer max\|Δ\|
summary-statistic artifact**, not a numeric ceiling: "No layer is bitwise identical to any
other" (`results/evo1_layer_freeze_check.json`). The word "rescue" appears four times in the
repo, none referring to Evo1 (three refer to GENERator PROK's impulse-KL dose-escalation
probe in `BLOCKED.md`, one to a not-yet-run Evo2 appendix idea, one to SAE dead-feature
resampling).

**What does already exist and is genuinely established:** C-009 — Evo1 is structurally
concentrated (‖U_k‖_F top 1.2–4.1%) but ablation gives ΔPPL = 0.0%. This is real, but it is
a weaker and differently-reasoned claim than "numerical saturation artifact" — it says
nothing about *why* Evo1's candidate is non-load-bearing beyond the existing structural/
systemic (mixer-redistribution) account already in the manuscript (`paper/main.tex:325-341`).

**Provenance status:** **MISSING, and partially CONTRADICTORY** — the specific 2^24 diagnosis
does not merely lack support, it names a number that does not match any measured saturation
point in the existing artifacts, and the "pinned/frozen" framing is a phenomenon this
project's own ledger (N-002) already re-examined and attributed to a statistic, not the
underlying hidden state.
**Ready for main text:** No. The existing, audited account (C-009 + the structural/systemic
mixer-redistribution story already in `main.tex`) remains the defensible statement about
Evo1; it should not be replaced by the 2^24 saturation narrative without new evidence.

## 11. NTv3: silent truncation-bug retirement of the positive splice replication

**Claimed statement:** old NTv3 checkpoints were trained under a silent max-length/truncation
bug preserving only ~20% of the intended window; after fixing the bug and retraining, NTv3
shows no functional SW effect; p≈0.48.

**Evidence found:** `models/ntv3_wrapper.py:21` hardcodes `max_length=512`, and
`scripts/evaluation/run_gue_multiseed.py:269` defaults to the same value — this is a real,
inspectable configuration fact — but no comment, docstring, commit message, or ledger entry
anywhere calls it a bug, states an "intended" larger context window, or gives a ~20%
retention figure. `results/gue_multiseed_ntv3_splice.json` gives `p_val_mcc = 0.0083` (5
seeds, 5/5 negative direction) — this is exactly the existing, ledgered C-029 result. **No
p≈0.48 value for NTv3 exists anywhere in the repository, under any metric.** No retrained
NTv3 splice checkpoint exists (`results/gue_checkpoints_multiseed/ntv3_prom_core_notata/`
holds only 2 seeds of a *different* task, prom_core_notata, and no splice checkpoints).

**Provenance status:** **MISSING, and directly CONTRADICTORY on the numeric claim** — the
only NTv3 splice p-value in the repository (p=0.008, C-029) is the opposite of what would be
needed to support "no functional SW effect," and it comes from the existing, already-flagged
5-seed run (flagged only for a *post-hoc MCC-vs-accuracy metric choice* — `PHASE_0_TRIAGE.md`
§0.4 — not for truncation).
**Ready for main text:** No, as a positive number. **Resolution adopted for this pass:** the
governing guardrails explicitly forbid presenting NTv3 as a confirmed positive splice
replication going forward, independent of whether the truncation-bug mechanism can be
verified locally — so C-029 is downgraded out of `supported`/"replicates" language in the
ledger (Task 2) and the "DNABERT-2 is the only strong functional encoder example" framing is
adopted for the outline. But because the only artifact in this repo (p=0.008, real controls,
5/5 seeds) directly contradicts the specific retraction mechanism and number offered, C-029 is
marked **`contested`**, not silently deleted or silently re-confirmed — this is flagged for
the author as a genuine conflict between instruction and measured evidence, not resolved by
this pass. See `CLAIMS_LEDGER.md` for the exact wording adopted.

## 12. PROK contamination (eukaryotic probe on prokaryotic model) + actual SW at L8/r260

**Claimed statement:** the old GENERator PROK SW was identified using a eukaryotic probe on
the prokaryotic model; this invalidates the old sign-reversal, kingdom, SAE, and composition
results; the actual PROK SW is ≈L8/r260.

**Evidence found:** confirmed **MISSING and directly CONTRADICTORY** by two independent
passes. The repository documents a **different** PROK layer/row problem, which must not be
conflated with this claim. `CLAIMS_LEDGER.md` N-009 (dated 2026-08-12, explicitly revising an
earlier, wrong "layer-mismatch" hypothesis) establishes: the claim's layer **is** 2, row
**1,927** (matching the checkpoint used throughout E4 and the main manuscript, verified
identical by name), but the *stored artifact's* ‖U_k‖_F value/rank (2648.48, rank 1/3,072)
does not reproduce against current weights at that same layer (recomputed 5.51/5.47, rank
1277/1289 — a ~480× magnitude discrepancy). N-009's own text rules out a probe/model mismatch
as the explanation: "the two current implementations agree with each other and both disagree
with the stored artifact, so this is not a formula difference... Cause not determined." No
file anywhere mentions a eukaryotic-probe-on-prokaryotic-model contamination mechanism, and
no file names layer 8 or row 260 in a PROK context — exhaustive grep for "probe" +
contamination/mismatch language, "L8," "layer 8" (in a PROK context), "row 260," "r260"
across every file type returns zero hits.

**More directly contradictory:** the claimed casualty of "contamination" — the EUK/PROK
r≈−0.710 sign relationship — is not merely un-invalidated in this repo, it is **actively
used, explained, and treated as an established, resolved positive result**: ledger C-020
("established... migrate"), `manuscript.txt:263-275`, `paper/main.tex:433-459`, and
`README.md:775-787` all state that the negative sign is a **write-direction convention**
(the PROK SW writes in the negative direction; on write *magnitude* the correlation is
r=+0.710, matching EUK's +0.437 in sign and interpretation) — explicitly not a
kingdom-inversion or contamination artifact. This is the opposite framing from the claim
handed down for this pass.

**Provenance status:** **MISSING**, and **CONTRADICTORY** — N-009 already re-examined this
exact layer/row for a different reason (stored-artifact reproducibility) and did not find a
probe-species error; the sign-convention explanation this claim would retract is currently
treated in the manuscript as established, resolved evidence (C-020), not as a live problem.
**Ready for main text:** No, as a positive replacement claim (the L8/r260 SW, the
eukaryotic-probe contamination mechanism, and any restated composition story built on it have
zero provenance and must not be written). **Resolution adopted for this pass:** the governing
guardrails separately forbid presenting the old kingdom/sign-reversal/composition story as
supported going forward, and N-009 *already* gives an independent, repo-native reason to
distrust the old PROK artifact (same layer/row, stored value does not reproduce, C-001 on
hold). So C-001 stays on hold (unchanged) and C-020 (the r=±0.710 sign-convention claim) is
downgraded from `established` to **`contested`** in the ledger (Task 2) rather than either
being left as established or being silently retracted in favor of the unverified new
narrative. The PROK SAE and composition claims built on the same row are cut from main text
per the existing `CUT_LIST.md` demotion logic, consistent with the instruction, but without
asserting *why* beyond what N-009 and the SAE clamp finding above actually show. This
conflict — instruction vs. an `established` ledger claim with real (if now-questioned)
controls — is flagged for the author, not silently resolved.

---

## Addendum — cluster 4 (PROK contamination, PROK SAE, quantization group-wise/destructive)

*(Filled in after the dedicated second-pass search for this cluster completed; see
provenance note at the foot of this file for the exact search performed.)*

### PROK SAE fp16-clamp diagnosis (item D of the task brief)

**Evidence found — this one is directly code-verifiable, independent of any agent report.**
`sae/collect.py:176-179`:

```python
# Clip before float16 cast to avoid Inf from super-weight overflow
# (float16 max = 65504; super-weight channels can exceed this).
h_f32 = h.detach().float().clamp(-60000.0, 60000.0)
_s["h"] = h_f32.cpu().to(torch.float16)
```

This is a real, hard clamp to ±60,000 applied uniformly to every channel (including the SW
channel) before every SAE training example is stored. GENERator PROK's own detected SW
`out_max` is 506,014 (`paper/main.tex:189-191`, C-008) — an order of magnitude above the
clamp ceiling — so if the SAE's residual-stream activations were collected at or near the
same layer/scale, the clamp would destroy the great majority of the SW channel's dynamic
range on its most extreme (and most informative) examples, consistent in kind with the "fp16
clamp destroyed ~98% of channel variance" description, though **the exact 98% figure is not
computed anywhere in the repo** and would need to be measured from the stored shards, not
asserted.

**Complication found by the second-pass agent, which weakens rather than strengthens the
"pathological" framing:** the manuscript's own reported PROK SAE health diagnostics describe
a *healthy* fit, not a pathological one — `manuscript.txt:389-393`: reconstruction MSE 4.78
vs. a random baseline of 7,238, with **97.4% of dictionary features active**. Neither figure
is what a "correlations driven by pathological/single-active features" description would
predict (that would look like most features dead, or reconstruction dominated by one or two
features). `CLAIMS_LEDGER.md` has no SAE claim row at all, so this was never brought under
the ledger's evidence discipline in the first place — it lives only in manuscript prose.

**Provenance status: PARTIAL, with a genuine tension.** The clamp mechanism is real and
code-verified (a real methodological weakness worth disclosing). Whether it actually produced
the specific failure mode described (98% variance destroyed, pathological/single-active
features) is not measurable from anything currently stored, and the one health metric that
*is* stored (97.4% active, low reconstruction MSE) points away from that specific failure
mode. The specific 98% figure should not be quoted as measured.
**Ready for main text:** the clamp mechanism, yes, as a disclosed methodological limitation
(consistent with cutting the PROK SAE result per the instruction's item D and the existing
`CUT_LIST.md` demotion). The "98% variance destroyed" / "pathological features" language,
no — unmeasured and in tension with the one diagnostic that was actually recorded.

### PROK contamination / L8-r260 / group-wise quantization / destructive-regime test

Second-pass search (dedicated `Explore` agent, run in parallel with the rest of this audit)
confirms the preliminary reads above: no file names layer 8 or row 260 for PROK under any
search term, no eukaryotic-probe-on-PROK contamination mechanism is documented anywhere, no
group-wise (as opposed to per-row) quantization scheme exists in this repo, and the only
"destructive"-adjacent quantization result is the existing C-031 INT4 exemption null already
covered under item 8 (standard INT4, not a deliberately destructive regime — a more
aggressive INT2 script, `run_int4_multimodel_benchmark.py`, exists but has no results
artifact anywhere, so it was apparently never completed or its output was never saved).
**All three remain MISSING.**

---

## Cross-cutting notes

- **Items already correctly reflected in existing artifacts (no audit gap):**
  - **DNABERT-2 KL≈0.31 retirement (task item H).** Fully audited and already actioned in
    this repository: `CLAIMS_LEDGER.md` N-004/N-007/X-006, `DECISIONS.md` D-014,
    `results/dnabert2_kernel_guard.json`, `results/impulse_determinism_dnabert2.json`. Two
    identical Triton passes differed by KL=0.305; eager attention is deterministic (0.000%
    spread) with clean primary-dose KL=1.8e-8; Triton PPL 687.9 (±8.48%) vs. eager PPL 176.9
    (0.000% spread). Retired as X-006, "not corrected to another large KL value" — matches
    the new-state instruction exactly. **AUDITED.**
  - **U_k/Frobenius demotion, DNABERT-2 706/768 row (task item A, partial).** The specific
    fact "DNABERT-2 includes a row ranked 706/768" is already in the manuscript
    (`paper/main.tex:290`, ledger C-010) with its own resolution (source-vs-propagator via
    residual attribution). **AUDITED** for this specific fact. The claim that the gated-FFN
    amplifier construction is "substantially related to prior work by Sun et al." is *not*
    locally verifiable — `REFERENCE_AUDIT.md` already flags the Sun et al. bibliography entry
    as unresolved (wrong author initial, wrong title, wrong year vs. the one candidate paper
    the local Yu et al. copy cites) and recommends external lookup, not a guess. Carried
    forward unchanged into this audit — see Task 8 section below.
  - **Old pruning/compression narrative retirement (task item G).** X-001 ("shadow
    redundancy" / SW-neighbourhood pruning tolerance) is already retired in
    `CLAIMS_LEDGER.md` on the project's own contradicting data (near-SW −1.60pp vs. random
    −0.82pp at 20%). **AUDITED for the retirement itself.** However, the specific new-state
    claim that "the far-from-SW result was largely reproduced by a layer-depth control with
    no SW information" has **no matching artifact** — no script in the repo isolates layer
    depth from SW proximity as a confound for the far-SW fragility number (5.9× worse than
    near-SW at 20%, `scripts/analysis/analyze_compression_downstream.py`). That specific
    sub-claim is **MISSING**; only the general "retire shadow redundancy" instruction is
    already satisfied.
  - **Histone "not a clean dissociation" (task item E, partial).** The manuscript already
    states this correctly and conservatively: `paper/main.tex:860-861` — "The splice/histone
    contrast is one of reliability and magnitude rather than a clean dissociation." **AUDITED**
    for the qualitative framing. The specific "2/5 seeds" quantification in the new-state
    brief has no match — the only histone data in the repo is a 3-seed DNABERT-2 result
    (−4.5±6.2pp, p=0.41, n.s., `paper/main.tex:551-552`), not a 5-seed collapse count. That
    specific number is **MISSING**.

## Search methodology (for reproducibility of this audit)

Every claim above was searched at least twice: once directly by me (`grep -rn` across
`.py/.md/.json/.csv/.log/.txt`, `find -newermt` against the last commit timestamp, `git log
--all`, `git stash list`, `git branch -a`), and once independently by a dedicated read-only
search agent per cluster, briefed with the claim text but not with my conclusions, and
explicitly instructed not to run or infer from filename resemblance alone. Where the two
passes agreed, the claim is reported as such above. No script was executed and no model was
loaded at any point in this audit.
