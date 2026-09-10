# E7 RESULTS — exact bilinear-operator dimensionality

**STATUS: FINAL.** Phase 5 (new-model confirmatory panel) reached **EXPLORATORY**, not a
formal Branch A/B/C/D — Evo 2 7B returned a Phase-1 detection **null** (no spike cleared the
preregistered threshold anywhere in the model), degrading the genomic group to one surviving
model, which the locked prereg explicitly disqualifies from Branch A. Phase 6 (legacy
reanalysis, run only after Phase 5 froze, exactly as ordered) then shows why forcing a branch
would have been the wrong move regardless: **there is no clean NLP-vs-genomic separation
under the exact metric across either panel.** Full reasoning below.

**Prereg:** `docs/prereg/PREREG_exact_operator_dimensionality.md`, sha256
`7da30f819312537e4b17edd9be3119eb199e85263e1b08e181a3b147a9b3a83b`, locked
2026-08-14T13:44:26+00:00, verified before every confirmatory step.

## Measurements

### New confirmatory panel (Phase 5)

| Model | Group | Candidate | Selection | Spectral status | q1 | PR_spec |
|---|---|---|---|---|---:|---:|
| Phi-3-mini-4k-instruct | NLP | 6 published rows (L2 x3, L4 x3) | ELIGIBLE-PUBLISHED | measured, aggregated | **0.9028** (median) | **1.2249** (median) |
| Qwen2.5-7B | NLP | L26/r458 | ELIGIBLE-PROSPECTIVE, ratio 639.2 | measured | **0.9529** | **1.0995** |
| GenomeOcean-4B | Genomic | L1/r2604 | ELIGIBLE-PROSPECTIVE, ratio 1076.0 | measured | **0.8989** | **1.2243** |
| Evo 2 7B | Genomic | L29/r1084 (rejected) | ELIGIBLE-PROSPECTIVE, ratio 2.22 | **NULL — below 5.0x threshold** | — | — |

Phi-3's six individual rows (not pseudoreplicated into the group comparison — see Model-Level
Aggregation below): L2/r525 q1=0.9529, L2/r1693 q1=0.9137, L2/r1113 q1=0.7505, L4/r525
q1=0.9278, L4/r1113 q1=0.5584, L4/r1693 q1=0.8919.

### Legacy discovery panel (Phase 6 — run only after Phase 5's decision was frozen)

| Model | Group | Row | q1 | PR_spec | published `f_cross` (E5/E6) | published diagonal PR (E5/E6) |
|---|---|---|---:|---:|---:|---:|
| Mistral-7B | NLP | L1/r2070 | 0.9922 | 1.0158 | 0.0670 | 1.0235 |
| Llama-7B | NLP | L2/r3968 | 0.9888 | 1.0227 | 0.1929 | 1.2443 |
| **GENERator EUK** | **Genomic** | L4/r2371 | **0.9689** | **1.0653** | 0.8291 | 4.5599 |
| OLMo-7B | NLP | L1/r269 | 0.9646 | 1.0747 | 0.0926 | 1.0944 |
| DNABERT-2 | Genomic | L5/r603 | 0.7933 | 1.5033 | 0.5074 | 3.6422 |
| NTv3 | Genomic | L11/r1472 | 0.3889 | 6.4803 | 0.2073 | 22.7194 |

### Combined ranked view (all 9 measured models, `q1` descending; Evo 2's null excluded)

```
1. Mistral-7B         (NLP, legacy)  q1=0.9922  |||||||||||||||||||||||||||||||||||||||||
2. Llama-7B           (NLP, legacy)  q1=0.9888  |||||||||||||||||||||||||||||||||||||||||
3. GENERator EUK      (GEN, legacy)  q1=0.9689  ||||||||||||||||||||||||||||||||||||||
4. OLMo-7B            (NLP, legacy)  q1=0.9646  ||||||||||||||||||||||||||||||||||||||
5. Qwen2.5-7B         (NLP, new)     q1=0.9529  |||||||||||||||||||||||||||||||||||||
6. Phi-3 (median)     (NLP, new)     q1=0.9028  ||||||||||||||||||||||||||||||||||
7. GenomeOcean-4B     (GEN, new)     q1=0.8989  ||||||||||||||||||||||||||||||||||
8. DNABERT-2          (GEN, legacy)  q1=0.7933  ||||||||||||||||||||||||||||||
9. NTv3                (GEN, legacy)  q1=0.3889  ||||||||||||||||
```

**Genomic models occupy the 3rd, 7th, 8th, and 9th ranks out of 9 — interleaved with NLP
models throughout the upper half of the range, not clustered at the bottom.** GENERator EUK
(a genomic SwiGLU decoder) is more rank-1 than one of the four NLP decoders (OLMo-7B).
GenomeOcean-4B (a genomic Mistral-architecture decoder) sits essentially on top of Phi-3.

## Preregistered decision, applied mechanically

**Step 1 (completeness):** NLP surviving = 2 (Phi-3, Qwen2.5-7B). Genomic surviving = 1
(GenomeOcean-4B; Evo 2 null). Per the locked rule, one group having exactly one surviving
candidate **degrades the entire confirmatory panel to an exploratory extension** — Branch A
is explicitly disqualified, and Steps 2/3 are not evaluated as a binding decision.

**Reported for context, not as a decision:** even treating the single surviving genomic model
at face value, `GenomeOcean-4B q1 (0.8989) < min(NLP q1) (0.9028)` — technically the "right"
direction — but this is not a group comparison with only one genomic data point, and the
prereg's own discipline forbids treating it as one.

**Mechanical Phase-5 result: EXPLORATORY, not A/B/C/D.**

## Why forcing a branch would have been wrong regardless — Phase 6 evidence

The legacy panel was reanalyzed only after the above was frozen, per the binding ordering
rule. It confirms that a clean branch call would have been the wrong instinct even with a
complete confirmatory panel:

- **GENERator EUK's exact `q1` (0.9689) exceeds OLMo-7B's (0.9646).** A genomic model
  is *more* rank-1 than an NLP model in this panel.
- **NTv3 is a genuine, large outlier (`q1`=0.389, `PR_spec`=6.48) — clearly the most
  distributed exact operator of all nine models measured** — but it is also the *only* model,
  new or legacy, that is unambiguously separated from the near-rank-1 cluster. DNABERT-2 sits
  at an intermediate value (`q1`=0.793), closer to the NLP/GENERator-EUK/GenomeOcean cluster
  than to NTv3.
- Both of the two models that remain clearly, substantially distributed under the exact
  metric (NTv3, DNABERT-2) are **bidirectional encoders**. Every near-rank-1 model, NLP or
  genomic, is a **causal decoder** (Llama, Mistral, OLMo, Phi-3, Qwen2.5, GENERator EUK,
  GenomeOcean-4B). **This is a descriptive, post-hoc pattern noticed while writing this
  document — not a preregistered test, not a new branch outcome, and not asserted as a
  finding.** Per this project's own discipline, a pattern noticed after seeing the data is
  discovery evidence for a *different* hypothesis (operator dimensionality tracks
  encoder/decoder architecture, not NLP/genomic domain), and would need its own, separately
  preregistered confirmatory test before being written as a claim anywhere. It is recorded
  here only so a future session does not have to rediscover it from raw numbers.

## Evo 2's null result, on its own terms

No layer in Evo 2 7B produced a down-projection output spike exceeding 5x its own layer's
median channel-max, anywhere across all 32 blocks, under the same ACTB-504 probe that produced
clean, extreme spikes (ratio 12x–1076x) in every other model this project has ever run this
detector on. This is reported as a genuine negative finding about Evo 2 under this specific
protocol and probe — not a failure of the protocol, and not a reason to search a second probe
or layer (which the prereg forbids). It does not, by itself, mean Evo 2 lacks a super-weight-
like phenomenon; it means this specific frozen detector, on this specific native-domain input,
does not find one. A different native input distribution could in principle behave
differently — untested here, and not tested by loosening this run's protocol.

## What this establishes

- Under the exact operator, the qualitative NLP-vs-genomic split reported by the old diagonal
  approximation **does not hold as a clean domain-level pattern** — not in the new
  confirmatory data (GenomeOcean-4B is indistinguishable from Phi-3) and not in the legacy
  panel (GENERator EUK is indistinguishable from — and by one comparison, more concentrated
  than — an NLP model).
- One genomic model (NTv3) remains genuinely, substantially more distributed under the exact
  metric than every other model measured, NLP or genomic — the effect is real for at least
  this one case, just not general.
- Phi-3's and Qwen2.5's model-level exact-operator concentration is high (median `q1` 0.90 and
  0.95) but visibly below Llama-7B/Mistral-7B's near-0.99 — the "near rank-1" NLP regime is
  itself not perfectly uniform across NLP models either.

## What this is merely consistent with

- The encoder/decoder pattern described above — plausible, visually clean in this small
  sample, entirely unconfirmed.
- The possibility that a different, better-powered genomic decoder panel (a working Evo 2
  measurement, a second Gene42-class model if one becomes available) could still show a
  cleaner separation than this one did — this session's null and single-survivor outcome
  limits what could be shown, it does not positively rule out a real effect existing
  elsewhere.

## What this does NOT establish

- That exact operator dimensionality is (or is not) a general NLP-vs-genomic distinction —
  the confirmatory panel could not test this to completion.
- Anything about causal dimensionality — out of scope for E7 throughout, untouched here.
- Anything about why Evo 2 produced no detectable spike under this protocol.
- The encoder/decoder pattern as a finding — explicitly flagged as unconfirmed above.

## Recommendation for existing claims (not enacted — no `CLAIMS_LEDGER.md` row is edited by
this document)

### C-002 / C-003 (row-ranking claims)

**Recommend: RETAIN, unchanged from E6's recommendation.** E7 does not bear on row-ranking
directly (it measures the operator's spectrum, not its rank position among other rows), and
nothing here contradicts E6's finding that exact-form row ranking survives on 12 of 13
independently-confirmed rows.

### C-032 (diagonal-PR granularity/dimensionality claim)

**Recommend: RETIRE**, in its current form (`PR 3.64 (DNABERT-2) / 4.56 (EUK) / 22.7 (NTv3) /
122 (Evo1) ... vs 1.02–1.24 for published NLP SWs`, presented as a general NLP-vs-genomic
granularity contrast). The exact spectral metric does not reproduce this as a domain-general
qualitative distinction — GENERator EUK's exact `PR_spec` (1.07) is squarely inside the
NLP range (1.02–1.24), not the "genomic" range the old claim implies. Every genomic model's
diagonal PR overstates its exact `PR_spec` by a large, consistent factor (EUK 4.3x, DNABERT-2
2.4x, NTv3 3.5x) — the direction of the bias is the same in all three cases, which is itself
informative (the diagonal approximation systematically overstates apparent dimensionality
specifically on the genomic side, matching E5's own `f_cross` finding) but means the old
number cannot be cited as a genomic-vs-NLP granularity measurement going forward.

### New claim — C-034 (proposed; not added to `CLAIMS_LEDGER.md` by this document)

*Proposed wording, for the author's judgment, not inserted into the ledger by this pass:*
"Exact bilinear-operator dimensionality (`q1`, `PR_spec`) of gated-FFN high-gain rows does not
separate cleanly by NLP-vs-genomic domain across nine models tested (6 legacy, 3 new): one
genomic model (NTv3) is substantially more distributed than every other model measured, one
(DNABERT-2) is intermediate, and two (GENERator EUK, GenomeOcean-4B) are statistically
indistinguishable from the NLP cluster. A visually clean but **unconfirmed, purely
descriptive** encoder-vs-decoder split coincides with the concentrated/distributed split in
this specific sample." This explicitly **supersedes** C-032's specific numeric claim (per the
distinction the governing instruction draws between RETIRE and SUPERSEDE) rather than sitting
alongside it — C-032's own diagonal PR numbers are preserved historically in the ledger, not
mutated.

## Provenance

| | |
|---|---|
| Producing scripts | `run_phi3_spectral.py`, `run_phase1_detection.py`, `run_phase1_detection_evo2.py`, `run_confirmatory_spectral_lite.py`, `run_phase5_decision.py`, `run_legacy_reanalysis.py` |
| Shared library | `spectral_lib.py` (6/6 synthetic tests green pre-lock) |
| Raw outputs | `results/experiments/E7/e7_phi3_spectral.json`, `results/e7_phase1_detection_{qwen25,genomeocean,evo2}.json`, `results/experiments/E7/e7_phase5_decision.json`, `results/experiments/E7/e7_legacy_reanalysis.json` |
| Checkpoints | `microsoft/Phi-3-mini-4k-instruct` (unpinned); `Qwen/Qwen2.5-7B` (unpinned); `DOEJGI/GenomeOcean-4B` (unpinned); `arcinstitute/evo2_7b` (`evo2_7b.pt`, resolved via the `evo2` package's own checkpoint loader, no separate hash captured); legacy panel checkpoints as pinned in E5/E6 |
| Dtype | float32 for all forward passes and weight-only extractions; float64 for every spectral/SVD computation |
| Device | CUDA, A100-PCIE-40GB; Evo 2 required `evo2.sif` with `--cleanenv --env PYTHONNOUSERSITE=1` (container's own `transformer_engine`, not the host `~/.local` one — see `MODEL_PANEL.md`) and the same FP8-autocast no-op monkeypatch already established in `models/evo2_wrapper.py` (A100 is compute capability 8.0; FP8 needs 8.9+) |
| Forward passes | one per prospective-detection model (Qwen2.5-7B, GenomeOcean-4B, Evo 2 7B); zero for Phi-3 and the legacy panel (weight-only) |
| Deviations disclosed | ACTB-504 probe was mistranscribed (truncated to 384bp) in the first draft of `run_phase1_detection.py`/`run_phase1_detection_evo2.py`; caught before any forward pass ran, fixed by deriving the slice programmatically from the full, already-published 1128bp CDS (byte-identical to `probes/dna_probes.py`'s own `actb_500`) |

## Mandatory stop

No causal follow-up feasibility document is written — conditional on Branch A only, which did
not occur. No replacement hypothesis is invented for the encoder/decoder pattern. No new model
is added to compensate for Evo 2's null (the model-expansion stop rule and "no post-hoc row/
model exclusions or additions" both forbid it). The result is preserved exactly as measured.
