# E8 RESULTS — encoder-vs-decoder exact operator dimensionality

**STATUS: FINAL. Mechanical branch: A.** Both independently selected text encoders
(MosaicBERT, ModernBERT) produced valid Phase-1 detection candidates and both shifted toward
the encoder side on the locked, pre-committed criterion. **MosaicBERT's shift is dramatic;
ModernBERT's is genuine but thin** (0.21% relative margin on `q1`) — read the full numbers
before treating "Branch A" as a clean, wide confirmation.

**Prereg:** `docs/prereg/PREREG_encoder_decoder_dimensionality.md`, sha256
`7a5dc793d35407a26be74aaeb79fb747e32669b51fd940f91035bd16857e4976`, locked
2026-08-14T15:14:09+00:00, verified before every forward pass.

## Deviation from the locked prereg text (mechanical, not scientific)

MosaicBERT ships no tokenizer files of its own (verified via the HF API file listing — only
`bert_layers.py`, `config.json`, `pytorch_model.bin`, etc., no `vocab.txt`/`tokenizer.json`).
Its own README states plainly: *"the tokenizer for this model is simply the Hugging Face
`bert-base-uncased` tokenizer."* The confirmatory script loads `BertTokenizer.from_pretrained
("bert-base-uncased")` instead of `AutoTokenizer.from_pretrained(repo)`. This changes zero
formulas, thresholds, or protocol steps — it supplies the tokenizer the model's own authors
specify, which `AutoTokenizer` cannot resolve automatically because the repo carries none.

## Measurements

### Detection (Phase 2)

| Model | Layer | Row | out_max | Ratio | Outcome |
|---|---|---|---:|---:|---|
| MosaicBERT | 9 | 287 | 253.3 | **288.65** | ACCEPTED |
| ModernBERT | 15 | 251 | 34,280 | **561.12** | ACCEPTED |

Both candidates cleared the 5.0x threshold by a wide margin — detection itself was not a close
call for either model.

### Spectral (Phase 3)

| Model | q1 | PR_spec |
|---|---:|---:|
| MosaicBERT | **0.476568** | **4.115865** |
| ModernBERT | **0.896989** | **1.233658** |

### Same-layer control rows (secondary, descriptive — not part of the decision)

| Model | Candidate q1 | Control q1 (5 seeded rows) | Candidate PR_spec | Control PR_spec |
|---|---:|---|---:|---|
| MosaicBERT | 0.4766 | 0.0473, 0.0351, 0.0393, 0.0348, 0.0445 | 4.12 | 139.7, 169.1, 154.8, 163.7, 146.5 |
| ModernBERT | 0.8970 | 0.0485, 0.0316, 0.0426, 0.0241, 0.0328 | 1.23 | 90.4, 116.3, 91.2, 120.9, 101.1 |

**Both candidates are dramatically more concentrated than their own layer's ordinary rows** —
control-row `q1` sits around 0.03–0.05 for both models (vs. candidates' 0.48 and 0.90), and
control-row `PR_spec` runs 90–170 (vs. candidates' 4.1 and 1.2). This is the opposite of what
"encoder rows are generally distributed, decoder rows generally concentrated" alone would
predict at the row level: **within these two encoders specifically, the detected high-gain
row is the striking exception (unusually concentrated), sitting atop a background of far more
distributed ordinary rows.** This is reported prominently because it complicates a simplistic
reading of the encoder/decoder story — the group-level Branch-A comparison below is
candidate-vs-candidate, not candidate-vs-typical-row, and the two questions have different
(here, opposite-direction) answers.

## Preregistered decision, applied mechanically

**Step 1 (completeness):** both models produced a valid candidate — no null. Proceed to Step 2.

**Step 2 (dimensionality shift), exact values, no rounding used in the actual comparison:**

```
decoder floor  (min NLP/genomic-decoder q1, GenomeOcean-4B)      = 0.8989077166811766
decoder ceiling (max NLP/genomic-decoder PR_spec, Phi-3)          = 1.224901693263825

MosaicBERT  q1 = 0.4765682244123624   < floor?  YES  (margin 0.4223, dramatic)
ModernBERT  q1 = 0.8969889514776379   < floor?  YES  (margin 0.0019, 0.21% relative -- thin)

MosaicBERT  PR_spec = 4.115864702588391   > ceiling?  YES (margin 2.891, dramatic)
ModernBERT  PR_spec = 1.233658471761313   > ceiling?  YES (margin 0.0088, 0.72% relative -- thin)
```

**Both models shift on both the primary (`q1`) and secondary (`PR_spec`) criteria.**

**Mechanical result: BRANCH A.** No threshold was adjusted after seeing these values — the
decoder floor/ceiling were fixed from the predeclared reference table before either model's
weight was loaded, exactly as locked.

## Why this is reported as "confirmed, not comfortably confirmed"

The rule that produced Branch A was deliberately lenient in one specific sense (no buffer
subtracted past the decoder floor — a model only had to fall outside the entire decoder range,
not approach NTv3's extreme). ModernBERT clears that lenient bar, but by a margin (0.21%
relative on `q1`) narrow enough that a differently-but-still-reasonably chosen anchor (e.g. a
small conservative buffer past the floor, which the prereg explicitly considered and rejected
in favor of the unadjusted floor) could plausibly have flipped this specific model to Branch B.
**The mechanical rule as actually locked says Branch A, and that is reported as the result —
but the closeness for ModernBERT specifically should temper how strongly this is read**, in
exactly the spirit of this project's standing practice of disclosing near-miss character even
on a nominal pass (matching E6's and E7's own disclosure discipline).

## The 2x2 table (written because, and only because, Branch A occurred)

All values `q1` (exact bilinear-operator top singular-energy share); higher = more rank-1.

| | **Decoder** | **Encoder** |
|---|---|---|
| **Text** | Mistral-7B 0.992 · Llama-7B 0.989 · OLMo-7B 0.965 · Qwen2.5-7B 0.953 · Phi-3 0.903 | **MosaicBERT 0.477 · ModernBERT 0.897 (new)** |
| **Genome** | GENERator EUK 0.969 · GenomeOcean-4B 0.899 | DNABERT-2 0.793 · NTv3 0.389 (discovery) |

No row-level statistics were computed on this table and none are claimed. The strongest
statement this project's discipline permits:

> Across the tested gated-FFN models, exact high-gain operator dimensionality tracks
> encoder/decoder organization more closely than sequence domain.

This is **not** claimed: *"causal masking creates distributed operators,"* *"all text encoders
are distributed,"* or any domain-general/architecture-causal statement.

## What this establishes

- Two independently selected, previously unmeasured text encoders (different codebases,
  different attention mechanisms, one sharing DNABERT-2's exact FFN class and one entirely
  independent) both produced a detectable high-gain candidate and both showed lower exact
  operator concentration than every decoder measured across this entire project, genomic or
  NLP.
- The encoder/decoder axis, not the text/genomic axis, is the one that separates cleanly in
  this specific 9-model comparison (7 decoders spanning `q1` 0.90–0.99; the two new encoders
  plus DNABERT-2/NTv3 spanning `q1` 0.39–0.90, all below every decoder).
- Within both new encoders, the detected high-gain candidate is far more concentrated than
  its own layer's ordinary rows — the "high-gain phenotype," where it exists in an encoder, is
  itself a locally exceptional, not a locally typical, row.

## What this is merely consistent with

- ModernBERT's own absolute `q1` (0.897) is closer to the decoder cluster's low end
  (Phi-3, 0.903) than to DNABERT-2 (0.793) or NTv3 (0.389) — consistent with a spectrum rather
  than a clean two-cluster split, even though it falls on the "encoder side" of this
  particular locked threshold.
- The candidate-vs-control asymmetry (candidates far more concentrated than their own layer)
  is consistent with — but does not establish — a picture where "high-gain phenotype present"
  and "encoder vs. decoder" are two partially independent axes.

## What this does NOT establish / is falsified

- No causal claim. No training-mechanism claim. No claim that attention masking causes the
  effect (explicitly disclaimed by the prereg and not tested here).
- Not falsified: unlike a Branch-C outcome, nothing here contradicts the encoder/decoder
  hypothesis. It is also not "cleanly" confirmed in the sense of a wide, unambiguous margin
  for both models — see the ModernBERT caveat above.

## Effect on claims

### C-002 / C-003

Unaffected — E8 does not measure row ranking.

### C-032

**Recommendation stands from the separate, decisive pre-E8 call**
(`experiments/E7_exact_dimensionality/C032_RETIREMENT_RECOMMENDATION.md`): RETIRE. E8 does not
change this either way — it was already decided independent of E8's outcome, and nothing in
E8 revisits it.

### C-034 (E7's proposed claim)

**Not restated as a headline mechanistic claim here either**, per the explicit instruction.
E8's finding is additive, descriptive evidence for the encoder/decoder pattern C-034 already
flagged as unconfirmed — it remains a proposal, not enacted, and any eventual ledger entry
should fold in E8's result rather than create a duplicate claim.

### New E8 claim (proposed only — not added to `CLAIMS_LEDGER.md`)

*Proposed wording, for author judgment:* "Exact bilinear-operator dimensionality of
prospectively detected gated-FFN high-gain rows tracks encoder/decoder organization more
closely than text-vs-genomic domain across the nine models tested (7 decoders, `q1` range
0.90–0.99; 4 encoders — 2 new, 2 discovery — `q1` range 0.39–0.90). The two newly confirmed
text encoders (MosaicBERT, ModernBERT) both cleared the pre-registered decoder-floor
criterion, though ModernBERT's margin was narrow (0.21% relative on `q1`)." This explicitly
inherits and narrows C-034's proposed wording rather than replacing it.

## Provenance

| | |
|---|---|
| Producing script | `experiments/E8_encoder_decoder/run_detection_and_spectral.py` |
| Shared library | `experiments/E7_exact_dimensionality/spectral_lib.py` (reused unmodified) |
| Raw outputs | `results/e8_detection_mosaicbert.json`, `results/e8_detection_modernbert.json` |
| Checkpoints | `mosaicml/mosaic-bert-base` (unpinned; tokenizer `bert-base-uncased`, also unpinned); `answerdotai/ModernBERT-base` (unpinned) |
| Dtype | float32 forward passes; float64 spectral computation |
| Device | CUDA, A100-PCIE-40GB |
| Forward passes | one per model (2 total) |
| Seed | control-row sampling only, `numpy.random.SeedSequence(42).spawn(2)`, fixed panel order (MosaicBERT=0, ModernBERT=1) |

## Mandatory stop / what was not done

No ablation, fine-tuning, pair perturbation, or LayerNorm test was run — the causal-
feasibility note (`CAUSAL_FOLLOWUP_FEASIBILITY.md`) is design-only and concludes the needed
intervention tooling does not yet exist in this repository. No connection is drawn to the
collaborator-reported DNABERT-2 "redundant pair" claim. No additional model was added after
seeing spectral values. Evo 2 was not retried.
