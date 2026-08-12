# PREREG — E1, prospective cold-weight prediction on an unseen NLP model

**Lock before running the forward-pass sweep:**
`python src/prereg_lock.py lock docs/prereg/PREREG_nlp_prospective.md`

---

## Question

Can ‖U_k‖_F and the c_{k,i} decomposition, computed from weights alone with zero forward
passes, predict the super-weight coordinate of a model whose super-weight has not been
published and which we have not inspected?

## Why NLP and not a genomic model

Every ablatable genomic model in the panel is already unblinded (GENERator EUK/PROK,
DNABERT-2, Evo1, HybridNA, MegaDNA). The only untouched one is Caduceus, which is blocked
by a CUDA-extension incompatibility — choosing it would trade a clean prospective test for
infrastructure debugging. See D-006.

## Design

**Retrospective arm (not prereg'd — published ground truth exists):**
Llama-7B, Mistral-7B, OLMo-7B against Yu et al. (arXiv:2411.07191). Level 1 = row rank,
Level 2 = scalar rank within row.

**Prospective arm (this preregistration):**

1. Select a model Yu et al. did not cover and that we have not inspected. Candidate: ______
   (Qwen3 / Gemma / Phi — whichever loads trivially). Record the exact HF revision.
2. Compute ‖U_k‖_F per layer from cold weights. No forward pass. No detection sweep.
3. Record below, before running anything else:

   - Predicted layer: ______
   - Predicted output row *k*: ______
   - Predicted scalar index *i*: ______
   - Predicted top-1 share within row: ______
   - Predicted participation ratio: ______

4. **Lock this file.**
5. Only then run the standard forward-pass detection sweep (max |input| / max |output| at
   down_proj) and the single-parameter ablation.

## Success criteria (state before unlocking)

- **Strong:** predicted `(layer, k, i)` matches the detection sweep exactly.
- **Partial:** layer and row *k* match; scalar *i* does not.
- **Failure:** row does not appear in the top 10 of the detected layer.

Report whichever occurs. A partial result is still a publishable and honest outcome — it
would mean the predictor is a row-level instrument, which is exactly what the wording rule
in CLAUDE.md already anticipates.

## Pre-committed reporting

**On success:** report as a prospective, timestamped prediction with the lock hash quoted
in Methods. Do not soften it into "consistent with."

**On failure:** report in R1 as a negative prospective test, with the locked prediction
shown. This is cheap honesty that materially strengthens every other claim in the paper —
a predictor that was allowed to fail and didn't, elsewhere, is worth more than one that
was never exposed.

---

_Locked: (filled by prereg_lock.py)_
