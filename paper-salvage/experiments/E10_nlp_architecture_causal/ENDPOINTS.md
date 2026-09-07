# E10 Phase 3 — intrinsic endpoint definitions

Defines the model-intrinsic endpoints for both arms, before any causal measurement, per
`e10_prompt.md` Phase 3. Written to be as close as possible to reusing existing E7/E8/E9
machinery rather than inventing new evaluation code — reuse is cited, gaps are disclosed.

## Decoder primary endpoint (ARM A)

**No pre-existing decoder-side causal/forward-pass endpoint exists in this repository**
(`MODEL_AND_BASIS_AUDIT.md` gap #6) — E1/E5/E7's decoder work was entirely weight-only. This
section therefore defines a new, but not novel-in-kind, endpoint: causal-LM loss/NLL over a
frozen WikiText-2 window set, following E9's existing `build_fixed_batches` /
`mlm_loss_per_batch` pattern (`tomography_lib.py`) generalized from masked to causal LM loss,
and reusing E7/E8's already-established WikiText-2-raw-v1 corpus choice
(`run_phase1_detection.py::build_wikitext_input`) rather than choosing a new corpus.

**Frozen parameters:**

| | |
|---|---|
| Corpus | `wikitext`, `wikitext-2-raw-v1`, `test` split (same dataset/split already used by E7/E8's detection forward passes) |
| Window construction | `N=100` windows, each built by concatenating consecutive non-empty lines from the test split (shuffled with a fixed RNG seed) until the tokenizer reaches `max_length` tokens, mirroring `build_wikitext_input`'s line-concatenation approach but drawing 100 independent windows instead of 1 |
| Context length | 512 tokens (matches the 512-token convention already used for all 3 forward-pass detections in E7/E8 — Qwen2.5, MosaicBERT, ModernBERT) |
| Tokenization | each decoder's own tokenizer (`AutoTokenizer.from_pretrained`, `use_fast=True` default), truncation at 512 tokens, no padding needed if windows are batched per-model with equal length via truncation |
| Batch size | 8 windows/batch (25/50/... batches of ≤8; chosen only for memory headroom on a single A100, not a measurement parameter) |
| dtype | `float32` (matches E7/E8's `torch_dtype=torch.float32` convention; no bf16/fp16 mixed-precision noise in the loss comparison) |
| Deterministic settings | `torch.manual_seed(42)`; window-shuffle RNG seed `42`; `model.eval()`, `torch.no_grad()` |
| Model revision | pinned per `DECODER_INTERVENTION_FREEZE.md` where available (Phi-3, Qwen2.5); Llama/Mistral/OLMo remain on the unpinned `main` resolution already used throughout E1/E5/E7 (audit gap #1) — ARM A will resolve and record the exact commit actually fetched at run time, but does not block on retroactively pinning historical E1/E7 results |

**Primary scalar:** mean per-token NLL over all 100 windows (equivalently, causal-LM loss as
already computed by `AutoModelForCausalLM(..., labels=input_ids)`).

**Secondary:** perplexity (`exp(mean NLL)`), logit KL vs. baseline (mean over positions of
`KL(softmax(logits_baseline) || softmax(logits_intervened))`), next-token entropy (mean
Shannon entropy of the next-token distribution).

**Not built:** any hand-selected semantic steering task. This endpoint is intentionally the
model's own training objective, not a downstream task, per the prompt's explicit instruction.

## Encoder primary endpoint (ARM B)

Reuses E9's exact, already-generic masking machinery unmodified
(`E9_mechanistic_tomography/tomography_lib.py::build_fixed_batches`,
`mlm_loss_per_batch`) — both are tokenizer-agnostic and already used for DNABERT-2's MLM loss;
only the *source* of `seqs` changes (WikiText-2 text windows instead of hg38 FASTA windows),
which is exactly the kind of domain substitution `build_fixed_batches` was written to not
care about.

**Frozen parameters (mirrors E9's DNABERT-2 settings, `PROVENANCE_AND_BASELINES.md` /
`run_dnabert2_measurements.py`, substituting text windows for genomic windows):**

| | |
|---|---|
| Corpus | `wikitext`, `wikitext-2-raw-v1`, `test` split — same corpus as E7/E8's detection and ARM A above |
| Sequences | `n_windows=256` (same as E9's DNABERT-2 setting), each a concatenation of shuffled test-split lines truncated to `max_len` tokens |
| Masking positions / seed | `mask_prob=0.15`, mask-generator seed `SEED=42` (`tomography_lib.SEED`, identical to E9) — one fixed mask realization reused across every condition so deltas are paired, exactly as E9's docstring states |
| Tokenization | each encoder's own tokenizer (`bert-base-uncased` tokenizer for MosaicBERT per `E8_encoder_decoder/RESULTS.md`; native tokenizer for ModernBERT) |
| Max length | 512 tokens (same as E7/E8 detection) |
| Batch size | 16 (memory-driven, not a measurement parameter; encoders are far smaller than the 7B decoders) |
| dtype | `float32` |
| Model revision | pinned to the resolved commits already recovered in `MODEL_AND_BASIS_AUDIT.md` §6-§7 (`c89bbadc...` MosaicBERT, `8949b909...` ModernBERT) — Open item #2 from the audit (confirm these are still `main` HEAD) must be checked at run time before treating them as a stable pin |

**Primary scalar:** mean MLM loss over masked positions across the 256 windows (weighted mean
via `mlm_loss_per_batch`'s `(sum_loss, n_masked)` pairs, exactly as E9 already does).

**Secondary:** logit KL at masked positions vs. baseline; residual norm at the intervention
layer (only if easy and predeclared per-model in the Phase 4 prereg — not committed here).

**Not built:** any downstream classification task, per the prompt's explicit instruction.

## Shared discipline across both arms

- One fixed mask/window realization per model, reused across every intervention condition
  (baseline, each target row, each control row) — this is what makes deltas paired rather than
  independently noisy, identical to E9's stated rationale.
- No endpoint, corpus, or window count changes after any response is observed (Phase 4 lock).
- Windows and masks are generated and frozen as raw artifacts (not regenerated per condition)
  before Phase 4 lock, so the same exact batches are replayable.
