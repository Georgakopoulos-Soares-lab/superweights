# Gate 0 RESULTS — weight-only mechanism of NLP-vs-genomic FFN scalarization

**STATUS: FINAL. Gate 0 STOPPED at §0B (the exactness robustness check), exactly as the
locked prereg specifies. §0C (factor decomposition) and §0D (permutation test) were never
run on any real weight — not on any of the six primary-panel models, genomic or NLP. Gate 1
is NOT authorized. No replacement hypothesis is proposed in this pass.**

**Prereg:** `docs/prereg/PREREG_dimensionality_gate0.md`
**Prereg lock:** sha256 `7e5083157f344a240f57e632b49c2bc99f3e919b68e973be9c078f99e753215b`,
locked 2026-08-13T18:13:24+00:00, commit `2132e9b9564818d7fb73734639c31060544410a6` (working
tree dirty at lock time — see caveat below), verified before the confirmatory run.
**Producing script:** `experiments/E5_dimensionality/run_gate0.py`
**Shared library:** `experiments/E5_dimensionality/dimensionality_lib.py` (12/12 synthetic
tests green before lock — see `test_dimensionality_lib.py`)
**Raw output:** `results/e5_gate0.json`

## Panel

| # | Group | Model | Layer | Row |
|---|---|---|---|---|
| 0 | NLP | Llama-7B | 2 | 3968 |
| 1 | NLP | Mistral-7B | 1 | 2070 |
| 2 | NLP | OLMo-7B | 1 | 269 |
| 3 | Genomic | GENERator EUK 3B | 4 | 2371 |
| 4 | Genomic | DNABERT-2 117M | 5 | 603 |
| 5 | Genomic | NTv3 650M | 11 | 1472 |

Excluded per prereg: GENERator PROK (C-001 on hold), Evo1 (contested SW interpretation).

## Deviations from the locked prereg text (mechanical, not scientific)

- The prereg's Weight-Extraction section names `transformers.AutoModel.from_pretrained` for
  genomic models. In the installed `transformers` version, the bare `AutoModel` class does
  not expose the module paths the named `uk_frobenius.py` adapters themselves require
  (`adapter_dnabert2` needs `.bert.encoder...`, which only the `*ForMaskedLM`/`*ForSequence-
  Classification` wrapper provides; `adapter_llama_swiglu` needs `.model.layers`, which only
  the `*ForCausalLM` wrapper provides on this GENERator checkpoint). The confirmatory script
  loads via `AutoModelForCausalLM` (GENERator EUK) and `AutoModelForMaskedLM` (DNABERT-2,
  NTv3) instead — matching the load pattern already tested and used elsewhere in this
  repository (`scripts/analysis/run_sw_mechanistic.py`, `scripts/interpretability/
  run_sw_broadcast_impulse.py`). This changes zero formulas, zero panel members, zero layers,
  zero rows, and zero thresholds; it is a loader-class correction only, disclosed here rather
  than by re-locking the prereg.
- An import-path bug (`SALVAGE` resolved one directory too shallow) was caught and fixed
  before any weight was loaded; the first invocation failed at import time with no computation
  performed and is not a partial/discarded result.

## §0B — Exactness robustness check: MEASURED, all six models, GPU (A100-40GB), float64

Raw output: `results/e5_gate0.json`. `f_cross = (S_exact² − S_diag²) / S_exact²`, computed
via the elementwise-Gram identity (`K = (W_gate W_gateᵀ) ⊙ (W_up W_upᵀ)`,
`S_exact² = W_down[k,:] K W_down[k,:]ᵀ`), verified pre-lock against a from-scratch brute-force
materialization of `U_k` (relative error 2.5×10⁻¹⁶, float64 precision floor) and against a
pairwise-orthogonal synthetic construction giving `f_cross = 0` exactly.

| Group | Model | S_diag | S_exact | **f_cross** | exact rank (of row *k*) | Stop trigger |
|---|---|---:|---:|---:|---:|---|
| NLP | Llama-7B | 123.636 | 137.616 | **0.1929** | 1 / 4096 (100.000 pct) | clear |
| NLP | Mistral-7B | 0.37511 | 0.38835 | **0.0670** | 1 / 4096 (100.000 pct) | clear |
| NLP | OLMo-7B | 0.86786 | 0.91108 | **0.0926** | 1 / 4096 (100.000 pct) | clear |
| Genomic | GENERator EUK | 215.860 | 522.132 | **0.8291** | 1 / 3072 (100.000 pct) | **FIRED** |
| Genomic | DNABERT-2 | 51.941 | 74.009 | **0.5074** | 1 / 768 (100.000 pct) | **FIRED** |
| Genomic | NTv3 | 393.094 | 441.517 | **0.2073** | 1 / 1536 (100.000 pct) | **FIRED** |

**Stop-the-study trigger** (`|f_cross| > 0.20` OR exact-rank outside the layer's top 1% —
both defined in the prereg before any real weight was loaded): fired for **all three genomic
models** and none of the three NLP models. The mechanical rule in
`PREREG_dimensionality_gate0.md` §0B is unambiguous: *"If cross terms radically alter the
target-row interpretation in the primary panel, STOP the entire dimensionality study and
report that the foundation is invalid."* That condition is met. §0C and §0D were not run on
any model — the script enforces this in code (Phase 1 / Phase 2 split), and the log confirms
it (`Gate 0 ABORTED at the 0B stop trigger`).

## What is, and is not, established by this measurement

**Established, mechanically, with controls:**
- The row-level finding is untouched and in fact reinforced: **all six primary-panel rows
  are still rank 1 of their layer under the exact (cross-term-complete) quadratic form**, not
  just under the diagonal approximation. Row-level recovery does not depend on which form is
  used, in this panel.
- **The diagonal approximation's fidelity itself splits cleanly by group in this panel.** For
  the three NLP rows, the diagonal captures 81–93% of the exact squared row norm
  (`f_cross` 0.067–0.193). For the three genomic rows, it captures as little as **17%**
  (GENERator EUK, `f_cross` = 0.829) up to 79% (NTv3, `f_cross` = 0.207). Every genomic model
  in the primary panel exceeds the preregistered 0.20 threshold; no NLP model does.

**Not established — explicitly out of scope for what was run:**
- *Why* the cross terms are larger in these three genomic rows. Gate 0 is a measurement, not
  a mechanism claim about cause.
- Whether this generalizes beyond these six specific (model, layer, row) coordinates.
- Anything about `D_i`/`A_i` alignment, realized activations, or causal dimensionality — §0C,
  §0D, and Gates 1–2 were never run.

## Why the study stops here rather than substituting the exact form and continuing

It would be tempting to simply re-run §0C/§0D on the exact quadratic form's own
implicit per-coordinate decomposition instead of the diagonal `C_i = D_i · A_i`. The locked
prereg does not permit this: no metric may be added and no formula may be substituted after
real weights are inspected (see "No post-hoc changes"), and the exact form does not have a
natural per-coordinate decomposition in the first place — `S_exact²` is a full quadratic form
over `K`, and attributing it to individual coordinates requires an off-diagonal allocation
rule that was not specified, tested, or locked before this run. Inventing one now, after
seeing that the diagonal form fails specifically on the genomic side, is exactly the kind of
post-hoc rescue the governing instruction and this repository's own standing discipline
(`CLAUDE.md`, the N-004/N-007 pattern) both forbid. The correct response to a failed
foundation is to stop and report it, not to patch the foundation until the original plan
becomes runnable again.

## Interpretive note for the author (flagged, not adjudicated, not written into the ledger)

This measurement has a direct implication for existing, already-`established`/`supported`
ledger claims that this pass does **not** resolve or act on, per instruction not to modify
manuscript prose from a single preliminary experiment:

- **C-002, C-003, C-032** (and the granularity comparison in
  `experiments/E4_granularity/CANONICAL_TABLE.md`) report participation ratio and top-1 share
  computed from the same diagonal `c_{k,i}` this experiment calls `C_i`. This experiment shows
  that quantity is a substantially worse approximation of the true bilinear amplifier
  structure for GENERator EUK, DNABERT-2, and NTv3 (17–79% of the exact squared norm captured)
  than for the three NLP models (81–93% captured). The existing "genomic rows are more
  distributed than NLP rows" granularity comparison may therefore be partly comparing a
  reliable statistic (NLP) against an unreliable one (genomic), rather than measuring a clean
  architectural difference. This is disclosed here as a **measurement-validity flag for the
  existing claims**, not as a retraction, a correction, or a new claim — no artifact in this
  pass re-derives C-002/C-003/C-032 under the exact form, and none of those claims is edited.
  A future session that wants to pursue this would need to define and preregister a
  cross-term-aware granularity metric before computing one.

## Decision, applied mechanically

Per the locked prereg: **Gate 0 does not pass or fail the primary alignment criterion — it
never reaches that decision.** The 0B stop trigger, itself part of the same locked
preregistration, fires first and takes precedence. Gate 1 is **not authorized**. No
replacement hypothesis is generated in this pass, per the governing instruction.

## Provenance

| | |
|---|---|
| Producing script | `experiments/E5_dimensionality/run_gate0.py` |
| Shared library | `experiments/E5_dimensionality/dimensionality_lib.py` |
| Pre-lock validation | `experiments/E5_dimensionality/test_dimensionality_lib.py` — 12/12 green |
| Raw output | `results/e5_gate0.json` |
| Checkpoints | `huggyllama/llama-7b` (no pinned revision — same gap noted in the earlier feasibility audit); `mistralai/Mistral-7B-v0.1` (no pinned revision); `allenai/OLMo-7B-0724-hf` (no pinned revision); `GenerTeam/GENERator-v2-eukaryote-3b-base` @ `7dc01bccce5b65e15141170538afdc2ff09d8dde`; `zhihan1996/DNABERT-2-117M` @ `7bce263b15377fc15361f52cfab88f8b586abda0`; `InstaDeepAI/NTv3_650M_pre` code rev. `0ecff3637f0d3ba5b686d1095083218157c2ca34` |
| Dtype | float64 throughout (weights promoted on load) |
| Device | CUDA, A100-PCIE-40GB (idle, confirmed available) |
| Environment | conda `grlm`; `LD_LIBRARY_PATH=$ENV/lib` required (PIL/GLIBCXX, same as E1's note); `HF_HUB_OFFLINE=1`, `SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE` unset at the boundary |
| Seed | n/a for §0B (deterministic weight algebra); §0D's `master_seed=42` scheme is specified but unused (never reached) |
| Forward passes | zero |
| GPU jobs beyond weight loads | none |
