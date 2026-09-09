# PROVENANCE_AND_BASELINES.md — E9 mechanistic tomography

Phase 0 of E9 (`next_prompt.md`, 2026-08-22). Pure provenance/audit — no new intervention
semantics are decided here (that is `INTERVENTION_BASIS.md`); no held-out response is
collected here. This document exists so that everything E9 later touches has a named,
checkable origin.

---

## GENERator EUK

| Field | Value | Source |
|---|---|---|
| `model_id` | `GenerTeam/GENERator-v2-eukaryote-3b-base` | `configs/generator.yaml:1` |
| Pinned revision | **none pinned** — loads whatever `main` resolves to at run time | `configs/generator.yaml`, `models/generator_wrapper.py:8-22` |
| Architecture | `AutoModelForCausalLM`, `trust_remote_code=True`, LlamaForCausalLM-shaped MLP, 30 layers | `configs/generator.yaml:1-10`, confirmed `num_hidden_layers=30` |
| dtype / device | `float32` / `cuda`, `device_map="auto"` | `configs/generator.yaml:9-10`, `models/generator_wrapper.py:16-21` |
| `down_proj_pattern` | `model.layers.{i}.mlp.down_proj` | `configs/generator.yaml:6-7` |

**High-gain row identity** — `results/super_weight_index.json["generator"]["results"]`:

| rank | layer | row | col | out_max |
|---|---|---|---|---|
| 1 | 4 | **2371** | 2536 | 375,361.31 |
| 2 | 4 | **1522** | 2536 | 143,880.20 |

Only these **two** candidates exist for GENERator EUK anywhere in this repository
(`scripts/detection/`, `results/`, `experiments/results/` all searched). Compare:
DNABERT-2 has 10 entries in the same index, Evo1 has 10. This fact directly drives the
Phase 1 basis-scope decision (see `INTERVENTION_BASIS.md`).

**Existing intervention scripts**

- `scripts/mechanism/run_sw_steering.py` (T2.2, primary source of claim C-040) — designated
  **canonical** for E9.
- `scripts/mechanism/run_steering_biological.py` (E3-labeled quality follow-up) — used for
  E9's quality-frontier secondary endpoints only, not for the primary GC endpoint.

**Exact intervention** (`run_sw_steering.py:68-101`): resolve
`mod = model.layers.4.mlp.down_proj`, clone `mod.weight.data[row, :]`, multiply the row
in place by a scalar `scale`, generate, restore from the clone in a `finally` block. This is
the down-projection **write itself** — after the SwiGLU activation, before the residual
stream add, before the next RMSNorm. It is exactly the `alpha_i` parameterization E9 needs:
`scale=1.0` = untouched, `scale=0.0` = full ablation.

**Scale grids already tested** (two scripts, not identical):
- `run_sw_steering.py`: `{0.0, 0.5, 1.0, 2.0, 5.0}` — **contains both of E9's frozen
  epsilon points** (`scale=0.5` ↔ `epsilon=0.5`; `scale=0.0` ↔ `epsilon=1.0`) and the
  natural baseline (`scale=1.0`).
- `run_steering_biological.py`: `{0.0, 0.25, 0.5, 1.0, 2.0}` — also contains all three.

**Endpoint / generation config** (`run_sw_steering.py:45-117`): primary = mean GC fraction
of generated continuations (`gc_frac`, from `run_ensemble_encoding.py`); secondary =
dinucleotide frequencies. `n_prompts=24`, `prompt_bp=120`, `max_new_tokens=64`,
`do_sample=True, top_k=50, temperature=1.0`, `torch.manual_seed(SEED+i)` per prompt,
`SEED=42`. Prompts are real hg38 windows drawn via
`data/regions/hg38/random_262kb.bed` against a `hg38.fa` FASTA
(**path fix**, see "Environment fixes applied" below).
`run_steering_biological.py` additionally reports PPL vs. the unmodified model, dinuc-KL vs.
hg38 background, longest homopolymer run, CpG O/E, with `n_prompts=32, max_new=96`.

**Random/matched controls**: `run_sw_steering.py` samples 5 non-SW rows from the same
down_proj tensor, `random.Random(42).sample(...)`, same scale grid;
`run_steering_biological.py` uses 3. E9 reuses the 5-row / seed-42 control set from
`run_sw_steering.py` as canonical.

**No raw output artifact currently exists** for either script
(`results/mechanism/sw_steering_generation.json` / `steering_biological.json` are both
absent) — consistent with `experiments/docs/MISSING_COLLEAGUE_ARTIFACTS.md`. C-040's
numbers are adopted as fact per `DECISIONS.md` D-024, but nothing here has been *run in this
repository* prior to E9.

---

## DNABERT-2

| Field | Value | Source |
|---|---|---|
| `model_id` | `zhihan1996/DNABERT-2-117M` | `configs/dnabert2.yaml:1` |
| Pinned revision | `7bce263b15377fc15361f52cfab88f8b586abda0` | `models/dnabert2_wrapper.py:6` |
| **Gap found**: `run_pretrained_epistasis.py` and the `run_gue_multiseed._load_model` path do **not** pass this revision by default (`--code_revision` defaults to `None`) — they would silently pull whatever HF `main` resolves to. **Fixed for E9**: every E9 script pins `--code_revision 7bce263b15377fc15361f52cfab88f8b586abda0` explicitly. | `scripts/mechanism/run_pretrained_epistasis.py:78-99,207`, `scripts/evaluation/run_gue_multiseed.py:87-126` |
| Architecture | `AutoModelForMaskedLM` (pretrained MLM path, no task head), 12 layers | `configs/dnabert2.yaml:5-9` |
| dtype / device | `float32` / `cuda` | `configs/dnabert2.yaml:8-9` |
| `down_proj_pattern` | `bert.encoder.layer.{i}.mlp.wo` (in=3072/out=768) | `configs/dnabert2.yaml:5-7` |
| Known compatibility fix | DNABERT-2's Triton flash-attn kernel is incompatible with the installed Triton (`dot() got an unexpected keyword argument trans_b`); MLM path (zero attention dropout) hits it, fine-tuned classification path does not. Forcing `flash_attn_qkvpacked_func=None` selects the same PyTorch fallback path all prior fine-tuned results used. **Not a change of method** — documented in-code. | `run_pretrained_epistasis.py:78-99` |

**Critical pair identity (C-036/037/038)**: **A = layer 9, row 264**, **B = layer 9, row
294**, both write to `col=1758` of the same down_proj tensor (same-layer/same-column pair,
different input rows).

**10-row ensemble** (the canonical pre-E9 basis; C-027/C-028) —
`results/super_weight_index.json["dnabert2"]["results"]`, all 10 entries, sorted by
`out_max` descending as `run_pretrained_epistasis.py:219` already does:

| rank | layer | row | col | out_max |
|---|---|---|---|---|
| 1 | 5 | 603 | 1062 | 944.56 |
| 2 | 3 | 86 | 2056 | 732.00 |
| 3 | 3 | 399 | 2056 | 703.01 |
| 4 | 9 | **264** | 1758 | 652.98 |
| 5 | 9 | **294** | 1758 | 620.57 |
| 6 | 3 | 603 | 2056 | 618.13 |
| 7 | 3 | 641 | 2056 | 413.52 |
| 8 | 7 | 603 | 2638 | 284.98 |
| 9 | 6 | 603 | — | — |
| 10 | 5 | 86 | — | — |

(row 603 recurs at layers 3/5/6/7 — "depth redundancy"; row 86 recurs at layers 3/5.)
Explicitly noted in `experiments/CLAUDE.md:62` / `CLAIMS_LEDGER.md:77`: this 10-row set is
**adjacent to, not the same as**, the critical-pair result — do not conflate.

**Existing intervention scripts** (`scripts/mechanism/`): `run_codominance.py` (MAKE-PAIR/
BREAK-PAIR ratio construction), `run_direction_vs_magnitude.py` (DOWN/UP rescale-then-ablate),
`run_norm_matched_control.py` (norm-matched non-SW pair falsification control),
`run_compensation_circuit.py` (single/A, single/B, joint additive-prediction test),
`run_pretrained_epistasis.py` (**pretrained MLM, no fine-tuning — designated canonical for
E9**, see below).

**Exact intervention** in every script: resolve
`m = bert.encoder.layer[i].mlp.wo`, `m.weight.data[row, :]` — a full down_proj output row,
never a single (row,col) scalar or a column. This is the same point in the forward pass as
GENERator: after the GLU nonlinearity, before the residual add, before the encoder
LayerNorm. `_save_row` / `_zero_row` / `_restore_row` (`scripts/evaluation/
run_gue_ablation.py:470-484`) implement pure zero-ablation (`scale=0.0`, i.e. `epsilon=1.0`).
**E9 needs a generalized scale variant for `epsilon=0.5` (`scale=0.5`)** — none of the
existing scripts implement partial-scale row intervention; only full zero-ablation and
norm-preserving rescale-to-target-ratio (`run_codominance.py`) exist. E9 adds
`_scale_row(model, pattern, layer, row, alpha)` (thin wrapper reusing `_save_row`/
`_restore_row`), not a new metric — same wire, generalized coefficient.

**Primary endpoint for E9 — pretrained MLM loss** (`run_pretrained_epistasis.py`,
source of C-037): `AutoModelForMaskedLM`, no fine-tuning, no task head. Held-out data: hg38
windows via `_read_fasta_windows(fasta, bed, n_windows=256, win_bp=600, rng)`, never used
for fine-tuning. **One fixed mask realization**, seed 42, `mask_prob=0.15`, reused across
every condition (`_build_fixed_batches`) so deltas are paired. Metric: token-weighted mean
MLM loss (`mlm_loss`); `dLoss = ablated − baseline` (positive = worse — inverted vs.
accuracy-based scripts). This script already implements singles, all-pairs epistasis,
random-pair floor, and a k-of-N cumulative curve over the 10-row index — i.e. almost exactly
E9's F0 + a partial F3 at `epsilon=1.0` only. E9 generalizes it to arbitrary masks `a` and to
`epsilon=0.5`.

**Chosen over the splice/GUE endpoint because**: (a) it is the prereg's own stated
preference — avoids downstream task-head confounding, continuous scalar, pair interaction
already shown to exist pre-fine-tuning; (b) **practical blocker** — the multiseed fine-tuned
splice checkpoints these scripts default to
(`results/gue_checkpoints_multiseed/dnabert2_reconstructed/seed_{0,1,2}/model_state.pt`) do
**not exist on this filesystem** (only a single non-multiseed checkpoint at
`results/gue_checkpoints/dnabert2_reconstructed/model_state.pt` is present, and the GUE
dataset itself — `/data/nvidia/data/gue/GUE` — is also absent here, see below). The splice
endpoint is therefore **out of scope for E9's primary/secondary analysis** and not pursued
as a rescue.

**Layer-9 residual norm** — **convention inconsistency found and resolved**:
`run_compensation_circuit.py::_last_hidden_channel` hooks the **last** encoder layer
generically (not literally layer 9), while `run_codominance.py`, `run_direction_vs_magnitude.py`,
and `run_norm_matched_control.py` all hook layer 9 **explicitly** via `--layer 9`. Since
DNABERT-2 has exactly 12 layers and the pair lives at layer 9 (not the last layer, index 11),
these are genuinely different measurement points. **E9 standardizes on the explicit
`--layer 9` hook** (the convention used by 3 of 4 scripts, and the one that actually targets
the pair's own layer) for its residual-norm secondary endpoint. This does not by itself
confirm or dispute the "17.20 → 7.14" figure quoted in C-038 — no raw artifact survives to
check which convention originally produced it (see `MISSING_COLLEAGUE_ARTIFACTS.md`); E9's
own measurement will be independently produced under the documented convention above.

**No raw output artifact currently exists** for any of the five DNABERT-2 mechanism
scripts — `results/mechanism/` holds only narrative `.md` reports, zero `.json`/`.csv`.
C-036/037/038 are adopted as fact per D-024; nothing here has been *run in this repository*
prior to E9.

---

## Environment fixes applied for E9 (recorded here, not silent)

1. **hg38 FASTA path.** Every mechanism script defaults `--fasta /data/nvidia/data/hg38/hg38.fa`
   — a path from the colleague's original machine. **Confirmed absent on this filesystem**
   (`/data` does not exist at all on this node; searched `/work/11034/atzanakak`, `/scratch`,
   `/home1/11034/atzanakak` for any existing hg38 FASTA — none found). **Fix**: downloaded
   `hg38.fa.gz` directly from UCSC
   (`https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz`, standard reference
   build, no login required) to `data/reference/hg38/hg38.fa` (2,980,020,224 bytes
   decompressed, 455 FASTA records — full primary assembly + alt/random contigs, matching
   every chromosome name referenced by `data/regions/hg38/random_262kb.bed`). Every E9
   script invocation passes `--fasta data/reference/hg38/hg38.fa` explicitly; this file is
   **not** committed to git (too large; `.gitignore`d alongside `results/`), only its
   provenance (URL + byte size + record count) is recorded here for reproducibility.
2. **GUE dataset** (`/data/nvidia/data/gue/GUE`) — also absent on this filesystem. Not
   fixed, because E9's primary DNABERT-2 endpoint (pretrained MLM loss) does not need it;
   see scope note above. Splice-endpoint scripts remain unrunnable here.
3. **DNABERT-2 revision pin** — see table above; every E9 script call passes
   `--code_revision 7bce263b15377fc15361f52cfab88f8b586abda0` explicitly.
4. **Python environment** — this repo's `requirements.txt`-described environment lives in
   the `grlm` conda env at
   `/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm` (not on `PATH` by
   default; no `conda`/`mamba` init was active in this shell — resolved via
   `MAMBA_ROOT_PREFIX`). Confirmed to contain `torch`, `transformers`, `pyfaidx` per
   `requirements.txt`. Running on an active SLURM interactive allocation
   (job 3382784, node `v330-011`, partition `gpu-a100-sm`) with one idle NVIDIA A100-PCIE-40GB
   visible via `nvidia-smi` — sufficient for both models (GENERator EUK ~3B params fp32,
   DNABERT-2 117M fp32).

---

## Baseline regression — status

Per Phase 0 instructions, the untouched condition and at least one known target
intervention endpoint must be reproduced under the current integrated implementation
**before** any new tomography measurement, with a pre-declared tolerance. This is recorded
separately once run: see `BASELINE_REGRESSION.md` (created immediately after this file, tolls
run before any mask is generated). If the qualitative causal phenotype fails to reproduce,
per instruction: **STOP**, write `E9_BLOCKED.md`, do not proceed to Phase 1.
