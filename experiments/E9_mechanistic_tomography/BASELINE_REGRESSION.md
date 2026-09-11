# BASELINE_REGRESSION.md — E9 Phase 0 verdict

Produced by `run_baseline_regression.py` under the environment fixes recorded in
`PROVENANCE_AND_BASELINES.md` (pinned DNABERT-2 revision, repo-local hg38 FASTA,
`grlm` conda env + `LD_LIBRARY_PATH` fix, HF cache pointed at the pre-existing
`/work/11034/atzanakak/ls6/huggingface` cache). Raw output:
`baseline_regression_results.json`.

## DNABERT-2 — PASSES

- Environment: `zhihan1996/DNABERT-2-117M` @ `7bce263b15377fc15361f52cfab88f8b586abda0`,
  pretrained MLM (no fine-tuning), Triton flash-attn patched to the PyTorch fallback
  (matches `run_pretrained_epistasis.py`'s documented compatibility fix).
- 256 hg38 windows (600bp) → 16 fixed batches, 4,453 masked tokens (seed 42,
  `mask_prob=0.15`), identical protocol to the pre-existing pretrained-epistasis script.
- **Baseline MLM loss = 4.695281.**
- **epsilon=1.0 (full ablation, the only scale the prior colleague work tested)**:
  `dA(L9r264)=+0.021819`, `dB(L9r294)=+0.006402`, `dAB=+2.040024`,
  **epistasis = +2.011803**. The previously-reported C-037 figure was **epistasis
  +2.0118** — this reproduces it to 4 decimal places under a from-scratch environment
  (fresh hg38 download, pinned revision, generalized intervention code). This is about
  as strong a Phase 0 pass as this kind of check can produce.
- **epsilon=0.5 (partial suppression, never previously measured)**: `dA=+0.003416`,
  `dB=-0.000997`, `dAB=+0.055204`, **epistasis = +0.052785** — still clearly
  superadditive/positive, ~38× smaller in magnitude than the full-ablation epistasis.
  Reported here as a new data point, not a reproduction target.
- **Qualitative check: PASS** (`epistasis > 0` at epsilon=1.0).

## GENERator EUK — PASSES

- Environment: `GenerTeam/GENERator-v2-eukaryote-3b-base`, no revision was ever pinned
  (see `PROVENANCE_AND_BASELINES.md`); this run resolved `main` via the pre-existing
  local HF cache (`/work/11034/atzanakak/ls6/huggingface`) to commit
  `7dc01bccce5b65e15141170538afdc2ff09d8dde` (multiple older snapshots exist in the
  cache from past sessions; this is the first time a revision has been pinned to a
  specific, recorded value for E9's purposes — all subsequent E9 GENERator work should
  use this same commit).
- 24 hg38-window prompts (120bp, seed 42), `max_new_tokens=64`, greedy-free sampling
  (`top_k=50, temperature=1.0`), per-prompt seeded (`SEED+i`) — identical generation
  config to `run_sw_steering.py`.
- **Row 2371 (primary, layer 4)**: GC = 0.3961 (untouched) → 0.3863 (`alpha=0.5`) →
  0.2944 (`alpha=0.0`, full ablation). Monotonic decrease with suppression.
- **Row 1522 (secondary, layer 4)**: GC = 0.3961 (untouched) → 0.4131 (`alpha=0.5`) →
  0.3466 (`alpha=0.0`). **Non-monotonic** — GC rises slightly at partial suppression
  before falling below baseline at full ablation. Reported as observed, not smoothed
  into a monotonic story; this is exactly the kind of result Phase 8's steerability
  rule exists to catch.
- **Random control (5 rows, seed 42)**: GC = 0.3961 → 0.3962 → 0.3960 across the same
  three alpha values — essentially flat, as expected.
- **GC span**: row 2371 = 0.10167, random control = 0.00026 — **ratio ≈ 390×**. (For
  reference, the previously-reported C-040 figure was "38.6×"; that number came from a
  different scale grid (`{0,0.5,1,2,5}`, span computed across 5 points, of which the
  extreme `scale=5` point is not part of E9's frozen `epsilon` grid) and a different,
  never-reproduced-in-this-repo run — this is not a discrepancy to reconcile, it is a
  different, narrower measurement (E9 uses only 3 alpha points: 1.0, 0.5, 0.0) that
  happens to also show a large, unambiguous effect.)
- **Qualitative check: PASS** (`span_primary=0.1017 ≫ span_random=0.0003`).

## Verdict

**Both models pass Phase 0. No `E9_BLOCKED.md` is written. Proceeding to the full
DNABERT-2 mask sweep (Phase 5's F0-F3 ladder).**

**This baseline regression run's GENERator data is also E9's final, frozen GENERator
dataset** — it already covers exactly the three alpha points the locked prereg
specifies (untouched, `epsilon=0.5`, `epsilon=1.0`) for both the primary and secondary
row plus the frozen 5-row random control, with no additional scales. No separate
GENERator "final measurement" run is needed or performed; re-running it would not add
information and would burn GPU time for no scientific reason. The GENERator dose-response
figure and Phase 8 write-up draw directly from `baseline_regression_results.json`.
