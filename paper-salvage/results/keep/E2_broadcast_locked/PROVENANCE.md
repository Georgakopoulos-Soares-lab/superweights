# PROVENANCE — E2 standardized broadcast, locked protocol

**Claims supported:** C-012, C-013 (`established`); C-015 (`established`, derived from
C-012/13 plus the re-measured DNABERT-2 C in this artifact).
**Artifact class:** **raw** — direct harness output, plus per-model noise-floor measurements.

| | |
|---|---|
| Source path | `results/sw_broadcast_impulse.json`, `results/impulse_determinism_*.json`, `logs/e2_run_*.log` |
| Producing script | `scripts/interpretability/run_sw_broadcast_impulse.py`; noise floors from `scripts/interpretability/impulse_determinism_check.py`; provenance stamped by `scripts/interpretability/stamp_e2_provenance.py` |
| **Prereg lock** | sha256 `3d7515d0b7889f65038acd8479e85976248e7dbd0e5a701303de3a1b37dbfdc3`, UTC 2026-08-13T02:59:20+00:00 — `LOCKS.jsonl` copied alongside. **All results generated after the lock.** |
| Protocol | D-011 (AC-relative ε) + D-013 (fp32, headroom) + D-014 (DNABERT-2 eager) + D-015 (T/C primary, KL secondary) |
| Checkpoints | `GenerTeam/GENERator-v2-{eukaryote,prokaryote}-3b-base` · `zhihan1996/DNABERT-2-117M @ 7bce263` · `InstaDeepAI/NTv3_650M_pre @ code_revision 0ecff36` · `togethercomputer/evo-1-8k-base @ 1.1_fix` |
| Config | `configs/generator.yaml`, `configs/generator_prokaryote.yaml`; others pinned in the harness |
| Seed | 42 · n_controls 5 |
| Dtype | float32 for all five models |
| git commit | `00cedd3` (four models) · `ee746de` (Evo1) · `e20d12de3273307f08e354c524c4d9f42dc7cf75` (migrated) |
| Environment | conda `grlm` for four models; Evo1 in `evo2.sif` sha256 `ecb001192eeee416239da526ceaf9a73685fc9609aad404a23a36ac0bc161706`, apptainer 1.1.8, with `evo-model` 0.5 and `stripedhyena` 0.2.2 resolved from `~/.local` **outside** the image |

## Caveats, all load-bearing

- **Evo1's secondary dose (α = 1.0) FAILED with CUDA OOM** and is absent. Its KL is
  **not measured**, not zero. No retry was performed.
- **matched-norm arm is empty** for GENERator EUK (0/3,072 rows within ±10% of the SW row's
  ‖W_down‖) and DNABERT-2 (0/768), and **failed** for NTv3 (harness cannot locate its
  down-projection). The band was not widened.
- **NTv3 T and C are undefined** — SW layer 11 of 12, no downstream layers.
- **DNABERT-2 uses the eager attention path.** Its Triton path is nondeterministic and casts
  q/k/v and bias to fp16; the kernel swap changed masked-LM perplexity by Δ = −74.3%, which
  **did not pass** the ±1% guard and is disclosed as such (D-014).
- The `evo1_fixed_eps_SUPERSEDED` key inside the JSON is the **void** pre-D-011 entry,
  retained deliberately. Do not read it as a result.
- **C-014 / X-006 is retired and is NOT covered by this artifact.**
