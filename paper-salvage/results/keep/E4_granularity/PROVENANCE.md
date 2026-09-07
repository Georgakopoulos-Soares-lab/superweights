# PROVENANCE — E4 c_{k,i} granularity decomposition

**Claims supported:** C-003 (`established`).
**Not covered:** C-032 (`supported`, not established) and C-001 (**on hold**) — see below.
**Artifact class:** **derived** — weight algebra only, no forward pass.

| | |
|---|---|
| Source path | `results/e4_granularity.json`, `results/e4_ntv3_shared_adapter.json` |
| Producing script | `paper-salvage/experiments/E4_granularity/run_e4_granularity.py` |
| Shared library | `paper-salvage/src/uk_frobenius.py` incl. `adapter_ntv3` (N-010 fix, 7 tests green) |
| Canonical table | `paper-salvage/experiments/E4_granularity/CANONICAL_TABLE.md` |
| Checkpoints | as listed per model in the JSON; NTv3 `InstaDeepAI/NTv3_650M_pre @ 0ecff36` |
| Seed | n/a — deterministic weight algebra |
| Dtype | float64 (`uk_frobenius` promotes) |
| git commit | `7c26a0a` (produced) · `3ca8bb6` (NTv3 adapter fix) · `0bc57ee` (canonical table) · `e20d12de3273307f08e354c524c4d9f42dc7cf75` (migrated) |
| Environment | conda `grlm`; Evo1 row computed inside `evo2.sif` |
| Caveats | Shape verified per model before any number was computed. **The GENERator PROK row is CONTESTED** — C-001 is on hold under N-009 and its row is **not** migrated as support for any established claim. NTv3's value reproduces exactly through the corrected shared adapter. |
