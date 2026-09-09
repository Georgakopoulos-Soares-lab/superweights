# E7 provenance addendum — immutable checkpoint identifiers

**This is a documentation addendum only. No E7 result is rerun, recomputed, or changed by
this file.** It resolves the exact commit/revision and file hashes for the four checkpoints
E7 downloaded, none of which were pinned at the time (disclosed as a gap in
`RESULTS.md`'s provenance table). All values below were read from the already-downloaded
local cache — no new download was triggered to produce this addendum.

## Phi-3-mini-4k-instruct

| | |
|---|---|
| Repo | `microsoft/Phi-3-mini-4k-instruct` |
| Resolved HF commit | `f39ac1d28e925b323eae81227eaba4464caced4e` |
| Shard used (layers 2, 4) | `model-00001-of-00002.safetensors` |
| Shard SHA256 | `b7492726c01287bf6e13c3d74c65ade3d436d50da1cf5bb6925bc962419d6610` |

## Qwen2.5-7B

| | |
|---|---|
| Repo | `Qwen/Qwen2.5-7B` |
| Resolved HF commit | `d149729398750b98c0af14eb82c78cfe92750796` |
| Shard used (layer 26, detected candidate) | `model-00004-of-00004.safetensors` |
| Shard SHA256 | `b5a2298dddcf228129975a9a271912a9f8dc817957deecde523e3154481ec3fb` |

Note: the Phase-1 full-model detection forward pass and the later shard-only spectral
re-fetch (`run_confirmatory_spectral_lite.py`) resolved to two different local cache
directories (`HF_HOME` vs. the legacy `TRANSFORMERS_CACHE`, both set simultaneously per this
project's standing environment requirement) — the shard content is byte-identical in both
locations (hash-verified above); this is a caching-path artifact of running two scripts with
slightly different env-var resolution, not a provenance discrepancy.

## GenomeOcean-4B

| | |
|---|---|
| Repo | `DOEJGI/GenomeOcean-4B` |
| Resolved HF commit | `2bed2fc3ed47c5f6955ba3e64563512c9b338dfb` |
| Shard used (layer 1, detected candidate) | `model-00001-of-00002.safetensors` |
| Shard SHA256 | `570ae8543a5b23850965d1812830c589d1eb16ae145482723333ea46104d89f3` |

## Evo 2 7B

| | |
|---|---|
| Repo | `arcinstitute/evo2_7b` |
| Resolved HF commit | `bda0089f92582d5baabf0f22d9fc85f3588f6b58` |
| Checkpoint file | `evo2_7b.pt` |
| File SHA256 | see `EVO2_SHA256.txt` in this directory (computed separately; ~14GB file) |

## What this changes

Nothing about E7's measurements, branch decision, or claim recommendations. This addendum
exists so a future session (or the manuscript's Methods section) can cite exact, immutable
checkpoint identifiers for all four E7 models without re-deriving them from cache-directory
archaeology.
