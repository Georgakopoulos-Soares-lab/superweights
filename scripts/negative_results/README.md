# Negative results

The experiments here produced nulls. They are collected in one place because a null is easy to
lose track of, and because each one is the evidence behind a retraction or rescope in the
top-level README — the claims they refute were, at some point, claims this project made.

| retraction | what was tested | what came back | code |
|---|---|---|---|
| 3 | Does exempting the super weight from quantisation help? | **No — a no-op by construction.** Per-row RTN already preserves the row max with 0.000e+00 error at INT8–INT2; per-tensor exemption benefit averaged +0.048 pp, t≈+0.15 | `run_per_tensor_sw_exemption.py`, `run_granularity_multimodel.py`, `run_group_scale_preservation.py`, `run_destructive_sw_protection.py`, `run_pair_aware_compression.py` |
| 2 | Is "shadow redundancy" real, or a layer-depth artifact? | **Artifact.** `prox_far` pruning confounded proximity with depth | `run_proximity_confound_control.py` |
| 7 | Does norm dominance predict causal criticality? | **No.** NTv3's super row is rank 1/1536 with a 29.4× gap — *more* norm-dominant than DNABERT-2's — and is functionally inert | `run_norm_matched_control.py`, `run_sw_health_check.py` |
| — | Do the DNABERT-2 pair rows compensate for each other? | **No.** Cross-ablation leaves the survivor at ×1.0000 | `run_compensation_circuit.py` |

The remaining quantisation scripts (`run_int4_*`, `run_whole_model_quantization.py`,
`run_compression_sweep.py`, `run_quantization_ablation.py`) are the benchmarks those
conclusions were drawn from.

Nothing here is imported by the reproduction pipelines: these are terminal analyses. That is
why they look unreachable to a dependency scan, and it is not a reason to delete them.
