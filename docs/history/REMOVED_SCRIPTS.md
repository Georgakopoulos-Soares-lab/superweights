# Removed scripts (2026-09-09)

The repository carried 385 code files, most of them TACC cluster job scripts and exploratory
one-off analyses that no manuscript claim depends on. 136 were removed so that what remains
is the code a reviewer needs. **Nothing is lost: every file below is retrievable from git
history** (`git log --diff-filter=D --name-only` to find its last commit, then
`git show <commit>^:<path>`).

## What was kept, and why

A file was kept if it met any of:

1. it is in the transitive import closure of the 20 experiments in `docs/EXPERIMENT_MAP.md`;
2. it lives in a protected directory -- the frozen experiment harnesses, the figure pipeline,
   model wrappers, configs, runtime shims, the audit trail, the closing round, or tests;
3. it is named in current prose (manuscript, README, figure provenance, audit reports) --
   `docs/history/` was excluded from this scan, since archived text must not keep code alive;
4. it is a Python analysis under `scripts/{compression,mechanism,detection,evaluation,`
   `diagnostics,sae}/` -- these produced the negative results behind the retraction items,
   and retraction evidence has to remain inspectable even when the README no longer cites it.

## What was removed

51 Slurm job scripts, 14 shell launchers, 71 Python files.

The Slurm scripts targeted a TACC allocation this project no longer runs on; their scientific
content is the Python they invoked, which is retained wherever a claim depends on it.

### `analysis/` — 3 file(s)

- `__init__.py`
- `probe_windows.py`
- `visualize_activations.py`

### `detection/` — 2 file(s)

- `__init__.py`
- `iterative_finder.py`

### `hooks/` — 1 file(s)

- `__init__.py`

### `probes/` — 1 file(s)

- `__init__.py`

### `scripts/` — 1 file(s)

- `run_caduceus_detection.sh`

### `scripts/analysis/` — 33 file(s)

- `build_channel_survival_all_models.py`
- `build_dnabert2_h3k4me3_aligned.py`
- `build_subspace_retention_summary.py`
- `diagnose_evo1_readout.py`
- `plot_activation_lifecycle.py`
- `plot_composite_figure.py`
- `plot_compression_multimodel.py`
- `plot_figure2_mechanism.py`
- `plot_model_ablation_comparison.py`
- `plot_quantization_sw_comparison.py`
- `plot_sw_mechanistic.py`
- `run_activation_lifecycle.sbatch`
- `run_dnabert2_uk_audit.py`
- `run_esm2_ablation.py`
- `run_esm2_uk_audit.py`
- `run_evo1_fp64_adjudication.py`
- `run_evo1_uk_audit.py`
- `run_generator_residual_attribution.py`
- `run_generator_uk_audit.py`
- `run_mixer_intervention_evo1_rescue.py`
- `run_multilayer_shuffle_probe.py`
- `run_prok_stratified_lifecycle.py`
- `run_splice_gc_confound.py`
- `run_uk_precision_recall.py`
- `submit_dnabert2_uk_audit.sh`
- `submit_esm2_ablation.sbatch`
- `submit_esm2_uk_audit.sbatch`
- `submit_generator_lifecycle.sbatch`
- `submit_generator_mechanistic_euk.sbatch`
- `submit_generator_mechanistic_prok.sbatch`
- `submit_multilayer_shuffle_probe.sbatch`
- `submit_prok_stratified_lifecycle.sbatch`
- `validate_ablation_trace_evo1.py`

### `scripts/compression/` — 9 file(s)

- `batch_int2_gue.sh`
- `int4_downstream_benchmark_splice.sbatch`
- `plot_multimodel_sweep.sh`
- `run_compression_sweep_evo2.sbatch`
- `run_compression_sweep_generator_euk.sbatch`
- `run_compression_sweep_generator_euk_all_tasks.sbatch`
- `run_compression_sweep_generator_prok.sbatch`
- `run_whole_model_quant_generator.sh`
- `run_whole_model_quant_generator_prokaryote.sh`

### `scripts/detection/` — 14 file(s)

- `run_ablation_evo2.sbatch`
- `run_ablation_generator_prokaryote.sbatch`
- `run_ablation_generator_prokaryote_1b.sbatch`
- `run_ablation_genomeocean.sbatch`
- `run_ablation_hybridna.sbatch`
- `run_ablation_megadna.sbatch`
- `run_detection.sbatch`
- `run_detection_generator_prokaryote_1b.sbatch`
- `run_detection_genomeocean.sbatch`
- `run_detection_hybridna.sbatch`
- `run_detection_megadna.sbatch`
- `run_hydra_test_evo2.sbatch`
- `run_hydra_test_evo2_local.sh`
- `submit_evo1_detection_ablation.sh`

### `scripts/dev/` — 27 file(s)

- `debug_dnabert2_masked_token.py`
- `debug_dnabert2_multi_probe.py`
- `debug_dnabert2_persistence.py`
- `debug_dnabert2_profile.py`
- `debug_evo2_shapes.py`
- `debug_gate_up.py`
- `debug_generator.py`
- `debug_generator.sbatch`
- `debug_layer4.py`
- `debug_layer4.sbatch`
- `debug_layer_persistence.py`
- `debug_row2371.py`
- `debug_stop_codon.py`
- `debug_super_row.py`
- `debug_sw_verify.py`
- `inspect_dnabert2.py`
- `inspect_evo2.py`
- `inspect_evo2.sbatch`
- `inspect_generator.py`
- `inspect_genomeocean.py`
- `inspect_hybridna.py`
- `inspect_megadna.py`
- `inspect_ntv3.py`
- `prefetch_generator_assets.sh`
- `probe_ablation.py`
- `run_generator_su_ablation.sbatch`
- `run_generator_su_ablation.sh`

### `scripts/interpretability/` — 30 file(s)

- `analyze_sw_kmer_motifs.py`
- `compare_attribution_methods.py`
- `dnabert2_kernel_guard.py`
- `plot_sw_task_diff_heatmap.py`
- `run_broadcast_impulse_dnabert2.sbatch`
- `run_broadcast_impulse_evo1.sbatch`
- `run_broadcast_impulse_generator.sbatch`
- `run_broadcast_impulse_ntv3.sbatch`
- `run_broadcast_impulse_prok.sbatch`
- `run_counterfactual_swap_generator.sbatch`
- `run_counterfactual_swap_prok.sbatch`
- `run_cross_kingdom_transfer.py`
- `run_neuron_base_vs_finetuned_dnabert2.py`
- `run_neuron_broadcast_impulse_dnabert2.py`
- `run_neuron_controls_dnabert2.py`
- `run_neuron_discovery_dnabert2.py`
- `run_neuron_structural_mapping_dnabert2.py`
- `run_relay_heads_generator.sbatch`
- `run_relay_heads_prok.sbatch`
- `run_svd_spectral_diagnostic.py`
- `run_sw_ablation_regression.py`
- `run_sw_activation_heatmap_euk.sbatch`
- `run_sw_causal_tracing.py`
- `run_sw_gradient_attribution.py`
- `run_sw_hexamer_causal.py`
- `run_sw_hexamer_extended.py`
- `run_sw_kmer_scan.py`
- `run_sw_shuffle_controls.py`
- `run_sw_token_omission.py`
- `single_neuron_pilot.sbatch`

### `scripts/sae/` — 15 file(s)

- `run_analyze_euk_v2.sbatch`
- `run_analyze_euk_v3.sbatch`
- `run_analyze_euk_v5.sbatch`
- `run_collect_euk.sbatch`
- `run_collect_euk.sh`
- `run_collect_prok.sbatch`
- `run_collect_prok.sh`
- `run_feature_analysis.sbatch`
- `run_feature_analysis.sh`
- `run_train_euk_v2.sbatch`
- `run_train_euk_v3.sbatch`
- `run_train_euk_v4.sbatch`
- `run_train_euk_v5.sbatch`
- `run_train_sae.sbatch`
- `run_train_sae.sh`
