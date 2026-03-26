# genomic-super-weights

Local module for detecting and validating super weights/super rows in genomic language models (GENERator, Evo2, NTv3, DNABERT2), inspired by Yu et al. (2024).

## Folder structure

- `configs/`: model-specific config files (`generator.yaml`, `evo2_7b.yaml`, `ntv3.yaml`, `dnabert2.yaml`)
- `models/`: wrapper classes that standardize model loading, forward pass, and target-module access
- `hooks/`: activation recorder hooks for per-layer input/output maxima
- `detection/`: sweep + spike-layer identification + iterative SW/SR discovery loop
- `analysis/`: ablation metrics and activation visualization utilities
- `probes/`: DNA probe sequences used for sweeps and destruction tests
- `scripts/`: runnable experiment entry points and SLURM launchers
- `tests/`: smoke tests for detection and hooks
- `debug_*.py`: focused diagnostics for specific hypotheses around the GENERator super row
- `run.sh`: convenience shell entrypoint for running detection/ablation workflows
- `logs/`, `results/`, `analysis/`: runtime artifacts (typically local outputs)

## Core scripts and what they do

### Top-level scripts

- `run.sh`: shell launcher that runs the main pipeline commands with project defaults.
- `inspect_generator.py`: inspects GENERator module names to locate MLP/down-proj candidates and layer count.

### Debug scripts

- `debug_super_row.py`: compares pruning row 2371 vs random rows and checks probe-invariance of the detected spike row.
- `debug_stop_codon.py`: tests whether removing the candidate super row inflates stop-codon next-token probability mass.
- `debug_sw_verify.py`: targeted verification around `(layer=4, row=2371, col=2536)` with scalar/row/layer ablations.
- `debug_gate_up.py`: diagnostic around gate/up-proj behavior and contribution to the detected anomaly.
- `debug_generator.py`: GENERator-focused sanity/debug utility (model loading + activation behavior checks).
- `debug_layer4.py`: layer-4-focused checks for suspected super-row localization.
- `debug_layer_persistence.py`: tests whether layer-level spike behavior persists under repeated forward passes/contexts.
- `debug_row2371.py`: row-2371-specific instrumentation and ablation diagnostics.

### scripts/

- `scripts/run_detection.py`: main CLI for running sweep-based super-weight/super-row detection.
- `scripts/run_ablation.py`: runs destruction/ablation experiments after candidate detection.
- `scripts/run_detection.sbatch`: SLURM batch wrapper for detection jobs.
- `scripts/debug_layer4.sbatch`: SLURM job for layer-4 debugging runs.
- `scripts/debug_generator.sbatch`: SLURM job for GENERator debugging runs.

### detection/

- `detection/sweep.py`: one forward pass across layers with hook-based activation statistics.
- `detection/identify_spikes.py`: picks earliest anomalous output spike layer and extracts `(layer, row, col)`.
- `detection/iterative_finder.py`: iterative loop that zeros scalar or full row and repeats sweep until suppression.
- `detection/__init__.py`: package marker.

### hooks/

- `hooks/activation_hooks.py`: forward pre/post hooks that record `in_max/in_channel` and `out_max/out_channel`.
- `hooks/__init__.py`: package marker.

### models/

- `models/base_wrapper.py`: abstract model wrapper API and in-place zeroing helpers.
- `models/generator_wrapper.py`: GENERator loader + 6-mer sequence preparation.
- `models/evo2_wrapper.py`: Evo2 causal LM wrapper.
- `models/ntv3_wrapper.py`: NTv3 masked-model wrapper.
- `models/dnabert2_wrapper.py`: DNABERT-2 wrapper.
- `models/__init__.py`: wrapper registry (`WRAPPER_MAP`).

### analysis/

- `analysis/ablation.py`: causal perplexity and masked-token entropy metrics + control-selection utilities.
- `analysis/visualize_activations.py`: plotting helpers for per-layer activation maxima.
- `analysis/__init__.py`: package marker.

### probes/

- `probes/dna_probes.py`: curated ACTB-based probe sequences and probe accessor.
- `probes/__init__.py`: package marker.

### tests/

- `tests/test_detection.py`: end-to-end smoke test for sweep + spike extraction on tiny fake model.
- `tests/test_hooks.py`: smoke test that hooks attach/record/detach correctly.

## Typical usage

1. Pick a config in `configs/`.
2. Run detection via `scripts/run_detection.py` (or `run.sh`).
3. Validate candidate with `scripts/run_ablation.py` and/or `debug_sw_verify.py`.
4. Use debug scripts for focused checks (`debug_super_row.py`, `debug_stop_codon.py`, etc.).

## Notes

- Some scripts assume HF model access and GPU availability.
- GENERator wrapper enforces 6-mer-compatible sequence length handling.
- This folder is intended to be self-contained inside the parent repository.
