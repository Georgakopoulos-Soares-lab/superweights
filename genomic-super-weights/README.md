# Super Weights in Genomic Language Models

Locating super weights in transformer-based genomic LMs following the data-free,
single-forward-pass method from Yu et al. (2024) *"The Super Weight in Large Language Models"*
(arXiv 2411.07191). Detection only — no quantization.

---

## Models Studied

| Model | Type | Params | HF ID |
|-------|------|--------|-------|
| GENERator eukaryote | Causal decoder | 3B | `GenerTeam/GENERator-eukaryote-3b-base` |
| GENERator prokaryote | Causal decoder | ~200M | `GenerTeam/GENERator-prokaryote-*` |
| GENERator prokaryote 1B | Causal decoder | 1B | `GenerTeam/GENERator-prokaryote-1b-base` |
| Evo 2 7B | StripedHyena2 SSM | 7B | `arcinstitute/evo2_7b` |
| NTv3 | Encoder (masked LM) | 50M | `InstaDeepAI/nucleotide-transformer-v3-50m-multi-species` |
| DNABERT-2 | Encoder (masked LM) | 117M | `zhihan1996/DNABERT-2-117M` |
| MegaDNA | Causal decoder | — | — |
| HybridNa | Hybrid SSM/Attn | — | — |
| GenomeOcean | Causal decoder | — | — |

---

## Results

### Super Weight Detection (perplexity / entropy ablation)

| Model | SW location (layer, row, col) | in_max | out_max | δ% pruned SW | δ% random (mean) |
|-------|-------------------------------|--------|---------|--------------|------------------|
| GENERator eukaryote | L4, r2371, c2536 | 67 551 | 375 361 | **+23 026%** | +0.01% |
| GENERator prokaryote | L2, r1927, c1769 | 7 383 | 506 014 | **+25 975%** | +0.03% |
| GENERator prokaryote 1B | L2, r1397, c63 | 12 526 | 85 983 | **+30.5%** | +0.03% |
| NTv3 | L11, r1472, c1579 | 145 | 1 582 | +4.8% | +0.03% |
| DNABERT-2 | L5, r603, c1062 (+ 9 more) | 240 | 945 | +1.5% | +0.009% |
| Evo 2 7B | — (no effect) | — | — | **+0.0007%** | +0.001% |
| MegaDNA | L1, r152, c225 | 43 | 1 873 | +0.34% | −0.09% |
| HybridNa | L31, r2893, c6187 | 508 | 61 | −1.3% | −0.003% |

Detection mode is "superrow" for all transformer models (max-activation row zero-out).

**Key finding**: Transformer-based genomic models (GENERator family, DNABERT-2, NTv3) exhibit
super weights with the Yu et al. phenotype. SSM-based models (Evo 2, MegaDNA, HybridNa) do not —
large activation outliers exist but zeroing them has no perplexity effect.

### GUE Downstream Task Ablation (DNABERT-2 and NTv3)

All 10 detected DNABERT-2 super rows zeroed simultaneously vs. 10 random-row controls (mean of 10 repeats).

| Model | Task | Baseline acc | Pruned SW acc (all rows) | Δacc | Δmcc | Rand ctrl Δacc |
|-------|------|-------------|--------------------------|------|------|----------------|
| DNABERT-2 | prom/prom_core_notata | 83.81% | 52.10% | **−37.84%** | −88.49% | ~0% |
| DNABERT-2 | EMP/H3K4me3 | 67.09% | 66.63% | −0.69% | −1.95% | ~0% |
| DNABERT-2 | splice/reconstructed | 92.46% | 59.12% | **−36.06%** | −77.57% | ~0% |
| NTv3 | prom_core_notata | 70.0% | 69.9% | −0.05% | −0.28% | ~0% |
| NTv3 | splice/reconstructed | 53.4% | 56.5% | +5.8% | **−86.2%** | ~0% |
| NTv3 | EMP/H3K4me3 | 47.0% | 47.0% | 0.0% | 0.0% | ~0% |

Random-weight controls are consistently within ±0.05% — the SW effect is specific.

**Finding**: Prom and splice tasks show catastrophic collapse (−35–38% accuracy) when all DNABERT-2
super rows are zeroed. The histone mark task (H3K4me3) is unaffected, consistent with epigenomic
signals being encoded diffusely rather than concentrated in MLP row clusters.

### DNABERT-2 Per-Row Ablation

Each of the 10 super rows zeroed individually on the fine-tuned checkpoint.

| Task | Row | out_max | Δacc (single row) | Δmcc |
|------|-----|---------|-------------------|------|
| prom_core_notata | L3 r603 | 618.1 | −0.11% | −0.23% |
| prom_core_notata | L5 r603 (detected top) | 944.6 | +0.06% | +0.14% |
| splice/reconstructed | **L3 r603** | 618.1 | **−1.45%** | **−2.47%** |
| splice/reconstructed | L5 r603 (detected top) | 944.6 | −0.15% | −0.31% |
| EMP/H3K4me3 | L3 r641 | 413.5 | +0.46% | +0.71% |
| EMP/H3K4me3 | L5 r603 (detected top) | 944.6 | +0.19% | +0.36% |

Key observations:
- **No individual row is a bottleneck** — maximum single-row effect is −1.45% (splice, L3r603).
  Removing all 10 together causes −36%: the rows function as a redundant ensemble.
- **The activation-detected top row (L5r603, out_max=944.6) is not the most functionally
  critical.** L3r603 (out_max=618.1) causes larger damage on splice, and L5r603 individually
  causes zero or positive effect on promoter/histone tasks. Activation magnitude ≠ functional
  importance.
- GENERator's single super row causes immediate +23,000% perplexity loss — a qualitatively
  different concentration level compared to DNABERT-2's distributed ensemble.

### DNABERT-2 Mechanistic Characterisation (`debug_dnabert2_*.py`)

Five targeted diagnostic scripts (`debug_dnabert2_profile.py`, `debug_dnabert2_multi_probe.py`,
`debug_dnabert2_persistence.py`, `debug_dnabert2_masked_token.py`) were run to compare
DNABERT-2 against the GENERator super weight phenotype.

| Property | GENERator (causal) | DNABERT-2 (masked encoder) |
|---|---|---|
| Super activation magnitude | out_max = 375,361 | out_max = 945 |
| Detection probe-stability (iterative) | Same rows every probe | L5r603 stable; secondary rows vary by probe |
| Intermediate channel persistence | Persists L4→L30 via skip connections | Decays to <0.1% at L6 (local spike only) |
| Degenerate token inflation after SW removal | Strong (stop-word analogue) | 1.04× vs 1.00× random (no effect) |
| Single-row functional damage | +23,026% perplexity | ≤1.45% accuracy drop |

**Architecture interpretation**: The super activation in causal decoders (Llama-style) propagates
through residual connections and is globally present at every layer, enabling a single row to
destroy generation. In DNABERT-2's encoder, the spike is confined to L5 MLP output and decays
immediately — the BERT bidirectional attention and MLM objective result in a more distributed
representation, requiring ensemble removal for functional damage.

---

## Directory Structure

```
genomic-super-weights/
├── configs/               # Per-model YAML configs
│   ├── generator.yaml
│   ├── generator_prokaryote.yaml
│   ├── generator_prokaryote_1b.yaml
│   ├── evo2.yaml / evo2_7b.yaml
│   ├── ntv3.yaml
│   ├── dnabert2.yaml
│   ├── megadna.yaml
│   ├── hybridna.yaml
│   └── genomeocean.yaml
├── models/                # HuggingFace wrappers (one per model)
│   ├── base_wrapper.py
│   ├── generator_wrapper.py
│   ├── evo2_wrapper.py
│   ├── ntv3_wrapper.py
│   ├── dnabert2_wrapper.py
│   ├── megadna_wrapper.py
│   ├── hybridna_wrapper.py
│   └── genomeocean_wrapper.py
├── hooks/
│   └── activation_hooks.py   # Forward hooks recording per-layer max activations
├── detection/
│   ├── sweep.py              # Single-pass activation sweep
│   ├── identify_spikes.py    # Spike layer + coordinate extraction
│   └── iterative_finder.py   # Iterative zero-out loop (Algorithm 1 of Yu et al.)
├── probes/
│   └── dna_probes.py         # 48-bp probe sequences (GENERator-compatible)
├── analysis/
│   ├── visualize_activations.py   # Per-layer activation profile plots
│   └── ablation.py               # Perplexity / masked-token entropy destruction test
├── scripts/
│   ├── run_detection.py           # CLI: detect super weights
│   ├── run_ablation.py            # CLI: ablation test
│   ├── run_gue_ablation.py        # GUE fine-tune + ablation
│   ├── run_gue_per_row_ablation.py
│   └── *.sbatch                   # SLURM job scripts
├── results/
│   ├── super_weight_index.json    # Detected SW coordinates (all models)
│   ├── ablation_results.json      # Perplexity delta results
│   ├── gue_ablation_results.json  # GUE task ablation (full fine-tune)
│   ├── gue_per_row_ablation.json  # GUE ablation (per SW row)
│   └── *_activation_profile.png  # Layer-wise activation plots
├── debug_generator.py          # GENERator activation profile + SW validation
├── debug_dnabert2_profile.py   # DNABERT-2: full 12-layer activation profile
├── debug_dnabert2_multi_probe.py # DNABERT-2: row importance + probe consistency
├── debug_dnabert2_persistence.py # DNABERT-2: skip-connection propagation test
├── debug_dnabert2_masked_token.py # DNABERT-2: degenerate token inflation test
├── inspect_*.py                # One-off model inspection / debug scripts
├── probe_ablation.py           # Interactive ablation playground
├── stubs/                      # Type stubs for models without type annotations
└── tests/
    ├── test_hooks.py
    └── test_detection.py
```

---

## Quick Start

### 1. Detect super weights

```bash
python scripts/run_detection.py --model generator
python scripts/run_detection.py --model dnabert2 --probe poly_a
python scripts/run_detection.py --model ntv3 --threshold 0.05
```

Results are written to `results/super_weight_index.json` and an activation profile PNG.

### 2. Run ablation (perplexity / entropy)

```bash
python scripts/run_ablation.py --model generator
python scripts/run_ablation.py --model dnabert2
```

Reads super weight coordinates from `super_weight_index.json` and writes deltas to
`results/ablation_results.json`.

### 3. GUE downstream ablation (DNABERT-2 / NTv3)

```bash
python scripts/run_gue_ablation.py --model dnabert2 --task prom/prom_core_notata
python scripts/run_gue_per_row_ablation.py --model ntv3 --task splice/reconstructed
```

Requires the GUE dataset directory on `$GUE_DATA_PATH`.

---

## Config Setup Note

Before running on a new model, inspect its module names:

```python
model = ...   # load the model
print([n for n, _ in model.named_modules()])
```

Then update `down_proj_pattern` in the corresponding YAML to match the actual
down-projection linear layer inside each MLP/FFN block.

---

## Method

1. **Single forward pass** on a 48-bp probe sequence.
2. **Activation recording** via forward hooks on every MLP down-projection layer,
   capturing `max|input|` and `max|output|` per layer.
3. **Spike identification**: the layer with the globally largest `max|input|` is the
   spike layer; its `out_channel` is the SW row and `in_channel` is the SW col.
4. **Iterative zeroing**: zero out `weight[row, col]`, re-run, repeat until
   `max|input| < 10%` of the initial max (or 10 iterations).
5. **Validation**: re-run perplexity (causal models) or masked-token entropy (encoder
   models) before and after zeroing. Compare against 10 random-weight controls.

---

## Compatibility Notes

**transformers ≥ 5.x + DNABERT-2**: The DNABERT-2 custom `bert_layers.py` relies on
`BertConfig.is_decoder` and `BertConfig.pad_token_id` which were removed as defaults in
transformers 5.x. Two fixes are required:

1. `models/dnabert2_wrapper.py` loads `AutoConfig` separately and injects missing defaults before
   passing `config=` to `from_pretrained`. It also uses `device_map={"":"cpu"}` to avoid the
   meta-device / ALiBi tensor conflict that arises during `__init__` with transformers ≥ 5.

2. For GUE scripts that load the model directly (without the wrapper), patch the cached
   `configuration_bert.py` in your HF cache:
   ```python
   # After super().__init__() in BertConfig.__init__:
   if not hasattr(self, "is_decoder"): self.is_decoder = False
   if not hasattr(self, "pad_token_id"): self.pad_token_id = 0
   ```
   Cache path: `~/.cache/huggingface/modules/transformers_modules/zhihan1996/
   DNABERT_hyphen_2_hyphen_117M/<revision>/configuration_bert.py`

---

## Reference

> Yu, T. et al. (2024). *The Super Weight in Large Language Models.* arXiv:2411.07191.
