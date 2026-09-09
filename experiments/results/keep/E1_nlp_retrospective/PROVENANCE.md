# PROVENANCE — E1 NLP cold-weight validation, retrospective arm

**Claims supported:** C-004, C-005 (both `established`).
**Artifact class:** **derived** — computed from published checkpoint weights. No raw model
outputs; the assay makes no forward pass.

| | |
|---|---|
| Source path | `results/e1_nlp_retrospective.json`, `logs/e1_retrospective.log` |
| Producing script | `paper-salvage/experiments/E1_nlp_validation/run_e1_retrospective.py` |
| Shared library | `paper-salvage/src/uk_frobenius.py` (self-test green at migration) |
| Checkpoints | `huggyllama/llama-7b` L2 · `mistralai/Mistral-7B-v0.1` L1 · `allenai/OLMo-7B-0724-hf` L1 |
| Ground truth | Yu et al. 2024 arXiv:2411.07191 Table 2, read from `docs/superweight_paper.txt` (local copy), not from memory |
| Config | none; coordinates hard-coded in the script from the table above |
| Seed | n/a — deterministic weight algebra, no sampling |
| Dtype | float64 (`uk_frobenius` promotes; documented in that module) |
| git commit | `a157686` (produced) · `e20d12de3273307f08e354c524c4d9f42dc7cf75` (migrated) |
| Environment | conda `grlm`; `LD_LIBRARY_PATH=$ENV/lib` required or PIL fails on GLIBCXX_3.4.29 |
| Caveats | Only the safetensors shards holding each layer's three MLP tensors were downloaded. Shape guards asserted per model before computing. **The prospective arm (C-006) was NOT run** and is not covered here. |
