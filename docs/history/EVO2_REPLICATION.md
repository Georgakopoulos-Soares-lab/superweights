# Evo2 Replication Plan — what to re-run for the StripedHyena claims

The manuscript's StripedHyena results are currently produced on **Evo1**
(`togethercomputer/evo-1-131k-base`, GPU node with `flash-attn` + custom
`stripedhyena` package). Evo2 lives in a different runtime
(`togethercomputer/evo-2-7B` packaged for Singularity / NGC) and was not
available on the local box. This document spells out exactly which
experiments need to be replayed on Evo2, with the same protocol, so that
the cross-architecture claim ("StripedHyena super-weights behave
qualitatively the same as transformer SWs in detection but not in
ablation") is the verified-on-the-frontier version rather than an
extrapolation from Evo1.

For each block below:
- **Input model** — exact HF repo to swap in.
- **Script** — file you can run unchanged after a single string edit.
- **Output JSON** — filename the figures expect.
- **Expected runtime** — measured on Evo1; Evo2-7B is ~3× larger so
  multiply by ≈ 3.
- **Status on Evo1** — what we have today (so the diff is obvious).

────────────────────────────────────────────────────────────────────────────

## 0. One-line swap

Every script reads the model id from a single top-of-file constant or
from `--model_id`. Replace:
```python
MODEL_ID = "togethercomputer/evo-1-131k-base"
```
with the Evo2 release the lab is benchmarking (e.g.
`togethercomputer/evo-2-7B` or whichever StripedHyena-2 checkpoint is
shared). Everything else — hook names, attribution loop, ablation
mechanics — is identical because both models expose
`backbone.blocks[i].mlp.{l1,l2,l3}.weight` (gate / up / down).

────────────────────────────────────────────────────────────────────────────

## 1. SW detection (activation lifecycle) — TIER 0, **required**

Find which residual-stream rows blow up on a held-out window.

| Field           | Value                                                                                                  |
|-----------------|--------------------------------------------------------------------------------------------------------|
| Evo1 script     | [scripts/analysis/run_activation_lifecycle_evo1.py](scripts/analysis/run_activation_lifecycle_evo1.py) |
| Evo1 output     | `results/activation_lifecycle_evo1.json`                                                               |
| Evo2 output     | `results/activation_lifecycle_evo2.json`                                                               |
| Status on Evo1  | DONE — 10 detected rows at block 11: 156, 411, 616, 682, 843, 1636, 3292, 3582, **3776**, 4033          |
| Expected runtime| Evo1: ~3 min on a single H100 (one forward).  Evo2-7B: ~9–12 min.                                       |
| Notes           | Use the same 2-kbp window of `data/genome/hg38_test.fa` as Evo1 (the script already references it).    |

────────────────────────────────────────────────────────────────────────────

## 2. ‖U_k‖_F weights-only audit — TIER 0, **required**

For every gated-MLP block, compute the row-wise Frobenius norm of
`U_k = l3[:, k] · l1[k, :] + l3[:, k] · l2[k, :]` and rank the detected
rows. Confirms whether Evo2 has the same "cold weights → load-bearing
nowhere" pattern.

| Field           | Value                                                                                              |
|-----------------|----------------------------------------------------------------------------------------------------|
| Evo1 script     | [scripts/analysis/run_uk_audit_evo1.py](scripts/analysis/run_uk_audit_evo1.py)                     |
| Evo1 output     | [results/sw_mechanistic_evo1.json](results/sw_mechanistic_evo1.json)                               |
| Evo2 output     | `results/sw_mechanistic_evo2.json`                                                                 |
| Status on Evo1  | DONE — detected rows at L11 land at ranks 48–168 / 4096 (1–4ᵗʰ percentile), NOT top.               |
| Runtime         | Evo1: ~1 min (weights only, no forward).  Evo2-7B: ~3 min. **No Singularity needed.**              |

After this lands, panel **E of Figure 2** can include an Evo2 marker
alongside the existing Evo1 open-triangles. If the same "low percentile,
ablation-null" pattern repeats, the cross-architecture claim is
strengthened with frontier-scale evidence.

────────────────────────────────────────────────────────────────────────────

## 3. Same-protocol detect + ablate — TIER 0, **required**

Same window, same rows, but now zero out the detected rows and re-score
held-out PPL.

| Field           | Value                                                                                                                                                            |
|-----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Evo1 script     | [scripts/analysis/run_evo1_detect_ablate.py](scripts/analysis/run_evo1_detect_ablate.py)                                                                          |
| Evo1 output     | [results/evo1_detect_ablate.json](results/evo1_detect_ablate.json)                                                                                                 |
| Evo2 output     | `results/evo2_detect_ablate.json`                                                                                                                                |
| Status on Evo1  | DONE — ΔPPL = 0.00 % (baseline 5.218, ablated 5.218) for the 10 detected rows.                                                                                   |
| Runtime         | Evo1: ~4 min (two forwards).  Evo2-7B: ~12 min.                                                                                                                  |
| Why on Evo2     | Reviewer-pre-empt: "does the ΔPPL = 0% hold on the frontier model?"                                                                                              |

────────────────────────────────────────────────────────────────────────────

## 4. Residual-stream attribution across all blocks — TIER 1, **highly recommended**

Track the candidate row through every block: `block_residual_in`,
`block_residual_out`, `block_delta`, `mlp_out_at_row`, `argmax_at_pos`.

| Field           | Value                                                                                                                                                                                     |
|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Evo1 script     | [scripts/analysis/run_evo1_residual_attribution_fp32.py](scripts/analysis/run_evo1_residual_attribution_fp32.py)                                                                          |
| Evo1 output     | [results/sw_residual_attribution_evo1_fp32.json](results/sw_residual_attribution_evo1_fp32.json) (bf16 — fp32 forbidden by `flash_attn`)                                                  |
| Evo2 output     | `results/sw_residual_attribution_evo2.json`                                                                                                                                              |
| Status on Evo1  | DONE — row 3776 enters at L10 (+165), L11/12 write 1.2 M / 1.6 × 10⁷, L13 onward MLP@row ≈ 0 and residual freezes at ≈ 3 × 10⁷, per-token argmax shifts away to other rows.              |
| Runtime         | Evo1: ~5 min (one forward + 32 block hooks).  Evo2-7B: ~15 min.                                                                                                                          |
| Interpretation  | The saturated row is a DC offset normalised away by the pre-unembed RMSNorm. **Evo2 should be checked to confirm this is generic StripedHyena behaviour vs. an Evo1-specific quirk.**     |
| Important       | Force the script into bf16 except for the SH `poles` / `residues` (kept fp32) — same dtype handling already in the fp32 script's `dtype` argument. flash-attn refuses pure fp32.            |

This output feeds the **bottom sub-panel of Figure 2 F**. Once the Evo2
JSON lands you can re-run [scripts/analysis/plot_figure2_panels_EF.py](scripts/analysis/plot_figure2_panels_EF.py)
with a third sub-panel for Evo2 (or replace the Evo1 sub-panel if the
paper claims are scoped to the frontier model).

────────────────────────────────────────────────────────────────────────────

## 5. (Optional, TIER 2) per-row ablation × held-out PPL fan

Run the ablation 10 times, zeroing out one detected row at a time, to
confirm none individually flip PPL by more than 1 %.

| Field           | Value                                                                                                              |
|-----------------|--------------------------------------------------------------------------------------------------------------------|
| Evo1 script     | (small modification of `run_evo1_detect_ablate.py` — loop over `[row]` instead of all rows)                        |
| Evo2 output     | `results/evo2_per_row_ablation.json`                                                                               |
| Runtime         | Evo1: ~30 min (11 forwards). Evo2-7B: ~90 min.                                                                     |
| Use             | Pre-empts "maybe one row was the load-bearing one and was rescued by another"; not in any figure but strong appendix.|

────────────────────────────────────────────────────────────────────────────

## Environment notes (for the Evo2 runner)

- `transformers >= 4.41`, `flash-attn >= 2.5.7`, the `stripedhyena` package
  that ships with the Evo2 release (the local Evo1 env is at
  `~/miniconda3/envs/evo` for reference).
- `HF_TOKEN` must be set; the Evo2 weights are gated.
- For attribution, set `torch.set_float32_matmul_precision("medium")` and
  cast weights with `model.to(torch.bfloat16)` except `poles` / `residues`.
- All scripts above accept `--out PATH` and `--device cuda:0` and write a
  single self-describing JSON.

────────────────────────────────────────────────────────────────────────────

## Mapping to manuscript claims

| Claim (current draft)                                                              | Evidence from Evo1 (have)                                                                  | Replication on Evo2 (needed)                                  |
|------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| StripedHyena SWs are detectable in activation                                      | activation_lifecycle_evo1.json                                                             | activation_lifecycle_evo2.json                                |
| StripedHyena gated-MLP cold weights do not host load-bearing rows                  | sw_mechanistic_evo1.json (ranks 48–168)                                                    | sw_mechanistic_evo2.json                                       |
| Same-protocol ablation of detected rows gives ΔPPL ≈ 0%                            | evo1_detect_ablate.json                                                                    | evo2_detect_ablate.json                                       |
| Candidate rows are *preserved-then-frozen* in the residual stream, not dispersed  | sw_residual_attribution_evo1_fp32.json                                                     | sw_residual_attribution_evo2.json                              |

If all four Evo2 outputs replicate the Evo1 pattern, the manuscript's
cross-architecture story stands as-is; if any diverge (especially #4), the
"DC offset normalised away" interpretation needs to be revised to the
narrative supported by the new Evo2 trace.
