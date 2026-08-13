# GENERator PROK super-weight contamination — blast-radius audit

**Date:** 2026-08-07 · **Phase C1 of Task Spec v2** · Status: index corrected, downstream artifacts catalogued below.

---

## 1. What happened

`scripts/detection/run_detection_generator_prokaryote.sbatch` was invoked with
`--probe human_promoter` — a **human ACTB sequence, i.e. eukaryotic** — against the
**prokaryote-specialised** GENERator 3B model. The resulting index entry:

> `generator_prokaryote: layer 2, row 1927, col 1769, out_max = 506,014`

is an artifact of feeding out-of-distribution input to the model.

**Evidence it is invalid**
- Row 1927 does not reproduce on **any** of the 4 canonical probes — all give `|row 1927| < 1.0` at layer 2.
- A fresh detection sweep with a **matching prokaryotic probe** (pseudomonadota) does not recover row 1927 anywhere in its top-10.
- `‖U_k‖_F` rank of row 1927 at layer 2 is **1289/3072** — statistical noise, not a structural outlier.

**The real super weight**

> `layer 8, row 260, col 4484, out_max = 30,167.07` (secondary: layer 8, row 1325)

- Primary row of the pseudomonadota sweep.
- Activation **identical across all 4 canonical probes** — content-invariant, the Yu et al. super-activation signature.
- `‖U_k‖_F` rank **1/3072**, matching GENERator EUK's row 2371 (also rank 1/3072).

Reference: `results/mechanism/generator_prokaryote_sw_corrected.json`.

---

## 2. Correction applied ✅

- `results/super_weight_index.json` → `generator_prokaryote` now holds **L8/r260** (+ secondary r1325), with a `provenance` block recording the old entry, why it was invalid, why the new one is right, and pointers to the corrected detection and this audit.
- Old entry archived verbatim → **`results/super_weight_index_generator_prokaryote_ORIGINAL_SUSPECT.json`**.
- Note the layer also moves: **L2 → L8**. Anything that hard-codes `layer 2` for PROK is wrong independently of the row index.

## 3. ⚠️ NEW — the 1B prokaryote sibling is also suspect

`generator_prokaryote_1b` = `layer 2, row 1397`, `source: detection_run_sbatch` — **the same pipeline and the same layer-2 signature** as the entry just proven contaminated. It has **not** been re-detected with a prokaryotic probe.

Flagged in the index as `provenance_warning.status = "SUSPECT - UNVERIFIED"`. **Do not use or cite it until re-detected.** Not corrected here because no corrected detection exists for it. This model does not appear in the current manuscript, so the fix is cheap if it is ever needed.

---

## 4. Blast radius

### 4a. Derived result files — ✅ ALL RERUN on L8/r260 (Phase D1, 2026-08-07)

All nine were recomputed on the corrected super weight; contaminated originals preserved in place
as `*_ORIGINAL_SUSPECT.{json,png}`. Findings in `D1_PROK_RERUN_REPORT.md`.

| File | Status | Outcome of the rerun |
|---|---|---|
| `results/prokaryote/sw_causal_tracing.json` | ✅ RERUN | `+9.47` → **`+1.25 ± 0.54`** log-PPL (7.6× inflated by contamination; still p ≈ 0 vs random rows) |
| `results/prokaryote/sw_hexamer_causal.json` | ✅ RERUN | 🔴 `r = −0.710` → **Spearman ρ = +0.0007, p = 0.96** — the kingdom sign-reversal is dead |
| `results/prokaryote/sw_shuffle_controls.json` | ✅ RERUN | 3 of 4 shuffles now n.s. (were all significant); the one "significant" effect is 4.7e-9 relative |
| `results/prokaryote/sw_kmer_scan.json` | ✅ RERUN | Activation flat at 621 ± 2 for 4094/4096 hexamers; GC corr −0.045 |
| `results/prokaryote/sw_token_omission.json` | ✅ RERUN | — |
| `results/prokaryote/sw_grad_attribution.json` | ✅ RERUN | — |
| `results/prokaryote/sw_kmer_motifs.json` | ✅ RERUN | **No enrichment: all q = 1.000** |
| `results/prokaryote/sw_ablation_regression.json` | ✅ RERUN | r(GC, ΔPPL) −0.520 → **−0.661**, R² 0.35 → **0.50** — this contrast *strengthened* |
| `results/prokaryote/attribution_comparison.json` | ✅ RERUN | Methods now disagree (mean ρ = −0.12, 0% above 0.3) |
| all matching `results/prokaryote/*.png` | ✅ REGENERATED | — |

**⚠️ Missed on the first pass — found 2026-08-07 during a results-directory audit:**

| File | Status | Issue |
|---|---|---|
| `results/cross_kingdom_transfer.json` | ✅ **RERUN — claim survives, numbers change** | Lives at `results/` top level, not `results/prokaryote/`, so the original directory-scoped grep missed it. Its second entry was `generator_prokaryote` at **L2/r1927**, backing `main.tex:494–495`. Rerun on L8/r260: **484.9** (E. coli) vs **484.7** (hg38), fold **0.9998**, MW **p = 0.768** — versus the contaminated 10,638 / 10,710 / 1.0068 / p = 0.644. The conclusion *"neither row distinguishes eukaryotic from prokaryotic input"* **holds**; only the magnitudes change. EUK half was always fine (L4/r2371, fold 1.0044, p = 0.538). Original archived as `cross_kingdom_transfer_ORIGINAL_SUSPECT.{json,png}`. |

⚠️ **Interpretive caveat for the manuscript:** this sentence now carries very different weight per
kingdom. For **EUK** it is a *meaningful* negative — that row genuinely varies with sequence
(Spearman ρ(act, KL) = +0.97) and still fails to separate kingdoms. For **corrected PROK** it is
*trivially* true, because the row is content-invariant and does not vary with anything. Do not
present them as two equivalent pieces of evidence.

Lesson: the contamination was scoped by *directory*, but contaminated PROK entries also live in
top-level multi-model result files. Any file containing a `generator_prokaryote` entry must be
checked individually, not assumed clean because it sits outside `results/prokaryote/`.

Already superseded in the prior session (archived, no action): `activation_lifecycle_generator_prokaryote`, `sw_mechanistic_generator_prokaryote`, `fig2_generator_lifecycle_prok`.

### 4b. Scripts that hard-code the contaminated coordinates

| File | Location | Action |
|---|---|---|
| `run_prokaryote_interpretability.sh` | lines 2, 43 | Update to `layer=8 row=260` — this driver produces all of §4a |
| `scripts/analysis/run_sw_mechanistic.py` | line 61 | `{"layer": 2, "rows": [1927]}` → `{"layer": 8, "rows": [260, 1325]}` |
| `scripts/analysis/run_activation_lifecycle.py` | line 55 | same |
| `scripts/analysis/plot_figure1.py` | line 191 | `sw_layer: 2, sw_rows: [1927]` → `8 / [260]` |
| `scripts/analysis/plot_architecture_schematic.py` | line 33 | `sw_rows: [1927]` → `[260]` |
| `scripts/interpretability/run_sw_activation_heatmap.py` | line 37 | docstring example |
| `scripts/interpretability/run_sw_task_activation_heatmap.py` | line 33 | docstring example |
| `scripts/interpretability/run_sw_activation_heatmap_prok.sbatch` | line 39 | `--sw_rows 1927` → `260` (+ layer) |

Benign — these *describe* the contamination and should keep the old number: `scripts/analysis/build_fig2_generator_panels.py`, `run_channel_survival.py`, `run_subspace_retention.py`.

### 4c. Manuscript items that must change

| Item | Where | Required action |
|---|---|---|
| **Fig 1 C/D** | `plot_figure1.py:191` | Regenerate — PROK panel uses L2/r1927 |
| **Fig 2 B/D** | `build_fig2_generator_panels.py` | Regenerate from corrected lifecycle data (already produced) |
| **Fig 3, all PROK panels** | `run_prokaryote_interpretability.sh` outputs | Regenerate after §4a reruns |
| **`r = −0.710` kingdom claim** | `main.tex:95`, `main.tex:434` | 🔴 **Rerun or cut.** Derived from `sw_hexamer_causal.json` on the contaminated row. The entire kingdom sign-reversal argument rests on this number. |
| **Shuffle controls** | `main.tex` PROK results | Rerun on L8/r260 |
| **PROK SAE** | `main.tex:447–465`, `:752–756`, `:925` | 🔴 **Invalid — retrain or cut.** `main.tex:925` states it was trained "on GENERator PROK at **layer 2**", i.e. on the contaminated layer's residual stream. Also: **no SAE checkpoint exists anywhere under `results/`**, so the artifact is not reproducible from this repo as it stands. |
| **`‖U_k‖_F = 2,648` for PROK row 1927** | `main.tex:265` | Replace — wrong row. Corrected row 260 is rank 1/3072 (see `uk_precision_recall.csv`). |
| **"comparable phenotype at layer 2, row 1,927"** | `main.tex:177` | Replace with layer 8, row 260 |
| **PROK ablation `+9.47` log-PPL** | `main.tex:399–402` | Rerun — from the contaminated causal tracing |
| **Layer-confound limitation** | Limitations, `main.tex:783+` | The EUK-vs-PROK layer comparison is now **L4 vs L8**, not L4 vs L2. Note: the six current limitations do **not** actually contain an "L4-vs-L2" sentence, so this is an item to *add*, not edit. |
| README PROK section | `README.md:138, 760, 763, 781, 804, 854` | Update all — incl. the "inverse GC preference" and `r = −0.710` narrative |
| `paper/main_old.tex` | lines 276, 311, 343, 372, 434 | Superseded draft — no action, do not resurrect |

---

## 5. What is *not* affected

- **GENERator EUK** (L4/r2371) — detected with a matching eukaryotic probe; rank 1/3072; unaffected.
- **DNABERT-2, NTv3, Evo1** — separate detections, unaffected.
- **Tier-1/Phase-A results** — `channel_survival_generator_prokaryote.json` and
  `subspace_retention_generator_prokaryote.json` already used the **corrected** L8/r260.
- The **splice dissociation** (DNABERT-2 / NTv3) — no PROK dependency.

## 6. Scope note

The contamination is confined to GENERator PROK (plus the unverified 1B sibling). But it lands on the
**kingdom sign-reversal** claim and the **PROK SAE** — both load-bearing for the manuscript's PROK
narrative.

**✅ Phase D1 has now been run (2026-08-07), and it confirms the cut for both.**

1. **`r = −0.710` sign reversal → CUT.** On the correct row, Spearman ρ = **+0.0007 (p = 0.96)**.
   There is no correlation, hence no sign to reverse. The naive Pearson reads +0.958, but that is
   carried entirely by 2 hexamers out of 4096 — do not substitute one artifact for another.
2. **PROK SAE → CUT.** Beyond being trained on the contaminated layer with no surviving checkpoint,
   retraining on layer 8 is pointless: the corrected row's activation is flat (621 ± 2 across all
   4096 hexamers), so there is no variance for a sparse autoencoder to decompose and no
   "compositional axis" in the activation for it to recover.

What *did* survive is a different quantity: the GC-dependence of ablation **cost**
(PROK r = −0.66 vs EUK −0.001), which strengthened under correction. Hypothesis-level only —
n = 2 models, with a corpus-GC confound.

Full evidence and corrected numbers: **`D1_PROK_RERUN_REPORT.md`**.
