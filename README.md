# Structure is not mechanism

### High-gain gated-FFN rows across text and genomic foundation models

Code, artifacts and preregistrations for the paper of this title. The manuscript itself is not
in this repository; it ships with the publication.

**The question.** A handful of "super weights" in text LLMs can be catastrophically important.
Do analogous structures recur in genomic foundation models — and, more sharply, does a
high-gain row's *structural* prominence tell you anything about its *causal* importance?

**The answer**, across a preregistered 22-model causal census and focused mechanistic case
studies, is that three quantities routinely conflated in this literature are distinct:

```
STRUCTURAL GEOMETRY  ≠  FUNCTIONAL CRITICALITY  ≠  CAUSAL RESPONSE COMPLEXITY
```

---

## Start here

| | |
|---|---|
| **Every claim → its code → its artifacts** | [`docs/EXPERIMENT_MAP.md`](docs/EXPERIMENT_MAP.md) — 21 experiments, machine-checked |
| **Reproduce anything** | [`pipelines/`](pipelines/) — idempotent, GPU-parallel runners |
| **Panel-by-panel figure provenance** | [`experiments/figures/FIGURE_PROVENANCE.md`](experiments/figures/FIGURE_PROVENANCE.md) |
| **Supplementary tables S1–S8** | [`results/supplementary/`](results/supplementary/) |
| **Preregistrations (12, content-locked)** | [`docs/prereg/`](docs/prereg/) — `python src/prereg_lock.py verify --all` |

## What the paper establishes

| finding | key numbers |
|---|---|
| High-gain gated-FFN rows are a **recurrent structural phenotype** across text and genomic models | 23-model structural panel; near-rank-1 candidates in both encoders and decoders |
| Activation-selected candidates are **functionally enriched** over same-layer controls | candidate beats the within-model control median in 18/22 models at ε=0.5 and 20/22 at ε=1.0 |
| That enrichment survives a **stricter, purely weight-space control** | vs the 5 highest-Frobenius same-layer rows: candidate retains its advantage in 20/22 (ε=0.5) and 19/22 (ε=1.0); those high-norm rows are themselves near-inert (+0.035%, +0.110%) |
| **Structural geometry does not predict effect size** — the central negative result | full-ablation effects span −0.79% to +698.87%; q1 vs signed effect ρ = 0.074 (95% CI −0.391 to 0.513); operator magnitude likewise fails |
| The activation ratio is **two-regime**, replicated in a genomic and a text decoder | below the ≥5 accept threshold ρ = +0.040 (p=0.86) genomic, **−0.584** (p=0.005) text; above it ρ = +0.639 and **+0.975**. n=36 rows/model |
| **Ratio rank ≠ causal importance** | SmolLM2-1.7B L7: the row ranked 4th by ratio is **130×** more damaging than the row ranked 2nd. OLMo-7B: the ratio-argmax coordinate is **47×** *less* damaging than the published one |
| Causal organization **differs by model**, and interaction sign is not fixed | DNABERT-2: two near-inert rows, joint epistasis **+2.0118**, held-out R² 0.888 (F3) vs 0.517 (F2). SmolLM2 L7: super-additive, sub-additive, and near-total **masking** all within one layer |
| GENERator's effect is **position-localized at BOS**, and not direction-specific | restoring the BOS contribution rescues essentially all damage; a damage-matched random direction reproduces the GC phenotype |
| Detector provenance is **closed for all 22 census rows** | 16 by the current rule; Llama-7B and Mistral-7B verified at global rank 1/131,072; OLMo-7B, DNABERT-2 and NTv3 characterised disagreements |
## Installation

```bash
git clone https://github.com/Georgakopoulos-Soares-lab/superweights.git
cd superweights
pip install -r requirements.txt
export PYTHONPATH=.          # required: scripts import the shared library as `src.*`
```

All commands below assume `PYTHONPATH=.` and the repository root as the working directory.

Three environments are needed, because the model families pin incompatible `transformers`:

| environment | `transformers` | used for |
|---|---|---|
| main | ≥ 4.40 (tested 5.5.0) | GENERator, NTv3, all text decoders, the census, every analysis |
| `dnabert` | 4.29.2 | DNABERT-2 — its remote code predates the Transformers-5 config API |
| `evo` | 4.48.1 | Evo1 |

Running DNABERT-2 under the main environment fails with
`BertConfig has no attribute pad_token_id`. That is the wrong-environment signature, not a bug.

## Data

| input | status |
|---|---|
| **WikiText-2** (text endpoints) | **Committed** at [`frozen_inputs/`](frozen_inputs/), sha256 `5f1bea06…`. The same file the census resolved through `datasets`, so text endpoints run offline. |
| **hg38** (genomic endpoints) | Not committed. Place or symlink at `data/reference/hg38/hg38.fa`. |
| **GUE** (downstream tasks) | Not committed. Point `$GUE_ROOT` at the extracted tree. |
| **Checkpoints** | Pinned Hugging Face revisions, one per model, in `audit/census_master.csv`. |

Other large public inputs are documented rather than vendored — see [`data/README.md`](data/README.md).

## Reproducing the paper

One row per object in the paper. Every artifact path below was verified to exist in this
repository; where an input is gitignored and therefore absent from a clone, the row says so and
names the command that regenerates it.

**A note on filenames.** Moving the within-layer section ahead of the DNABERT-2 section shifted
the figure numbers after it by one, but the *script filenames* were not renamed — renaming them
risks breaking the literal-path `sys.path` inserts this repository depends on (see
`tests/test_entrypoints.py`). So `fig3_dnabert.py` renders **Figure 4** and `fig4_generator.py`
renders **Figure 5**. The offset is intentional and is flagged on each affected row.

### Main figures

| paper object | command | artifact | notes |
|---|---|---|---|
| **Figure 1** — structural panel | `python experiments/figures/fig1_structural.py` | `experiments/figures/main/fig1_structural.{pdf,png}` | Panel A is the retrospective diagonal-proxy calibration; B and C the exact operator metrics |
| **Figure 2** — 22-model causal census | `python experiments/figures/fig2_causal.py` | `experiments/figures/main/fig2_causal.{pdf,png}` | Panels A/B effects, C the q₁ dissociation, D the top-norm control |
| **Figure 3** — within-layer sweep | `bash pipelines/fig5_within_layer.sh` | `results/analyses/within_layer_sweep/fig_within_model_slope_2panel.{pdf,png}` | ⚠ pipeline filename retains the earlier figure number |
| **Figure 4** — DNABERT-2 multi-row causal response | `python experiments/figures/fig3_dnabert.py` | `experiments/figures/main/fig3_dnabert.{pdf,png}` | ⚠ **script filename says `fig3`** |
| **Figure 5** — GENERator BOS mediation | `python experiments/figures/fig4_generator.py` | `experiments/figures/main/fig4_generator.{pdf,png}` | ⚠ **script filename says `fig4`**; companion panel from `fig4_generator_specificity.py` |

### Supplementary figures

| paper object | command | artifact |
|---|---|---|
| **Fig S1** — structural/control detail | `python experiments/figures/supplement/fig_s1_structural_detail.py` | `experiments/figures/supplement/fig_s1_structural_detail.{pdf,png}` |
| **Fig S2** — damage-matched random direction | `python experiments/figures/supplement/fig_s2_random_direction.py` | `experiments/figures/supplement/fig_s2_random_direction.{pdf,png}` |

### Supplementary tables

S2, S3 and S6–S8 are generated; S1, S4 and S5 are reformatted verbatim from the audit and are
not rebuilt. One command produces all five generated tables.

```bash
python scripts/census_analysis/build_supplementary_tables.py
```

| table | artifact | rows |
|---|---|---|
| **S1** — model panel and provenance | `results/supplementary/S1_model_panel_provenance.{tsv,md,csv}` | 22 |
| **S2** — structural metrics + 24-input stability | `results/supplementary/S2_structural_metrics_stability.{tsv,md}` | 22 |
| **S3** — causal census + disagreement coordinates | `results/supplementary/S3_causal_census_disagreements.{tsv,md}` | 49 |
| **S4** — structure–function correlations | `results/supplementary/S4_structure_function_correlations.{tsv,md,csv}` | 16 |
| **S5** — GENERator conditions | `results/supplementary/S5_generator_conditions.{tsv,md,csv}` | 106 |
| **S6** — DNABERT-2 tomography | `results/supplementary/S6_dnabert2_tomography.{tsv,md}` | 98 |
| **S7** — within-layer sweep | `results/supplementary/S7_within_layer_sweep.{tsv,md}` | 72 |
| **S8** — two critical rows + geometry | `results/supplementary/S8_two_critical_rows.{tsv,md}` | 13 |

### Results in the text, not in a figure

| claim | command | artifact |
|---|---|---|
| Two critical rows in one layer, and the masking interaction | `python scripts/within_layer_sweep/run_smollm2_second_row_epistasis.py` | `results/analyses/within_layer_sweep/smollm2_second_row_epistasis.json` |
| Row geometry against a same-layer null (data-free, no GPU) | `python scripts/within_layer_sweep/run_smollm2_row_geometry.py` | `results/analyses/within_layer_sweep/smollm2_161_749_geometry.json` |
| Selection-rule resolution for every legacy candidate | `bash pipelines/detector_provenance.sh` | `results/analyses/detector_provenance/EXP2_LEGACY_DETECTOR_RESOLUTION.md` |
| Cohort structure–function correlations | `python scripts/census_analysis/run_activation_vs_causality.py` | `results/analyses/census_analysis/activation_vs_causality.tsv` |

The authoritative, machine-checked mapping is [`docs/EXPERIMENT_MAP.md`](docs/EXPERIMENT_MAP.md)
(21 experiments) and, panel by panel,
[`experiments/figures/FIGURE_PROVENANCE.md`](experiments/figures/FIGURE_PROVENANCE.md). This
table surfaces them; it does not replace them. `python scripts/build_experiment_map.py`
re-verifies that every script and artifact named there still exists.

### Running the pipelines

Prefer the pipelines. Each skips stages whose output already exists, so an interrupted run
resumes and a finished one costs nothing; `FORCE=1` recomputes.

```bash
bash pipelines/census.sh                 # the preregistered 22-model causal census (restartable)
bash pipelines/census.sh dnabert2 ntv3   # ...or a subset
bash pipelines/detector_provenance.sh    # selection-rule resolution; runs its positive control first
bash pipelines/fig5_within_layer.sh      # the within-layer sweeps and second critical row
                                         # (filename retains an earlier figure number)
```

Individual stages:

```bash
# structural panel and causal census
python experiments/E11_scale_ladder/run_model.py --model <slug>
python experiments/E13_full_cohort_causal_census/run_singleton_census.py --model <slug>

# mechanism case studies
python experiments/E9_mechanistic_tomography/run_fit_observers.py
python experiments/E12_generator_degradation_control/run_bos_mediation_main.py

# within-layer sweeps
python scripts/within_layer_sweep/run_within_model_slope_text.py     # SmolLM2-1.7B, ~7 min, 1 GPU
python scripts/within_layer_sweep/run_smollm2_row_geometry.py        # data-free, no GPU

# supplementary tables S2, S3, S6-S8 (rebuilt from committed artifacts)
python scripts/census_analysis/build_supplementary_tables.py

# two flags that are REQUIRED and silently wrong if omitted
python scripts/detection/run_detection.py --model ntv3 --pad_to_multiple 256   # stride-256 U-Net
python scripts/evaluation/run_gue_multiseed.py --model ntv3 --max_length 400   # nucleotide tokenizer
```

Verify the repository's own invariants:

```bash
python scripts/build_experiment_map.py    # every claim's code and artifacts exist
python src/prereg_lock.py verify --all    # 12 preregistrations, content-locked
pytest tests/                             # unit tests + every CLI entry point starts
```

## Repository map

**`experiments/` contains the preregistered harnesses** (E1–E13, each with the preregistration
it was run under in [`docs/prereg/`](docs/prereg/)); **`scripts/` contains analyses developed
after preregistration.** The split is deliberate: it lets a reader see at a glance which
analyses were locked in advance and which came later.

| path | contents |
|---|---|
| `experiments/E1`–`E13` | The **preregistered harnesses**, one directory per experiment line. Not edited after the fact to make later results come out differently. `E5_dimensionality/` is a cut line whose library is still imported by three audit builders — see [its README](experiments/E5_dimensionality/README.md). |
| `experiments/figures/` | Figure render scripts, output, `source_data/` (the exact plotted values), and `FIGURE_PROVENANCE.md` |
| `scripts/detection/` | Super-row detection and selection-rule resolution |
| `scripts/mechanism/` | BOS mediation, attention sink, special-token dependence, matched replacements |
| `scripts/within_layer_sweep/` | The graded within-layer sweeps, the second critical row, Figure 5 |
| `scripts/census_analysis/` | Cohort structure–function correlations; the supplementary-table builder |
| `scripts/negative_results/` | Experiments whose only result is a null — see its [README](scripts/negative_results/README.md) |
| `scripts/evaluation/` | GUE downstream evaluation |
| `pipelines/` | Reproduction runners |
| `src/` | Shared library: activation capture, ablation, spike detection, DNA probes, the prereg lock |
| `models/`, `configs/` | Per-model wrappers and YAML, loaded dynamically via `WRAPPER_MAP` |
| `stubs/` | **Import shims, not PEP 484 type stubs** — they let wrappers for models outside the paper's panel import without heavy optional dependencies. See [`stubs/README.md`](stubs/README.md). |
| `audit/` | The verification record: `census_master.csv` (22 models × 44 columns) is the canonical table. Three adversarial passes; see its [README](audit/README.md). |
| `results/` | Artifacts in four groups — `experiments/` (per harness), `analyses/` (derived), `negative_results/`, `supplementary/` (S1–S8). Gitignored by default; files backing a reported number are force-added. See its [README](results/README.md). |
| `docs/` | The experiment map and the 12 locked preregistrations |
| `frozen_inputs/`, `data/`, `tests/` | Content-hashed inputs, small reference files, tests |

## Provenance and known limitations

Stated plainly, because a reader will find them anyway.

1. **Some raw artifacts are not committed.** Several were produced on a cluster and never
   transferred. For every affected item the committed figure source-data snapshot under
   `experiments/figures/source_data/` carries the plotted values, so each reported number
   remains inspectable.
2. **NTv3's original weight revision was not recorded** and is not recoverable. Resolved Hub
   commits exist for 21 of 22 models.
3. **One n=10 subgroup is fragile.** The text-decoder correlation moves from ρ=0.770 to 0.673
   under an OLMo coordinate substitution, with a 95% lower bound of +0.032. It should not be
   read as independently robust; the 22-model result is insensitive to the substitution.
4. **Experiment lines that the paper does not report are not in the working tree.** Their
   preregistrations are retained in `docs/prereg/`, each with an explicit disposition, rather
   than deleted — see [`docs/prereg/README.md`](docs/prereg/README.md). `E5_dimensionality/`
   keeps one file, a library three audit builders import.
5. **The preregistrations record what was planned**, including paths that predate later
   directory renames. They are content-locked and were deliberately never rewritten to match.

## Citation

See [`CITATION.cff`](CITATION.cff). Released under the MIT License ([`LICENSE`](LICENSE)).

Prior work this builds on: Yu M, Wang D, Shan Q, Reed CJ, Wan A. *The Super Weight in Large
Language Models.* [arXiv:2411.07191](https://arxiv.org/abs/2411.07191) (2024).
