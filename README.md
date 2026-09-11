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
| **What we withdrew, and why** | [Retractions and rescopes](#retractions-and-rescopes) — read before citing any earlier number |

Two properties of this repository are worth stating up front, because they are unusual and
they are deliberate.

**Every plotted number is loaded from a raw artifact at render time.** None is transcribed
from prose. Where a narratively interesting number had no raw artifact, the panel was not
rendered rather than approximated — `FIGURE_PROVENANCE.md` lists those cases.

**The reproduction scripts gate themselves against stored values.** Each re-derives quantities
already recorded in `audit/census_master.csv` — baseline loss, candidate effect at both
intervention strengths, the seeded control rows — and *aborts* on mismatch, on the principle
that a failed gate means the harness is wrong rather than that the science changed. Observed
reproduction errors are 1e-9 to 1e-6 relative.

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
## Retractions and rescopes


1. **NTv3 splice results were produced on 20%-truncated inputs.** `_MAX_LEN["reconstructed"]
   = 80` is tuned for DNABERT-2's BPE (400 bp → 86 tokens). NTv3 is *nucleotide-level*
   (400 bp → 400 tokens), so 80 truncated every splice window to its first 80 bp — the
   junction was never seen, and all 5 seeds sat at the 0.5658 majority-class floor.
   **With the fix, NTv3 reaches MCC 0.86–0.91 — and its SW ablation effect disappears
   (−0.02 pp).** The previously reported ΔMCC = −0.119 ± 0.054 (p = 0.0083) is an artifact
   of the truncated model and is **withdrawn**. Functional replication of the ensemble
   effect is therefore **n = 1 (DNABERT-2)**; structural replication is n = 2.
2. **"Shadow redundancy" is a layer-depth artifact.** `prox_far` prunes all 768 rows of
   layer 0; `layer_matched_random` (no SW information) reproduces it on all three tasks
   (promoter −9.09 vs −10.04; histone −2.87 vs −2.74; splice −34.87 vs −34.88).
3. **SW quantisation-exemption experiments test a no-op by construction** — per-row RTN sets
   `s = max|w|/qmax`, and the SW *is* that max.
4. **"Histone-mark prediction intact" is false** — the same k=5 cliff fires in 2/5 seeds.
5. **Composition claim rescoped** — R² = 0.373 (GC alone 0.035); 2/12 motifs survive a
   GC-matched null, so "no canonical motifs are enriched" is also false.
6. **PROK SAE withdrawn** — layer-2 contamination, an fp16 clamp destroying 98% of
   SW-channel variance, an unnormalised objective, and degenerate `n_active ≈ 1`
   correlations.
7. **Norm dominance does not predict criticality** — NTv3's SW is rank 1/1536 with a 29.4×
   gap (more dominant than DNABERT-2's) and is functionally inert. The joint-norm-carriage
   mechanism explains DNABERT-2 and does **not** generalise.

8. **"Activation extremeness *calibrates* causal severity" is withdrawn** (Sep 2026). The
   between-model ρ = +0.766 is real but must be read as *"models with a more extreme top row
   have a more damaging top row"* — **not** as a dose-response law. Two within-model graded
   sweeps (36 rows inside one layer, one genomic and one text decoder) show the relation is
   **two-regime**: below the detector's ≥5 accept rule the ratio carries no positive graded
   signal (ρ=+0.040, p=0.86 genomic; ρ=−0.584, p=0.005 text), and although it *orders* rows
   well above the threshold, magnitudes do not follow — in SmolLM2-1.7B layer 7 the row
   ranked **4th** by ratio is **130× more damaging** than the row ranked **2nd**. Supported:
   the ratio **detects** and **orders**. Withdrawn: smooth severity calibration.
   Evidence: `results/analyses/within_layer_sweep/within_model_slope_comparison.json`.

9. **The one-candidate-per-model census design undercounts critical rows** (Sep 2026, scope
   limit rather than a retraction). Sweeping 36 rows instead of 1 found a **second**
   independently catastrophic row in SmolLM2-1.7B layer 7 (r161, +287 %) that appears in no
   census artifact, plus a **masking** interaction: ablating r749 — which costs +2.2 % alone
   — *abolishes* r161's catastrophe (joint +2.07 %). Independently re-verified with an
   alternative implementation, weight drift 0.00e+00. No mechanism is claimed; the link to
   the rows' extreme geometric alignment (cos = +0.3952, z = +17.6, the maximum of 19,900
   layer pairs) is **[UNTESTED]**. Counts of "the super row" per model should be read as
   *"the census's single frozen candidate"*, not as a complete inventory.

10. **The ratio-argmax rule is a detector, not a definition** (Sep 2026). EXP2 closed
    provenance for all 22 census rows, and in doing so falsified the reading that the
    layer-relative-ratio argmax identifies the most causally important coordinate. In
    OLMo-7B the ratio prefers L2/r269 while the census's L1/r269 is **47× more damaging**
    (+1.1178 vs +0.0237); in SmolLM2-1.7B layer 7 the row ranked 4th by ratio is **130×
    more damaging** than the row ranked 2nd; and DNABERT-2 runs the *opposite* way, so the
    rule is **neither uniformly better nor worse** than the legacy absolute-activation rule.
    Specify "global argmax of the layer-relative ratio, accept if ≥5" as the **candidate
    detector** — reproducible and worth stating — and not as a claim about causal importance.
    Evidence: `results/analyses/detector_provenance/EXP2_LEGACY_DETECTOR_RESOLUTION.md`.

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

## Reproducing

Prefer the pipelines. Each skips stages whose output already exists, so an interrupted run
resumes and a finished one costs nothing; `FORCE=1` recomputes.

```bash
bash pipelines/census.sh                 # the preregistered 22-model causal census (restartable)
bash pipelines/census.sh dnabert2 ntv3   # ...or a subset
bash pipelines/detector_provenance.sh    # selection-rule resolution; runs its positive control first
bash pipelines/fig5_within_layer.sh      # the within-layer sweeps, second critical row, Figure 5
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

| path | contents |
|---|---|
| `experiments/E1`–`E13` | The **preregistered harnesses**, one directory per experiment line, each with the prereg it was run under. Not edited after the fact to make later results come out differently. |
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
| `stubs/` | Import shims so models with heavy optional dependencies load without them (`mamba_ssm`, a HybriDNA config). Not type stubs, despite the name. |
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
4. **Superseded experiment lines were removed from the working tree**, not merely unreported:
   E2 (Evo1 broadcast), E4 (quantisation granularity), E6 (cross-geometry), the
   sparse-autoencoder package, and the exploratory interpretability trees. None is cited by the
   figure-provenance manifest or the experiment map, and their preregistrations are retained in
   `docs/prereg/` with an explicit disposition rather than deleted (see
   [`docs/prereg/README.md`](docs/prereg/README.md)). `E5_dimensionality/` retains one file, a
   library three surviving audit builders import. All removed code remains in git history.
5. **The preregistrations record what was planned**, including paths that predate later
   directory renames. They are content-locked and were deliberately never rewritten to match.

## Citation

See [`CITATION.cff`](CITATION.cff). Released under the MIT License ([`LICENSE`](LICENSE)).

Prior work this builds on: Yu M, Wang D, Shan Q, Reed CJ, Wan A. *The Super Weight in Large
Language Models.* [arXiv:2411.07191](https://arxiv.org/abs/2411.07191) (2024).
