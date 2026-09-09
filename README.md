# Structure is not mechanism

### Code and data for *"High-gain gated-FFN rows across text and genomic foundation models"*

This repository is the computational companion to the paper. It contains the code that
produced every reported number, the artifacts those runs emitted, the figure pipeline, and
the preregistrations — but **not the manuscript text**, which lives with the publication.

The project asks whether the "super weight" phenomenon reported in text LLMs recurs in
genomic foundation models, and — the sharper question — whether a high-gain row's
**structural** prominence tells you anything about its **causal** importance. Across a frozen
22-model census and focused mechanistic case studies, the answer is that structural geometry,
functional criticality, and causal response complexity are three different things.

```
STRUCTURAL GEOMETRY  !=  FUNCTIONAL CRITICALITY  !=  CAUSAL RESPONSE COMPLEXITY
```

---

## Start here

| what | where |
|---|---|
| **Every experiment → its code → its artifacts** | [`docs/EXPERIMENT_MAP.md`](docs/EXPERIMENT_MAP.md) — 20 experiments |
| **Verify that map against this tree** | `python scripts/build_experiment_map.py` — exits non-zero if any script is missing |
| **Panel-by-panel figure provenance** | [`experiments/figures/FIGURE_PROVENANCE.md`](experiments/figures/FIGURE_PROVENANCE.md) |
| **Reproduce anything** | [`pipelines/`](pipelines/) — idempotent runners, GPU-parallel |
| **Supplementary tables S1–S5** | [`audit/round2/tables/`](audit/round2/tables/) |
| **Supplementary tables S6–S7** | [`results/paper_closing/supplementary/`](results/paper_closing/supplementary/) |
| **Preregistrations (12, content-locked)** | [`experiments/docs/prereg/`](experiments/docs/prereg/) — `python experiments/src/prereg_lock.py verify --all` |
| **What we withdrew and why** | [Retractions and rescopes](#retractions-and-rescopes) — please read before citing any older number |

Every plotted number in Figures 1–4 is loaded programmatically from a raw artifact at render
time; none is transcribed from prose. Where a narratively interesting number had no raw
artifact, the panel was not rendered rather than approximated.

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
   Evidence: `results/paper_closing/within_model_slope_comparison.json`.

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
    Evidence: `results/paper_closing/EXP2_LEGACY_DETECTOR_RESOLUTION.md`.

## Reproducing

### Environments

Three conda environments, because the model families pin incompatible `transformers`:

| env | `transformers` | used for |
|---|---|---|
| `generator` | 5.5.0 | GENERator, NTv3, all text decoders, the census, everything in `scripts/paper_closing/` |
| `dnabert` | 4.29.2 | DNABERT-2 (its remote code predates the Transformers 5 config API) |
| `evo` | 4.48.1 | Evo1 |

`pip install -r requirements.txt` covers the shared dependencies. Running DNABERT-2 jobs under
`generator` fails with `BertConfig has no attribute pad_token_id` — that is the wrong-env
signature, not a code bug.

### Data

* **WikiText-2** — committed at [`frozen_inputs/`](frozen_inputs/) (sha256 `5f1bea06…`), so text
  endpoints run offline. This is the same file the census resolved through `datasets`, not a
  substitute corpus; see [`frozen_inputs/README.md`](frozen_inputs/README.md).
* **hg38** — not committed (size). Place or symlink at `data/reference/hg38/hg38.fa`.
* **GUE** — not committed. Point `$GUE_ROOT` at it.
* **Checkpoints** — pinned Hugging Face revisions, listed per model in `audit/census_master.csv`.

### Commands

Prefer the pipelines. Each skips stages whose output already exists, so an interrupted run
resumes and a finished one costs nothing; set `FORCE=1` to recompute.

```bash
bash pipelines/fig5_within_layer.sh      # Fig 5: both within-layer sweeps, second row, geometry, plot
bash pipelines/detector_provenance.sh    # positive control first, then the three 7B decoders
bash pipelines/census.sh                 # the frozen 22-model census (restartable)
bash pipelines/census.sh dnabert2 ntv3   # ...or just these
```

Individual stages:

```bash
# structural panel and the causal census
python experiments/frozen/E11_scale_ladder/run_model.py --model <slug>
python experiments/frozen/E13_full_cohort_causal_census/run_singleton_census.py --model <slug>

# detector provenance (self-validating: 4 gates from census_master.csv)
python scripts/paper_closing/run_uniform_detector_text.py --model smollm2-1.7b   # harness check
python scripts/paper_closing/run_detector_published_protocol.py --model olmo

# within-layer sweeps
python scripts/paper_closing/run_within_model_slope_text.py        # SmolLM2-1.7B, ~7 min, 1 GPU
python scripts/paper_closing/run_smollm2_second_row_epistasis.py   # second critical row + masking
python scripts/paper_closing/run_smollm2_row_geometry.py           # data-free, no GPU needed

# mechanism case studies
python experiments/frozen/E9_mechanistic_tomography/run_fit_observers.py
python experiments/frozen/E12_generator_degradation_control/run_bos_mediation_main.py

# two flags that are REQUIRED and silently wrong if omitted
python scripts/detection/run_detection.py --model ntv3 --pad_to_multiple 256   # stride-256 U-Net
python scripts/evaluation/run_gue_multiseed.py --model ntv3 --max_length 400   # nucleotide tokenizer
```

The paper-closing scripts validate themselves before reporting: each reproduces stored census
quantities (`baseline_loss`, `R_cand` at both ε, and the seeded control rows) and **aborts** on
mismatch, on the principle that a gate failure means the harness is wrong rather than that the
science changed. Observed reproduction errors are 1e-9 to 1e-6 relative.

## Repository layout

| path | contents |
|---|---|
| `experiments/frozen/E1`–`E13` | The **frozen** experiment harnesses — the code that produced the census and the case studies. *Frozen* means not edited to make later results come out differently. |
| `experiments/figures/` | The figure pipeline: render scripts, output, `source_data/` (the exact plotted values), and `FIGURE_PROVENANCE.md` |
| `experiments/docs/prereg/` | 12 content-locked preregistrations, with the ledger and verifier |
| `experiments/src/` | `prereg_lock.py` |
| `experiments/results/keep/` | Provenance-locked artifacts for E1 and E2 |
| `pipelines/` | One idempotent runner per experiment group. **Start here to reproduce anything.** |
| `scripts/paper_closing/` | Detector provenance, within-layer sweeps, BOS and attention analyses |
| `scripts/mechanism/`, `scripts/compression/`, `scripts/diagnostics/` | Mechanism, quantisation, and super-row health-check tooling. Several produced the negative results behind the retractions above. |
| `scripts/detection/`, `scripts/evaluation/` | Super-row detection and GUE downstream evaluation |
| `scripts/analysis/`, `scripts/interpretability/` | Analyses feeding the figures and the audit |
| `src/` | Shared libraries: activation capture, ablation, spike detection, DNA probes |
| `models/`, `configs/` | Per-model wrapper classes and YAML, loaded dynamically via `WRAPPER_MAP` |
| `stubs/` | Import shims so models with heavy optional dependencies load without them (`mamba_ssm` for Caduceus, a HybriDNA config). Put on `sys.path` at runtime by `scripts/evaluation/run_gue_ablation.py`; not type stubs, despite the name. |
| `audit/` | The verification record across three adversarial rounds. `census_master.csv` is the canonical census table (22 models × 44 columns); `round2/tables/` holds Supplementary S1–S5. See [`audit/README.md`](audit/README.md). |
| `results/` | Artifacts. Gitignored by default; files backing a manuscript claim are force-added. |
| `results/paper_closing/supplementary/` | Supplementary Tables S6–S7 |
| `sae/` | Sparse-autoencoder tooling. Its PROK result is **withdrawn** (retraction 6); kept so the withdrawal is inspectable. |
| `frozen_inputs/` | Content-hashed evaluation inputs |
| `docs/` | The experiment map |
| `tests/` | Includes a test that fails if the experiment map goes stale |

## Provenance and known gaps

Stated plainly, because a reader will find them anyway:

1. **Some raw artifacts are not committed.** Several were produced on a TACC cluster and never
   transferred. For every affected item the committed figure source-data snapshot under
   `experiments/figures/source_data/` carries the plotted values, so each reported number
   remains inspectable.
2. **NTv3's original weight revision was not recorded** and is not recoverable. Resolved Hub
   commits exist for 21 of 22 models.
3. **One n=10 subgroup is fragile.** The text-decoder correlation moves from ρ=0.770 to 0.673
   under an OLMo coordinate substitution, with a CI lower bound of +0.032. It should not be
   reported as independently robust; the 22-model result is insensitive.
4. **The manuscript text is not in this repository.** Earlier revisions of it, and the internal
   process documents, decision logs and session reports that accompanied its preparation, were
   removed on 2026-09-09 to leave a repository that is only code, data, and provenance. All of
   it remains in git history.
5. **Some retained notes cite documents that are no longer here.** A number of per-experiment
   `RESULTS.md` files and the figure-provenance manifest reference internal decision logs
   (`DECISIONS.md`, `CLAIMS_LEDGER.md`) and manuscript drafts that were removed in the same
   cleanup. They were kept because the figure manifest cites them as evidence; their onward
   references are stale. Everything they point at is in git history.
6. **No licence file yet.** Add one before publication; the appropriate choice is the authors'.

## Reference

Yu M, Wang D, Shan Q, Reed CJ, Wan A. *The Super Weight in Large Language Models.*
[arXiv:2411.07191](https://arxiv.org/abs/2411.07191) (2024).
