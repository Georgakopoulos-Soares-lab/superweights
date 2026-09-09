# Structure is not mechanism

### High-gain gated-FFN rows across text and genomic foundation models

Code and artifacts for the manuscript of the same name. The project asks whether the
"super weight" phenomenon reported in text LLMs recurs in genomic foundation models, and — the
sharper question — whether a high-gain row's **structural** prominence tells you anything
about its **causal** importance. Across a frozen 22-model census and focused mechanistic case
studies, the answer is that structural geometry, functional criticality, and causal response
complexity are three different things.

```
STRUCTURAL GEOMETRY  !=  FUNCTIONAL CRITICALITY  !=  CAUSAL RESPONSE COMPLEXITY
```

---

## For reviewers — start here

| what | where |
|---|---|
| **Manuscript** | [`paper-salvage/actual_manuscript.md`](paper-salvage/actual_manuscript.md) and [`.tex`](paper-salvage/actual_manuscript.tex) — kept content-identical |
| **Every experiment → its code → its artifacts** | [`docs/EXPERIMENT_MAP.md`](docs/EXPERIMENT_MAP.md) (20 experiments) |
| **Verify that map against this tree** | `python scripts/build_experiment_map.py` — exits non-zero if any script is missing |
| **Panel-by-panel figure provenance** | [`paper-salvage/figures/FIGURE_PROVENANCE.md`](paper-salvage/figures/FIGURE_PROVENANCE.md) |
| **What we withdrew and why** | [Retractions and rescopes](#retractions-and-rescopes) below — please read before citing any older number |
| **Supplementary tables** | [`results/paper_closing/supplementary/`](results/paper_closing/supplementary/) |

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

```bash
# ── structural panel and the causal census ────────────────────────────────────
python paper-salvage/experiments/E11_scale_ladder/run_model.py --model <slug>
python paper-salvage/experiments/E13_full_cohort_causal_census/run_singleton_census.py --model <slug>

# ── detector provenance (self-validating: 4 gates from census_master.csv) ─────
python scripts/paper_closing/run_uniform_detector_text.py --model smollm2-1.7b   # harness check
python scripts/paper_closing/run_uniform_detector_text.py --model llama          # then mistral, olmo
python scripts/paper_closing/run_detector_published_protocol.py --model olmo

# ── within-layer graded sweep (Figure 5) ─────────────────────────────────────
python scripts/paper_closing/run_within_model_slope_text.py        # SmolLM2-1.7B, ~7 min, 1 GPU
python scripts/paper_closing/run_within_model_slope.py             # GENERator-EUK-3B
python scripts/paper_closing/run_smollm2_second_row_epistasis.py   # second critical row + masking
python scripts/paper_closing/plot_within_model_slope_2panel.py     # figure; checks legend/data overlap

# ── mechanism case studies ───────────────────────────────────────────────────
python paper-salvage/experiments/E9_mechanistic_tomography/run_fit_observers.py
python paper-salvage/experiments/E12_generator_degradation_control/run_bos_mediation_main.py
python scripts/mechanism/run_pretrained_epistasis.py

# ── two flags that are REQUIRED and silently wrong if omitted ────────────────
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
| `paper-salvage/` | The manuscript, its figures, and the **frozen experiment harnesses** (`experiments/E1`–`E13`). Frozen means: not edited to make later results come out differently. |
| `paper-salvage/figures/` | Figure scripts, rendered output, and `source_data/` — the exact plotted values |
| `scripts/paper_closing/` | The closing round: detector provenance, within-layer sweeps, BOS/attention analyses |
| `scripts/mechanism/`, `scripts/compression/`, `scripts/diagnostics/` | Mechanism, quantisation, and health-check tooling |
| `scripts/detection/`, `scripts/evaluation/` | Super-row detection and GUE downstream evaluation |
| `scripts/analysis/`, `scripts/interpretability/`, `scripts/dev/` | Exploratory and superseded analyses, kept for provenance |
| `audit/` | `census_master.csv` (22 models × 44 columns), detector provenance, and the round-2 audit |
| `results/` | Artifacts. Gitignored by default; files backing manuscript claims are force-added |
| `results/paper_closing/supplementary/` | Supplementary Tables S5–S6 |
| `configs/`, `models/` | Per-model YAML and wrapper classes |
| `docs/` | Experiment map, evidence packets, collaborator instructions |
| `docs/history/` | Superseded material, kept deliberately: the 1,443-line previous README, old session notes |
| `paper/` | **Superseded** earlier draft under a different title ("A Structural Predictor of Super-Weights"). Retained for history; not the manuscript. |
| `frozen_inputs/` | Content-hashed evaluation inputs |

## Provenance and known gaps

Stated plainly, because a reviewer will find them anyway:

1. **Some raw artifacts are not committed.** Several were produced on a TACC cluster and never
   transferred; `docs/EXPERIMENT_MAP.md` marks the affected rows and points at the committed
   figure source-data snapshot that carries the same plotted values. Inventory:
   [`paper-salvage/docs/MISSING_COLLEAGUE_ARTIFACTS.md`](paper-salvage/docs/MISSING_COLLEAGUE_ARTIFACTS.md).
2. **NTv3's original weight revision was not recorded** and is not recoverable. Resolved Hub
   commits exist for 21 of 22 models.
3. **Supplementary Tables S1–S4 are described in the manuscript but not yet built as files.**
   Their values live in `audit/census_master.csv` (S1–S3) and
   `results/paper_closing/activation_vs_causality.tsv` (S4). S5 and S6 are built.
4. **The manuscript's Data-availability commit hash is stale** and must be updated to the
   submission commit before submission.
5. **One n=10 subgroup is fragile.** The text-decoder correlation moves from ρ=0.770 to 0.673
   under an OLMo coordinate substitution, with a CI lower bound of +0.032. It should not be
   reported as independently robust; the 22-model result is insensitive.

## History

This repository accreted across many sessions, including experiments that were later
withdrawn. That history is kept rather than rewritten:

* [`docs/history/README_ARCHIVE.md`](docs/history/README_ARCHIVE.md) — the full previous README
* [`paper-salvage/docs/DECISIONS.md`](paper-salvage/docs/DECISIONS.md), `CLAIMS_LEDGER.md` — dated decisions and per-claim status
* [`results/paper_closing/PAPER_CLOSING_REPORT.md`](results/paper_closing/PAPER_CLOSING_REPORT.md) — the closing round in full
* [`results/mechanism/`](results/mechanism/) — the mechanism/negative-results session reports

## Reference

Yu M, Wang D, Shan Q, Reed CJ, Wan A. *The Super Weight in Large Language Models.*
[arXiv:2411.07191](https://arxiv.org/abs/2411.07191) (2024).
