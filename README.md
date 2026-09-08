# Super Weights in Genomic Language Models

Locating super weights in transformer-based genomic LMs following the data-free,
single-forward-pass method from Yu et al. (2024) *"The Super Weight in Large Language Models"*
(arXiv 2411.07191), plus mechanistic interpretability and compression analyses.

---

## Paper-closing program (Sep 2026) — start here

Closing analyses for *"Structure is not mechanism: high-gain gated-FFN rows across text and
genomic foundation models."* Full report:
[`results/paper_closing/PAPER_CLOSING_REPORT.md`](results/paper_closing/PAPER_CLOSING_REPORT.md);
one row per question in
[`results/paper_closing/paper_closing_summary.tsv`](results/paper_closing/paper_closing_summary.tsv);
execution notes for the overseeing agent in
[`results/paper_closing/EXECUTION_REPORT_FOR_OVERSEER.md`](results/paper_closing/EXECUTION_REPORT_FOR_OVERSEER.md).

### 🔬 Open item — delegated, needs a machine with the 7B weights

**EXP2 on Llama-7B, Mistral-7B and OLMo-7B.** These are the last 3 of 22 census candidates
whose selection rule is unverified (legacy absolute-activation argmax, never re-checked
against the current layer-relative ratio rule). Two of the three legacy models already
checked *disagreed* with the ratio rule, so this is a real gap, not a formality.

👉 **Instructions:
[`docs/EXP2_INSTRUCTIONS_FOR_COLLABORATOR.md`](docs/EXP2_INSTRUCTIONS_FOR_COLLABORATOR.md)**
— self-contained; three commands, ~15 min of GPU each, no training, no downloads if the
weights are cached.

```bash
# harness check first (1.7B, ~45 s) -- MUST print "uniform rule AGREES"
python scripts/paper_closing/run_uniform_detector_text.py --model smollm2-1.7b
# then the three targets
python scripts/paper_closing/run_uniform_detector_text.py --model llama
python scripts/paper_closing/run_uniform_detector_text.py --model mistral
python scripts/paper_closing/run_uniform_detector_text.py --model olmo
```

Each run self-validates against four quantities stored in `audit/census_master.csv`
(`baseline_loss`, `R_cand_eps1.0`, `R_cand_eps0.5`, and the 5 control rows redrawn from
`SeedSequence(42).spawn(23)[panel_index]`) and **aborts** on mismatch. The detector positive
control on SmolLM2-1.7B passes: frozen coordinate returned as global argmax, rank 1/49152,
stable in 100 % of 24 inputs, all four gates within 1.6e-07.

### What this round established

| finding | key numbers | status |
|---|---|---|
| Activation ratio is **two-regime**, replicated in a genomic *and* a text decoder | below the detector's ≥5 rule: ρ=+0.040 (p=0.86) genomic, ρ=**−0.584** (p=0.005) text. Above it: ρ=+0.639 (p=0.010) genomic, ρ=**+0.975** (p=7.1e-10) text. n=36 rows/model, n=21 sub- and 15 supra-threshold | [MEASURED]; supra/sub split **post hoc** |
| Ratio **orders** rows but is **not** a dose-response law | SmolLM2-1.7B L7: the row ranked **4th** by ratio (r161, ratio 63.5) is **130× more damaging** (+287 %) than the row ranked 2nd (r749, ratio 358.6, +2.2 %) | [MEASURED] |
| A **second critical row** the census cannot see, and a **masking** interaction | r227 +698.9 %, r161 +287.2 % alone; joint(161,749) = **+2.07 %** — ablating r749 (harmless alone, +2.2 %) **abolishes** r161's catastrophe. Independently re-verified, weight drift 0.00e+00 | [MEASURED]; mechanism **[UNTESTED]** |
| The three rows are the layer's geometric extremes | cos(161,749)=**+0.3952** (z=+17.6, percentile 100.00 of 19,900 pairs); cos(227,749)=**−0.3980** (z=−17.8, percentile 0.00). Aligned, **not** opposed | [MEASURED] |

Earlier wording that **"activation extremeness calibrates causal severity"** is retired — see
the retractions block below. Supported: the ratio *detects* and *orders*. Not supported:
damage scaling smoothly with ratio.

> ⚠️ The sub-/supra-threshold decomposition was added **after** the pre-specified statistics
> (ρ all rows, ρ excluding the frozen candidate, ρ ratio<5) were computed and plotted. The
> split point is the detector's own pre-existing ≥5 rule and was not fitted to these data,
> but the decision to decompose came post hoc and is labelled as such everywhere.

Figure: `results/paper_closing/fig_within_model_slope_2panel.{png,pdf}`.

### Reproducing the paper-closing results

```bash
# Within-model graded sweep, text decoder (SmolLM2-1.7B, ~7 min, 1 GPU)
python scripts/paper_closing/run_within_model_slope_text.py

# ...and the genomic counterpart (GENERator-EUK-3B; needs the frozen E12 harness + hg38)
python scripts/paper_closing/run_within_model_slope.py

# The second critical row and its masking interaction (~2 min)
python scripts/paper_closing/run_smollm2_second_row_epistasis.py

# Two-panel cross-domain figure
python scripts/paper_closing/plot_within_model_slope_2panel.py
```

Text-decoder endpoints need no dataset download: the frozen WikiText-2 test parquet is
committed at `frozen_inputs/wikitext_repo/` (sha256 `5f1bea06…`, provenance in
[`frozen_inputs/README.md`](frozen_inputs/README.md)). It is the *same* file the census
resolved via `datasets.load_dataset`, not a substitute corpus — verified by reproducing the
census's stored `baseline_loss` to 1.5e-08 relative.

---

## Mechanism session findings (Aug 2026)

A separate mechanism/negative-results session produced a large set of markdown reports and
new scripts under [`results/mechanism/`](results/mechanism/) and `scripts/mechanism/`,
`scripts/compression/`, `sae/`. Start with
[HANDOFF_ANALYTICAL_SUMMARY.md](results/mechanism/HANDOFF_ANALYTICAL_SUMMARY.md).

**Status (updated 2026-08-17):** the raw JSON/CSV/PNG outputs behind these reports are still
not committed to this repository (only the `.md` reports and the `.py` scripts that would
produce them were pushed) — see
[`paper-salvage/docs/MISSING_COLLEAGUE_ARTIFACTS.md`](paper-salvage/docs/MISSING_COLLEAGUE_ARTIFACTS.md)
for the recovery checklist. **The findings themselves are adopted as established results**,
per explicit author decision
([`paper-salvage/docs/DECISIONS.md`](paper-salvage/docs/DECISIONS.md) D-024) — recovering the
raw artifacts remains desirable for independent reproducibility but is no longer a
precondition for citing these numbers. Full provenance audit and per-claim evidence table:
[`paper-salvage/docs/COLLEAGUE_BRANCH_AUDIT.md`](paper-salvage/docs/COLLEAGUE_BRANCH_AUDIT.md);
enacted claim rows: `paper-salvage/docs/CLAIMS_LEDGER.md` C-036–C-043.

### What we established

| finding | key numbers | claim |
|---|---|---|
| DNABERT-2's functional unit is a **redundant pair**, not a single row | top-7 pairs 7/7 structurally related, p = 0.00014 (n=5); splice −26.84 ± 2.56 vs sum-of-parts −3.51; critical pair joint −33.76 pp vs. separate −0.02/−0.11 pp | C-036 |
| The pair is **intrinsic to pretraining** (MLM loss, no task head) | epistasis +2.0118; top pair = 136,521× the random-pair sd; k=5 enrichment cliff | C-037 |
| Mechanism = **joint norm carriage**, DNABERT-2-specific — does **not** generalize to NTv3 | pair carries 58% of layer-9 residual norm (17.20 → 7.14); co-dominance intervention moves epistasis −33.6→−0.7; NTv3 MAKE-PAIR produces nothing; NTv3's SW is *more* norm-dominant (29.4× gap) yet functionally inert (−0.02pp) | C-038 |
| **Decoder/encoder attention-sink dissociation** | GENERator: 37.96% attention mass at BOS, 33× uniform, 45,585× activation. DNABERT-2: 0/40 windows peak at [CLS] | C-039 |
| **Causal steering** of generated composition (non-monotonic — do not read as a linear knob) | GC span 38.6× random rows; saturates by 2× scale, reverses at 5× | C-040 |
| **Corrected PROK super-weight**: L8/r260, not the old L2/r1927 (detected via a mismatched eukaryotic probe) | rank 1/3072, content-invariant, out_max=30,167.07; corrected hexamer causal test ρ=+0.0007 (p=0.96, no relationship); ΔPPL +1.25±0.54 vs random | C-001 (updated), C-041 |
| Corrected PROK: GC-dependence of ablation **cost** survives as the real kingdom contrast | PROK r=−0.661, EUK r=−0.001 | C-042 |
| Compression: SW-aware exemption is empirically a **no-op** | per-row RTN preserves the row max with 0.000e+00 error at INT8–INT2; per-tensor exemption benefit mean +0.048pp, t≈+0.15 — no reliable benefit at any granularity/precision | C-043 |
| **NTv3 splice ablation does not functionally replicate** | old ΔMCC=−0.119/p=0.008 was produced under a confirmed truncation bug (`max_length` effectively 80, not 400); refit reaches MCC 0.86–0.91 with SW ablation effect −0.02pp | C-029 (updated) |

**The old NTv3 splice number and the old L2/r1927 PROK story are retired**, not merely
superseded in place — see `CLAIMS_LEDGER.md` `X-008`, `X-009`. DNABERT-2 remains the sole
functional replication (n=1) of the SW-ensemble effect across the models tested.

### 🔴 Retractions and rescopes (read before citing older numbers)

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


### New scripts shipped this round

New, non-colliding additions — safe to use, not yet run end-to-end in this repository's own
CI/tests:

- `scripts/mechanism/` — `run_attention_sink.py`, `run_codominance.py`,
  `run_compensation_circuit.py`, `run_direction_vs_magnitude.py`, `run_ensemble_encoding.py`,
  `run_norm_matched_control.py`, `run_pretrained_epistasis.py`, `run_steering_biological.py`,
  `run_sw_steering.py`
- `scripts/compression/` — `run_destructive_sw_protection.py`,
  `run_group_scale_preservation.py`, `run_pair_aware_compression.py`,
  `run_per_tensor_sw_exemption.py`, `run_proximity_confound_control.py`
- `scripts/evaluation/run_sw_pairwise_epistasis.py`
- `sae/analyze_real_sequence.py`, plus float32-storage/standardization fixes to
  `sae/collect.py`, `sae/train.py`, `sae/model.py`, `sae/analyze.py` (avoids an fp16 clamp
  that was destroying SW-channel activation variance)
- `scripts/detection/run_detection.py --pad_to_multiple` and
  `scripts/evaluation/run_gue_ablation.py`'s `_NTv3Classifier` padding fix — both address the
  same architectural fact (NTv3's conv/deconv U-Net skip connections require sequence length
  to be an exact multiple of `2**num_downsamples`, not merely above a minimum)

---

## Pre-submission status (Nov 2026 — superseded in part by the section above)


The 5-arc manuscript has been restructured (v4) with two new cross-architecture experiments
landing in Figures 2 & 4:

- ~~**5-seed NTv3 splice** — ΔMCC = −0.119 ± 0.054, t = −4.86, p = 0.0083, sign-neg 5/5.~~
  🔴 **WITHDRAWN (Aug 2026)** — produced on 20%-truncated inputs (`_MAX_LEN` is
  tokenizer-specific; NTv3 is nucleotide-level). All 5 seeds were at the majority-class
  floor. Refit models reach MCC 0.86–0.91 and show **no** SW ablation effect (−0.02 pp).
- **NTv3 ‖U_k‖_F per-layer audit** (all 12 transformer blocks) — only L11 row 1472 ranks
  1 / 1536 (100th percentile); no other layer hosts a registered SW row.
- **Evo1 bf16-clean residual attribution** across all 32 StripedHyena blocks — corrects
  the earlier fp16 trace: row 3776 is *preserved-then-frozen* at ≈ 3 × 10⁷ from layer 13
  onward (MLP@row ≈ 0), normalised away by the pre-unembed RMSNorm. Same-protocol
  ablation gives ΔPPL ≈ 0% on the held-out window.

**Repo-local workflow:**
- Per-figure data-status & generation guide:
  [scripts/analysis/FIGURES_README.md](scripts/analysis/FIGURES_README.md)
- Evo2 replication plan (what to re-run with frontier StripedHyena access):
  [EVO2_REPLICATION.md](EVO2_REPLICATION.md)

Generated this round:
- [paper/media/image_fig4.png](paper/media/image_fig4.png) (5-seed NTv3 panel B)
- [paper/media/image_fig2_panels_EF.png](paper/media/image_fig2_panels_EF.png)
  (cross-arch ‖U_k‖_F percentile + dual-panel residual attribution)

Figure cleanup (commit `bc5575f`):
- All plot scripts now share [scripts/analysis/_figstyle.py](scripts/analysis/_figstyle.py)
  (no suptitles, no per-panel titles, compact bold A/B/C labels in corners,
  sans-serif, no top/right spines, editable-text PDF).
- Fig 3 redesigned from an 8-cell stack to 5 clean panels (A composition bars,
  B EUK scatter, C PROK scatter, D shuffle controls, E motif butterfly).
  Panel labels B/D and C/E are vertically aligned via figure-coordinate
  `fig.text()` to handle Panel E's wide motif tick labels.
- Fig 4 dropped the schematic comparative-phenotype panel D — now 3 panels A/B/C.
- Fig 5 dropped the splice-INT4 B2 sidebar — now 3 panels A/B/C in a single row.
- Fig 2 (E+F) legend collapsed to a single block with the GENERator (pending)
  marker; panel F1 source/propagator regimes shown as background shading; panel
  F2 has inline annotations for the "MLP writes blow up" and "frozen ≈ 3·10⁷"
  regions.

Still pending (GENERator EUK / PROK access required from collaborator):
- Fig 2 panels A–D — activation lifecycle + ‖U_k‖_F bar at step-up layer.
  Expected file names listed in
  [scripts/analysis/FIGURES_README.md](scripts/analysis/FIGURES_README.md).

---

## Paper Status (May 2026)

Restructured 5-part-arc manuscript at `paper/main.tex` (~200 lines, replaces 900-line
`main_old.tex`). Compiled headline figures live under `paper/media/`.

| # | Figure | Source script | Source data | Status |
|---|--------|---------------|-------------|--------|
| 1 | Architecture restriction (transformer decoders only) | `scripts/analysis/plot_figure1.py` | `super_weight_index.json`, `ablation_results.json` | ✅ |
| 2 | Quadratic amplifier mechanism | `scripts/analysis/plot_sw_mechanistic.py` | `sw_causal_tracing.json`, `sw_grad_attribution.json` | ✅ |
| 3 | Composition encoding (shared magnitude-scaled mechanism; EUK \|write\|–KL r = +0.437, PROK r = +0.710; the PROK −0.710 is a signed-activation write-direction convention, not an opposite mechanism) | `scripts/analysis/plot_figure2.py` | `sw_hexamer_causal.json`, `sw_shuffle_controls.json`, `sw_kmer_scan.json` | ✅ |
| 4 | GUE functional consequences (splice −25.5% across 3 seeds) | `scripts/analysis/plot_figure4.py` | `gue_multiseed_results.json`, `gue_per_row_ablation.json` | ✅ NEW |
| 5 | Compression — shadow redundancy + whole-model INT4 | `scripts/analysis/plot_figure5.py` | `compression_sweep_*.json`, `quant_ablation_generator_int4.json`, `int4_downstream_benchmark_splice.json`, `whole_model_quant_generator{,_prokaryote}_100k.json` | ✅ NEW |

### Latest result — extended 100k-token whole-model INT4 (May 26 2026)

14× higher resolution than the original 7,200-nt probe; **resolves the Yu et al. SW-exemption question**:

| Model | Baseline PPL | Naïve INT4 ΔPPL | Yu-exempt ΔPPL | SW marginal cost | SW-only fragility |
|-------|-------------:|----------------:|---------------:|-----------------:|------------------:|
| GENERator EUK 3B  | 8.427 | **+0.0870** | +0.0879 | **−0.00083** | +0.00013 |
| GENERator PROK 3B | 8.783 | **+0.4669** | +0.4664 | **+0.00041** | −0.00007 |

**Non-replication of Yu et al. confirmed**: SW marginal cost (Yu+SW − Yu-exempt) is below the
±0.001-PPL noise floor in both kingdoms — protecting SW rows yields **no measurable benefit** at
INT4 in genomic transformer decoders. This is now Part 5's headline result.

Reproduce with:
```bash
CUDA_VISIBLE_DEVICES=0 conda run -n generator --live-stream python3 \
  scripts/compression/run_whole_model_quantization.py --model generator \
  --scopes full --fracs 100 --bits 4 --n_rand_seeds 0 \
  --n_probe_seqs 500 --probe_seq_len 1200 --probe_seed 42 \
  --sw_index results/super_weight_index.json \
  --out results/whole_model_quant_generator_100k.json
```
(`--model generator_prokaryote` for PROK.)

### Priority of results (for paper)

1. **Architecture restriction** (Fig 1) — transformer decoders only; SSM/Hyena/Mamba show no SW.
2. **Splice collapse** (Fig 4A) — DNABERT-2 SW ablation: −25.5 ± 0.7% across 3 seeds, p = 0.0004.
3. **Shared magnitude-scaled mechanism** (Fig 3) — in both kingdoms ablation cost scales with SW write magnitude (EUK r = +0.437, PROK r = +0.710); the PROK −0.710 reflects a signed-activation write-direction convention, not an opposite "suppressive gate."
4. **Quadratic amplifier mechanism** (Fig 2) — single early FFN row + residual propagation.
5. **Yu et al. non-replication at scale** (Fig 5C) — 100k-token INT4 SW marginal cost ≈ 0.
6. **Shadow redundancy** (Fig 5A,B) — near-SW rows are the *most* INT4-tolerant; far-SW rows collapse.

### Test suite

```
tests/
├── test_detection.py   # Super-weight detection regression tests
└── test_hooks.py       # Forward-hook integrity / shape tests
```

Run with `pytest tests/`.

---

## Models Studied

| Model | Type | Params | HF ID |
|-------|------|--------|-------|
| GENERator eukaryote | Causal decoder | 3B | `GenerTeam/GENERator-v2-eukaryote-3b-base` |
| GENERator prokaryote | Causal decoder | 3B | `GenerTeam/GENERator-v2-prokaryote-3b-base` |
| GENERator prokaryote 1B | Causal decoder | 1.2B | `GenerTeam/GENERator-v2-prokaryote-1.2b-base` |
| Evo 2 7B | StripedHyena2 SSM | 7B | `arcinstitute/evo2_7b` |
| NTv3 | Encoder (masked LM) | 50M | `InstaDeepAI/nucleotide-transformer-v3-50m-multi-species` |
| DNABERT-2 | Encoder (masked LM) | 117M | `zhihan1996/DNABERT-2-117M` |
| MegaDNA | Causal decoder | — | — |
| HybridNa | Hybrid SSM/Attn | — | — |
| GenomeOcean | Causal decoder | — | — |
| Caduceus ps | Bidirectional Mamba (SSM) | 7M | `kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16` |

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

Multi-seed results (seeds 0/1/2, DNABERT-2, mean ± std across 3 seeds):

| Task | Baseline acc | SW-ablated acc | Δacc | p (t-test vs 0) | Rand ctrl |
|------|-------------|----------------|------|-----------------|----------|
| prom/prom_core_notata | 83.82% ± 0.19% | 71.25% ± 16.87% | −12.56% ± 16.73% | p=0.40 (n.s.) | ±0.19% |
| EMP/H3K4me3 | 64.20% ± 2.84% | 59.67% ± 5.27% | −4.53% ± 6.18% | p=0.41 (n.s.) | ±2.81% |
| splice/reconstructed | **92.74% ± 0.11%** | **67.19% ± 0.69%** | **−25.54% ± 0.73%** | **p=0.0004** | ±0.13% |

NTv3 single-seed reference:

| Model | Task | Baseline acc | Pruned SW acc | Δacc | Rand ctrl Δacc |
|-------|------|-------------|---------------|------|----------------|
| NTv3 | prom_core_notata | 70.0% | 69.9% | −0.05% | ~0% |
| NTv3 | splice/reconstructed | 53.4% | 56.5% | +5.8% (Δmcc **−86.2%**) | ~0% |
| NTv3 | EMP/H3K4me3 | 47.0% | 47.0% | 0.0% | ~0% |

Random-weight controls are consistently within ±0.05% — the SW effect is specific.

**Finding**: The splice detection result is the robust finding across all 3 seeds: SW ablation
causes **−25.5% ± 0.7% accuracy** (p=0.0004). The promoter result is seed-dependent (seed 0:
−36.2%, seeds 1–2: ~−1%) and not statistically reliable with 3 seeds (p=0.40). The histone mark
task (H3K4me3) is unaffected in all seeds, consistent with epigenomic signals being distributed.
SW rows are causally necessary for splice site recognition regardless of fine-tuning trajectory.

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

### Fairness of the Random Control: Super-Row Proximity Analysis

To verify the random ablation baseline is fair, we analysed whether the 10 detected
DNABERT-2 super-rows are spatially clustered relative to a random set of 10 rows.

Key structural properties of the 10 super-rows:
- Only **6 unique row indices** across 10 entries — row 603 appears at layers 3, 5, 6, 7;
  row 86 at layers 3 and 5.
- Only **5 unique layers** used (3, 5, 6, 7, 9) out of 12.

50 000 Monte-Carlo random sets of 10 (layer, row) pairs were compared using mean
pairwise Euclidean distance in normalised coordinate space:

| | Mean pairwise dist (normalised) |
|--|--|
| Random sets (mean ± std) | 0.545 ± 0.065 |
| Super-rows | 0.468 |
| z-score | −1.19 (12th percentile) |

**Conclusion**: the super-rows are only mildly more clustered than chance (z = −1.19),
well within the normal range. The random control is geometrically fair.

### Structured-Random Control

Because the super-rows share row indices across layers (a property random sampling
almost never produces), a *structured-random* control was added that mirrors:
- the same **layer distribution** (`{3:4, 5:2, 6:1, 7:1, 9:2}`)
- the same **row-repetition pattern** (one row repeated 4×, one 2×, four singletons)

Results on `prom_core_notata`:

| Condition | Accuracy | MCC | Δacc |
|-----------|----------|-----|------|
| Baseline | 83.46% | 0.6697 | — |
| Pruned SW (10 rows) | 72.34% | 0.4728 | **−13.3%** |
| Random control (n=10 mean) | 83.40% | 0.6687 | −0.07% |
| Structured-random control (n=10 mean) | 83.44% | 0.6695 | −0.02% |

The structured control is indistinguishable from purely random, confirming the
super-row effect is not a clustering or repetition artifact — *which* rows they are
matters, not *where* they cluster.

Run with:
```bash
python scripts/evaluation/run_gue_ablation.py --model dnabert2 --task prom/prom_core_notata \
    --gue_root $GUE_DATA_PATH --structured_rand 10
```

### Progressive Compression Sweep (DNABERT-2, `prom_core_notata`)

A pruning sensitivity analysis was run with four ranking criteria over fractions
0.5%–30% of the 9 206 non-SW rows (12 layers × 768 rows − 10 SW rows):

| Criterion | What it removes first |
|-----------|----------------------|
| `l1_low` | Smallest L1-norm rows (standard magnitude pruning) |
| `l1_high` | Largest L1-norm rows (sanity upper-bound) |
| `prox_far` | Rows furthest from any super-row in (layer, row) space |
| `prox_near` | Rows closest to any super-row |
| `random` | Uniform random (10 seeds, mean ± std) |

Selected Δacc results:

| Frac | n rows | l1_low | l1_high | prox_far | prox_near | random |
|------|--------|--------|---------|----------|-----------|--------|
| 1% | 92 | −0.25% | −0.14% | −0.09% | −0.07% | −0.21% |
| 5% | 460 | −3.41% | −0.63% | −0.36% | −0.32% | −0.41% |
| 10% | 921 | −3.59% | −0.63% | −0.47% | **−0.32%** | −0.69% |
| 15% | 1 381 | −3.00% | −0.72% | −3.97% | −1.63% | −0.82% |
| 20% | 1 841 | −3.12% | −1.29% | **−9.39%** | −1.60% | −0.82% |
| 30% | 2 762 | −3.93% | −1.20% | **−8.15%** | −1.65% | −1.38% |

**Key findings:**

1. **`prox_near` is the most stable curve** — removing up to 921 rows (10%) from the
   super-row neighbourhood causes only −0.32% accuracy loss. The SW region is the
   *safest* part of the model to compress.

2. **`prox_far` collapses catastrophically at 20%** (−9.4% acc, −23.3% MCC), worse
   than pruning the 10 super-rows themselves. Important rows are distributed throughout
   the rest of the model, not co-located with the super-rows.

3. **`l1_low` (magnitude pruning) breaks early** — −1.1% at just 2%, plateauing
   at ~−3.5% thereafter. Small-norm rows are not safely prunable.

4. **Shadow redundancy hypothesis**: the super-row is so disproportionately strong
   in its local region that neighbouring rows become redundant during training
   (their gradients are dominated by the super-row's output). This makes the SW
   neighbourhood appear safe to compress — but only because the super-row itself is
   intact. Removing it alone causes −13.3% accuracy.

Run the sweep with:
```bash
python scripts/compression/run_compression_sweep.py --model dnabert2 --task prom/prom_core_notata \
    --gue_root $GUE_DATA_PATH --ckpt_dir results/gue_checkpoints/dnabert2_prom_core_notata
```

Results: `results/compression_sweep_dnabert2_prom_core_notata.{json,png}`

### Quantization Sensitivity Ablation (INT4 / INT8)

Simulated round-to-nearest (RTN) quantization applied selectively to `mlp.wo` rows
to test whether the shadow-redundancy finding from the compression sweep extends to
precision loss. Four conditions:

| Condition | Description |
|---|---|
| `baseline` | No quantization |
| `yu_all` | INT{4,8} all non-SW rows (SW-exempt, after Yu et al.) |
| `sw_fragility` | INT{4,8} only the detected SW rows themselves |
| `near_sw` | INT{4,8} rows closest to SW coordinates (most redundant per compression sweep) |
| `random` | INT{4,8} same count of random non-SW rows (10 seeds, mean ± std) |

#### GENERator (perplexity, INT4) — positive result

| Condition | Δ PPL | p vs random (t-test) |
|---|---|---|
| `yu_all` INT4 (all non-SW rows) | +0.159 | — |
| `sw_fragility` INT4 (2 SW rows) | −0.013 | n.s. |
| `near_sw` 30% (27 647 rows) | **−0.023** | **p = 0.0003** |
| `random` 30% (27 647 rows) | +0.044 ± 0.037 | — |

At INT4, **shadow-redundant (near-SW) rows tolerate lower precision better than random
rows** — near_sw PPL holds flat or slightly decreases while random causes measurable
degradation. The effect is consistent from 10%–30% (p ≤ 0.016) and absent at INT8
(all Δ < 0.001), confirming that INT8 is too mild to reveal differential sensitivity
in a 3B model. The SW rows themselves are insensitive to INT4 quantization when taken
in isolation (n = 2, effect negligible), consistent with the model distributing the
SW signal through the residual stream at all downstream layers.

#### DNABERT-2 (GUE classification, INT4) — negative result

INT4 quantization ablation across three GUE tasks (prom/prom_core_notata,
splice/reconstructed, EMP/H3K4me3) shows no consistent signal:

- All Δacc values are within ±0.002 (i.e. within test-set noise for a 117M model).
- Scattered p < 0.05 values appear but with **inconsistent directionality** — near_sw
  is sometimes worse than random, sometimes better, with no monotonic trend.
- The cause is structural: DNABERT-2's 5 307–4 562 test-set samples yield accuracy
  resolution of ~0.02%; INT4 RTN error (~1.5% relative per weight) cannot move the
  classification boundary enough to differentiate row sensitivity at this granularity.

**Interpretation**: classification accuracy on GUE tasks is too coarse a metric to
detect differential quantization sensitivity. The GENERator perplexity result is the
primary finding; these DNABERT-2 results are a documented negative.

Run with:
```bash
# GENERator (perplexity mode, INT4)
python scripts/compression/run_quantization_ablation.py \
    --model generator --bits 4 \
    --out results/quant_ablation_generator_int4.json

# DNABERT-2 (GUE classification, INT4)
python scripts/compression/run_quantization_ablation.py \
    --model dnabert2 --task prom/prom_core_notata \
    --gue_root $GUE_DATA_PATH \
    --ckpt_dir results/gue_checkpoints/dnabert2_prom_core_notata \
    --bits 4 \
    --out results/quant_ablation_dnabert2_prom_core_notata_int4.json
```

Output files: `results/quant_ablation_{model}_{task}_int{bits}.{json,png}`

#### Whole-Model INT4 Quantization — SW-Exemption Comparison (`run_whole_model_quantization.py`)

Extends the down-proj-only ablation above to all 7 linear projection types
(`down_proj`, `gate_proj`, `up_proj`, `q_proj`, `k_proj`, `v_proj`, `o_proj`) —
806 396 / 806 397 rows total across 30 layers.  RTN INT4 is applied; SW rows are
excluded from the candidate pool using the following criterion:

- `down_proj`  — exclude `(sw_layer, sw_row)` (the actual super-row)
- `gate_proj` / `up_proj` — exclude `(sw_layer, sw_col)` (intermediate feature index)
- Attention projections — no exclusion (no known SW in attention layers)

Three single-shot conditions are measured per model:

| Condition | Rows quantized |
|---|---|
| `yu_all` | all eligible non-SW rows |
| `yu_all_including_sw` | all eligible rows **+ SW rows** |
| `sw_fragility` | SW rows only |

Evaluation supports a configurable probe via `--n_probe_seqs`, `--probe_seq_len`, `--probe_seed`.
The headline run uses a **500-sequence × 1,200-nt hg38 probe (~100,000 tokens, seed 42)** — 14×
the original 7,200-nt probe, sufficient to resolve marginal costs below 0.001 PPL.

**Results — GENERator eukaryote 3B (INT4, full scope, 100k tokens)**

| Condition | n rows | PPL | Δ PPL |
|---|---|---|---|
| FP16 baseline | — | 8.4274 | — |
| `yu_all` (SW exempt) | 806 396 | 8.5153 | **+0.0879** |
| `yu_all_including_sw` (naïve INT4) | 806 402 | 8.5145 | **+0.0870** |
| SW marginal cost (`yu_all_inc_sw − yu_all`) | +6 SW rows | — | **−0.00083** |
| `sw_fragility` (SW rows only) | 6 | 8.4276 | +0.00013 |

**Results — GENERator prokaryote 3B (INT4, full scope, 100k tokens)**

| Condition | n rows | Δ PPL |
|---|---|---|
| `yu_all` (SW exempt) | 806 397 | **+0.4664** |
| `yu_all_including_sw` (naïve INT4) | 806 400 | **+0.4669** |
| SW marginal cost | +3 SW rows | **+0.00041** |
| `sw_fragility` (SW rows only) | 3 | −0.00007 |

**Key finding (updated May 2026)**: At a properly powered 100k-token evaluation, the SW rows in
GENERator are **definitively not quantization-sensitive**. Protecting them (Yu et al.-style
exemption) changes PPL by ≤ 0.001 in both models — within the noise floor — and the EUK
marginal cost is actually slightly *negative*. SW-only INT4 fragility is negligible
(±0.0001 PPL). This is a **confirmed non-replication** of the Yu et al. (2024) SW-exemption
heuristic in genomic LMs.

Note: the prokaryote model shows a larger overall INT4 cost (+0.444 vs +0.117).
This is consistent with the prokaryote SW being more deeply integrated into the
residual stream (earlier layer L2, higher ΔPPL from zeroing: +9.47 vs +2.92) — the
whole model is more fragile to precision loss, but this fragility is distributed
across ordinary rows, not concentrated in the SW.

Run with:
```bash
# Eukaryote
bash scripts/compression/run_whole_model_quant_generator.sh
# Prokaryote
bash scripts/compression/run_whole_model_quant_generator_prokaryote.sh
```

Output files (extended 100k-token runs):
- `results/whole_model_quant_generator_100k.{json,png}`
- `results/whole_model_quant_generator_prokaryote_100k.{json,png}`

Legacy 7,200-nt runs (preserved for reference):
- `results/whole_model_quant_generator_sw_comparison.{json,png}`
- `results/whole_model_quant_generator_prokaryote_sw_comparison.{json,png}`

---

### INT4 Downstream Benchmark — Practical Compression (`run_int4_downstream_benchmark.py`)

Translates the shadow-redundancy finding from the perplexity-level sensitivity analysis
into a practical downstream task question: **does SW-aware INT4 quantization preserve
GUE accuracy better than naive INT4?**

This experiment uses DNABERT-2 on `splice/reconstructed` — the only robust multi-seed
result — and evaluates four conditions at a fixed compression target (10% of non-SW rows):

| Condition | What is INT4'd | What stays FP16 |
|---|---|---|
| `fp16_baseline` | nothing (reference) | everything |
| `naive_int4` | ALL rows including SW rows | nothing |
| `yu_all_int4` | all non-SW rows | 10 SW rows only |
| `near_sw_int4` | near-SW rows only (10% of non-SW pool) | everything else |
| `random_int4` | same count as near_sw, random non-SW rows (10 seeds) | everything else |

Per condition, the script reports: accuracy, MCC, F1, Δ vs FP16 baseline, peak GPU
memory (MB) during inference, inference time (ms/sample), and theoretical weight memory
savings (MB) for the quantized rows assuming FP16→INT4 (4× compression on those rows).

> **Note**: RTN simulation keeps weights in FP16 dtype — actual runtime memory and
> speed gains require hardware INT4 GEMM kernels. Theoretical savings are reported as
> a proxy for expected real-world gains.

**What to look for:**
- `yu_all_int4` holding accuracy close to FP16 — replicates Yu et al. in a genomic model
- `near_sw_int4` ≥ `random_int4` at matched compression — shadow-redundancy transfers to downstream
- `naive_int4` showing clear accuracy drop from quantizing the SW rows — confirms SW fragility

Run:
```bash
python scripts/compression/run_int4_downstream_benchmark.py \
    --task splice/reconstructed \
    --ckpt_dir results/gue_checkpoints/dnabert2_splice_reconstructed \
    --near_sw_frac 10.0 \
    --out results/int4_downstream_benchmark_splice.json \
    --plot results/int4_downstream_benchmark_splice.png
```

Or on SLURM:
```bash
sbatch scripts/compression/int4_downstream_benchmark_splice.sbatch
```

Requires: fine-tuned checkpoint from `run_gue_ablation.py` on `splice/reconstructed`.
Output: `results/int4_downstream_benchmark_splice.{json,png}`

---

### Mechanistic Interpretability: Gradient Saliency + Causal Ablation

Two complementary analyses targeting GENERator's layer-4 SW rows (2371 and 1522)
over real genomic sequences from hg38 (promoters, enhancers, random regions).

#### Gradient saliency (`run_sw_gradient_attribution.py`)

Computes per-token saliency maps by backpropagating from the SW row activation
back to the input embeddings:

```
saliency[t] = || d( |sw_act[..., sw_row]|.sum() ) / d(embed[t]) ||₂
```

Each token covers 6 bp (GENERator k-mer tokenizer). Saliency profiles are
interpolated to a common fractional-position axis and averaged by genomic context.

Full run on 30 sequences per context (hg38, 3 072 bp windows):

| Context | n | Peak saliency (mean ± std) | Mean saliency |
|---|---|---|---|
| Enhancer | 30 | 1 311 584 ± 896 735 | 2 763 ± 1 838 |
| Promoter | 30 | 1 093 148 ± 788 296 | 2 284 ± 1 508 |
| Random   | 30 |   883 867 ± 595 994 | 1 876 ± 1 150 |

Enhancer saliency is significantly higher than random (t=2.14, p=0.037).
Promoter vs random is not significant (p=0.26).
High within-context variance: the SW row responds heterogeneously,
suggesting sensitivity depends on specific k-mer content rather than
gross region class.

Run:
```bash
python scripts/interpretability/run_sw_gradient_attribution.py \
    --n_seqs 30 \
    --promoters data/regions/hg38/promoters_262kb.bed \
    --enhancers data/regions/hg38/enhancers_ccre_262kb.bed \
    --random    data/regions/hg38/random_262kb.bed \
    --out results/sw_grad_attribution.json \
    --plot results/sw_grad_attribution.png
```

Output: per-token saliency JSON + saliency profile PNG (mean ± std by context).

#### Causal ablation + activation shift (`run_sw_causal_tracing.py`)

Two mechanistic sub-experiments:

**A. SW row ablation (primary)**

For each clean genomic sequence, zeros the SW rows at the MLP `down_proj` output
and measures the perplexity increase vs. zeroing matched-count random rows:

Full run (n=90 sequences: 30 per context, hg38 3 072 bp windows):

| Condition | ΔPPL (mean ± std) | t-test vs random |
|---|---|---|
| SW rows zeroed (2 rows, rows 1522+2371) | **+2.92 ± 1.33** | t=20.74, p<0.0001 |
| Random rows zeroed (same count, 10 seeds) | +0.000 ± 0.0003 | — |

Zeroing 2 out of 3 072 MLP rows — the two SW rows — raises perplexity by ~2.9 nats
on average (ranging from +1.0 to +7.8 depending on sequence), while zeroing any other
pair of rows has zero measurable effect. The SW rows are causally necessary for prediction.

**B. Activation distribution shift**

Compares the SW row activation distributions between real genomic sequences and their
dinucleotide-shuffled counterparts (shuffling preserves di-nucleotide frequency but
destroys higher-order sequence context). Full-run result: mean shift = +0.0002 (p = 0.15, n.s.,
n=90 sequences) — the SW rows fire with equal intensity on real and shuffled sequences.
This is consistent with GENERator's 6-mer tokenizer limiting sensitivity to local k-mer
composition, which dinucleotide shuffling largely preserves.

Run:
```bash
python scripts/interpretability/run_sw_causal_tracing.py \
    --n_seqs 30 \
    --promoters data/regions/hg38/promoters_262kb.bed \
    --enhancers data/regions/hg38/enhancers_ccre_262kb.bed \
    --random    data/regions/hg38/random_262kb.bed \
    --out results/sw_causal_tracing.json \
    --plot results/sw_causal_tracing.png
```

Output: ablation ΔPPL + activation shift JSON + dual-panel PNG.

---

### Shuffle Controls — All Four Types (`run_sw_shuffle_controls.py`)

Four progressively stronger shuffle types test whether the SW is sensitive to any level
of sequence context. The script preserves the Altschul-Erickson De Bruijn algorithm for
trinucleotide-preserving shuffle (the strictest control, preserves all 3-mer frequencies).

| Shuffle type | What it preserves | What it destroys |
|---|---|---|
| `dinuc` | Per-dinucleotide frequency | k-mer order and composition above 2-mers |
| `mono` | Per-base GC composition only | All k>1 context, 6-mer frequencies |
| `kmer_block` | Per-token identity distribution (same bag of 6-mers) | Token order and all inter-token context |
| `trinuc` *(new)* | Per-trinucleotide frequency (3-mer) | k-mer order and composition above 3-mers |

**Results** (n=90 sequences, GENERator euk 3B, hg38, 3072 bp windows):

| Shuffle type | Mean activation shift | t | p |
|---|---|---|---|
| `dinuc` | −0.000009 | −0.087 | 0.931 |
| `mono` | +0.000101 | 1.012 | 0.314 |
| `kmer_block` | +0.000032 | 0.288 | 0.774 |
| `trinuc` | +0.000020 | 0.202 | **0.841** |

**All four are flat.** Even the trinucleotide shuffle — which preserves exact 3-mer composition
and thus heavily constrains 6-mer content — leaves SW activation statistically unchanged
(p=0.841). Combined with the k-mer block result (which preserves exact token-identity bag),
the SW fires on **token identity alone**, not on token order or any higher-order context.
This is the strictest possible shuffle control for a 6-mer tokenizer.

Run:
```bash
python scripts/interpretability/run_sw_shuffle_controls.py \
    --n_seqs 30 \
    --promoters data/regions/hg38/promoters_262kb.bed \
    --enhancers data/regions/hg38/enhancers_ccre_262kb.bed \
    --random    data/regions/hg38/random_262kb.bed \
    --n_shuffles 5 \
    --shuffle_types dinuc mono kmer_block trinuc \
    --out  results/sw_shuffle_controls.json \
    --plot results/sw_shuffle_controls.png
```

Output: `results/sw_shuffle_controls.json` — per-sequence activation shift by shuffle type,
t-tests, and comparison bar chart.

---

### Causal Hexamer Test (`run_sw_hexamer_causal.py`) *(new)*

Closes the interpretability loop by establishing that the SW is **causally responsible**
for CC/CT hexamer next-token predictions — not merely correlated with those activations.

**Method**: For each of the 4,096 possible GENERator 6-mer tokens (presented in a poly-A
context), compute:
1. Clean forward → predicted next-token distribution p_clean
2. SW rows zeroed mid-forward-pass → p_ablated
3. KL(p_clean ‖ p_ablated) — the prediction cost of SW ablation for each k-mer

Then correlate KL divergence with SW activation across all 4,096 k-mers.

**Results** (all 4,096 k-mers, GENERator euk 3B):

| Metric | Value |
|---|---|
| Pearson r(SW activation, KL divergence) | **0.437** |
| KL[top SW quartile] (CC/CT-rich k-mers) | **1.7844 ± 0.0012** |
| KL[bottom SW quartile] (CpG/polyA k-mers) | 1.7756 ± 0.0555 |
| Group Welch's t-test | t = 5.065, **p = 4.46 × 10⁻⁷** |

Top 5 k-mers by ablation cost (KL), with SW activation:

| Rank | k-mer | KL(clean ‖ ablated) | SW activation |
|------|-------|---------------------|---------------|
| 1 | `CCTGGT` | 1.7894 | 302,634 |
| 2 | `CCTGGC` | 1.7889 | 300,446 |
| 3 | `CCTGGG` | 1.7887 | 299,320 |
| 4 | `CCAGGT` | 1.7887 | 299,721 |
| 5 | `GCTGGT` | 1.7886 | 298,732 |

Bottom k-mer (negative control): `AAAAAA` KL = 0.0018, SW activation = 24 (effectively zero).

**Interpretation**: The causal chain is fully established:
1. SW fires maximally on CC/CT-rich hexamers (k-mer scan)
2. SW ablation **specifically** disrupts next-token predictions for those same k-mers (r = 0.437, p = 4.46 × 10⁻⁷)
3. AAAAAA — a k-mer that barely activates the SW — shows near-zero prediction change when SW is removed

The SW is not a passive bystander: it is the mechanism through which the model generates predictions for CC/CT hexamers.

Run:
```bash
python scripts/interpretability/run_sw_hexamer_causal.py \
    --out results/sw_hexamer_causal.json
```

No BED files needed — uses all 4,096 vocabulary tokens directly. Output: per-k-mer KL + correlation
stats saved to `results/sw_hexamer_causal.json`.

---

### K-mer Motif Cross-Reference (`analyze_sw_kmer_motifs.py`)

After running `run_sw_kmer_scan.py`, cross-references the top-K SW-activating 6-mers
against a curated dictionary of known regulatory DNA motifs (11 classes):

| Motif class | Description |
|---|---|
| `TATA_box` | TATAAA and common variants |
| `Kozak` | GCC[R]CC and ATG-context k-mers |
| `splice_donor` | GT-containing exon-intron junction k-mers |
| `splice_acceptor` | Pyrimidine-rich + AG junction k-mers |
| `Shine_Dalgarno` | AGG/AGGAGG (prokaryote ribosome binding) |
| `CCAAT_box` | Eukaryotic −80 promoter element |
| `GC_box_Sp1` | GGGCGG / Sp1 binding site |
| `E_box` | CA[ACGT][ACGT]TG — bHLH binding |
| `AP1_TRE` | TGA[C/G]TC |
| `NFkB` | GGG[R]NN family |
| `CpG_rich` | ≥ 2 CpG dinucleotides in the 6-mer |
| `AT_rich` | ≥ 5/6 A or T bases |
| `homopolymer_run` | ≥ 4 consecutive identical bases |

Enrichment is computed via one-sided Fisher's exact test with Benjamini-Hochberg correction.
Also reports per-position nucleotide log₂FC (foreground vs background) to identify structural
biases in SW-activating k-mers.

Run:
```bash
python scripts/interpretability/analyze_sw_kmer_motifs.py \
    --kmer_scan results/sw_kmer_scan.json \
    --top_k 200 \
    --out  results/sw_kmer_motifs.json \
    --plot results/sw_kmer_motifs.png
```

No GPU required. Output: enrichment table JSON + 4-panel figure.

---

### Attribution Method Comparison (`compare_attribution_methods.py`)

Measures per-sequence agreement between gradient saliency (from
`run_sw_gradient_attribution.py`) and token omission (from `run_sw_token_omission.py`)
by computing Spearman ρ, Kendall τ, and Pearson r between the two attribution vectors
on matched sequences.

Both methods ask "which positions drive the SW activation?" but differ fundamentally:
gradient saliency backpropagates linear sensitivity; token omission measures functional
perturbation impact. High agreement validates both; divergence reveals gradient saturation
or non-local synergy effects.

Run:
```bash
python scripts/interpretability/compare_attribution_methods.py \
    --grad results/sw_grad_attribution.json \
    --omit results/sw_token_omission.json \
    --out  results/attribution_comparison.json \
    --plot results/attribution_comparison.png
```

No GPU required. Requires both attribution result files to exist.
Output: per-sequence Spearman ρ, aggregate t-tests by genomic context, overlay profile plots.

---

## Prokaryote Interpretability — GENERator Prokaryote 3B SW Row 1927 (superseded, kept for history)

> **⚠ SUPERSEDED 2026-08-17.** This whole section (layer 2 / row 1927, including the
> `r = −0.710` write-direction-convention resolution below) was detected with a mismatched
> eukaryotic probe (`--probe human_promoter` against the prokaryote model) and is retired —
> see `paper-salvage/docs/CLAIMS_LEDGER.md` `X-008`, `paper-salvage/docs/DECISIONS.md` D-024.
> **The corrected super-weight is layer 8 / row 260** (rank 1/3072, content-invariant,
> out_max=30,167.07): the corrected hexamer causal test finds **no** sign relationship at all
> (Spearman ρ=+0.0007, p=0.96) — the sign-convention question this section resolves does not
> arise at the corrected channel, because there is no correlation to have a sign. A different,
> real kingdom contrast survives at the corrected channel instead: GC-dependence of ablation
> *cost* (PROK r=−0.661 vs. EUK r=−0.001) — see `CLAIMS_LEDGER.md` C-001/C-041/C-042. Kept
> below verbatim as project history, not as current findings.

Parallel six-step interpretability pipeline applied to the prokaryote model (SW at layer 2,
row 1927, out\_max = 506 014) on *E. coli* K-12 sequences (promoters, terminators, random).

### Per-experiment findings (prokaryote)

| Step | Experiment | Key result |
|------|-----------|-----------|
| 1 | K-mer scan | Top k-mers produce **negative** SW activations; AT-rich hexamers dominate. GC correlation r = **−0.09** (opposite sign to EUK r = +0.19). |
| 2 | Token omission | 30 sequences (10/label); no significant context-specificity across promoter/terminator/random. |
| 3 | Shuffle controls | **All four shuffles significantly perturb SW activation (p < 0.01, n = 90)**, the opposite of EUK behaviour. PROK SW is context-sensitive above the single k-mer level. |
| 4 | K-mer motif enrichment | AT-rich motif hint (OR = 1.5, p_adj = 0.043); below FDR threshold. |
| 5 | Gradient attribution | 150 sequences (50/label), all valid; saliency profiles computed. |
| 6 | Attribution comparison | Gradient vs omission agreement: mean Spearman ρ = −0.077, all contexts n.s. Methods disagree in prokaryote context. |
| 7 | Hexamer causal test | r(signed SW activation, KL divergence) = **−0.710** (p ≈ 0, t = −118). The negative sign is a write-direction convention: the PROK SW writes with **negative** activations, so the most strongly (most negative) activating AT-rich tokens carry the largest write magnitude and the largest ablation KL. On **\|write\| magnitude** the correlation is **r = +0.710**, matching EUK (+0.437) — the same magnitude-scaled mechanism, not an inversion. |
| 8 | Causal tracing | Mean ΔPPL = **+9.47 log-PPL units** (random control Δ = 0.00033). Uniform across contexts: promoter=9.15, terminator=9.34, random=9.93. No context-specificity — SW is a general-purpose component for E. coli sequences. |
| 9 | Ablation regression | R² = **0.347** (higher than EUK R² = 0.158). GC fraction (β = −0.306) and k-mer entropy (β = −0.307) dominate: AT-rich, low-complexity sequences suffer most from SW ablation, consistent with AT-rich hexamers being the top activators. |

### Biological interpretation (prokaryote)

PROK SW row 1927 shows **inverse GC preference** (AT-rich activators, negative activations)
and **context-sensitivity** (shuffle controls significant, p < 0.01). The hexamer causal
test (signed-activation r = −0.710) initially looks inverted versus EUK, but the sign is a
write-direction convention: because the PROK SW writes with negative activations, the most
strongly (most negative) activating AT-rich tokens carry the largest write magnitude and
their predictions are most disrupted by ablation. On |write| magnitude the correlation is
**r = +0.710**, the same magnitude-scaled mechanism as EUK (+0.437). The PROK SW is therefore
not a "suppressive gate" opposite to EUK but the same magnitude-driven component with a
flipped activation sign.

The high causal tracing ΔPPL (+9.47 vs EUK ~+3 log-PPL) and higher composition-explained
variance (R² = 0.347) indicate the PROK SW is more deeply integrated into the model's
residual stream than the EUK SW, despite lying at an earlier layer (L2 vs L4). The mechanism
remains fundamentally composition-driven (GC/AT content, k-mer entropy) rather than
biologically regulatory, consistent with the cross-kingdom transfer null result.

---

## Cross-Kingdom Transfer Experiment

Tests whether each model's super-weight activates equally on sequences from the opposite
biological domain. Run with `scripts/interpretability/run_cross_kingdom_transfer.py`.

| Model | In-domain | In-domain mean | Out-domain | Out-domain mean | Fold-change | Mann-Whitney p |
|-------|-----------|---------------|-----------|----------------|-------------|----------------|
| GENERator eukaryote (SW L4 r2371) | hg38 | 6 581.6 | E. coli | 6 610.4 | 1.004 | 0.538 (n.s.) |
| GENERator prokaryote (SW L2 r1927) | E. coli | 10 637.8 | hg38 | 10 709.7 | 1.007 | 0.644 (n.s.) |

**Finding**: Both super-weights fire at statistically indistinguishable levels on in-domain
and out-of-domain sequences (~0.4–0.7% fold-change, p > 0.5). Super-weights are not
kingdom-specific detectors — they encode low-level statistical features (k-mer composition,
sequence length statistics) that are present across both eukaryotic and prokaryotic genomes.

---

## Overall Synthesis — GENERator 3B SW Row 2371

Six interpretability experiments consistently characterise SW row 2371 (layer 4,
out_max = 375 361) as a **context-insensitive housekeeping super-weight**.

> **Note (canonical index):** the EUK detector converged on **two** layer-4 super-rows —
> the dominant row 2371 (characterised below) and a secondary row 1522 at the same layer.
> Causal tracing zeroes both rows together; the per-experiment characterisation below was
> run on the dominant row 2371.

### Per-experiment findings

| Step | Experiment | Key result |
|------|-----------|-----------|
| 1 | K-mer scan | Nearly all 4,096 6-mers activate the SW (IQR 387K–409K). Only poly-A (AAAAAA) ≈ 0. Top-5: CCT/CAG-family GC-rich k-mers. GC correlation r = +0.19. |
| 2 | Token omission | No significant context specificity (enhancer vs random p = 0.087, promoter p = 0.29). Peak omission tokens cluster at sequence start — likely a 384 bp window edge artifact. |
| 3 | Shuffle controls (all 4) | **All four shuffles (mono, dinuc, kmer_block, trinuc) produce zero mean activation shift (p > 0.31, n=90).** Trinucleotide-preserving shuffle (strictest control): p=0.841. SW encodes token identity, not context at any level above single k-mer. |
| 4 | Causal hexamer test *(new)* | r(SW activation, KL divergence) = **0.437** across all 4,096 k-mers (p=4.46×10⁻⁷). SW ablation specifically disrupts predictions for CC/CT k-mers (top SW activators); AAAAAA KL ≈ 0. **Causal chain closed.** |
| 5 | Motif enrichment | No motif reaches FDR < 0.05 (Fisher exact, BH-corrected). Strongest hint: NF-κB (OR = 3.1, p_adj = 0.16). AT-rich, CpG-rich, TATA-box, splice donor all depleted (OR ≈ 0). |
| 6 | Attribution comparison | Gradient vs omission agreement: mean Spearman ρ = +0.043 (p = 0.040), 46.7% of sequences ρ < 0. Enhancers show the best agreement (ρ = +0.10). Gradient saliency is unreliable for this neuron, likely due to activation saturation. |
| 7 | Ablation regression | Sequence composition explains only R² = 0.158 of variance in ΔSW. Complexity (β = −1.48) and k-mer entropy (β = +1.05) are the dominant predictors. Ablation disrupts perplexity by +2.7–3.3 log-PPL units equally across all genomic contexts. |

### Biological interpretation

SW row 2371 functions as a **token-identity detector for CC/CT hexamers**, not a
biologically tuned regulatory-element detector. The causal hexamer test (r=0.437,
p=4.46×10⁻⁷) now confirms that the correlation between SW activation and k-mer identity
is mechanistically meaningful: SW ablation specifically destroys the model's ability to
predict from CC/CT tokens. The trinucleotide shuffle (p=0.841) rules out any sensitivity
to sequence context above the individual 6-mer level.

The mechanism in full:
- Large input activation (in_max ≈ 67,550) from CC/CT k-mer embeddings
× normal weight magnitude (max ≈ 1.85 in down_proj)
= extreme output activation (out_max = 375,361)
→ propagates through residual stream → catastrophic PPL when removed

The mild NF-κB / CTCF k-mer enrichment (OR = 3.1, p_adj = 0.16) remains an
interesting but unvalidated secondary signal.

### Recommended next steps

1. ~~Prokaryote k-mer scan: test if prok SW fires on Shine-Dalgarno (AGGAGG) instead of CC/CT.~~ ✅ Done — AT-rich preference, not Shine-Dalgarno.
2. Nucleotide Transformer v2: provides a byte-level transformer to decouple architecture from tokenizer.
3. Test NF-κB enrichment at larger k (top-500 foreground) and cross-reference with CTCF ChIP-seq.
4. SW-aware INT4 downstream benchmark: retain SW rows in FP16, INT4 all else → report GUE accuracy.
5. ~~Investigate PROK SW causal paradox (r = −0.710)~~ ✅ Resolved — the negative sign is a write-direction convention; on |write| magnitude r = +0.710, matching EUK's magnitude-scaled mechanism. PROK SW row 1927 is the same magnitude-driven component with a flipped activation sign, not a suppressive gate.

---

## Directory Structure

```
/  (repo root)
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
│   ├── detection/                # SW detection (--pad_to_multiple required for NTv3)
│   ├── evaluation/               # GUE fine-tune / ablation / epistasis harnesses
│   ├── analysis/                 # figure + composition/motif analyses
│   ├── mechanism/                # ── mechanism experiments (Aug 2026) ──
│   │   ├── run_pretrained_epistasis.py     # is the pair intrinsic to pretraining?
│   │   ├── run_compensation_circuit.py     # how does compensation work? (two geometries)
│   │   ├── run_norm_matched_control.py     # falsification: SW-specific or just norm?
│   │   ├── run_direction_vs_magnitude.py   # E1: intervention breaking that confound
│   │   ├── run_codominance.py              # E2: break-pair / make-pair co-dominance test
│   │   ├── run_ensemble_encoding.py        # what the ensemble encodes (sae | direct modes)
│   │   ├── run_attention_sink.py           # attention-sink / implicit-bias diagnostics
│   │   ├── run_sw_steering.py              # causal steering of generated composition
│   │   └── run_steering_biological.py      # steering + genomic-realism readouts
│   └── compression/              # ── compression experiments ──
│       ├── run_proximity_confound_control.py   # kills "shadow redundancy" (layer-depth)
│       ├── run_pair_aware_compression.py       # INT2-8 dose-response on the SW pair
│       ├── run_per_tensor_sw_exemption.py      # exemption across 4 granularities
│       ├── run_group_scale_preservation.py     # group (top-M) preservation, g=64/128
│       └── run_destructive_sw_protection.py    # destructive regime + block-size axis
├── sae/
│   ├── collect.py                # --store_dtype float32 (fp16 clamp destroys SW variance)
│   ├── train.py                  # --standardize (SW channel is 6,888x median sd)
│   ├── model.py                  # persists data_scale in the checkpoint
│   ├── analyze.py                # applies data_scale; hexamer-probe analysis
│   └── analyze_real_sequence.py  # real-sequence features, n_active beside every r
├── data/
│   └── regions/hg38/         # BED files for hg38 genomic regions
│       ├── promoters_262kb.bed
│       ├── enhancers_ccre_262kb.bed
│       └── random_262kb.bed
├── results/
│   └── mechanism/            # mechanism .md reports (raw JSON/CSV not yet committed, see MISSING_COLLEAGUE_ARTIFACTS.md)
├── docs/
│   └── superweight_paper.txt  # Reference paper (Yu et al. 2024)
├── paper/
│   ├── main.tex                  # Restructured 5-part-arc manuscript (current)
│   ├── main_old.tex              # 900-line prior version (reference)
│   └── media/
│       ├── image_fig4.{png,pdf}  # Figure 4 — GUE functional consequences
│       └── image_fig5.{png,pdf}  # Figure 5 — Compression: shadow redundancy + INT4
├── scripts/
│   ├── analysis/
│   │   ├── plot_figure1.py             # Fig 1 — architecture restriction
│   │   ├── plot_figure2.py             # Fig 3 — kingdom asymmetry composite
│   │   ├── plot_figure4.py             # Fig 4 — GUE functional consequences (NEW)
│   │   ├── plot_figure5.py             # Fig 5 — compression + 100k-INT4 (NEW)
│   │   ├── plot_sw_mechanistic.py      # Fig 2 — quadratic amplifier mechanism
│   │   ├── plot_activation_lifecycle.py
│   │   ├── plot_architecture_schematic.py
│   │   ├── plot_composite_figure.py
│   │   ├── plot_model_ablation_comparison.py
│   │   └── plot_quantization_sw_comparison.py
│   ├── detection/
│   │   ├── run_detection.py          # CLI: detect super weights
│   │   ├── run_ablation.py           # CLI: ablation test (perplexity)
│   │   └── *.sbatch                  # SLURM job scripts for detection
│   ├── evaluation/
│   │   ├── run_gue_ablation.py             # GUE fine-tune + 3/4-condition ablation
│   │   ├── run_gue_per_row_ablation.py     # GUE ablation per individual SW row
│   │   ├── run_gue_multiseed.py            # GUE fine-tuning + ablation over 3 seeds
│   │   └── analyze_superrow_proximity.py   # Monte-Carlo clustering of SW coordinates
│   ├── compression/
│   │   ├── run_compression_sweep.py           # Progressive pruning sweep (5 criteria)
│   │   ├── run_quantization_ablation.py       # Per-row RTN INT4/INT8 quantization sensitivity
│   │   ├── run_whole_model_quantization.py    # Full-model INT4 with configurable probe (NEW: --n_probe_seqs / --probe_seq_len / --probe_seed)
│   │   ├── run_int4_downstream_benchmark.py   # Practical INT4 benchmark (4 conditions, splice)
│   │   └── int4_downstream_benchmark_splice.sbatch  # SLURM job for benchmark
│   ├── interpretability/
│   │   ├── run_sw_gradient_attribution.py  # Gradient saliency maps for SW rows
│   │   ├── run_sw_causal_tracing.py        # SW row ablation + activation-shift
│   │   ├── run_sw_shuffle_controls.py      # Shuffle controls (dinuc, mono, kmer_block, trinuc)
│   │   ├── run_sw_hexamer_causal.py        # Causal KL test for all 4,096 hexamers
│   │   ├── run_sw_kmer_scan.py             # K-mer activation scan
│   │   ├── run_sw_token_omission.py        # Token omission attribution
│   │   ├── run_sw_ablation_regression.py   # Ablation regression analysis
│   │   ├── analyze_sw_kmer_motifs.py       # Motif enrichment in top SW-activating k-mers
│   │   └── compare_attribution_methods.py  # Spearman ρ: gradient saliency vs token omission
│   └── dev/
│       ├── debug_*.py                      # Per-model activation debug scripts
│       ├── inspect_*.py                    # One-off model inspection scripts
│       ├── probe_ablation.py               # Interactive ablation playground
│       └── *.sbatch                        # Dev/debug SLURM job scripts
├── results/
│   ├── super_weight_index.json                          # Detected SW coordinates (all models)
│   ├── ablation_results.json                            # Perplexity delta results
│   ├── gue_ablation_results.json                        # GUE task ablation (single-seed)
│   ├── gue_multiseed_results.json                       # GUE task ablation (3-seed, mean±std)
│   ├── gue_per_row_ablation.json                        # GUE ablation (per SW row)
│   ├── compression_sweep_dnabert2_prom_core_notata.json # Compression sweep results
│   ├── compression_sweep_dnabert2_prom_core_notata.png  # Compression sweep plot
│   ├── quant_ablation_generator_int4.json               # Generator INT4 quant ablation
│   ├── quant_ablation_dnabert2_*_int4.json              # DNABERT-2 INT4 quant ablation
│   ├── sw_grad_attribution.json / .png                  # SW gradient saliency (hg38)
│   ├── sw_causal_tracing.json / .png                    # SW ablation + activation shift
│   ├── sw_shuffle_controls.json                         # All 4 shuffle types (incl. trinuc)
│   ├── sw_hexamer_causal.json                           # Causal KL test for all 4,096 k-mers
│   ├── superrow_proximity.png                           # SW clustering analysis plot
│   └── *_activation_profile.png                        # Layer-wise activation plots
├── stubs/                      # Type stubs for models without type annotations
├── reports/
│   └── manager_briefing.md
└── tests/
    ├── test_hooks.py
    └── test_detection.py
```

---

## Reproducing the mechanism results (Aug 2026)

All commands assume `PYTHONPATH=.`. DNABERT-2 needs the `dnabert` env
(transformers 4.29.x); GENERator/NTv3 need the `generator` env.

```bash
# Is the redundant pair intrinsic to pretraining? (MLM loss, no task head)
python scripts/mechanism/run_pretrained_epistasis.py \
    --out results/mechanism/epistasis_pretrained_vs_finetuned.json

# NTv3 detection — --pad_to_multiple 256 is REQUIRED (conv/deconv U-Net)
python scripts/detection/run_detection.py --model ntv3 --pad_to_multiple 256 \
    --mode superrow --threshold 0.02 --max_iter 30 \
    --out results/mechanism/super_weight_index_ntv3_deep.json

# NTv3 splice fine-tuning — --max_length 400 is REQUIRED (nucleotide-level tokenizer)
python scripts/evaluation/run_gue_multiseed.py --model ntv3 --task splice/reconstructed \
    --gue_root $GUE --seeds 0 --max_length 400 --batch 8 --epochs 5

# Direction vs magnitude (breaks the norm/super-weight confound)
python scripts/mechanism/run_direction_vs_magnitude.py --max_seeds 3

# Co-dominance: does breaking the pair move criticality to a singleton?
python scripts/mechanism/run_codominance.py --mode break_pair --model dnabert2 \
    --layer 9 --anchor 264 --partner 294 --ratios 1.0 0.5 0.25 0.1 0.03 \
    --ckpt results/gue_checkpoints_multiseed/dnabert2_reconstructed/seed_0/model_state.pt \
    --out results/mechanism/codominance_break_dnabert2.json

# Compression: the "shadow redundancy" control, and exemption across granularities
python scripts/compression/run_proximity_confound_control.py --model dnabert2 \
    --task prom/prom_core_notata --gue_root $GUE --ckpt_dir <seed_dir> --out <json>
python scripts/compression/run_per_tensor_sw_exemption.py --task splice/reconstructed \
    --granularities per_row group_64 group_128 per_tensor --bits 4 3 2 --out <json>

# SAE — float32 collection and standardised training are both required
python sae/collect.py --model generator --layer 4 --store_dtype float32 --max_tokens 1000000 ...
python sae/train.py --acts_dir <dir> --standardize --k 64 --dict_mult 4 --steps 50000
python sae/analyze_real_sequence.py --sae_ckpt <ckpt> --acts_dir <dir> --min_active 100
```

---

## Quick Start

### 1. Detect super weights

```bash
python scripts/detection/run_detection.py --model generator
python scripts/detection/run_detection.py --model dnabert2 --probe poly_a
python scripts/detection/run_detection.py --model ntv3 --threshold 0.05
```

Results are written to `results/super_weight_index.json` and an activation profile PNG.

### 2. Run ablation (perplexity / entropy)

```bash
python scripts/detection/run_ablation.py --model generator
python scripts/detection/run_ablation.py --model dnabert2
```

Reads super weight coordinates from `super_weight_index.json` and writes deltas to
`results/ablation_results.json`.


# with structured-random control (matches SW layer distribution + row-repetition pattern)
python scripts/evaluation/run_gue_ablation.py --model dnabert2 --task prom/prom_core_notata \
    --structured_rand 10
```

Requires the GUE dataset directory on `$GUE_DATA_PATH`.

### 4. Progressive compression sweep

```bash
python scripts/compression/run_compression_sweep.py \
    --model dnabert2 --task prom/prom_core_notata \
    --gue_root $GUE_DATA_PATH \
    --ckpt_dir results/gue_checkpoints/dnabert2_prom_core_notata \
    --fracs 0.5 1 2 5 10 15 20 30
```

Ranks all non-SW rows by five criteria and sweeps pruning fractions, producing
a sensitivity curve and JSON + PNG output.

### 5. Quantization sensitivity ablation

```bash
# GENERator — perplexity mode (INT4 recommended for visible signal)
python scripts/compression/run_quantization_ablation.py \
    --model generator --bits 4 \
    --out results/quant_ablation_generator_int4.json

# DNABERT-2 — GUE classification mode (requires fine-tuned checkpoint)
python scripts/compression/run_quantization_ablation.py \
    --model dnabert2 --task prom/prom_core_notata \
    --gue_root $GUE_DATA_PATH \
    --ckpt_dir results/gue_checkpoints/dnabert2_prom_core_notata \
    --bits 4
```

Sweeps five conditions (baseline, yu_all, sw_fragility, near_sw, random) at increasing
quantization fractions. Use `--bits 8` for INT8; default fracs are 1, 5, 10, 20, 30%.
See Results section for interpretation.

### 5b. INT4 downstream benchmark (practical compression)

```bash
python scripts/compression/run_int4_downstream_benchmark.py \
    --task splice/reconstructed \
    --ckpt_dir results/gue_checkpoints/dnabert2_splice_reconstructed \
    --near_sw_frac 10.0
```

Four conditions (fp16_baseline, naive_int4, yu_all_int4, near_sw_int4, random_int4).
Reports accuracy, MCC, memory, inference time, and theoretical weight savings per condition.

### 6. Mechanistic interpretability (gradient saliency + causal ablation)

Requires hg38 FASTA and the BED region files under `data/regions/hg38/`.

```bash
# Gradient saliency: which token positions drive the SW activation?
python scripts/interpretability/run_sw_gradient_attribution.py \
    --n_seqs 30 \
    --promoters data/regions/hg38/promoters_262kb.bed \
    --enhancers data/regions/hg38/enhancers_ccre_262kb.bed \
    --random    data/regions/hg38/random_262kb.bed

# Causal ablation: are the SW rows necessary? Do they differentiate real vs. shuffled seqs?
python scripts/interpretability/run_sw_causal_tracing.py \
    --n_seqs 30 \
    --promoters data/regions/hg38/promoters_262kb.bed \
    --enhancers data/regions/hg38/enhancers_ccre_262kb.bed \
    --random    data/regions/hg38/random_262kb.bed
```

Smoke test with `--n_seqs 2` first to confirm FASTA/BED paths are accessible.

### 7. Super-row proximity analysis

```bash
python scripts/evaluation/analyze_superrow_proximity.py
```

Monte-Carlo clustering analysis of SW coordinates. Produces
`results/superrow_proximity.png`.

```bash
python scripts/evaluation/run_gue_ablation.py --model dnabert2 --task prom/prom_core_notata
python scripts/evaluation/run_gue_per_row_ablation.py --model ntv3 --task splice/reconstructed
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
