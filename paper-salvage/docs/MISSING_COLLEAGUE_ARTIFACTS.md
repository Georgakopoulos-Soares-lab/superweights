# Missing colleague-branch raw artifacts

Produced during integration of `origin/mechanism-and-negative-results` (colleague commit
`5b0220c513b1cd7db379f3023ae04ad3fd5801aa`) onto `integrate/mechanism-and-negative-results`.
See `paper-salvage/docs/COLLEAGUE_BRANCH_AUDIT.md` for the full provenance audit this manifest
is derived from.

**Purpose.** Every finding below has a real, inspectable producing script and a detailed
markdown report, but the raw JSON/CSV/PNG output the report's numbers came from is not present
anywhere in this repository (confirmed via `git ls-tree -r` on the colleague branch and a
filesystem search — see audit §2). This is a recovery/reproduction checklist, not a claim
that the work didn't happen. **Nothing here is reproduced or recreated in this pass** — this
document only records what is missing and what would be needed to close each gap.

> **UPDATE 2026-08-17, see `docs/DECISIONS.md` D-024.** The author has since decided to adopt
> the findings below as established results facts, notwithstanding the artifacts remaining
> physically absent — see `docs/CLAIMS_LEDGER.md` C-036–C-043, C-001, C-029. **This manifest
> is no longer a precondition for citing these numbers.** It stays open as a genuine
> reproducibility gap: recovering or reproducing these artifacts is still worthwhile for
> independent verification, checkpoint provenance, and anything that would need the actual
> data (plots, error bars beyond what the reports state, re-analysis) — just not as a gate on
> claim status anymore.

---

## Priority order

- **P0** — DNABERT-2 redundant pair, pretrained-intrinsic pair, norm/codominance mechanism
  (claims A, B, C — the core new mechanistic result of the session)
- **P1** — GENERator causal steering, attention sink (claims D, E)
- **P2** — corrected NTv3, corrected PROK, destructive quantization controls, Evo1
  supplementary artifacts (claims F, G, H, I-Q4)

---

## A. DNABERT-2 redundant pair (P0)

| Expected artifact | Producing script | Supports |
|---|---|---|
| Individual-row ablation results (10 rows, 3 tasks, multiseed) | `scripts/evaluation/run_sw_pairwise_epistasis.py` | Sum-of-parts −3.51 pp baseline |
| Joint all-10-row ablation (5 seeds) | same | −26.84 ± 2.56 pp headline |
| Critical-pair (L9/r264, L9/r294) separate + joint ablation | same | −0.02/−0.11 pp separate, −33.76 pp joint |
| Random-pair control set | same | +0.006 to +0.007 pp null |
| Full pairwise epistasis matrix, 3 tasks, top-7 ranking | same | Structural-relatedness enrichment (p=0.00014/0.0035/0.0035) |

External-existence check: not attempted this pass (no scientific rerun). Priority: **P0.1**
(the single highest-value recovery — this is the anchor result for the whole session).

## B. Pretrained-intrinsic epistasis (P0)

| Expected artifact | Producing script | Supports |
|---|---|---|
| Pretrained MLM-loss epistasis for the critical pair | `scripts/mechanism/run_pretrained_epistasis.py` | epistasis +2.0118 |
| Random-pair floor distribution | same | sd 0.0000147, 136,521× ratio |
| Top-k enrichment output (k=3,5,7) | same | p=0.032/0.0025/0.0035 |
| Pretrained-vs-finetuned rank correlation | same | ρ=+0.316, p=0.034 |

Priority: **P0.2**.

## C. Norm/codominance mechanism (P0)

| Expected artifact | Producing script | Supports |
|---|---|---|
| Layer-9 residual norm measurements (baseline vs. both-ablated) | `scripts/mechanism/run_compensation_circuit.py` | 17.20 → 7.14 |
| Depth-redundancy / asymmetric-head-readout tests | `run_compensation_circuit.py`, `run_norm_matched_control.py` | 91.9% survival; percentile 98 vs 31 |
| `FALSIFICATION_NORM_VS_SUPERWEIGHT.md` backing data (NTv3 vs DNABERT-2 norm-gap comparison) | `run_norm_matched_control.py` | 29.4× vs 4.5× gap; −0.02 vs −17.72 pp |
| E1/E2 codominance intervention (BREAK-PAIR, MAKE-PAIR) | `scripts/mechanism/run_direction_vs_magnitude.py`, `run_codominance.py` | Full 5-ratio epistasis migration table; NTv3 MAKE-PAIR null |

Priority: **P0.3** — this is the result the audit's §8 "chain" question hinges on; recovering
it would let the DNABERT-2/NTv3 comparison be independently re-verified rather than taken from
the report's own numbers.

## D. GENERator attention sink (P1)

| Expected artifact | Producing script | Supports |
|---|---|---|
| EUK/PROK attention-mass-at-BOS distributions | `scripts/mechanism/run_attention_sink.py` | 37.96%/28.72% at pos 0, 33.0×/25.0× uniform |
| Per-head argmax-at-BOS counts | same | 78.3% of heads; 60/60 probes |
| DNABERT-2 contrast (argmax-at-CLS) | same | 0/40 |
| Activation-magnitude-at-BOS ratio | same | 45,585× |

Priority: **P1.1**.

## E. GENERator causal steering (P1)

| Expected artifact | Producing script | Supports |
|---|---|---|
| Dose-response GC output (scales 0.0/0.5/1.0/2.0/5.0) | `scripts/mechanism/run_sw_steering.py` | 0.2944→0.3961→0.3549 |
| Random-row control | same | flat 0.396–0.399 |
| PROK replication | same | 17.87× span |
| Generated-sequence quality metrics (perplexity, dinuc KL, homopolymer, entropy) | `scripts/mechanism/run_steering_biological.py` | "quality flat across range" claim |

Priority: **P1.2**.

## F. Evo1 supplementary artifacts (P2)

| Expected artifact | Producing script | Supports |
|---|---|---|
| fp64 vs bf16 adjudication trace | (per `CHECKPOINT_A_REPORT.md` §A2 — script not identified in this pass) | 1,247,245.4 vs 1,245,184.0; ULP match |
| Per-block MLP/mixer contribution decomposition | same | +1.25e6 / +14.68e6 / +13.43e6 |
| Forced-survival rescue intervention | (per `CHECKPOINT_REPORT.md` T1.2) | rescued 0.0026% ≈ natural 0.0032% |

Priority: **P2.4** — lowest urgency; this branch's own N-001/N-002 already independently cover
the same territory (audit §4.1), so this is corroboration, not a load-bearing gap.

## G. NTv3 corrected experiment (P2)

| Expected artifact | Producing script | Supports |
|---|---|---|
| Corrected training config (max_length=400, per `sh` block in README) | `scripts/evaluation/run_gue_multiseed.py` | intended fix |
| Retrained checkpoint(s) | (launch script not identified in this pass) | MCC 0.86–0.91 |
| Corrected multiseed evaluation JSON, with `max_length` explicitly recorded | `run_gue_multiseed.py` | −0.02 pp ablation effect |

**This is the one claim in this manifest where the underlying bug is already AUDITED**
(audit §4.3, confirmed independently from code) — only the corrected replacement numbers are
missing. Priority: **P2.1** (highest within P2, precisely because the bug diagnosis is no
longer in question).

## H. PROK corrected experiment (P2)

| Expected artifact | Producing script | Supports |
|---|---|---|
| Corrected detection output using the `pseudomonadota` probe | `scripts/detection/run_detection_generator_prokaryote.sbatch` (probe arg changed) | L8/r260, out_max=30,167.07 |
| Updated `results/super_weight_index.json` entry + provenance block | manual/index-merge step | Referenced but **not present**; current file confirmed byte-identical to pre-colleague version |
| Archived old entry (`super_weight_index_generator_prokaryote_ORIGINAL_SUSPECT.json`) | same | Referenced but absent |
| D1 full biology rerun (hexamer KL, shuffle controls, motif enrichment, GC correlation) | (D1 rerun scripts not individually identified in this pass) | ρ=+0.0007, p=0.96; ΔPPL +1.25±0.54; GC-cost r=−0.661 |
| Retrained layer-8 SAE checkpoint | `sae/*.py` | PROK SAE claims |

Priority: **P2.2** — the missing index-file edit is the single most concretely falsifiable gap
found in the whole audit (report claims it was done; the pushed file shows it wasn't).

## I. Quantization Q2/Q4 (P2)

| Expected artifact | Producing script | Supports |
|---|---|---|
| Per-row SW-element quantization error (2 rows × 4 precisions) | `scripts/compression/run_pair_aware_compression.py` | 0.000e+00 at INT8/4/3/2 |
| 10-row `row_max/tensor_max` table | same or `run_per_tensor_sw_exemption.py` | 5/10 rows = 1.000 |
| Per-tensor exemption dose-response (14-cell table) | `scripts/compression/run_per_tensor_sw_exemption.py` | mean +0.048pp, t≈+0.15 |
| Group-wise (g=64/16) and destructive-regime outputs | `scripts/compression/run_group_scale_preservation.py`, `run_destructive_sw_protection.py` | Q4 destructive-regime null |

Priority: **P2.3** — Q1–Q3 are already independently established from code (no recovery
needed); only the empirical Q2 10-row table and Q4 destructive-regime numbers are missing.

---

## Recovery paths, in order of preference

1. **Ask the colleague directly** whether the raw outputs exist locally/on a scratch
   filesystem/on TACC and were simply not committed (the most likely explanation, given
   `results/` is gitignored by default and the commit's own message states only `.md` reports
   were carved out as an exception).
2. **Reproduce from the pushed scripts**, which are real and inspectable, once (1) is
   exhausted for a given artifact. This is explicitly **out of scope for this integration
   pass** — no reproduction was attempted here, per the governing instruction not to run any
   scientific experiment or rerun in this pass.
3. Only if the colleague confirms an artifact is permanently unrecoverable should a bounded,
   explicitly-scoped reproduction of that specific P0 item be planned — as its own session,
   with its own preregistration, following this project's standing discipline.
