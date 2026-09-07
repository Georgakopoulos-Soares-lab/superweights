# CUT_LIST.md

**Nothing is deleted from the repository.** Cuts are demotions. Every item below keeps its
data, its script and its provenance in `results/keep/`; only its position in the manuscript
changes.

---

## Main text (keep)

- ‖U_k‖_F derivation, per-*i* decomposition, genomic ranks
- NLP external validation (E1) and prospective lock
- Detect-then-ablate panel, ΔPPL, random-row controls
- DNABERT-2 L7/r603 source-vs-propagator resolution
- Impulse T/C/KL across architectures + control arms
- Hexamer scan and activation → ablation-KL relationship
- Compact shuffle experiment (D-007)
- Bidirectional steering (E3)
- GENERator 35-task dissociation — 4 representative rows + compact distribution
- DNABERT-2 splice −25.5% and per-row ablation
- NTv3 five-seed replication
- Compression: one paragraph

## Supplement (demote)

| Item | Why demoted |
|---|---|
| PROK sparse autoencoder | Recovers the same GC/AT axis the causal hexamer test already establishes. Adds a second method, not a second finding. Keep as supporting evidence. |
| EUK SAE | Incomplete (training instabilities). Do not mention as forthcoming in main text — reviewers read that as a gap. |
| Motif enrichment (Fisher + BH) | All null except EUK homopolymers. One sentence in main text, full analysis to supplement. |
| OLS regression on composition features (R² = 0.158 / 0.347) | Modest variance explained; the conclusion it supports (token identity dominates) is better made by the hexamer test. |
| Kingdom-specificity test | Both directions null (p = 0.538, p = 0.644). One clause in main text at most. |
| Monte-Carlo proximity analysis (n = 50,000) + structured-random layout control | Good controls, but control-of-a-control detail. |
| DNABERT-2 promoter result | −12.6 ± 16.7 pp, p = 0.40, strong seed dependence. Reporting it as "inconclusive" in main text still invites argument. |
| Full pruning sweep (5 criteria × 8 fractions) | Supports one sentence about far-SW fragility. |
| INT4 machinery, U_k-guided precision allocation, whole-model quantization | Mostly null at the fractions tested. |
| Relay-head ablations | Negative vs. random-head controls. Report as a negative control in supplement. |
| GC↔AT counterfactual coordinate swaps | Small but non-zero KL. Only promote if E3 makes it interpretable — a swap result is much more meaningful next to a working steering result. |
| Full 35-task GENERator table | Four representative rows in main text; full table here. |
| Caduceus static weight analysis | Not protocol-symmetric. Methods footnote only. |

## Removed as claims

| Claim | Replacement |
|---|---|
| "Shadow redundancy" | Nothing. The pruning data contradict it. |
| "Extremely tolerant to pruning and INT4 quantization compared with random rows" | "The NLP SW-preservation heuristic did not transfer" |
| "We scanned eight of the best genomic language models" | Coverage table |
| Any claim that C is necessary for criticality | "C describes routing geometry, not criticality" |
| "Suggesting refined SW-aware compression schemes are possible" | Removed — the results do not establish it |

## Retained from v15 verbatim

- The Limitations block. It is well written and honest; update it against the new
  structure rather than rewriting it.
