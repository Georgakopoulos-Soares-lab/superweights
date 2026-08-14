# NEXT_SESSION.md

**Session:** 2026-08-14, E6 Stage A (fourth session, direct continuation of the E5 Gate-0
session on 2026-08-13).
**Status: a bounded, preregistered weight-only confirmatory experiment ran to completion and
was assigned Branch C (no independent replication) by its own locked, mechanical rule.** This
was a near-miss on effect-size magnitude (0.28% short), not a clean direction failure. No
forward pass was run in this session either. No manuscript prose was touched. The prior
session's handoff (writing R1/R6, or resolving N-013/N-014/N-015) is **unchanged and still the
default next step** unless whoever reads this deliberately picks up a fresh E6 continuation.

---

## 1. What happened this session

E5 (2026-08-13) found that its diagonal `c_{k,i}` decomposition captures the exact
gated-FFN amplifier much more reliably for three NLP rows than for three genomic rows
(`f_cross` 0.067–0.193 vs. 0.207–0.829). That comparison used only six *discovery* rows. This
session asked whether the pattern independently replicates on rows chosen from evidence that
predates E5 entirely — and, if it did, whether the excess cross-pathway strength in genomic
rows is layer-specific (not just "the whole layer is like that") and constructively organized
(not just large numbers that mostly cancel).

New, self-contained experiment: `paper-salvage/experiments/E6_cross_geometry/`. Does not
modify E5's files, `results/`, `CLAIMS_LEDGER.md`, `PAPER_OUTLINE.md`, or `CLAUDE.md`.

1. **Panel frozen blind to cross-term values**
   (`experiments/E6_cross_geometry/CONFIRMATION_PANEL.md`): 3 NLP rows (OLMo-7B, layers 2, 7,
   24 — all the *same* output row, 269, that Yu et al.'s own Table 2 records recurring across
   layers; the E5 primary was layer 1, excluded) and 10 genomic rows (9 additional DNABERT-2
   rows + 1 additional GENERator EUK row, all from this repo's pre-existing, May-2026
   activation-detection sweep). GENERator PROK and Evo1 mechanically excluded, as E5 already
   required.
2. **Prereg locked**: `paper-salvage/docs/prereg/PREREG_cross_geometry_stageA.md`
   (sha256 `4cbe4416719b33a7a6eac256dbba49d40ef5df822cf15d72167374e58fc51bb5`, UTC
   2026-08-14T12:08:49+00:00) after 7/7 synthetic tests and a 6/6 exact-reproduction
   regression check against E5's own published numbers, all pre-lock.
3. **Confirmatory run, weight-only, all 13 candidates:**
   - **Complete separation held**: every genomic `f_cross` (0.113–0.830) exceeds every NLP
     `f_cross` (−0.004–0.076). Exact, non-asymptotic rank-sum test: **p = 0.0035** (the single
     most extreme of 286 possible arrangements).
   - **The locked 1.5× effect-size margin narrowly failed**: genomic minimum 0.1134 vs. the
     required 0.1137 — a 0.28% shortfall.
   - **Mechanical result: Branch C.** Per the prereg's own rule, this stops the experiment —
     Q2 (layer-specificity) and Q3 (coherence) were computed as descriptive measurements but
     were never part of the binding decision.
4. **The near-miss traces to one row**: DNABERT-2 L7/r603 (`f_cross` 0.1134, far below the
   next-smallest genomic value of 0.297) — a row **already flagged in this repository's own
   ledger** (C-010: "propagator, not source") as structurally atypical, independently of E6.
   **This row was not excluded and the decision was not recomputed without it** — the
   near-miss is reported exactly as measured, per the governing instruction against post-hoc
   rescues.
5. **Task 6 (causal-feasibility audit) was not written** — conditional on Branch A only.
6. **Claim-ledger recommendations, not enacted:**
   - **C-002 / C-003 (row-ranking): recommend RETAIN**, with a Methods clarification. Exact-
     form row ranking held on 12 of 13 confirmation-panel rows (all rank ≤ 4 of their layer),
     not just E5's original six.
   - **C-032 (granularity/PR): recommend HOLD.** E5's measurement-validity flag is neither
     confirmed nor refuted by a result this close to its own pre-committed bar. No
     cross-term-aware replacement metric was invented — any future one gets a new claim ID.

Full numbers, the complete decision trace, and the reasoning behind every threshold:
`paper-salvage/experiments/E6_cross_geometry/RESULTS.md`.

## 2. Blockers

**None mechanical for E6 itself** — it completed cleanly and reached its own preregistered
decision, which is a successful (if inconclusive) outcome of the discipline, not a bug.

Carried over, unchanged from the prior two sessions:

- **N-013, N-014, N-015** (`paper-salvage/docs/CLAIMS_LEDGER.md`) — unchanged.

New, optional, not queued by default:

- **A freshly-independent E6 continuation.** The near-miss leaves open whether a larger,
  genuinely cross-model NLP arm (this session's was one model's recurring channel across
  layers, disclosed as a limitation up front) or a different, more powered genomic panel
  would clear the 1.5× margin. This would need its **own new preregistration** — not a
  re-run or threshold adjustment of the locked `PREREG_cross_geometry_stageA.md`.
- **A principled cross-term-aware granularity/PR metric to replace C-032's diagonal-based
  one.** Not attempted in this pass, deliberately — see `RESULTS.md`'s C-032 recommendation.
- **Whether DNABERT-2 L7/r603's atypical status (C-010) extends beyond that one earlier
  finding.** E6 independently reproduced the same qualitative diagnosis (poor exact-form
  rank, 686/768) via a completely different computation than C-010 used. Worth noting for
  whoever eventually writes about the DNABERT-2 10-row ensemble, but not investigated further
  here.

## 3. Canonical report paths (this session)

| What | Path |
|---|---|
| Panel provenance + freeze | `paper-salvage/experiments/E6_cross_geometry/CONFIRMATION_PANEL.md` |
| Locked Stage-A prereg | `paper-salvage/docs/prereg/PREREG_cross_geometry_stageA.md` |
| Shared library + synthetic tests | `paper-salvage/experiments/E6_cross_geometry/cross_geometry_lib.py`, `test_cross_geometry_lib.py` |
| E5 regression check | `paper-salvage/experiments/E6_cross_geometry/validate_against_e5.py` |
| Confirmatory script | `paper-salvage/experiments/E6_cross_geometry/run_stageA.py` |
| **Results, decision, claim-ledger recommendations** | `paper-salvage/experiments/E6_cross_geometry/RESULTS.md` |
| Raw output | `results/e6_stageA.json` |

Everything from the three prior sessions (E1/E2/E4 results, the reconciliation pass, E5 Gate
0) is unchanged — see `paper-salvage/docs/PROJECT_STATUS.md`.

## 4. What this session did NOT do (deliberately)

- Did not touch E5's files, or re-run E5 Gate 1/Gate 2.
- Did not edit `PREREG_cross_geometry_stageA.md` after locking it.
- Did not exclude DNABERT-2 L7/r603, adjust the 1.5× margin, or otherwise recompute the
  decision after seeing it was a near-miss.
- Did not invent a replacement geometry hypothesis or a cross-term-aware granularity metric.
- Did not run any causal experiment, ablation, or forward pass.
- Did not touch `CLAIMS_LEDGER.md`, `PAPER_OUTLINE.md`, `CLAUDE.md`, or any manuscript file —
  the C-002/C-003/C-032 recommendations above are recommendations only.

## 5. What a future session should actually do — pick one lane

**Lane A — write** (unchanged from prior sessions). R1 and R6 have enough real evidence to
draft now. Does not require any new experiment or touch E5/E6.

**Lane B — close one provenance gap** (unchanged). Pick exactly one of N-013/N-014/N-015.

**Lane C — a fresh, separately preregistered E6 continuation.** Only if explicitly
instructed. Two independent directions, not to be conflated: (a) a more powered/independent
panel (more NLP models, not just more layers of one), or (b) a principled cross-term-aware
replacement for the participation-ratio metric C-032 uses. Neither continues or edits the
locked `PREREG_cross_geometry_stageA.md` — each needs its own lock.

**Do not** mix lanes in one pass, and do not treat Lane C as queued — it needs its own
go-ahead, same as the prior session's Lane C did.

## 6. Unresolved scientific decisions carried over (unchanged, still open)

Identical to the prior two `NEXT_SESSION.md` versions' §6 (Evo1 Branch A vs B, C-017,
C-001/N-009, Evo1's missing secondary-dose KL, empty matched-norm arms, C-010 on eager,
ref [11]) — nothing in this session touched any of them, though item 4 above notes E6
independently corroborated C-010's existing diagnosis of DNABERT-2 L7/r603 without changing
its status.

## 7. Exact next commands

```bash
cd /work/11034/atzanakak/glm_super_weight/genomic-super-weights

# verify all three prereg locks still hold (E2, E5 Gate-0, E6 Stage-A)
python3 paper-salvage/src/prereg_lock.py verify --all

# re-run E6's synthetic tests if you touch cross_geometry_lib.py
ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python3 \
  paper-salvage/experiments/E6_cross_geometry/test_cross_geometry_lib.py   # expect: 7/7

# re-run E5's synthetic tests if you touch dimensionality_lib.py (E6 imports it)
LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python3 \
  paper-salvage/experiments/E5_dimensionality/test_dimensionality_lib.py  # expect: 12/12

# re-run the shared uk_frobenius.py tests if you touch it (untouched this session)
LD_LIBRARY_PATH=$ENV/lib PYTHONPATH=$PWD/paper-salvage/src \
  $ENV/bin/python -m pytest paper-salvage/src -q            # expect: 7 passed
```

## 8. Environment traps (unchanged from prior sessions, plus two new notes)

- Container has **no `python`**, only `python3`.
- Host `SSL_CERT_FILE` points outside the container; use `env -u SSL_CERT_FILE -u
  REQUESTS_CA_BUNDLE` for any `huggingface_hub` network call.
- `grlm` needs `LD_LIBRARY_PATH=$ENV/lib`.
- `grlm` also needs both `HF_HOME=/work/11034/atzanakak/ls6/huggingface/.hf-cache` (NLP
  shards) **and** `TRANSFORMERS_CACHE=/work/11034/atzanakak/ls6/nonbdna/cache/hf` (genomic
  checkpoints) set simultaneously to load the full E5/E6 panel.
- **New:** `HF_HUB_OFFLINE=1` is safe (and preferred, for reproducibility) for any row whose
  shard is already cached, but will hard-fail with `LocalEntryNotFoundError` on a genuinely
  new shard (e.g. OLMo-7B's layers 7/24, which live in different shard files than layer 1/2).
  Unset it only for the specific fetch that needs a new shard, and disclose the fetch in the
  results doc — do not leave it unset for the whole run by default.
- **New:** on this node, `torch`/CUDA import alone can take 1–2+ minutes under the visible
  single-core CPU allocation — a long silent startup before any script output is normal, not
  a hang. Building a `[d_ffn, d_ffn]` pair matrix on **CPU** (rather than moving weights to
  the GPU first) is also much slower than the same operation on the GPU (e.g. ~51s vs. an
  expected low single digits of seconds for `d_ffn=11008`) — correctness is unaffected either
  way, but a future session doing many more such rows should move tensors to CUDA before
  calling `pair_matrix`/`exact_uk_all_rows`.
- Loading GENERator EUK for weight-only inspection needs
  `AutoModelForCausalLM.from_pretrained(...)`; DNABERT-2/NTv3 need
  `AutoModelForMaskedLM.from_pretrained(...)` — a bare `AutoModel.from_pretrained(...)` does
  not expose the module paths `uk_frobenius.py`'s adapters require for these three.
- No `evo` conda env; Evo1/StripedHyena work needs the `evo2.sif` Apptainer container.
- `paper-salvage/results/` is gitignored via the `results/` pattern — use `git add -f`.
- No LaTeX toolchain on this node.
