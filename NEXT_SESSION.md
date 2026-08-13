# NEXT_SESSION.md

**Session:** 2026-08-13, E5 Gate 0 (third session of the day).
**Status: a bounded, preregistered weight-only experiment was opened and stopped at its own
first gate.** No forward pass was run. No manuscript prose was touched. The prior session's
handoff (writing R1/R6, or resolving N-013/N-014/N-015) is **unchanged and still the default
next step** unless whoever reads this deliberately picks up E5 instead.

---

## 1. What happened this session

A new, narrower hypothesis than the earlier feasibility audit was scoped and tested: does the
extreme within-row scalarization of canonical NLP super-weight rows (top-1 share 0.89–0.99)
versus the more distributed clean genomic rows (top-1 share 0.18–0.39) arise from unusually
strong multiplicative alignment between the down-projection term and the gated-amplifier term
— as opposed to either factor simply having a more extreme marginal distribution in NLP? This
is **not** a re-test of row-level recovery, which is already strong in both groups (see
`paper-salvage/experiments/E5_dimensionality/README.md`'s premise correction).

New, self-contained experiment: `paper-salvage/experiments/E5_dimensionality/`. Does not
modify `results/`, `CLAIMS_LEDGER.md`, `PAPER_OUTLINE.md`, or `CLAUDE.md`.

1. Preregistration locked: `paper-salvage/docs/prereg/PREREG_dimensionality_gate0.md`
   (sha256 `7e5083157f344a240f57e632b49c2bc99f3e919b68e973be9c078f99e753215b`, UTC
   2026-08-13T18:13:24+00:00). 12/12 synthetic-tensor tests green beforehand
   (`experiments/E5_dimensionality/test_dimensionality_lib.py`), cross-validated exactly
   against the existing `uk_frobenius` predictor and a from-scratch brute-force `U_k`.
2. Confirmatory Gate 0 run on the six-model primary panel (Llama-7B, Mistral-7B, OLMo-7B,
   GENERator EUK, DNABERT-2, NTv3 — PROK and Evo1 excluded per the prereg) — weight-only, GPU
   (A100-40GB, idle on this node), float64 throughout.
3. **Gate 0 stopped at §0B**, the preregistered exactness-robustness check, before the
   factor-decomposition/permutation-test machinery (§0C/§0D) ever ran on a real weight. Cross
   terms in the exact quadratic form exceed the locked 20% threshold for **all three genomic
   models** (GENERator EUK 82.9%, DNABERT-2 50.7%, NTv3 20.7%) and **none** of the three NLP
   models (6.7%–19.3%). This is exactly the mechanical stop condition the prereg specifies;
   it fired correctly, not by judgment call. Full numbers:
   `paper-salvage/experiments/E5_dimensionality/GATE0_RESULTS.md`.
4. **Gate 1 was not run and is not authorized.** No replacement hypothesis was generated in
   this pass, per the governing instruction for E5.

**Flagged, not acted on:** the same diagonal `c_{k,i}` decomposition whose fidelity Gate 0
just showed splits cleanly by group (81–93% of the exact norm for NLP vs. 17–79% for genomic)
is exactly what existing claims C-002, C-003, and C-032 (the granularity/participation-ratio
comparison) are built from. This is disclosed as a measurement-validity flag in
`GATE0_RESULTS.md`'s interpretive note. **No ledger row was edited, no claim was retracted or
downgraded, and no manuscript text was changed** — this pass does not adjudicate it.

## 2. Blockers

**None mechanical for E5 itself** — it stopped cleanly on its own preregistered rule, which is
a successful (if negative) outcome of the discipline, not a failure needing debugging.

Carried over, unchanged from the prior session:

- **N-013, N-014, N-015** (`paper-salvage/docs/CLAIMS_LEDGER.md`) — same status as before this
  session; nothing here resolves them.

New, optional, not queued by default:

- **The C-002/C-003/C-032 measurement-validity flag** (item 1.4 above / `GATE0_RESULTS.md`'s
  interpretive note). Closing this would mean defining and **separately preregistering** a
  cross-term-aware granularity metric — it is not a small patch to the existing one, and
  inventing one now (after seeing the diagonal form fail specifically on the genomic side)
  would be exactly the kind of post-hoc rescue this project's discipline forbids. Needs an
  explicit author decision to pursue, with its own new prereg, not a continuation of the
  locked `PREREG_dimensionality_gate0.md`.

## 3. Canonical report paths (this session)

| What | Path |
|---|---|
| E5 premise correction + panel + exclusions | `paper-salvage/experiments/E5_dimensionality/README.md` |
| Locked Gate-0 prereg | `paper-salvage/docs/prereg/PREREG_dimensionality_gate0.md` |
| Gate-0 shared library + synthetic tests | `paper-salvage/experiments/E5_dimensionality/dimensionality_lib.py`, `test_dimensionality_lib.py` |
| Gate-0 confirmatory script | `paper-salvage/experiments/E5_dimensionality/run_gate0.py` |
| **Gate-0 results (the stop, the numbers, the interpretive flag)** | `paper-salvage/experiments/E5_dimensionality/GATE0_RESULTS.md` |
| Raw output | `results/e5_gate0.json` |

Everything from the two prior sessions (E1/E2/E4 results, the E2 prereg lock, the
reconciliation pass) is unchanged — see `paper-salvage/docs/PROJECT_STATUS.md`.

## 4. What this session did NOT do (deliberately)

- Did not run §0C (factor decomposition) or §0D (permutation test) on any real weight — the
  0B stop trigger fired first, and the script enforces the Phase-1/Phase-2 split in code.
- Did not edit `PREREG_dimensionality_gate0.md` after locking it.
- Did not touch `CLAIMS_LEDGER.md`, `PAPER_OUTLINE.md`, `CLAUDE.md`, or any manuscript file.
- Did not propose or start a replacement hypothesis for E5 in this pass.
- Did not run E3 (steering), retrain NTv3, re-run Evo1's secondary dose, search PROK layers,
  or touch Evo1 at all (excluded from E5's panel by its own prereg).

## 5. What a future session should actually do — pick one lane

**Lane A — write** (unchanged from the prior session). R1 and R6 have enough real evidence to
draft now. Does not require any new experiment or touch E5.

**Lane B — close one provenance gap** (unchanged). Pick exactly one of N-013/N-014/N-015.

**Lane C — new: define a cross-term-aware granularity metric for the genomic side, then
preregister it separately.** Only if explicitly instructed. `f_cross` (§0B) already gives a
per-model summary of how much the diagonal form misses; a full replacement would need a
principled way to attribute the exact quadratic form's off-diagonal mass to coordinates (or to
groups of coordinates), which does not currently exist in this repository and was correctly
not invented on the fly during Gate 0.

**Do not** mix lanes in one pass, and do not treat Lane C as queued — it needs its own
go-ahead.

## 6. Unresolved scientific decisions carried over (unchanged, still open)

Identical to the prior `NEXT_SESSION.md`'s §6 (Evo1 Branch A vs B, C-017, C-001/N-009, Evo1's
missing secondary-dose KL, empty matched-norm arms, C-010 on eager, ref [11]) — nothing in
this session touched any of them.

## 7. Exact next commands

```bash
cd /work/11034/atzanakak/glm_super_weight/genomic-super-weights

# verify BOTH prereg locks still hold (E2 unaffected; E5's Gate-0 lock is new this session)
python3 paper-salvage/src/prereg_lock.py verify --all

# re-run E5's synthetic-tensor tests if you touch dimensionality_lib.py
ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python3 \
  paper-salvage/experiments/E5_dimensionality/test_dimensionality_lib.py   # expect: 12/12

# re-run the shared uk_frobenius.py tests if you touch it (untouched this session)
LD_LIBRARY_PATH=$ENV/lib PYTHONPATH=$PWD/paper-salvage/src \
  $ENV/bin/python -m pytest paper-salvage/src -q            # expect: 7 passed
```

## 8. Environment traps (unchanged from prior sessions, plus one new note)

- Container has **no `python`**, only `python3`.
- Host `SSL_CERT_FILE` points outside the container; use `env -u SSL_CERT_FILE -u
  REQUESTS_CA_BUNDLE` for any `huggingface_hub` network call.
- `grlm` needs `LD_LIBRARY_PATH=$ENV/lib`.
- **New:** `grlm` also needs both `HF_HOME=/work/11034/atzanakak/ls6/huggingface/.hf-cache`
  (NLP shards) **and** `TRANSFORMERS_CACHE=/work/11034/atzanakak/ls6/nonbdna/cache/hf`
  (genomic checkpoints) set simultaneously, plus `HF_HUB_OFFLINE=1`, to load the full E5
  panel without any network access.
- **New:** loading GENERator EUK for weight-only inspection needs
  `AutoModelForCausalLM.from_pretrained(...)`, and DNABERT-2/NTv3 need
  `AutoModelForMaskedLM.from_pretrained(...)` — a bare `AutoModel.from_pretrained(...)` does
  not expose the module paths `uk_frobenius.py`'s adapters require for these three (confirmed
  by direct test this session; see `GATE0_RESULTS.md`'s deviations note).
- No `evo` conda env; Evo1/StripedHyena work needs the `evo2.sif` Apptainer container.
- `paper-salvage/results/` is gitignored via the `results/` pattern — use `git add -f`.
- No LaTeX toolchain on this node.
