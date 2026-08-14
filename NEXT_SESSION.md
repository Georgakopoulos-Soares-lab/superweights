# NEXT_SESSION.md

**Session:** 2026-08-14, E7 exact operator dimensionality (fifth session, direct continuation
of E5/E6).
**Status: a bounded, preregistered model-level confirmatory experiment ran to completion and
reached EXPLORATORY status (not a formal Branch A/B/C/D) because one of its four models
returned a legitimate detection null.** The follow-up legacy reanalysis then showed no clean
NLP-vs-genomic separation survives under the exact metric regardless. No manuscript prose was
touched. The original (2026-08-13) handoff — writing R1/R6, or resolving N-013/N-014/N-015 —
is unchanged and still the default next step unless someone picks up an E7 continuation.

---

## 1. What happened this session

E6's near-miss (row-level, pseudoreplicated across OLMo-7B and DNABERT-2) motivated E7: test
the same NLP-vs-genomic dimensionality question on the **exact** `U_k` operator's singular
spectrum (not the diagonal approximation), with **model/checkpoint as the primary replication
unit**, explicitly fixing E6's pseudoreplication.

New, self-contained experiment: `paper-salvage/experiments/E7_exact_dimensionality/`.

1. **Model audit** (`MODEL_PANEL.md`): Gene42 has **no public checkpoint anywhere on
   HuggingFace**, despite the paper's claim — verified directly against the `inceptionai` org
   and a HF-wide search (only empty placeholder repos found under an unrelated `madog` org).
   **Evo 2 7B initially appeared broken** — `import evo2` raised a `transformer_engine`
   ABI-mismatch error from a stale host-side (`~/.local`) package — **the user corrected this
   mid-session**: this repo's own existing `run_detection_evo2.sbatch` already solves it with
   `--cleanenv --env PYTHONNOUSERSITE=1`, which uses the container's own, compatible
   `transformer_engine`/`evo2` instead. Confirmed working once corrected. GenomeOcean-4B
   (`DOEJGI/GenomeOcean-4B`, `MistralForCausalLM`) was selected as Gene42's one authorized
   replacement, chosen specifically because it can carry the same "LLaMA-style architecture on
   genomic data" bridge-diagnostic role Gene42 was meant to serve. Gemma 2 is gated with no
   valid HF token available in this environment — excluded.
2. **Spectral library + lock**: `spectral_lib.py` (`q1`, `PR_spec` from `U_k`'s exact singular
   spectrum), 6/6 synthetic tests green pre-lock. `docs/prereg/
   PREREG_exact_operator_dimensionality.md` locked (sha256 `7da30f819312537e4b17edd9be3119eb1
   99e85263e1b08e181a3b147a9b3a83b`, UTC 2026-08-14T13:44:26+00:00) with a 4-model panel
   (Phi-3-mini-4k-instruct, Qwen2.5-7B / Evo 2 7B, GenomeOcean-4B), a frozen prospective
   activation-detection protocol for the three models lacking published coordinates, and a
   mechanical Branch A/B/C/D decision tree.
3. **Confirmatory run:**
   - Phi-3 (6 published Yu et al. rows, weight-only): model-level median `q1`=0.9028,
     `PR_spec`=1.2249.
   - Qwen2.5-7B: cleanly detected at L26/r458 (spike ratio 639), `q1`=0.9529.
   - GenomeOcean-4B: cleanly detected at L1/r2604 (spike ratio 1076), `q1`=0.8989.
   - **Evo 2 7B: NULL.** No layer's down-projection output spike cleared the preregistered
     5x max/median ratio anywhere across all 32 blocks (best candidate ratio 2.22). Per the
     frozen protocol, no alternative probe or layer was tried and the threshold was not
     loosened.
4. **Mechanical Phase-5 result: EXPLORATORY**, not a formal branch — the prereg's own rule
   disqualifies Branch A once a group degrades to one surviving model (genomic: only
   GenomeOcean-4B survived).
5. **Phase 6 (legacy reanalysis, run only after Phase 5 froze):** applied the identical code
   to E5/E6's six discovery rows. **GENERator EUK's exact `q1` (0.9689) exceeds OLMo-7B's
   (0.9646)** — a genomic model is more rank-1 than an NLP model in this panel. NTv3 remains a
   genuine, substantial outlier (`q1`=0.389, `PR_spec`=6.48). DNABERT-2 is intermediate
   (`q1`=0.793). **No clean NLP-vs-genomic separation survives under the exact metric across
   either panel** — full ranked table in `RESULTS.md`.
6. **No causal-feasibility document was written** (conditional on Branch A only, not reached).
7. **A visually clean but explicitly unconfirmed pattern was noticed and recorded, not
   asserted**: the two models that remain clearly distributed under the exact metric (NTv3,
   DNABERT-2) are both bidirectional encoders; every near-rank-1 model, NLP or genomic, is a
   causal decoder. This is discovery-only, post-hoc, and would need its own separately
   preregistered test before being written as a claim anywhere.
8. **Claim-ledger recommendations, not enacted:**
   - **C-002 / C-003 (row-ranking): unchanged from E6** (retain, with a Methods clarification).
   - **C-032 (diagonal-PR granularity claim): recommend RETIRE.** The exact metric does not
     reproduce its NLP-vs-genomic qualitative distinction (GENERator EUK's exact `PR_spec`
     is inside the NLP range).
   - **New claim C-034 (proposed, not added to `CLAIMS_LEDGER.md`)**: captures the actual
     pattern — heterogeneous exact dimensionality within the genomic group, no clean domain
     split, an unconfirmed encoder/decoder correlation noted for future testing.

## 2. Blockers

**None mechanical for E7 itself** — it completed cleanly and reached its own preregistered
(degraded) outcome; a null detection is a valid, informative result under the locked protocol,
not a bug.

Carried over, unchanged: **N-013, N-014, N-015** (`paper-salvage/docs/CLAIMS_LEDGER.md`).

New, optional, not queued by default:

- **Evo 2's null result.** The ACTB-504 probe (the same one used successfully for every other
  genomic model in this project) does not produce a detectable down-projection spike anywhere
  in Evo 2 7B under this protocol. A different native-domain input distribution might behave
  differently, but testing that is a **new, separately preregistered** detection run — not a
  retry, threshold adjustment, or second layer search under the current lock.
- **The encoder/decoder observation.** Would need its own confirmatory design (candidate
  models, panel, decision rule) before being written anywhere as a finding.
- **C-032's retirement and C-034's proposal** are recommendations only — an author needs to
  actually edit `CLAIMS_LEDGER.md` (append-only discipline: C-032 stays in the ledger with its
  original numbers, a new row is added for the recommendation, C-034 gets created only if the
  author agrees).

## 3. Canonical report paths (this session)

| What | Path |
|---|---|
| Model compatibility audit | `paper-salvage/experiments/E7_exact_dimensionality/MODEL_PANEL.md` |
| Locked prereg | `paper-salvage/docs/prereg/PREREG_exact_operator_dimensionality.md` |
| Spectral library + synthetic tests | `spectral_lib.py`, `test_spectral_lib.py` |
| Confirmatory + legacy scripts | `run_phi3_spectral.py`, `run_phase1_detection.py`, `run_phase1_detection_evo2.py`, `run_confirmatory_spectral_lite.py`, `run_phase5_decision.py`, `run_legacy_reanalysis.py` |
| **Results, ranked table, claim recommendations** | `paper-salvage/experiments/E7_exact_dimensionality/RESULTS.md` |
| Raw outputs | `results/e7_*.json` |

Everything from the four prior sessions (E1/E2/E4, the reconciliation pass, E5 Gate 0, E6
Stage A) is unchanged — see `paper-salvage/docs/PROJECT_STATUS.md`.

## 4. What this session did NOT do (deliberately)

- Did not touch E5's or E6's files.
- Did not edit `PREREG_exact_operator_dimensionality.md` after locking it.
- Did not retry Evo 2 with a different probe, layer, or threshold after seeing the null.
- Did not add a replacement genomic model to compensate for Evo 2's null (the model-expansion
  stop rule forbids adding models after seeing results).
- Did not write a causal-feasibility document (Branch A only; not reached).
- Did not assert the encoder/decoder pattern as a finding, or edit `CLAIMS_LEDGER.md`,
  `PAPER_OUTLINE.md`, `CLAUDE.md`, or any manuscript file.

## 5. What a future session should actually do — pick one lane

**Lane A — write** (unchanged). R1 and R6 have enough real evidence to draft now.

**Lane B — close one provenance gap** (unchanged). Pick exactly one of N-013/N-014/N-015.

**Lane C — a fresh, separately preregistered E7/E8 continuation.** Only if explicitly
instructed. Candidates: (a) retry Evo 2 detection with a different, still-non-cherry-picked
native input distribution; (b) pursue a genuinely new genomic decoder model if one becomes
available (Gene42, if it is ever actually released; a fresh search for another candidate);
(c) design a proper confirmatory test of the encoder-vs-decoder observation. None of these
continues the locked `PREREG_exact_operator_dimensionality.md` — each needs its own lock.

**Do not** mix lanes, and do not treat Lane C as queued.

## 6. Unresolved scientific decisions carried over (unchanged, still open)

Identical to the prior `NEXT_SESSION.md` versions' §6 (Evo1 Branch A vs B, C-017,
C-001/N-009, Evo1's missing secondary-dose KL, empty matched-norm arms, C-010 on eager,
ref [11]) — nothing in this session touched any of them.

## 7. Exact next commands

```bash
cd /work/11034/atzanakak/glm_super_weight/genomic-super-weights

# verify all four prereg locks still hold (E2, E5 Gate-0, E6 Stage-A, E7)
python3 paper-salvage/src/prereg_lock.py verify --all

# re-run E7's synthetic tests if you touch spectral_lib.py
ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python3 \
  paper-salvage/experiments/E7_exact_dimensionality/test_spectral_lib.py   # expect: 6/6

# re-run E5's and E6's synthetic tests if you touch their shared libs (untouched this session)
LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python3 \
  paper-salvage/experiments/E5_dimensionality/test_dimensionality_lib.py  # expect: 12/12
LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python3 \
  paper-salvage/experiments/E6_cross_geometry/test_cross_geometry_lib.py  # expect: 7/7
```

## 8. Environment traps (unchanged from prior sessions, plus new Evo 2 / download notes)

- Container has **no `python`**, only `python3`.
- Host `SSL_CERT_FILE` points outside the container; use `env -u SSL_CERT_FILE -u
  REQUESTS_CA_BUNDLE` for any `huggingface_hub` network call.
- `grlm` needs `LD_LIBRARY_PATH=$ENV/lib`, and both `HF_HOME=/work/11034/atzanakak/ls6/
  huggingface/.hf-cache` and `TRANSFORMERS_CACHE=/work/11034/atzanakak/ls6/nonbdna/cache/hf`
  set simultaneously.
- **New — Evo 2 specifically:** `import evo2` **must** run with
  `--cleanenv --env PYTHONNOUSERSITE=1` (matches `scripts/detection/run_detection_evo2.sbatch`'s
  existing convention) or it picks up a broken, ABI-incompatible `transformer_engine` from
  `~/.local` (installed there for Evo1/StripedHyena, not Evo2/StripedHyena2). Also needs the
  same FP8-autocast no-op monkeypatch already in `models/evo2_wrapper.py` (A100 is compute
  capability 8.0; FP8 needs 8.9+) — apply it **before** `from evo2 import Evo2`.
- **New — Evo 2 checkpoint loading is slow even from cache**: ~11-29 minutes just for
  `torch.load` of the 7B `.pt` state dict (not a download issue; observed even with the
  checkpoint already fully cached). Budget for this.
- **New — background shell `cd` does not persist** across separate Bash tool calls in this
  environment; always `cd <absolute path> &&` at the start of every background command rather
  than relying on a prior `cd`.
- **New — GPU memory budget**: a fp32 7B model (~28GB) and a fp32 4B model (~16GB) together
  exceed the single A100's 40GB — serialize large model loads rather than running them
  concurrently, or reduce precision.
- No `evo` conda env (Evo1); Evo 2 needs `evo2.sif` specifically as above.
- `paper-salvage/results/` is gitignored via the `results/` pattern — use `git add -f`.
- No LaTeX toolchain on this node.
