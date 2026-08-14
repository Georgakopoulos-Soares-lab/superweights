# NEXT_SESSION.md

**Session:** 2026-08-14, E8 encoder-vs-decoder dimensionality (sixth session, direct
continuation of E5/E6/E7).
**Status: a bounded, preregistered confirmatory experiment ran to completion and reached
Branch A** — both of two independently selected text encoders shifted toward the
"distributed" side of the pre-committed criterion, though one margin was thin. No manuscript
prose was touched. The original (2026-08-13) handoff — writing R1/R6, or resolving
N-013/N-014/N-015 — remains available; this session adds a new option instead of replacing it.

---

## 1. What happened this session

Two housekeeping items first, both explicitly independent of the new experiment:

1. **E7 provenance addendum** (`paper-salvage/experiments/E7_exact_dimensionality/
   PROVENANCE_ADDENDUM.md`, `EVO2_SHA256.txt`): resolved immutable HF commit hashes and
   weight-shard SHA256 for Phi-3, Qwen2.5-7B, GenomeOcean-4B, and Evo2-7B's local checkpoint.
   Documentation only.
2. **C-032 retirement recommendation, decided now** (`paper-salvage/experiments/
   E7_exact_dimensionality/C032_RETIREMENT_RECOMMENDATION.md`): GENERator EUK's exact
   `PR_spec` (1.0653) already sits inside the published-NLP range (1.02–1.24), and every
   genomic model's diagonal PR overstates its exact value in the same direction. This did not
   wait for E8's outcome, per instruction.

New experiment: `paper-salvage/experiments/E8_encoder_decoder/`. Tests the one clean
hypothesis E7's post-hoc observation generated: does exact bilinear-operator dimensionality
track encoder-vs-decoder organization rather than text-vs-genomic domain? DNABERT-2 and NTv3
(both encoders, both distributed) were E7's *discovery* evidence for this; E8 needed
genuinely new, independently selected text encoders to confirm it.

1. **Model audit** (`MODEL_AUDIT.md`): **MosaicBERT** (`mosaicml/mosaic-bert-base`) uses the
   **identical** `BertGatedLinearUnitMLP` class DNABERT-2 uses (verified from actual
   remote-code source — DNABERT-2 is itself built on the Mosaic BERT codebase) — the
   strongest possible architecture bridge to DNABERT-2 while changing domain to natural
   language. **ModernBERT** (`answerdotai/ModernBERT-base`) uses an independent codebase
   (`ModernBertMLP`, Answer.AI/LightOn, verified from installed `transformers` source). Both
   ELIGIBLE; the hard gate (≥2 independent bidirectional encoder families) passes without
   needing a replacement model.
2. **Prereg locked**: `paper-salvage/docs/prereg/PREREG_encoder_decoder_dimensionality.md`
   (sha256 `7a5dc793d35407a26be74aaeb79fb747e32669b51fd940f91035bd16857e4976`, UTC
   2026-08-14T15:14:09+00:00). Reuses E7's frozen WikiText-2 NLP detection protocol and 5.0x
   ratio threshold **verbatim** (not re-derived). Fixes the Branch-A criterion as `q1`
   strictly below **0.8989** — the decoder-cluster's own observed floor (GenomeOcean-4B),
   used unadjusted with no buffer, anchored entirely to the predeclared reference table from
   E5/E6/E7 (7 decoders, 2 discovery encoders).
3. **Detection + spectral results:**
   - MosaicBERT: detected at L9/r287 (ratio 288.65). `q1`=0.4766, `PR_spec`=4.1159 —
     dramatically below the decoder floor.
   - ModernBERT: detected at L15/r251 (ratio 561.12). `q1`=0.8970, `PR_spec`=1.2337 — below
     the floor, but by only **0.21% relative** (0.0019 absolute) — thin, genuine, and
     reported as such rather than smoothed over.
   - **Mechanical result: BRANCH A.** Both models shift on both the primary (`q1`) and
     secondary (`PR_spec`) criteria; no threshold was adjusted after seeing these values.
4. **Secondary finding, reported prominently (not part of the decision):** in both encoders,
   5 seeded same-layer control rows are far *more* distributed than the detected high-gain
   candidate (control `q1` ≈ 0.03–0.05 vs. candidates' 0.48/0.90) — the opposite of what a
   naive "encoder rows are generally distributed" reading would predict. Where a high-gain
   phenotype exists in an encoder, it is a local exception, not typical of the layer.
5. **Design-only causal-feasibility note written** (`CAUSAL_FOLLOWUP_FEASIBILITY.md`,
   triggered because Branch A occurred; no run authorized or performed): concludes that
   testing whether `PR_spec` predicts causal dimensionality would need a genuinely new
   activation-space (singular-vector-basis) intervention tool — the existing scalar/row
   ablation infrastructure is keyed to the diagonal decomposition this whole E5→E8 arc has
   been showing is unreliable, so it cannot be reused as-is.
6. **2x2 table presented** (descriptive only, no row-level statistics, no factorial claim):
   7 decoders (`q1` 0.90–0.99) vs. 4 encoders — 2 new, 2 discovery — (`q1` 0.39–0.90).
7. **Claim-ledger note:** C-034's proposed wording (from E7) is narrowed with E8's
   confirmatory result rather than restated as a separate new claim. Still not added to
   `CLAIMS_LEDGER.md`.

Full numbers, the exact-precision decision trace, and every caveat: `paper-salvage/
experiments/E8_encoder_decoder/RESULTS.md`.

## 2. Blockers

**None mechanical for E8 itself** — it completed cleanly and reached a genuine (if partly
thin-margin) Branch A.

Carried over, unchanged: **N-013, N-014, N-015** (`paper-salvage/docs/CLAIMS_LEDGER.md`).

New, optional, not queued by default:

- **Author action on C-032/C-034.** Both are decisive recommendations now, not tentative —
  but neither is enacted. An author needs to actually edit `CLAIMS_LEDGER.md` (move C-032 to
  the "Retired claims" table with its historical values preserved; decide whether to create
  C-034 with E8's narrowed wording).
- **The causal-feasibility gap.** Testing whether spectral dimensionality predicts causal
  dimensionality needs new activation-space intervention infrastructure this repository does
  not have. Scoping and building that is a substantial new effort, not a quick follow-up.
- **ModernBERT's thin margin.** A future session could ask whether a different, still
  non-cherry-picked natural-language input (not WikiText-2) gives a materially different
  `q1` for ModernBERT — this would need its **own new preregistration**, not a rerun of the
  locked `PREREG_encoder_decoder_dimensionality.md`.

## 3. Canonical report paths (this session)

| What | Path |
|---|---|
| E7 provenance addendum | `paper-salvage/experiments/E7_exact_dimensionality/PROVENANCE_ADDENDUM.md`, `EVO2_SHA256.txt` |
| C-032 retirement recommendation | `paper-salvage/experiments/E7_exact_dimensionality/C032_RETIREMENT_RECOMMENDATION.md` |
| E8 model audit | `paper-salvage/experiments/E8_encoder_decoder/MODEL_AUDIT.md` |
| Locked prereg | `paper-salvage/docs/prereg/PREREG_encoder_decoder_dimensionality.md` |
| Detection + spectral script | `paper-salvage/experiments/E8_encoder_decoder/run_detection_and_spectral.py` |
| **Results, 2x2 table, claim notes** | `paper-salvage/experiments/E8_encoder_decoder/RESULTS.md` |
| Causal-feasibility note (design-only) | `paper-salvage/experiments/E8_encoder_decoder/CAUSAL_FOLLOWUP_FEASIBILITY.md` |
| Raw outputs | `results/e8_detection_{mosaicbert,modernbert}.json` |

Everything from the five prior sessions is unchanged — see `paper-salvage/docs/
PROJECT_STATUS.md`.

## 4. What this session did NOT do (deliberately)

- Did not touch E5/E6/E7's files, or rerun any of their results.
- Did not edit `PREREG_encoder_decoder_dimensionality.md` after locking it.
- Did not adjust the decoder-floor threshold after seeing ModernBERT's thin margin.
- Did not run any ablation, fine-tune, pair perturbation, or LayerNorm test — the causal-
  feasibility note is design-only, exactly as instructed.
- Did not retry Evo 2 (explicitly out of scope for E8).
- Did not add a third text-encoder model to strengthen the margin after seeing results.
- Did not edit `CLAIMS_LEDGER.md`, `PAPER_OUTLINE.md`, `CLAUDE.md`, or any manuscript file.

## 5. What a future session should actually do — pick one lane

**Lane A — write** (unchanged). R1 and R6 have enough real evidence to draft now.

**Lane B — close one provenance gap** (unchanged). Pick exactly one of N-013/N-014/N-015.

**Lane C — enact the C-032/C-034 ledger edits.** Straightforward author action: move C-032 to
"Retired claims," decide on C-034. Does not require new experiments.

**Lane D — scope the causal-feasibility gap as new infrastructure.** Only if explicitly
instructed. Needs its own design pass before any code is written — `CAUSAL_FOLLOWUP_
FEASIBILITY.md` is a starting point, not a spec.

**Do not** mix lanes, and do not treat Lane D as queued.

## 6. Unresolved scientific decisions carried over (unchanged, still open)

Identical to the prior `NEXT_SESSION.md` versions' §6 (Evo1 Branch A vs B, C-017,
C-001/N-009, Evo1's missing secondary-dose KL, empty matched-norm arms, C-010 on eager,
ref [11]) — nothing in this session touched any of them.

## 7. Exact next commands

```bash
cd /work/11034/atzanakak/glm_super_weight/genomic-super-weights

# verify all five prereg locks still hold (E2, E5 Gate-0, E6 Stage-A, E7, E8)
python3 paper-salvage/src/prereg_lock.py verify --all
```

No new synthetic tests were added this session — E8 reuses E7's `spectral_lib.py` unmodified
(already 6/6 green, verified in the E7 session).

## 8. Environment traps (unchanged from prior sessions, plus one new note)

- Container has **no `python`**, only `python3`.
- Host `SSL_CERT_FILE` points outside the container; use `env -u SSL_CERT_FILE -u
  REQUESTS_CA_BUNDLE` for any `huggingface_hub` network call.
- `grlm` needs `LD_LIBRARY_PATH=$ENV/lib`, and both `HF_HOME=/work/11034/atzanakak/ls6/
  huggingface/.hf-cache` and `TRANSFORMERS_CACHE=/work/11034/atzanakak/ls6/nonbdna/cache/hf`
  set simultaneously — a full-model download via `transformers.AutoModel*.from_pretrained`
  and a targeted `huggingface_hub.hf_hub_download` call can land in **different** cache
  directories even in the same session, depending on which env var each code path respects
  (observed for Qwen2.5-7B and GenomeOcean-4B in the E7 session) — both are valid, just
  disclose which one a script actually used.
- **New — MosaicBERT ships no tokenizer files of its own.** Its own README states it uses
  the plain `bert-base-uncased` tokenizer; `AutoTokenizer.from_pretrained(repo)` fails
  (`vocab_file` resolves to `None`). Load `BertTokenizer.from_pretrained("bert-base-uncased")`
  instead.
- **New — background shell `cd` does not persist** across separate Bash tool calls; always
  `cd <absolute path> &&` at the start of every background command.
- `paper-salvage/results/` is gitignored via the `results/` pattern — use `git add -f`.
- No LaTeX toolchain on this node.
