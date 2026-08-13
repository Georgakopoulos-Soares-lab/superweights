# CLEANUP_LOG.md — item 7c, mechanical only

**Date:** 2026-08-13. Non-interpretive changes only. Every scientific sentence altered
because of C-014/D-014 is listed with its exact reason.

## Changed

### 1. `paper/main.tex:111` — typo
`extremelly` → `extremely`. Pure spelling.

**Flagged, not fixed:** the sentence is *"The SW neighbourhood is extremely tolerant to
pruning and INT4 quantization compared with random rows…"*, which asserts **retired claim
X-001** (contradicted by the project's own pruning sweep: near-SW −1.60 pp vs random
−0.82 pp at 20%). Rewriting it is a scientific act, so it was left alone; it is already
tracked as submission blocker B2.

### 2. `paper/main.tex` Methods, model-loading paragraph — precision statement
**Reason: mechanically invalidated by D-012, D-014 and N-008.** The text said the model set
was "loaded from HuggingFace Hub in full precision where compatible", which is false for two
of them. Two factual sentences were added, no framing changed:

- **Evo1** ran in bfloat16 for every parameter except `poles`/`residues` (float32). float16
  cannot represent the layer-10→13 residual excursion and yields non-finite values (D-012).
- **DNABERT-2** parameters were float32, **but its default Triton flash-attention kernel
  casts query, key, value and attention-bias to float16 internally**, so its attention was
  never evaluated in full precision at inference. Impulse-response measurements use the
  eager PyTorch path, which is deterministic and stays float32 (D-014).

This is the disclosure D-014 requires. It does not describe the perplexity guard as passed.

### 3. `docs/PHASE_1_BLOCKING.md` §E2 Design — superseded-protocol annotation
Annotated rather than rewritten, to preserve the planning record. Notes that ε = 1.0 was
replaced by AC-relative ε (α = 0.01 primary / 1.0 secondary) under D-011/D-013/D-015, that
the run is fp32 with a headroom column, and that **"the same four control arms" was
inaccurate when written** — the harness implemented three; matched-norm was added only for
the locked run.

### 4. `docs/PROJECT_STATUS.md` — stale state
- Header said "E2 halted at STEP 3" and "Prereg NOT locked, re-run NOT started". E2 is
  locked, run and reported. Corrected; `BLOCKED.md` retained as the record of the stop.
- The lock box said "X-003 stays **live**". N-006's rule has since fired on the re-measured
  DNABERT-2 C ≈ 0, so X-003 **stays retired** (N-011). Corrected.
- Phase board said Phase 1 "not started".
- Open questions 1 and 2 said "resolved by E2 / E4". E2 and E4 are *measured*; question 1's
  relation is explicitly **not** assigned. Corrected to say so.
- Blocker list: typo item closed; ref [11] and Figure 2 items point at `REFERENCE_AUDIT.md`.

## Checked and found clean — no change needed

- **No stale C-014 / "largest impulse KL ≈ 0.31" anywhere in the manuscript.** The `0.31`
  matches in `paper/main.tex` and `manuscript.txt` are shuffle p-values (`p > 0.31`), not
  the retired KL. Nothing to retract in the manuscript text on that count.
- **"Pending Figure 2 panels" language left as written.** The rule permits removing it only
  if the panels are actually present. They are not rendered — the render is blocked on N-009
  (see `REFERENCE_AUDIT.md`) — so the caption stays.
- **No four/three control-arm statement in the manuscript.** The only occurrence was in
  `PHASE_1_BLOCKING.md`, handled above.

## Build

**Not run — no LaTeX toolchain on this node.** `pdflatex`, `xelatex`, `lualatex`, `latexmk`
and `tectonic` are all absent, and `paper/` has no Makefile or build script.

Substitute verification performed on `paper/main.tex` after editing:

- braces balanced: 379 `{` / 379 `}`
- exactly one `\begin{document}` / `\end{document}`
- the edited passage renders as plain text with two `\texttt{}` groups, both closed

**The manuscript has not been compiled and the edits are unverified against a real build.**
