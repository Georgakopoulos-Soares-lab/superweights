# NEXT_SESSION.md

**Session:** 2026-08-13 overnight queue #3.
**Status: the queue is finished.** Items 6, 7c, 8 and 9 complete. Nothing is mid-flight and
the tree is clean. E3 and E5 were not started, as instructed.

---

## 1. Completed this session

| Item | Outcome | Commit |
|---|---|---|
| 6 — canonical E4 table | Built from stored artifacts only; PROK marked CONTESTED | `0bc57ee` |
| 7c — mechanical cleanup | Typo, precision disclosure, stale state; **build could not run** | `e20d12d` |
| 8 — provenance backfill | 5 established claims migrated, 12 logged unmigrated | `25606c6` + fixup |
| 9 — this handoff | — | (this commit) |

Earlier in the campaign: `d7fb27f` (E2 blocked at 3d), `a157686` (E1 retrospective),
`7c26a0a` (E4 raw), `16e80f6` (D-015), `b9e3ab1` (N-009), `3ca8bb6` (N-010),
`695eb92`/`128b4ee` (prereg amended + **locked**), `00cedd3`/`ee746de` (E2 five-model run),
`1098ca9` (claim statuses), `791813e` (ref [11] + Figure 2 audit).

## 2. Blockers

**None blocking further mechanical work.** Everything remaining is a scientific judgment
(§5). Two things were attempted and could not be completed:

- **Manuscript build not run.** No LaTeX toolchain on this node — `pdflatex`, `xelatex`,
  `lualatex`, `latexmk`, `tectonic` all absent, no Makefile in `paper/`. The edits are
  verified only by brace balance (379/379), one `\begin{document}`/`\end{document}`, and
  closed `\texttt{}` groups. **`paper/main.tex` has not been compiled since editing.**
- **Evo1 secondary-dose KL** remains unmeasured (CUDA OOM). Not retried, per instruction.

## 3. Canonical report paths

| What | Path |
|---|---|
| **E2 results** | `paper-salvage/experiments/E2_evo1_broadcast/RESULTS.md` |
| E2 instrument validation | `paper-salvage/experiments/E2_evo1_broadcast/INSTRUMENT_VALIDATION.md` |
| E2 STEP-2 gate | `paper-salvage/experiments/E2_evo1_broadcast/GATE_RESULTS.md` |
| E2 STEP-3d stop record | `paper-salvage/experiments/E2_evo1_broadcast/BLOCKED.md` (historical) |
| **E4 canonical table** | `paper-salvage/experiments/E4_granularity/CANONICAL_TABLE.md` |
| E4 N-009 resolution | `paper-salvage/experiments/E4_granularity/N009_RESOLUTION.md` |
| E1 results | `paper-salvage/experiments/E1_nlp_validation/RESULTS.md` |
| Cleanup log | `paper-salvage/docs/CLEANUP_LOG.md` |
| Reference / Figure 2 audit | `paper-salvage/docs/REFERENCE_AUDIT.md` |
| Environment for Methods | `paper-salvage/docs/ENVIRONMENT.md` |

## 4. Migrated provenance artifacts

Under `paper-salvage/results/keep/` (copied, not moved; each with `PROVENANCE.md`):

| Batch | Claims | Class |
|---|---|---|
| `E1_nlp_retrospective/` | C-004, C-005 | derived |
| `E2_broadcast_locked/` | C-012, C-013, C-015 | raw (+ `LOCKS.jsonl`) |
| `E4_granularity/` | C-003 | derived |

**`results/keep/UNMIGRATED.md`** lists the 12 established claims left behind — all have an
**empty evidence path** in the ledger, so picking a candidate file would be a guess at the
mapping, which is precisely how N-009 arose. C-001 and C-014/X-006 are excluded by rule.

Note: `paper-salvage/results/` matches the repo's `results/` ignore pattern, so everything
there is tracked via `git add -f`.

## 5. Unresolved scientific decisions — all yours, none decided

1. **Evo1 Branch A vs B**, and how to read its T relative to controls. Branch C and D are
   excluded (headroom 26× at L13+, 0/20 under-powered, noise floor 0.0). Peak T: EUK
   3.79e-02, **Evo1 1.20e-01**, DNABERT-2 8.01e-01, PROK 6.18e+00, NTv3 undefined. Evo1's SW
   arm (1.1997e-01) sits within ~9% of all three of its control arms (1.10–1.17e-01).
2. **Whether T has any relationship to criticality** — C-017 remains the thesis slot,
   unassigned.
3. **C-001 / PROK rank discrepancy (N-009).** Same layer (2), same row, same checkpoint
   *name*; exact and decomposed formulas agree (rank 1277 / 1289) while the stored artifact
   says rank 1. EUK reproduces exactly. C-001 on hold; Figure 2's PROK panel blocked on it.
4. **Evo1's missing secondary-dose KL** — accept as unmeasured, or add a per-dose flag and
   run the secondary alone in a fresh process.
5. **Empty matched-norm arms** (EUK 0/3,072, DNABERT-2 0/768 within ±10%) — whether an empty
   arm is acceptable or the definition needs revisiting in a future protocol version.
6. **Whether/when to re-run C-010 on eager** (Phase 1b; it was measured through the Triton
   path at inference).
7. **E1 prospective model choice** — needs your pick plus a lock. C-006 stays `pending`.
8. **E3 steering prereg and predictions** — not started; needs your predictions and a lock.

Also open, from the audit: **ref [11]** needs an external arXiv lookup (`2603.05498` vs
`2402.17762`; it is additionally uncited in the body), and **Figure 2 A–D** rendering is an
authorial call gated on N-009.

## 6. Current claim statuses

| Claim | Status |
|---|---|
| C-001 | **on hold** — do not migrate, do not render its Figure 2 panel |
| C-003, C-004, C-005 | established |
| C-012, C-013 | **established, confirmed under the locked protocol** |
| C-014 | **RETIRED as X-006** — never resurrect |
| C-015 | **restored: established** (N-006's rule fired → N-011) |
| C-016 | supported — Evo1 routing regime measured |
| C-017 | pending — thesis slot, **branch unassigned** |
| C-032 | supported — canonical table |
| X-003 | **stays retired** — re-measured DNABERT-2 C ≈ 0 |
| N-009 | open (scientific) · N-010 fixed · N-011 applied |

## 7. Exact next commands

Nothing is required to resume — the queue is done. Useful entry points:

```bash
cd /work/11034/atzanakak/glm_super_weight/genomic-super-weights

# verify the prereg lock still holds
python3 paper-salvage/src/prereg_lock.py verify --all      # expect: OK

# re-run the shared-library tests after any uk_frobenius change
ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
LD_LIBRARY_PATH=$ENV/lib PYTHONPATH=$PWD/paper-salvage/src \
  $ENV/bin/python -m pytest paper-salvage/src -q            # expect: 7 passed

# regenerate the canonical E4 table (reads stored artifacts only, loads no model)
python3 paper-salvage/experiments/E4_granularity/build_canonical_table.py
```

**Prereg lock, for citation:**
`3d7515d0b7889f65038acd8479e85976248e7dbd0e5a701303de3a1b37dbfdc3`,
UTC `2026-08-13T02:59:20+00:00`, git `695eb92`. Do **not** edit the locked prereg — filling
its `<hash>`/`<date>` placeholders changes its hash and breaks `verify`.

## 8. Environment traps

- Container has **no `python`**, only `python3`.
- Host `SSL_CERT_FILE` points outside the container; any `huggingface_hub` network call dies
  in `ssl.create_default_context`. Use `env -u SSL_CERT_FILE -u REQUESTS_CA_BUNDLE`.
- `grlm` needs `LD_LIBRARY_PATH=$ENV/lib` or PIL fails on `GLIBCXX_3.4.29`.
- No `evo` conda env; the `.sbatch`/`nohup` launchers saying `conda activate evo` are stale.
- GENERator 3B checkpoints take ~6 min to load; tqdm shard progress does not flush to a
  redirected log, so a static log is **not** evidence of a hang — check `ps -o etime,time`.
- Evo1 at fp32 nearly fills a 40 GB A100; a second dose in the same process OOMs.
- `paper-salvage/results/` is gitignored via the `results/` pattern — use `git add -f`.
- No LaTeX toolchain on this node.
