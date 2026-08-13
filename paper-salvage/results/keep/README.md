# results/keep/

Curated artifacts the manuscript actually cites. One subdirectory per experiment, each
with a `PROVENANCE.md` recording: original path, generating script, git commit if known,
date, and the environment the run used.

**Copy — do not move.** The old tree is never modified and never deleted from
(`CLAUDE.md`, `docs/CUT_LIST.md`). Nothing in the new tree may cite a path that exists
only in the old tree (`docs/PHASE_0_TRIAGE.md` §0.2).

## Status

Created 2026-08-12, empty at creation. Per the STEP 2b instruction, the provenance
discipline **starts with the E2 re-run** rather than being backfilled now. Migration of
the pre-existing artifacts named in `docs/CLAIMS_LEDGER.md` is Phase 2 work and is tracked
as an open gap in `docs/PROJECT_STATUS.md`.

The audit called for in D-012 — "does any existing Evo1 artifact in `results/keep/` come
from the fp16 path?" — returns **nothing to mark**: this directory did not exist when the
audit ran and no `PROVENANCE.md` existed anywhere in the repository. The fp16 Evo1 run was
never promoted, so nothing needed superseding. The one artifact that did need demoting was
the fixed-ε entry in the repo-level `results/sw_broadcast_impulse.json`, which is now keyed
`evo1_fixed_eps_SUPERSEDED`.
