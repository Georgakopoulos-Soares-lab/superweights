# docs/

| file | purpose |
|---|---|
| `EXPERIMENT_MAP.md` / `.tsv` | **Start here.** Every manuscript experiment → the code that produced it → its artifacts, with a per-row status. Generated and verified by `scripts/build_experiment_map.py`; `tests/test_experiment_map.py` fails if it goes stale. |
| `PART1_STRUCTURAL_MANUSCRIPT_PACKET.md` | Evidence assembly for the structural results. Cited by the round-1 and round-2 audit reports, so it is part of the audit trail rather than superseded prose. |
| `PART2_EVIDENCE_PACKET.md` | The same for the functional-criticality results and Figure 2. |
| `EXP2_INSTRUCTIONS_FOR_COLLABORATOR.md` | Self-contained protocol for the legacy-detector experiment. Banner-marked ALREADY RUN — retained as the reproduction protocol, not as an open task. |
| `history/` | Superseded material, kept deliberately. |

## history/

| file | purpose |
|---|---|
| `README_ARCHIVE.md` | The previous repository README, 1,443 lines, in full and unedited. Numbers in it predate the retractions list; consult the current README first. |
| `REMOVED_SCRIPTS.md` | The 136 files removed in the 2026-09-09 consolidation, the rule used to decide, and how to retrieve any of them from git history. |
| `COAUTHOR_LLM_CONTEXT.md`, `NEXT_SESSION.md`, `EVO2_REPLICATION.md` | Session notes and a replication plan from earlier rounds. |
| `manuscript_prism.txt` | A stale LaTeX export under the superseded title. |
| `e2_logs/` | E2 broadcast run logs. Evidence for a locked experiment, so they stay tracked despite the global `*.log` ignore rule. |

Two things were deleted rather than archived on 2026-09-09, both recoverable from git history:
the earlier manuscript draft and its 2.7 MB of artwork (`superseded_draft/`), which no current
claim depends on; and a full text dump of Yu et al. 2024, which is a third-party preprint that
should be cited rather than redistributed.
