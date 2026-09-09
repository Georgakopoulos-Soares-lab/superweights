# PHASE 0 — Triage

**Goal:** know exactly what we have, migrate what we're keeping, and clear the
submission blockers that are independent of any new science.

**Exit criteria:** every row of `CLAIMS_LEDGER.md` has an evidence path that resolves;
`results/keep/` is populated with provenance; the five submission blockers below are
either fixed or have an owner and a date.

---

## 0.1 Inventory the old tree

Produce `results/INVENTORY.md` listing, for every artifact currently referenced by v15:

- artifact path in the old tree
- what claim it supports
- keep / demote-to-supplement / orphan
- whether it is reproducible from a script still in the repo

Known artifacts to locate (from the session notes):

```
scripts/interpretability/run_sw_broadcast_impulse.py
run_all_grlm_nohup.sh
run_evo1_nohup.sh
results/sw_broadcast_impulse.json
logs/nohup_grlm.log
logs/nohup_ntv3_broadcast*.log
```

Plus the GENERator JSONs that Figure 2 A–D depend on. **Finding these is the single
highest-priority item in Phase 0** — the figure cannot be rendered without them and the
v15 caption currently admits they are missing.

Also inventory (all confirmed to exist from v15 results text):
- 35-task GENERator fine-tune + ablation results
- DNABERT-2 3-seed GUE results, per-row ablation on splice checkpoint
- NTv3 5-seed splice results incl. per-seed confusion matrices
- PROK SAE checkpoint (200k steps, k=64, dict 12288)
- hexamer scan / causal-hexamer KL tables (4,096 hexamers, both models)
- shuffle control outputs (4 variants × 90 sequences × 2 models)
- pruning sweep, INT4 sweeps
- relay-head ablations, GC↔AT counterfactual swap results

## 0.2 Migrate

Copy — do not move — kept artifacts into `results/keep/<experiment>/`, each with a
`PROVENANCE.md` recording original path, generating script, git commit if known, and date.
Nothing in the new tree should reference a path that exists only in the old tree.

## 0.3 Clear the submission blockers

| # | Blocker | Fix | Owner | Date |
|---|---|---|---|---|
| B1 | Figure 2 panels A–D absent, caption admits it | **JSONs located — all four present.** Caption's stated reason is out of date. Render still blocked: the PROK ‖U_k‖_F panel would come from the artifact N-009 found unreproducible. EUK reproduces. Authorial call. See `docs/REFERENCE_AUDIT.md`. | | 2026-08-12 |
| B2 | Abstract claims SW-neighbourhood pruning tolerance; Results conclude far-SW fragility dominates and random pruning is safer than near-SW | rewrite abstract; delete "shadow redundancy" | | |
| B3 | "Eight models" implies uniform benchmark | replace with coverage table (PAPER_OUTLINE) | | |
| B4 | NTv3 MCC chosen post-hoc after seeds 4–5 collapsed | see 0.4 | | |
| B5 | Ref [11] Sun et al. arXiv:2603.05498 | **FLAGGED — needs external lookup, NOT corrected.** Different author initial (P. vs Mingjie), different title, different year; the local Yu et al. bibliography cites Sun et al. 2024 by OpenReview ID `1ayU4fMqme`, giving no arXiv ID, so `2402.17762` is uncorroborated locally. The entry is also **never cited in the body**. See `docs/REFERENCE_AUDIT.md`. | | 2026-08-12 |

## 0.4 Resolve the NTv3 metric problem

Disclosure is the right instinct but does not fix it, and the paper is now framed around a
*predictor* — a post-hoc metric choice sitting in R6 is a cheap thing to remove.

Two acceptable resolutions. Pick one and log it in DECISIONS.md:

- **(a)** Report accuracy *and* MCC for all five seeds, and let the per-class confusion
  matrices carry the argument. Majority-class collapse in seeds 4–5 is visible directly;
  the metric stops doing rhetorical work.
- **(b)** Preregister MCC and run a fresh seed batch (n ≥ 10, including the excluded
  seed 3 or a documented replacement).

(a) is cheaper and probably sufficient. (b) is stronger if compute allows.

## 0.5 Freeze the outline

Once 0.1–0.4 are done, `PAPER_OUTLINE.md` is frozen. Any change requires a DECISIONS entry.
