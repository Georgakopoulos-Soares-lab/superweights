# NEXT_SESSION.md

**Session:** 2026-08-12/13 overnight queue #2.
**Stopped at:** queue item 4 (five-model E2 run) — **launched and in flight when the session
ended**. Items 1, 2, 3, 7a, 7b complete. Items 5, 6, 8 not started.

---

## 1. What completed

| Item | Outcome | Commit |
|---|---|---|
| Session decision | D-015 recorded: T/C primary, KL secondary/descriptive, pre-lock | `9e35b1b`* |
| 1 — N-009 / C-001 | Layer resolved to **L2**; stored artifact **does not reproduce**; C-001 stays on hold | `b9e3ab1` |
| 2 — N-010 | `adapter_ntv3` added to shared `uk_frobenius.py` + 7 tests; NTv3 E4 reproduces **exactly** | `3ca8bb6` |
| 3a — prereg amendment | D-015 endpoints written in; confidence **3/5** | (in `128b4ee` lineage) |
| 3b — **PREREG LOCKED** | see hash below | `128b4ee` |
| 7a/7b — audit | Ref [11] and Figure 2 both **flagged, neither changed** | `791813e` |

\* the D-015 commit is the one immediately preceding `b9e3ab1`; `git log --oneline` shows the
full chain.

### PREREG LOCK — record this everywhere it is cited

```
sha256 3d7515d0b7889f65038acd8479e85976248e7dbd0e5a701303de3a1b37dbfdc3
utc    2026-08-13T02:59:20+00:00
commit 695eb9279bbdbbe824721e9f1ab39c7f82af9b4e
ledger paper-salvage/docs/prereg/LOCKS.jsonl   (first and only entry)
verify paper-salvage/src/prereg_lock.py verify --all  ->  OK
```

**Do not edit `PREREG_evo1_broadcast.md`.** Its Methods `<hash>`/`<date>` placeholders are
deliberately left unfilled: writing the hash into the file changes the file's hash and makes
`verify` report CHANGED. LOCKS.jsonl is authoritative.

## 2. What is in flight / not done

- **Item 4 — five-model run: STARTED, INCOMPLETE.** Four non-Evo1 models were launched
  sequentially in the background; only GENERator EUK had begun (still loading checkpoint
  shards) when the session ended. Evo1 was never launched. **Assume nothing finished.**
- **Item 5 — E2 RESULTS.md: not written.**
- **Item 6 — canonical E4 cross-model table: not written.**
- **Item 8 — provenance backfill into `results/keep/`: not started.**
- Item 7c (typo sweep) not done.

### Before rerunning item 4, check what survived

```bash
cd /work/11034/atzanakak/glm_super_weight/genomic-super-weights
python3 -c "
import json; d=json.load(open('results/sw_broadcast_impulse.json'))
[print(k, 'NEW-PROTOCOL' if 'epsilon_meta' in v else 'old fixed-eps') for k,v in d.items()]"
```

The pre-D-011 baseline is preserved at
`results/sw_broadcast_impulse_PRE_D011_fixed_eps.json` — do not overwrite it.

## 3. Exact next commands

```bash
cd /work/11034/atzanakak/glm_super_weight/genomic-super-weights
ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
export LD_LIBRARY_PATH=$ENV/lib HF_HOME=/work/11034/atzanakak/ls6/huggingface/.hf-cache
export HF_HUB_CACHE=$HF_HOME/hub PYTHONPATH=$PWD

# (a) four non-Evo1 models, one at a time (each 3B model needs ~6 min just to load)
for M in generator generator_prokaryote dnabert2 ntv3; do
  $ENV/bin/python -u scripts/interpretability/run_sw_broadcast_impulse.py \
      --model $M --n_controls 5 --seed 42 --device cuda --out_dir results \
      > logs/e2_run_$M.log 2>&1
done

# (b) Evo1, in the container
source /opt/apps/lmod/lmod/init/bash && module load tacc-apptainer
export HF_HUB_OFFLINE=1
env -u SSL_CERT_FILE -u REQUESTS_CA_BUNDLE apptainer exec --nv \
    /work/11034/atzanakak/ls6/containers/evo2.sif \
    python3 -u scripts/interpretability/run_sw_broadcast_impulse.py \
    --model evo1 --n_controls 5 --seed 42 --device cuda --out_dir results \
    > logs/e2_run_evo1.log 2>&1

# (c) Evo1 noise floor — required by the null rule, not yet measured for Evo1
env -u SSL_CERT_FILE -u REQUESTS_CA_BUNDLE apptainer exec --nv \
    /work/11034/atzanakak/ls6/containers/evo2.sif \
    python3 -u scripts/interpretability/impulse_determinism_check.py --model evo1

# (d) stamp provenance (written this session, not yet run)
$ENV/bin/python scripts/interpretability/stamp_e2_provenance.py
```

`stamp_e2_provenance.py` adds checkpoint, dtype, attention path, seed, git commit,
container identity, lock hash and each model's measured noise floor. It is **additive** and
never touches a measured value — it exists so the harness was not edited mid-run.

Noise floors already measured and on disk: `results/impulse_determinism_{generator,
generator_prokaryote,dnabert2,ntv3}.json`. DNABERT-2's is the **eager** one. Evo1's is missing.

## 4. Claim statuses touched this session

| Claim | Status now |
|---|---|
| C-001 | **on hold** — PROK half has no reproducible support; must not be migrated |
| C-003 | established, confirmed by E4 |
| C-004, C-005 | established (E1 retrospective, previous session) |
| C-014 | **RETIRED as X-006.** Never resurrect |
| C-015 | unsupported pending re-run |
| C-032 | supported |
| X-003 | **LIVE** — do not treat "C is not necessary for criticality" as settled |
| N-009 | resolved-as-blocker (see below) · N-010 **fixed** |

## 5. Unresolved scientific decisions — yours

1. **N-009: the PROK structural artifact does not reproduce.** Same layer (2), same row
   (1927), same checkpoint *name*, and both the exact and the decomposed formulas agree with
   each other (rank 1277 / 1289) while the stored artifact says rank 1. EUK reproduces
   exactly. Possible causes — an upstream HF repo change since 29 May 2026, a difference in
   how weights were loaded/dtyped originally, something else — need a call, not a trace.
   Until settled, C-001's PROK half is unsupported and Figure 2's PROK panel cannot be drawn.
2. **Ref [11]** needs an external arXiv lookup of `2603.05498` and `2402.17762`. Local
   sources cannot separate "corrupted reference to Sun M. 2024" from "correct reference to a
   real 2026 paper". It is also uncited in the body.
3. **Figure 2 A–D**: inputs all exist; whether to render EUK-only, or wait on N-009, is
   authorial.

## 6. Environment traps (unchanged, still true)

- Container has **no `python`**, only `python3`.
- Host `SSL_CERT_FILE` points outside the container; any `huggingface_hub` network call dies
  in `ssl.create_default_context`. Use `env -u SSL_CERT_FILE -u REQUESTS_CA_BUNDLE`.
- `grlm` needs `LD_LIBRARY_PATH=$ENV/lib` prepended or PIL fails on `GLIBCXX_3.4.29`.
- No `evo` conda env; the `.sbatch`/`nohup` launchers saying `conda activate evo` are stale.
- GENERator 3B checkpoints take ~6 minutes to load from the shared filesystem; tqdm shard
  progress does not flush to a redirected log, so a static log is **not** evidence of a hang.
  Check `ps -o etime,time` and `nvidia-smi` instead.
- `prereg_lock.py` warns "working tree is dirty" because of pre-existing `paper/main.tex`
  changes and two untracked manuscript files that predate this work and are not E2's.

## 7. Not started, deliberately

E1 prospective, E3, E5, corpus-prior, extra DNABERT-2 seeds, Phase 1b / C-010 eager
re-verification, AC/DC decomposition, R3 thesis sentence, abstract/Discussion rewrites.
