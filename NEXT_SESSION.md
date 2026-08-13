# NEXT_SESSION.md

**Session:** 2026-08-12 overnight queue. **Queue finished.** E3 and E5 not started, as
instructed.

---

## Completed

| Item | Outcome | Commit |
|---|---|---|
| D-014 — adopt eager attention for DNABERT-2 | recorded; C-014 **retired** as X-006; X-003 **live** | `d7fb27f` |
| STEP 3d — secondary-dose probe | **BLOCKED** — KL flat for PROK at both doses | `d7fb27f` |
| E1 retrospective arm | **3/3 at both levels**; scalar recovery claimable | `a157686` |
| E4 granularity | **5/5 models**, all shape-verified | `7c26a0a` |

Earlier in the session: `0cbca69` bf16 harness fix, `e705497` E2 bookkeeping, `2c83692`
restructure + D-012a/D-013, `40b2d29` STEP 3a/3b/3c.

## Blocked — needs your decision

### 1. E2, at STEP 3d: which functional metric for R3

`paper-salvage/experiments/E2_evo1_broadcast/BLOCKED.md`

The stop condition fired. **GENERator PROK KL is flat at both doses**: 0.0 at α=0.01 and
9.94e-7 at α=1.0. A 100× dose increase moved nothing in the output — top-10 mean rank shift
**0.000**, top-1 unchanged, top-10 overlap 1.00. Both candidate replacement metrics were
measured and are *also* flat for PROK: KL at the SW token position is 1.06e-6 (same order as
the sequence mean), top-k rank displacement is exactly 0.

DNABERT-2 on eager *is* measurable at α=1.0 (KL 4.63e-5, top-1 changes), but it is the only
one, and it is the model whose kernel was just replaced.

Per your rule the dose was **not** escalated and no metric was substituted. **The prereg is
not locked and the five-model re-run has not started.** Everything else about the protocol
is settled and the prereg is ready to lock the moment a metric is chosen.

Not decided here: whether the metric is wrong, or the PROK perturbation genuinely has no
small-signal functional effect. The probe does not distinguish them.

### 2. C-001 — GENERator PROK ranks 1289/3,072, not 1 (N-009)

E4 measured PROK's SW row at layer 2 (the `SW_TARGETS` layer) and got rank **1289 / 3,072**.
EUK at layer 4 ranks 1, as C-001 claims. Probably a layer mismatch — `SW_TARGETS` layer 2 is
the *impulse source* layer, C-001 says *step-up* layer — but choosing which layer C-001 means
is a call about an existing claim. **C-001 is on hold and must not be migrated until it is
pinned down.**

### 3. `uk_frobenius` NTv3 adapter is wrong (N-010)

`ADAPTERS["ntv3"] = adapter_llama_swiglu` would raise. NTv3 holds its FFN inline as
`fc1` (12288, 1536) + `fc2` (1536, 6144) with SiLU — `fc1` is packed gate+up, the DNABERT-2
layout. Fixed in the E4 script only; `src/uk_frobenius.py` deliberately untouched because it
is shared with E1 and the fix belongs with a test.

## Exact next command

Once you have chosen the R3 functional metric:

```bash
cd /work/11034/atzanakak/glm_super_weight/genomic-super-weights

# 1. fill the confidence field in docs/prereg/PREREG_evo1_broadcast.md, then:
python paper-salvage/src/prereg_lock.py lock \
    paper-salvage/docs/prereg/PREREG_evo1_broadcast.md

# 2. four non-Evo1 models
ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
export LD_LIBRARY_PATH=$ENV/lib HF_HOME=/work/11034/atzanakak/ls6/huggingface/.hf-cache
export HF_HUB_CACHE=$HF_HOME/hub PYTHONPATH=$PWD
$ENV/bin/python scripts/interpretability/run_sw_broadcast_impulse.py \
    --model all --n_controls 5 --seed 42 --device cuda --out_dir results

# 3. Evo1 must run inside the container (see paper-salvage/docs/ENVIRONMENT.md)
source /opt/apps/lmod/lmod/init/bash && module load tacc-apptainer
export HF_HUB_OFFLINE=1
env -u SSL_CERT_FILE -u REQUESTS_CA_BUNDLE apptainer exec --nv \
    /work/11034/atzanakak/ls6/containers/evo2.sif \
    python3 scripts/interpretability/run_sw_broadcast_impulse.py \
    --model evo1 --n_controls 5 --seed 42 --device cuda --out_dir results
```

The harness already does dual dose, four control arms, fp32, the headroom column, eager
DNABERT-2, and Evo1 with `use_flash_attn=False` set before `StripedHyena(config)`.

## Environment gotchas that cost time

- The container has **no `python`**, only `python3`.
- The host's `SSL_CERT_FILE` points at a path that does not exist inside the container; any
  `huggingface_hub` network call dies in `ssl.create_default_context` with a bare
  `FileNotFoundError`. Use `env -u SSL_CERT_FILE -u REQUESTS_CA_BUNDLE`.
- The `grlm` conda env needs `LD_LIBRARY_PATH=$ENV/lib` prepended or PIL fails on
  `GLIBCXX_3.4.29`.
- There is **no `evo` conda env**; the `.sbatch` and `nohup` launchers under
  `scripts/interpretability/` still say `conda activate evo` and are stale for Evo1.

## Still open from earlier

- Phase 0 §0.2 migration: `results/keep/` exists but is empty; backfill is Phase 2.
- Phase 1b queue (**logged, not started**): C-010's L7/r603 source-vs-propagator resolution
  ran through Triton at inference and needs re-verification on eager. C-002 is weights-only
  and unaffected. C-027/C-028 are fine-tuning with dropout > 0 and already took eager.
- E1 prospective arm — needs your model choice and a lock. C-006 stays `pending`.
- The Methods disclosure from D-014 (Δ = −74.3%, fp16 cast as mechanism) must appear
  verbatim and must not be presented as a passed guard.
