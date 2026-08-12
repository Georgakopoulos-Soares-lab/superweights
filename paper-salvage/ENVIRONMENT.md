# ENVIRONMENT.md — reproducibility record for Methods

Cite these identifiers in the Methods section. Append, do not rewrite: if an environment
changes, add a new block and mark the old one superseded.

---

## Evo1 (StripedHyena) — impulse assay and residual attribution

There is **no `evo` conda environment on this node.** The `.sbatch` and `nohup` launchers
under `scripts/interpretability/` still say `conda activate evo`; they are stale for Evo1
and were not the path used. Evo1 runs go through the Apptainer container below.

| Item | Value |
|---|---|
| Container | `/work/11034/atzanakak/ls6/containers/evo2.sif` |
| Container sha256 | `ecb001192eeee416239da526ceaf9a73685fc9609aad404a23a36ac0bc161706` |
| Container size | 12,603,383,808 bytes |
| Container mtime | 2025-09-18 |
| Apptainer | `tacc-apptainer/1.1.8` (`/opt/apps/tacc-apptainer/1.1.8/bin/apptainer`) |
| Host | LS6, `gpu-a100` partition, NVIDIA A100-PCIE-40GB, driver 570.195.03 |
| Python | 3.11.12 (`/usr/bin/python3` inside the container) |
| torch | 2.5.1+cu124 (CUDA 12.4) |
| transformers | 5.15.0 |
| `evo-model` | 0.5 — installed to `~/.local`, **not** in the container image |
| `stripedhyena` | 0.2.2 — installed to `~/.local`, **not** in the container image |
| Checkpoint | `evo-1-8k-base` (HF snapshot, 15 files) |

The container image hash alone does **not** pin the environment: `evo-model` and
`stripedhyena` are resolved from `~/.local/lib/python3.11/site-packages`, which is outside
the image. Methods must state both the image hash and the two package versions.

### Invocation

```bash
source /opt/apps/lmod/lmod/init/bash
module load tacc-apptainer
export HF_HOME=/work/11034/atzanakak/ls6/huggingface/.hf-cache
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_HUB_OFFLINE=1          # weights are cached; see note
export PYTHONPATH=$PWD
env -u SSL_CERT_FILE -u REQUESTS_CA_BUNDLE \
  apptainer exec --nv /work/11034/atzanakak/ls6/containers/evo2.sif \
  python3 scripts/interpretability/<script>.py
```

Two things bite if omitted:

- The host sets `SSL_CERT_FILE` to a path that does not exist inside the container, so any
  `huggingface_hub` call that touches the network dies in `ssl.create_default_context`
  with a bare `FileNotFoundError`. Unset it (and `REQUESTS_CA_BUNDLE`) at the boundary.
- The container has no `python` on `PATH`, only `python3`.

### Numerical precision

Parameters are cast to **bf16 except `poles` and `residues`**, which stay fp32 (D-012).
Do not use fp16: its exponent range cannot hold Evo1's layer-10→13 residual excursion and
the run returns `KL = nan` at every layer. The reference implementation of this pattern is
`scripts/analysis/run_evo1_residual_attribution_fp32.py`.

> Note: D-012 gives that reference script's path as `scripts/interpretability/…`. The
> actual path is `scripts/analysis/…`. DECISIONS.md is append-only, so the entry is left
> as written and the correction is recorded here.

---

## Other four models (GENERator EUK/PROK, DNABERT-2, NTv3)

Run outside the container under the `grlm` conda environment. Pinned revisions are in
`scripts/interpretability/run_sw_broadcast_impulse.py`:

| Model | Pin |
|---|---|
| DNABERT-2 117M | revision `7bce263b15377fc15361f52cfab88f8b586abda0` |
| NTv3 650M | code revision `0ecff3637f0d3ba5b686d1095083218157c2ca34` |
| GENERator EUK / PROK | `configs/generator.yaml`, `configs/generator_prokaryote.yaml` |

_To be completed with the exact conda environment record at the STEP 4 re-run._
