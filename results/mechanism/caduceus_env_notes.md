# Caduceus environment setup notes (T2.1)

What was required to get Caduceus running locally, for anyone reproducing this or
extending Tier 2 to HybridNA/MegaDNA.

## Environment

Created a dedicated conda env (`caduceus`) rather than modifying the existing,
validated `evo` env — `mamba-ssm` pulls in `triton==3.7.1`, which would have
upgraded the `evo` env's `triton==3.2.0` and risked breaking the already-working
flash-attn/StripedHyena setup used for Evo1.

```
conda create -n caduceus python=3.10 -y
conda activate caduceus
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.2 pyyaml pyfaidx numpy accelerate packaging ninja psutil einops
```

## Building mamba-ssm (CUDA extension)

No system-wide `nvcc` was available. The `evo` conda env happens to have one
(installed for building flash-attn), so it was reused **only as a compiler**,
without installing anything into the `evo` env itself:

```
export CUDA_HOME=/home/nvidia/miniconda3/envs/evo
export PATH=$PATH:/home/nvidia/miniconda3/envs/evo/bin   # APPEND, not prepend --
                                                           # prepending shadows this
                                                           # env's own python/pip and
                                                           # silently builds against
                                                           # the wrong torch
pip install mamba-ssm==2.2.2 --no-build-isolation
```

`mamba-ssm==2.2.2` (not the latest 2.3.x) was chosen because its build/runtime
requirements are more compatible with an older, widely-supported torch/transformers
combo (2.4.0 / 4.44.2) — the latest mamba-ssm pulls in `quack-kernels`,
`nvidia-cutlass-dsl`, `tilelang`, and other bleeding-edge CUDA-13-era packages that
would have needed a much newer torch, in turn incompatible with the `transformers`
version below.

## Version pin needed for `transformers`

`mamba_ssm.utils.generation` imports `GreedySearchDecoderOnlyOutput` /
`SampleDecoderOnlyOutput` from `transformers.generation` — these were removed in
modern `transformers` (renamed/merged into `GenerateDecoderOnlyOutput` around the
4.4x/5.x transition). Pinning `transformers==4.44.2` (still has both classes) was
simpler and more robust than monkeypatching, and it happens to also be new enough
to load Caduceus's `trust_remote_code=True` custom modeling files without the
`torch.distributed.tensor.DTensor` import error that `transformers>=5` throws
against `torch==2.4.0`.

## Gated weights

None needed — `kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16` is a
public model.

## What was NOT needed

No container/Singularity/NGC image was required (unlike Evo2, see
`EVO2_REPLICATION.md`) — a plain conda env with a compiled CUDA extension was
sufficient, once the PATH/CUDA_HOME/version-pinning issues above were resolved.
