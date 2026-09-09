# E13 Stage-1 pre-launch resource estimate

Recorded before launching the full-cohort singleton/control measurements.

## Hardware actually available

`nvidia-smi` exposes **one NVIDIA A100-PCIE-40GB** to this process. The task prompt's
anticipated four H100 GPUs are not visible on this node, so the queue is serial and never
places two large checkpoints on one GPU. Every model invocation is separately checkpointed;
the detached queue continues past a failed model and is restartable without repeating
completed JSON artifacts.

## GPU and elapsed-time estimate

Each ordinary one-candidate model requires 13 complete endpoint passes: one baseline, two
candidate interventions, and ten control interventions (five controls at two epsilons).
Phi-3 requires 33 passes because its frozen basis has six candidates across two layers plus
ten controls. Across 23 models this is **319 complete endpoint passes** (296 intervention
conditions plus 23 baselines), with 100
decoder windows or 16 fixed MLM batches per pass.

- Estimated aggregate GPU compute: **approximately 5–9 A100 GPU-hours**.
- Estimated wall time on the one visible GPU: **approximately 8–16 hours**.
- Main uncertainty: shared `/work` cache I/O. During Stage 0, ModernBERT inference itself
  took about 14 seconds after loading, while the legacy unsharded Llama-7B checkpoint was
  still streaming tensors after 15 minutes. This is elapsed I/O time, not GPU compute, and
  may dominate the four legacy 7B checkpoints.
- The estimate excludes locked E10b Phi-3 tomography, which is reused exactly after
  mechanical basis/revision/endpoint/design equality checks rather than rerun.

These ranges are planning estimates, not a reason to alter sample counts or omit models.

## Disk estimate

- Raw paired per-unit responses: expected **well below 100 MB** (296 conditions, each with
  13–100 `(loss_sum, token_count)` units plus metadata).
- CSV/JSON summaries and figures: expected **below 25 MB**.
- Nohup logs: expected **below 100 MB** even with progress output.
- No model download is permitted. Existing weights are read with `local_files_only=True`
  from `/work/11034/atzanakak/ls6/nonbdna/cache/hf`; WikiText reads from the existing
  `/scratch/11034/atzanakak/huggingface_cache/datasets` artifact.
- Durable operational logs are written under quota-free
  `/scratch/11034/atzanakak/genomic-super-weights/E13/logs/`.

## Special loading/evaluation code

- **MosaicBERT:** remote-code model; `bert-base-uncased` tokenizer; down projection
  `bert.encoder.layer.{i}.mlp.wo`.
- **ModernBERT base/large:** packed GeGLU `Wi` with down projection
  `model.layers.{i}.mlp.Wo`; pinned revisions.
- **EuroBERT ladder:** remote-code masked-LM models with Llama-style gated projection
  names.
- **DNABERT-2:** narrow compatibility loader for the current torch/transformers environment;
  the existing fixed-mask genomic MLM endpoint is otherwise reused unchanged.
- **NTv3:** pinned remote-code revision, NTv3 tokenizer, fixed-mask genomic MLM endpoint,
  and `core.transformer_blocks.{i}.fc2` intervention path.
- **GENERator EUK/PROK:** fixed-6bp-k-mer sequence preparation and teacher-forced nucleotide
  NLL from E12.
- **GenomeOcean:** character/BPE tokenizer; teacher-forced nucleotide NLL without
  GENERator's 6bp trim or forced BOS preparation.
- **Phi-3:** packed gate/up model layout, six structural candidates, and two separately
  seeded same-layer control panels.

## Orchestration

The queue is launched using `nohup` plus `setsid`, writes one log per model on scratch, and
writes each completed raw JSON atomically. It begins with small checkpoints to expose
architecture problems cheaply, then runs the larger models. Scientific table order remains
the frozen E11/E13 panel order regardless of operational execution order.
