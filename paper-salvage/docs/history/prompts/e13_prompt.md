# Fix and Complete the E13 Multicandidate Structural Census

Stop changing the execution environment or rebuilding data-loading paths that already worked in previous experiments.

We have already successfully run these same model families in E11/E13. The current failures are engineering regressions introduced by the new runner, not scientific blockers.

## Current state

The detached run ended prematurely.

Completed successfully:

- DNABERT-2 — K=12
- GENERator-EUK-3B — K=172
- GenomeOcean-4B — K=410
- GENERator-PROK-1.2B — K=490
- GENERator-PROK-3B — K=333

Failed:

- 17/22 models

Primary failure:
- text models are trying to load WikiText through a new/current Hugging Face dataset-loader path;
- cached metadata contains `hf://datasets/wikitext/...`;
- the installed `datasets` / Hub stack cannot parse it.

Separate failure:
- NTv3 has a sequence-length/padding mismatch.

No Part 2B tomography has been run. Good. Keep it that way until the structural census is complete and frozen.

---

# What I want you to do

## 1. Do not "fix WikiText" by inventing a new data pipeline

We already had successful E11/E13 runs.

Find the **exact materialized evaluation inputs and preprocessing path that those successful experiments used**.

Search the repository, prior result artifacts, configs, scripts, cached arrays/files, logs, commits, and manifests.

The correct solution is preferably:

> reuse the exact already-materialized WikiText examples/windows/tokens from the successful previous experiment

NOT:

> call Hugging Face `load_dataset()` again and hope the environment works.

There is no scientific reason to redownload or reconstruct WikiText.

The structural census must use the same frozen calibration inputs used previously wherever possible.

### Explicitly trace:

- which script produced the previous successful text-model structural/causal inputs;
- where the actual tokenized/materialized examples live;
- tokenizer revision;
- sequence lengths;
- masking if applicable;
- batch construction;
- padding/truncation behavior;
- model revision;
- relevant package/environment details.

Then make the new structural runner consume those exact frozen inputs directly.

If necessary, vendor/copy the already-materialized examples into the new experiment directory in a simple local format such as `.pt`, `.npz`, `.jsonl`, etc.

Do not depend on Hugging Face dataset resolution during the run.

---

# 2. Fix NTv3 by reproducing the exact previous successful NTv3 preprocessing

Do not guess at sequence length.

Find the latest **valid successful NTv3 forward-pass pipeline** in the project and reproduce exactly:

- tokenizer
- tokenization convention
- max sequence length
- padding
- truncation
- attention mask
- batch shape
- model revision

We already know this model has previously suffered from an incorrect DNABERT-style truncation configuration, so be especially careful not to reuse the wrong preprocessing.

First run one forward-pass smoke and verify tensor dimensions before putting NTv3 back into the queue.

---

# 3. Do not rerun the five completed genomic models unless required

Their raw structural census outputs already exist.

Validate their artifacts and preserve them.

Only rerun them if you find an actual implementation error affecting their measurements.

Otherwise resume from the remaining failed models.

The pipeline must be restartable and skip completed valid outputs.

---

# 4. Add a real preflight test before launching 17 expensive jobs

Before relaunching the cohort, test:

### One text decoder
Prefer a small one, e.g. Qwen2.5-0.5B or SmolLM2-135M.

Verify:
- local frozen input loads;
- tokenizer/model inputs are correct;
- forward pass succeeds;
- row-wise detector produces output;
- output schema is valid.

### One text encoder
Prefer a small one.

Verify:
- frozen masked inputs work;
- forward pass succeeds;
- row detector output is valid.

### NTv3
Verify its corrected sequence path separately.

Do not launch the full queue until these three preflight cases succeed end-to-end.

---

# 5. Preserve the scientific protocol

Do not change:

- the 22-model cohort;
- model revisions;
- the >=5 detector threshold;
- detector definition;
- selected calibration data;
- structural metrics;
- candidate eligibility logic.

This is an engineering repair only.

Do not tweak the detector because some models produce large K.

Do not cap K.

Do not introduce top-K selection.

Do not start tomography.

---

# 6. Run everything detached and resiliently

The jobs must survive:

- closing VS Code;
- losing the SSH tunnel;
- losing my laptop connection.

On Vista, GPU work must run through Slurm.

Set up the complete remaining queue before leaving it.

Use persistent Slurm job scripts and/or a detached `tmux` orchestration session as appropriate.

The required behavior is:

1. queue all remaining models;
2. assign jobs to available GPU resources;
3. checkpoint each completed model;
4. skip already completed valid models on restart;
5. if one model fails, record it and continue;
6. write persistent stdout/stderr logs;
7. leave the node/job running independently of VS Code.

Do not leave the campaign dependent on an interactive foreground terminal.

---

# 7. Make the queue observable

Maintain a status file such as:

`results/E13_multicandidate_structural/status.json`

with:

- model
- status: pending/running/completed/failed
- Slurm job ID
- GPU/node if known
- start time
- finish time
- output path
- failure reason

Also keep logs under:

`results/E13_multicandidate_structural/logs/`

At launch, print the exact commands I can later use to:

- check Slurm jobs;
- check GPU utilization;
- inspect the status file;
- tail logs;
- attach to tmux if used;
- verify whether all 22 models are complete.

---

# 8. Do not stop after merely explaining another failure

Your task is not to report that a library version is incompatible.

Your task is to **recover the already-working project input path and complete the experiment**.

If the current runner conflicts with the previous successful pipeline, adapt the new runner to the old proven pipeline.

Prefer reuse over reinvention.

Only stop if there is a genuine scientific or hardware blocker that cannot be resolved from existing project artifacts.

---

# Completion criteria

Do not call this done until:

- all 22 cohort models either have valid structural census outputs or a genuinely irrecoverable model-specific blocker is documented;
- all successful outputs pass schema/integrity checks;
- the full row-wise detector results are assembled;
- candidate K is reported for every completed model;
- the resulting basis summary is generated;
- the basis is ready for scientific review;
- **no Part 2B tomography has been launched yet**.

Then produce:

- updated `full_cohort_rowwise_detector.csv`
- `full_cohort_multicandidate_basis.csv`
- `basis_summary_by_model.csv`
- final status table
- `MULTICANDIDATE_STRUCTURAL_CENSUS_SUMMARY.md`

And tell me plainly:

1. how many of 22 completed;
2. K for every model;
3. whether any failures remain;
4. whether the >=5 row-wise detector behaved sensibly across architectures;
5. whether the resulting basis is suitable for Part 2B tomography.

Do the repair, run the preflights, queue the remaining cohort detached, and let it finish.