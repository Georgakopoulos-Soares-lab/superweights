"""
experiments/E13_full_cohort_causal_census/genomic_encoder_lib.py

NTv3 masked-nucleotide-LM loss. No NTv3 equivalent of E9's DNABERT-2 MLM stack
(`tomography_lib.py::load_dnabert2_pretrained`/`read_fasta_windows_mlm`/
`build_fixed_batches`/`mlm_loss_per_batch`/`mlm_loss`) exists anywhere in this repo before
this file. `build_fixed_batches`/`mlm_loss_per_batch`/`mlm_loss`/`read_fasta_windows_mlm`
are already architecture-agnostic (they only touch `tok.cls_token_id`/`sep_token_id`/
`pad_token_id`/`mask_token_id` and a generic `model(input_ids=..., attention_mask=...,
labels=...)` call) -- imported and reused HERE UNMODIFIED. The only new code is the
NTv3-specific loader (`load_ntv3_pretrained`), mirroring `load_dnabert2_pretrained`'s
structure with NTv3's own repo/revision/tokenizer/model class in place of DNABERT-2's, and
the row-scaling pattern reused from `configs/ntv3.yaml`'s already-confirmed
`down_proj_pattern` (== `uk_frobenius.adapter_ntv3`'s `core.transformer_blocks.{i}.fc2`,
independently verified in that adapter's own test suite,
`paper-salvage/src/test_ntv3_adapter.py`).

Same masking convention as DNABERT-2 (mask_prob=0.15), same fixed-seed batch construction
(seed 42), same row-scaling intervention (`tomography_lib._scale_rows`/`with_mask`, imported
unmodified). This is a small adapter, not new methodology.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

E9_DIR = Path(__file__).resolve().parents[1] / "E9_mechanistic_tomography"
sys.path.insert(0, str(E9_DIR))
import tomography_lib as tl  # noqa: E402

# Reused unmodified -- already architecture-agnostic apart from tokenizer/model.
read_fasta_windows_mlm = tl.read_fasta_windows_mlm
build_fixed_batches = tl.build_fixed_batches
mlm_loss_per_batch = tl.mlm_loss_per_batch
aggregate_per_batch = tl.aggregate_per_batch
mlm_loss = tl.mlm_loss
with_mask = tl.with_mask  # re-export for callers of this module
_scale_rows = tl._scale_rows
_restore_rows = tl._restore_rows
alphas_for_mask = tl.alphas_for_mask

ROOT = tl.ROOT
HG38_FASTA = tl.HG38_FASTA
HG38_BED = tl.HG38_BED
SEED = tl.SEED

NTV3_REPO = "InstaDeepAI/NTv3_650M_pre"
NTV3_CODE_REVISION = "0ecff3637f0d3ba5b686d1095083218157c2ca34"
NTV3_DOWN_PROJ_PATTERN = "core.transformer_blocks.{i}.fc2"  # configs/ntv3.yaml, confirmed


def load_ntv3_pretrained(device: str = "cuda", local_files_only: bool = False):
    """Load NTv3 via AutoModelForMaskedLM (confirmed loadable this way in
    E7_exact_dimensionality/run_legacy_reanalysis.py's `ntv3` loader), pinned
    `code_revision` per that same script. Returns (model, tokenizer)."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        NTV3_REPO, trust_remote_code=True, code_revision=NTV3_CODE_REVISION,
        local_files_only=local_files_only,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        NTV3_REPO, trust_remote_code=True, code_revision=NTV3_CODE_REVISION,
        torch_dtype=torch.float32, local_files_only=local_files_only,
    )
    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    return model, tok


def ntv3_response(model, coords, a, epsilon, batches) -> float:
    """MLM loss under mask `a` (list of 0/1, same length/order as coords) at scale
    epsilon, for NTv3's down-proj pattern. Mirrors tomography_lib.dnabert2_response."""
    pattern = NTV3_DOWN_PROJ_PATTERN
    active = [(c, ai) for c, ai in zip(coords, a) if ai]
    if not active:
        return mlm_loss(model, batches)
    ac, aa = zip(*active)
    alphas = alphas_for_mask(aa, epsilon)
    return with_mask(model, pattern, list(ac), alphas, lambda: mlm_loss(model, batches))
