"""
experiments/E13_full_cohort_causal_census/genomic_decoder_lib.py

GenomeOcean-4B teacher-forced causal-LM NLL / damage, reusing E12's `e12_lib.py` machinery
(row scaling, damage-metric structure, control/candidate scoring) with ONE change:
sequence preparation.

`e12_lib.window_nll` calls a module-level `_prepare_sequence(tok, seq)` that trims the
sequence to a multiple of 6bp and prepends BOS -- this is GENERator-specific, because
GENERator uses a fixed 6bp-k-mer tokenizer that needs a length aligned to its k-mer stride
(mirrors `GeneratorWrapper._prepare_sequence` in `models/generator_wrapper.py`). GenomeOcean
uses a different tokenizer entirely: character/BPE, no fixed-length alignment requirement
(`configs/genomeocean.yaml`: "Tokenizer: character-level / BPE; no fixed-length alignment
needed"; confirmed again in `models/genomeocean_wrapper.py`, which tokenizes raw sequences
directly with `add_special_tokens=False`, no trim, no forced BOS). Applying GENERator's
6bp-trim to GenomeOcean would silently corrupt every window's start by up to 5bp for no
reason -- this file exists specifically so that mistake is impossible.

Because `e12_lib.window_nll`/`damage` call `_prepare_sequence` directly rather than
accepting it as a parameter, this file duplicates their ~10-line bodies with the one
substituted line (same "copy the ~40-line idiom with attribution, not `import`" convention
already used elsewhere in this repo when a function isn't factored to take the variation
point as an argument -- see INVENTORY.md item 2 on `DownProjRecorder`). Everything else
(row save/restore, `alphas_for_mask`, `with_mask`/`_scale_rows`/`_restore_rows`,
`find_matching_scale`) is imported and reused unmodified from `e12_lib.py`/`tomography_lib.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

E12_DIR = Path(__file__).resolve().parents[1] / "E12_generator_degradation_control"
sys.path.insert(0, str(E12_DIR))
import e12_lib as e12  # noqa: E402  -- row scaling, with_mask, alphas_for_mask, etc.

E9_DIR = Path(__file__).resolve().parents[1] / "E9_mechanistic_tomography"
sys.path.insert(0, str(E9_DIR))

ROOT = e12.ROOT
HG38_FASTA = e12.HG38_FASTA
HG38_BED = e12.HG38_BED

GENOMEOCEAN_REPO = "DOEJGI/GenomeOcean-4B"
GENOMEOCEAN_REVISION = "2bed2fc3ed47c5f6955ba3e64563512c9b338dfb"


def load_genomeocean(device: str = "cuda", local_files_only: bool = False):
    """Mirrors `models/genomeocean_wrapper.py::GenomeOceanWrapper.load` (left-padding
    tokenizer, sdpa attention) but returns bare (model, tokenizer) rather than a wrapper
    instance, and loads float32 (not the wrapper's default bfloat16) so this endpoint is
    computed at the same precision E10/E12's other causal-LM NLL endpoints use."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        GENOMEOCEAN_REPO, revision=GENOMEOCEAN_REVISION, trust_remote_code=True,
        padding_side="left", use_fast=True, local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        GENOMEOCEAN_REPO, revision=GENOMEOCEAN_REVISION, trust_remote_code=True,
        torch_dtype=torch.float32, attn_implementation="sdpa",
        local_files_only=local_files_only,
    )
    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    return model, tok


def prepare_sequence_genomeocean(tok, sequence: str) -> str:
    """GenomeOcean's own tokenizer needs no fixed-length alignment and no forced BOS
    prepend beyond whatever the tokenizer's own defaults add -- this is the identity
    function, kept as a named function (rather than inlining `seq` directly at each call
    site) so the "GENERator's trim does NOT apply here" decision is visible and citable at
    exactly one place, and so a future contributor cannot "helpfully" copy GENERator's
    `_prepare_sequence` over this one without first reading this docstring."""
    return sequence


@torch.no_grad()
def window_nll_genomeocean(model, tok, seq: str) -> tuple[float, int]:
    """GenomeOcean analogue of e12_lib.window_nll -- identical teacher-forced shifted
    causal-LM loss, only the sequence-prep step differs (see module docstring)."""
    dev = next(model.parameters()).device
    prep = prepare_sequence_genomeocean(tok, seq)
    enc = tok(prep, return_tensors="pt", add_special_tokens=False).to(dev)
    ids = enc["input_ids"]
    if ids.shape[1] < 2:
        return 0.0, 0
    out = model(input_ids=ids, labels=ids)
    n_tok = ids.shape[1] - 1
    return float(out.loss) * n_tok, n_tok


@torch.no_grad()
def damage_genomeocean(model, tok, windows: list[tuple[str, int, str]]) -> float:
    """Same aggregation as e12_lib.damage (mean per-token NLL over a window pool), calling
    window_nll_genomeocean instead of the GENERator-specific window_nll."""
    tot, n = 0.0, 0
    for _c, _s, seq in windows:
        s, k = window_nll_genomeocean(model, tok, seq)
        tot += s
        n += k
    return tot / max(n, 1)


def damage_under_row_alpha_genomeocean(model, tok, pattern, layer, row, alpha, windows) -> float:
    """Same structure as e12_lib.damage_under_row_alpha, GenomeOcean damage fn."""
    from run_gue_ablation import _resolve_module, _save_row, _restore_row  # noqa: E402
    saved = _save_row(model, pattern, layer, row)
    m = _resolve_module(model, pattern, layer)
    with torch.no_grad():
        m.weight.data[row, :] = saved * alpha
    try:
        return damage_genomeocean(model, tok, windows)
    finally:
        _restore_row(model, pattern, layer, row, saved)
