#!/usr/bin/env python3
"""Priority-2 multi-input stability check on the FROZEN E13 candidates.

Never re-selects a candidate. For each model, loads its frozen (layer, row) from
results/E13/candidate_manifest.json, builds N independent domain-appropriate inputs
using the exact same window-sampling machinery / per-model tokenizer conventions
already used by run_singleton_census.py and run_rowwise_detector.py (build_windows for
text, read_fasta_windows_mlm for genomic encoders, e12.build_corpora's damage pool for
genomic decoders), and records, per input, the frozen row's activation_max, its rank
and percentile among that layer's channels, and the sequence position of its own max.

No ablation, no masking, no backward pass -- forward passes only. Restartable/atomic
per-model output, same convention as run_rowwise_detector.py.
"""
from __future__ import annotations

import argparse, json, os, random, sys, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SALVAGE, ROOT = HERE.parents[1], HERE.parents[1].parent
sys.path[:0] = [str(HERE), str(SALVAGE / "experiments" / "E10_nlp_architecture_causal"),
                str(SALVAGE / "experiments" / "E9_mechanistic_tomography"),
                str(SALVAGE / "experiments" / "E12_generator_degradation_control"),
                str(SALVAGE / "experiments" / "E7_exact_dimensionality"),
                str(SALVAGE / "experiments" / "E11_scale_ladder")]
import e10_lib as L  # noqa: E402
import tomography_lib as tl  # noqa: E402
import e12_lib as e12  # noqa: E402
from dnabert2_compat import load_dnabert2_patched, load_mosaicbert_patched  # noqa: E402
from genomic_encoder_lib import load_ntv3_pretrained  # noqa: E402
from genomic_decoder_lib import load_genomeocean, window_nll_genomeocean, prepare_sequence_genomeocean  # noqa: E402
from run_singleton_census import ALIASES, load_generic, load_structural_row, model_pattern  # noqa: E402
from run_model import _ACTB_500  # noqa: E402

N_INPUTS = int(os.environ.get("E13_STABILITY_N_INPUTS", "24"))
OUT_DIR = ROOT / "results" / "E13_candidate_stability"
MANIFEST = ROOT / "results" / "E13" / "candidate_manifest.json"
SEED = 42


def frozen_candidate(model_name: str) -> tuple[int, int]:
    manifest = json.loads(MANIFEST.read_text())
    rows = [c for c in manifest["candidates"] if c["model"] == model_name and c.get("primary", True)]
    assert len(rows) == 1, f"expected exactly one primary candidate for {model_name}, got {rows}"
    return rows[0]["layer"], rows[0]["row"]


def load(name, structural):
    # Absent-path fix, no science changed: allow a fresh HF download when the shared
    # /work cache env_cached.sh assumes isn't present on this filesystem (default
    # unchanged -- local_files_only=True -- wherever that cache does exist).
    local_only = os.environ.get("E13_LOCAL_FILES_ONLY", "1") != "0"
    if name == "DNABERT-2":
        model, tok, cfg, _ = load_dnabert2_patched(local_files_only=local_only)
        return model, tok, cfg["down_proj_pattern"]
    if name == "MosaicBERT":
        rev = None if structural["revision"].startswith("unpinned") else structural["revision"]
        model, tok, _ = load_mosaicbert_patched(structural["repo"], revision=rev, local_files_only=local_only)
        return model, tok, model_pattern(name)
    if name == "NTv3":
        model, tok = load_ntv3_pretrained(local_files_only=local_only)
        return model, tok, model_pattern(name)
    if name == "GenomeOcean-4B":
        model, tok = load_genomeocean(local_files_only=local_only)
        return model, tok, model_pattern(name)
    model, tok, _ = load_generic({"model": name, "repo": structural["repo"], "revision": structural["revision"]},
                                  structural["architecture"])
    return model, tok, model_pattern(name)


def build_inputs(name, domain, architecture, tok):
    """Returns a list of already-tokenized [1, L] cuda LongTensors, one per input,
    each built with the SAME per-model preprocessing convention run_rowwise_detector.py
    uses for this model's single frozen probe -- just applied to N independent texts
    instead of one fixed one."""
    if domain == "text":
        # e10_lib.build_windows already applies each model's OWN tokenizer/special-token
        # convention (ENDPOINTS.md); text selection is seeded identically across models.
        windows = L.build_windows(tok, N_INPUTS, 512, seed=SEED)
        return [w.unsqueeze(0).cuda() for w in windows]

    if name == "NTv3":
        seqs = tl.read_fasta_windows_mlm(tl.HG38_FASTA, tl.HG38_BED, n_windows=N_INPUTS,
                                          win_bp=600, rng=random.Random(SEED))
        out = []
        for seq in seqs:
            while len(tok(seq, add_special_tokens=False)["input_ids"]) % 256:
                seq += "A"
            enc = tok(seq, return_tensors="pt", add_special_tokens=False)
            out.append(enc["input_ids"].cuda())
        return out

    if name == "DNABERT-2":
        seqs = tl.read_fasta_windows_mlm(tl.HG38_FASTA, tl.HG38_BED, n_windows=N_INPUTS,
                                          win_bp=600, rng=random.Random(SEED))
        out = []
        for seq in seqs:
            sequence = (tok.bos_token or "") + seq
            enc = tok(sequence, return_tensors="pt", add_special_tokens=False)
            out.append(enc["input_ids"].cuda())
        return out

    # genomic decoder: GENERator-EUK/PROK, GenomeOcean -- reuse E12's disjoint damage
    # pool (already vetted, non-overlapping with the prompt pool used elsewhere) and
    # each model's own established sequence-prep function.
    _, damage_windows, _ = e12.build_corpora(seed=SEED, n_prompt=96, prompt_win_bp=170,
                                              n_damage=max(N_INPUTS, 100), damage_win_bp=512)
    seqs = [seq for _, _, seq in damage_windows[:N_INPUTS]]
    prep_fn = prepare_sequence_genomeocean if name == "GenomeOcean-4B" else e12._prepare_sequence
    out = []
    for seq in seqs:
        prep = prep_fn(tok, seq)
        enc = tok(prep, return_tensors="pt", add_special_tokens=False)
        out.append(enc["input_ids"].cuda())
    return out


class RowHook:
    """Captures per-window stats for ONE fixed (layer,row) without ever touching the
    weights -- pure forward-hook readout on the existing down_proj/wo/Wo module."""

    def __init__(self, row: int):
        self.row = row
        self.records = []

    def __call__(self, _module, _inp, output):
        y = output[0] if isinstance(output, tuple) else output
        y = y.detach()
        # [B, L, H] expected (B==1 for every input this script builds).
        flat = y.reshape(-1, y.shape[-1]).abs().float()
        channel_max = flat.max(dim=0).values
        n = channel_max.shape[0]
        row_val = float(channel_max[self.row])
        median = float(channel_max.median())
        rank = 1 + int((channel_max > row_val).sum().item())
        percentile = 100.0 * (n - rank + 1) / n
        row_series = y.reshape(-1, y.shape[-1])[:, self.row].abs().float()
        pos = int(row_series.argmax().item())
        seq_len = int(row_series.shape[0])
        self.records.append(dict(
            activation_max=row_val, activation_median=median,
            activation_ratio=(row_val / median if median > 0 else float("inf")),
            rank=rank, n_rows=n, percentile=percentile,
            max_position=pos, seq_len=seq_len,
            max_position_frac=(pos / max(seq_len - 1, 1)),
        ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(k for k in ALIASES if k != "phi3"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    name = ALIASES[args.model]
    out_path = OUT_DIR / f"{args.model}.json"
    if out_path.exists() and not args.force:
        print(f"SKIP existing {out_path}")
        return

    layer, row = frozen_candidate(name)
    structural = load_structural_row(name)
    domain, architecture = structural["domain"], structural["architecture"]

    t0 = time.time()
    model, tok, pattern = load(name, structural)
    load_seconds = time.time() - t0

    module = L._resolve_module(model, pattern, layer)
    hook = RowHook(row)
    handle = module.register_forward_hook(hook)

    inputs = build_inputs(name, domain, architecture, tok)
    with torch.no_grad():
        for i, ids in enumerate(inputs):
            model(input_ids=ids)
            print(f"  {args.model} input {i+1}/{len(inputs)}", flush=True)
    handle.remove()

    recs = hook.records
    ratios = [r["activation_ratio"] for r in recs if r["activation_ratio"] != float("inf")]
    import statistics as st
    n = len(recs)
    frac_rank1 = sum(1 for r in recs if r["rank"] == 1) / n
    frac_top1pct = sum(1 for r in recs if r["percentile"] >= 99.0) / n
    summary = dict(
        model=name, slug=args.model, domain=domain, architecture=architecture,
        layer=layer, row=row, n_inputs=n, load_seconds=load_seconds,
        activation_ratio_median=st.median(ratios) if ratios else None,
        activation_ratio_iqr=[float(x) for x in (
            st.quantiles(ratios, n=4)[0], st.quantiles(ratios, n=4)[2])] if len(ratios) >= 4 else None,
        activation_ratio_min=min(ratios) if ratios else None,
        activation_ratio_max=max(ratios) if ratios else None,
        percentile_median=st.median(r["percentile"] for r in recs),
        frac_rank1=frac_rank1, frac_top1pct=frac_top1pct,
        max_position_frac_median=st.median(r["max_position_frac"] for r in recs),
        max_position_at_0_fraction=sum(1 for r in recs if r["max_position"] == 0) / n,
    )
    payload = dict(summary=summary, per_input=recs, elapsed_seconds=time.time() - t0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, out_path)
    print(f"WROTE {out_path} frac_rank1={frac_rank1:.3f} frac_top1pct={frac_top1pct:.3f} "
          f"ratio_median={summary['activation_ratio_median']} elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
