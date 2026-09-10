#!/usr/bin/env python3
"""Restartable Stage-1 singleton/control causal census, one model per invocation.

The candidate/control identities come exclusively from candidate_manifest.json.  Each
output is written atomically to results/experiments/E13/raw/<slug>.json only after the complete model
finishes; the nohup queue skips already-valid outputs on restart.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SALVAGE = HERE.parents[1]
ROOT = SALVAGE.parent
sys.path[:0] = [str(HERE), str(SALVAGE / "experiments" / "E10_nlp_architecture_causal"),
                str(SALVAGE / "experiments" / "E9_mechanistic_tomography"),
                str(SALVAGE / "experiments" / "E12_generator_degradation_control")]
import e10_lib as L  # noqa: E402
import tomography_lib as tl  # noqa: E402
import e12_lib as e12  # noqa: E402
from dnabert2_compat import load_dnabert2_patched, load_mosaicbert_patched  # noqa: E402
from genomic_encoder_lib import load_ntv3_pretrained, NTV3_DOWN_PROJ_PATTERN  # noqa: E402
from genomic_decoder_lib import load_genomeocean, window_nll_genomeocean  # noqa: E402
from panel import PANEL_ORDER  # noqa: E402

PREREG = SALVAGE / "docs" / "prereg" / "PREREG_full_cohort_causal_census.md"
MANIFEST = ROOT / "results" / "E13" / "candidate_manifest.json"
STRUCTURAL = ROOT / "results" / "E11" / "scale_ladder_backfilled.csv"
RAW_DIR = ROOT / "results" / "E13" / "raw"
EPSILONS = (0.5, 1.0)

ALIASES = {
    "llama": "Llama-7B", "mistral": "Mistral-7B", "olmo": "OLMo-7B-0724-hf",
    "phi3": "Phi-3-mini-4k-instruct", "qwen25-7b": "Qwen2.5-7B",
    "mosaicbert": "MosaicBERT", "modernbert-base": "ModernBERT-base",
    "ntv3": "NTv3", "dnabert2": "DNABERT-2", "generator-euk-3b": "GENERator-EUK-3B",
    "genomeocean-4b": "GenomeOcean-4B",
    "qwen25-0.5b": "Qwen/Qwen2.5-0.5B", "qwen25-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen25-3b": "Qwen/Qwen2.5-3B", "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M", "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B",
    "generator-prok-1.2b": "GenerTeam/GENERator-v2-prokaryote-1.2b-base",
    "generator-prok-3b": "GenerTeam/GENERator-v2-prokaryote-3b-base",
    "eurobert-210m": "EuroBERT/EuroBERT-210m", "eurobert-610m": "EuroBERT/EuroBERT-610m",
    "eurobert-2.1b": "EuroBERT/EuroBERT-2.1B",
    "modernbert-large": "answerdotai/ModernBERT-large",
}


def verify_lock() -> None:
    r = subprocess.run([sys.executable, str(SALVAGE / "src" / "prereg_lock.py"), "verify",
                        str(PREREG)], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode or "OK" not in r.stdout:
        raise SystemExit("E13 prereg lock verification failed")


def slug(model: str) -> str:
    return next(k for k, v in ALIASES.items() if v == model)


def load_structural_row(model: str) -> dict:
    import csv
    rows = list(csv.DictReader(STRUCTURAL.open()))
    return next(r for r in rows if r["model"] == model)


def model_pattern(model: str) -> str:
    if model == "MosaicBERT":
        return "bert.encoder.layer.{i}.mlp.wo"
    if model.startswith("ModernBERT") or model == "answerdotai/ModernBERT-large":
        return "model.layers.{i}.mlp.Wo"
    if model == "NTv3":
        return NTV3_DOWN_PROJ_PATTERN
    if model == "DNABERT-2":
        return "bert.encoder.layer.{i}.mlp.wo"
    return "model.layers.{i}.mlp.down_proj"


def load_generic(spec: dict, architecture: str):
    from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer, BertTokenizer
    # EuroBERT's pinned remote code uses the pre-Transformers-5 "default" RoPE
    # registry entry.  Transformers 5 removed that entry while retaining the other
    # scaling variants.  Restore the exact legacy (unscaled) definition process-locally;
    # never modify the shared cached model source.
    if spec["model"].startswith("EuroBERT/"):
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        def legacy_default_rope(config, device=None, seq_len=None, **_kwargs):
            del seq_len
            dim = getattr(config, "head_dim", None) or (
                config.hidden_size // config.num_attention_heads
            )
            partial = getattr(config, "partial_rotary_factor", 1.0)
            dim = int(dim * partial)
            base = getattr(config, "rope_theta", 10000.0)
            inv_freq = 1.0 / (
                base
                ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
            )
            return inv_freq, 1.0

        ROPE_INIT_FUNCTIONS.setdefault("default", legacy_default_rope)
    repo, rev = spec["repo"], spec["revision"]
    rev = None if rev.startswith("unpinned") else rev
    # Absent-path fix, no science changed (same discipline as E9's hg38/revision-pin
    # fixes): this session's filesystem does not have the shared /work HF cache
    # env_cached.sh assumes, so local_files_only defaults True (unchanged behavior
    # wherever that cache DOES exist) but can be opted out of via env var when it
    # doesn't, allowing a fresh download instead of a hard failure.
    _local_only = os.environ.get("E13_LOCAL_FILES_ONLY", "1") != "0"
    # Phi-3 is deliberately loaded through the native Transformers implementation,
    # exactly as in the bit-reproducing Stage-0 path.  Its pinned legacy remote code
    # expects rope_scaling["type"], which Transformers 5 normalizes to "rope_type".
    kw = dict(local_files_only=_local_only,
              trust_remote_code=spec["model"] != "Phi-3-mini-4k-instruct")
    if rev:
        kw["revision"] = rev
    if spec["model"] == "MosaicBERT":
        tok = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=_local_only)
    else:
        tok = AutoTokenizer.from_pretrained(repo, **kw)
    loader = AutoModelForCausalLM if architecture == "decoder" else AutoModelForMaskedLM
    model = loader.from_pretrained(repo, dtype=torch.float32, **kw)
    model.eval().cuda()
    return model, tok, getattr(model.config, "_commit_hash", None)


def condition_list(manifest: dict, model: str) -> list[dict]:
    out = []
    for c in manifest["candidates"]:
        if c["model"] == model:
            for eps in EPSILONS:
                out.append({"kind": "candidate", "candidate_index": c["candidate_index"],
                            "layer": c["layer"], "row": c["row"], "epsilon": eps})
    for cs in manifest["control_sets"]:
        if cs["model"] == model:
            for row in cs["control_rows"]:
                for eps in EPSILONS:
                    out.append({"kind": "control", "layer": cs["layer"], "row": row,
                                "epsilon": eps})
    return out


def score_text_decoder(model, tok, pattern, conds):
    windows = L.build_windows(tok, 100, 512)
    # Conservative batches leave room for the full-vocabulary logits of every 7B model.
    batch_size = 4 if sum(p.numel() for p in model.parameters()) > 3_000_000_000 else 8
    units = L.batch_windows(windows, batch_size)
    base, pert = [], [[] for _ in conds]
    for i, ids in enumerate(units):
        ids = ids.cuda()
        s, n, lp, _, _ = L.causal_lm_batch(model, ids)
        base.append([s, n]); del lp
        for j, c in enumerate(conds):
            alpha = L.alphas_for_mask([1], c["epsilon"])[0]
            with L.masked(model, pattern, [(c["layer"], c["row"])], [alpha]):
                s2, n2, lp2, _, _ = L.causal_lm_batch(model, ids)
            pert[j].append([s2, n2]); del lp2
        print(f"unit {i+1}/{len(units)}", flush=True)
    return base, pert, {"n_windows": 100, "batch_size": batch_size, "max_tokens": 512}


def score_text_encoder(model, tok, pattern, conds):
    windows = L.build_windows(tok, 256, 512)
    units = L.build_fixed_mlm_batches(tok, windows, "cuda", 16, mask_prob=0.15, seed=42)
    base, pert = [], [[] for _ in conds]
    for i, batch in enumerate(units):
        s, n = L.mlm_loss_per_batch(model, [batch])[0]; base.append([s, n])
        for j, c in enumerate(conds):
            alpha = L.alphas_for_mask([1], c["epsilon"])[0]
            with L.masked(model, pattern, [(c["layer"], c["row"])], [alpha]):
                s2, n2 = L.mlm_loss_per_batch(model, [batch])[0]
            pert[j].append([s2, n2])
        print(f"unit {i+1}/{len(units)}", flush=True)
    return base, pert, {"n_windows": 256, "batch_size": 16, "max_tokens": 512,
                        "mask_prob": 0.15}


def score_genomic_encoder(model, tok, pattern, conds):
    import random
    seqs = tl.read_fasta_windows_mlm(tl.HG38_FASTA, tl.HG38_BED, n_windows=256, win_bp=600,
                                      rng=random.Random(42))
    units = tl.build_fixed_batches(tok, seqs, "cuda", max_len=256, batch_size=16,
                                   mask_prob=0.15, seed=42)
    base = [[s, n] for s, n in tl.mlm_loss_per_batch(model, units)]
    pert = []
    for j, c in enumerate(conds):
        alpha = tl.alphas_for_mask([1], c["epsilon"])[0]
        with L.masked(model, pattern, [(c["layer"], c["row"])], [alpha]):
            pb = [[s, n] for s, n in tl.mlm_loss_per_batch(model, units)]
        pert.append(pb); print(f"condition {j+1}/{len(conds)}", flush=True)
    return base, pert, {"n_windows": 256, "win_bp": 600, "max_tokens": 256,
                        "batch_size": 16, "mask_prob": 0.15}


def score_genomic_decoder(model, tok, pattern, conds, genomeocean=False):
    _, windows, meta = e12.build_corpora(seed=42, n_prompt=96, prompt_win_bp=170,
                                         n_damage=100, damage_win_bp=512)
    fn = window_nll_genomeocean if genomeocean else e12.window_nll
    base = [[*fn(model, tok, seq)] for _, _, seq in windows]
    pert = []
    for j, c in enumerate(conds):
        alpha = L.alphas_for_mask([1], c["epsilon"])[0]
        with L.masked(model, pattern, [(c["layer"], c["row"])], [alpha]):
            pb = [[*fn(model, tok, seq)] for _, _, seq in windows]
        pert.append(pb); print(f"condition {j+1}/{len(conds)}", flush=True)
    return base, pert, {"n_windows": len(windows), "win_bp": 512, "corpus_meta": meta}


def weighted(pb):
    return sum(x[0] for x in pb) / max(sum(x[1] for x in pb), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(ALIASES))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    verify_lock()
    model_name = ALIASES[args.model]
    outpath = RAW_DIR / f"{args.model}.json"
    if outpath.exists() and not args.force:
        print(f"SKIP existing {outpath}"); return
    manifest = json.loads(MANIFEST.read_text())
    conds = condition_list(manifest, model_name)
    structural = load_structural_row(model_name)
    architecture, domain = structural["architecture"], structural["domain"]
    spec = {"model": model_name, "repo": structural["repo"], "revision": structural["revision"]}
    pattern = model_pattern(model_name)
    t0 = time.time()
    if model_name == "DNABERT-2":
        model, tok, cfg, patched = load_dnabert2_patched(local_files_only=True)
        pattern = cfg["down_proj_pattern"]; resolved = structural["revision"]
    elif model_name == "MosaicBERT":
        revision = None if structural["revision"].startswith("unpinned") else structural["revision"]
        model, tok, resolved = load_mosaicbert_patched(
            structural["repo"], revision=revision, local_files_only=True)
    elif model_name == "NTv3":
        model, tok = load_ntv3_pretrained(local_files_only=True); resolved = structural["revision"]
    elif model_name == "GenomeOcean-4B":
        model, tok = load_genomeocean(local_files_only=True); resolved = structural["revision"]
    else:
        model, tok, resolved = load_generic(spec, architecture)
    load_seconds = time.time() - t0
    if domain == "text" and architecture == "decoder":
        base, pert, endpoint = score_text_decoder(model, tok, pattern, conds)
    elif domain == "text" and architecture == "encoder":
        base, pert, endpoint = score_text_encoder(model, tok, pattern, conds)
    elif domain == "genomic" and architecture == "encoder":
        base, pert, endpoint = score_genomic_encoder(model, tok, pattern, conds)
    else:
        base, pert, endpoint = score_genomic_decoder(model, tok, pattern, conds,
                                                      genomeocean=model_name == "GenomeOcean-4B")
    baseline = weighted(base)
    results = []
    for c, pb in zip(conds, pert):
        loss = weighted(pb)
        results.append({**c, "perturbed_loss": loss, "absolute_delta_loss": loss-baseline,
                        "relative_loss_change": (loss-baseline)/baseline, "per_unit": pb})
    payload = {"model": model_name, "slug": args.model, "repo": structural["repo"],
               "requested_revision": structural["revision"], "resolved_revision": resolved,
               "domain": domain, "architecture": architecture, "pattern": pattern,
               "dtype": "float32", "load_seconds": load_seconds, "endpoint": endpoint,
               "baseline_loss": baseline, "baseline_per_unit": base, "conditions": results}
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tmp = outpath.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, outpath)
    print(f"WROTE {outpath} baseline={baseline:.8f} elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
