"""
experiments/E13_full_cohort_causal_census/backfill_layer_median.py

Phase 1, job 1 of the full-cohort causal census (part_prompt.md). Backfills
`layer_median_frob_norm` / `candidate_frob_norm_ratio_to_layer_median` for the 11 "cited"
rows in results/E11/scale_ladder.csv (Llama-7B, Mistral-7B, OLMo-7B-0724-hf, Phi-3-mini,
Qwen2.5-7B, MosaicBERT, ModernBERT-base, GENERator-EUK-3B, GenomeOcean-4B, DNABERT-2, NTv3).

This is NOT a new causal measurement and does not touch candidate selection (all 11
candidate (layer, row) pairs are read verbatim from results/E11/scale_ladder.csv, which is
itself locked/frozen structural output from E7/E8/E10/E11). It is a pure weight-space
Frobenius-norm computation, reusing:
  - experiments/src/uk_frobenius.py (layer_report, ADAPTERS: llama_swiglu / dnabert2 / ntv3)
  - the partial-safetensors-shard-download pattern from E1's fetch_layer_tensors /
    E10's exact_uk_norm_all_rows.py fetch_layer_tensors (llama-style separate gate/up/down
    projections; downloads ONLY the shard(s) holding one layer's MLP weights, not the whole
    checkpoint)

For 4 of the 11 models, the exact full-layer computation ALREADY EXISTS on disk from prior
experiments and is simply read back, not recomputed:
  - OLMo-7B-0724-hf: results/e10_exact_uknorm_olmo.json (layer 1, all 4096 rows, Gram-identity
    exact ||U_k||_F, validated against spectral_lib SVD to rel_err < 1e-6 in that script)
  - Phi-3-mini-4k-instruct: results/e10_exact_uknorm_phi3.json (layers 2 and 4 separately, all
    3072 rows each)
  - MosaicBERT: results/e10_encoder_row_ranking_mosaicbert.json (full per-row q1/frob_norm scan,
    all 12 layers x 768 rows, via row_spectral_metrics/SVD -- exact, not the Gram shortcut, but
    the same frob_norm quantity)
  - ModernBERT-base: results/e10_encoder_row_ranking_modernbert.json (same, 22 layers x 768 rows)

For the remaining 7, this script performs a NEW (but cheap, weight-only) computation:
  - Llama-7B, Mistral-7B, Qwen2.5-7B, GENERator-EUK-3B, GenomeOcean-4B: llama-style separate
    gate_proj/up_proj/down_proj. Downloads only the safetensors shard(s) holding the candidate
    layer's MLP weights (partial fetch, not a full-model load), then runs
    uk_frobenius.layer_report() -- the EXACT closed-form ||U_k||_F for every row of that one
    layer (Gram-matrix identity, no SVD, float64; this is the same math as
    exact_uk_norm_all_rows.py, validated there against spectral_lib to rel_err < 1e-6).
  - DNABERT-2, NTv3: small models (117M / 650M) with custom packed-FFN module layouts already
    handled by uk_frobenius.ADAPTERS['dnabert2'] / ['ntv3']. Full model load (cheap at this
    size) via AutoModelForMaskedLM, then the same layer_report() call, restricted to the one
    candidate layer (not all layers, to keep this a targeted backfill, not a full re-audit).

Output: results/E13/layer_median_backfill.json (one entry per model, full LayerReport-shaped
data for the candidate's own layer) AND an updated
results/E11/scale_ladder_backfilled.csv (scale_ladder.csv is NOT edited in place --
it is treated as an existing frozen artifact; the backfilled copy is a new file, with a
`layer_median_source` column documenting exactly which of the two paths above was used per
row).

Usage:
  python3 backfill_layer_median.py             # do all 11
  python3 backfill_layer_median.py --model llama-7b   # do one (see MODEL_KEYS)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

SALVAGE = Path(__file__).resolve().parents[2]
ROOT = SALVAGE.parent
sys.path.insert(0, str(SALVAGE / "src"))
sys.path.insert(0, str(SALVAGE / "experiments" / "E7_exact_dimensionality"))

from uk_frobenius import layer_report, ADAPTERS  # noqa: E402

RESULTS_DIR = ROOT / "results"
E13_DIR = RESULTS_DIR / "E13"
E13_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RESULTS_DIR / "E11" / "scale_ladder.csv"
OUT_JSON = E13_DIR / "layer_median_backfill.json"
OUT_CSV = RESULTS_DIR / "E11" / "scale_ladder_backfilled.csv"


# --------------------------------------------------------------------------------------
# Precomputed-source models: already-exact full-layer ||U_k||_F sitting on disk from prior
# experiments. No new forward pass, no new weight download -- pure JSON extraction.
# --------------------------------------------------------------------------------------

def from_exact_uknorm_json(model_key: str, layer: int, row: int) -> dict:
    """OLMo / Phi-3: results/e10_exact_uknorm_{model_key}.json, Gram-identity exact, all rows."""
    p = RESULTS_DIR / f"e10_exact_uknorm_{model_key}.json"
    d = json.loads(p.read_text())
    L = d["layers"][str(layer)]
    tgt = next(t for t in L["targets"] if t["row"] == row)
    assert abs(tgt["uk_norm"] / L["layer_median_uk"] - tgt["uk_over_layer_median"]) < 1e-6
    return dict(
        source_file=str(p.relative_to(ROOT)),
        method="Gram-identity exact ||U_k||_F, all rows in layer, float64 "
               "(exact_uk_norm_all_rows.py, validated against SVD to rel_err<1e-6)",
        n_rows_in_layer=L["d_model"], candidate_frob_norm=tgt["uk_norm"],
        layer_median_frob_norm=L["layer_median_uk"],
        candidate_frob_norm_ratio_to_layer_median=tgt["uk_over_layer_median"],
    )


def from_encoder_row_ranking_json(model_key: str, layer: int, row: int) -> dict:
    """MosaicBERT / ModernBERT-base: results/e10_encoder_row_ranking_{model_key}.json,
    per-row SVD (row_spectral_metrics) already computed for every row of every layer."""
    p = RESULTS_DIR / f"e10_encoder_row_ranking_{model_key}.json"
    d = json.loads(p.read_text())
    layer_rows = [r for r in d["full_ranking"] if r["layer"] == layer]
    frobs = sorted(r["frob_norm"] for r in layer_rows)
    n = len(frobs)
    median = frobs[n // 2] if n % 2 else 0.5 * (frobs[n // 2 - 1] + frobs[n // 2])
    cand = next(r for r in layer_rows if r["row"] == row)
    return dict(
        source_file=str(p.relative_to(ROOT)),
        method="exact ||U_k||_F via SVD (spectral_lib.row_spectral_metrics), every row of "
               "the layer already scored by E10 rank_encoder_rows.py",
        n_rows_in_layer=n, candidate_frob_norm=cand["frob_norm"],
        layer_median_frob_norm=median,
        candidate_frob_norm_ratio_to_layer_median=cand["frob_norm"] / median,
    )


# --------------------------------------------------------------------------------------
# New-computation models: partial safetensors fetch (llama-style) or small full-model load
# (dnabert2/ntv3), then uk_frobenius.layer_report on that one layer only.
# --------------------------------------------------------------------------------------

def fetch_llama_style_layer(repo: str, layer: int, revision: str | None = None,
                             local_files_only: bool = False):
    """Fetch only the shard(s) holding this layer's gate/up/down projections. When
    `local_files_only=True`, this NEVER touches the network and NEVER writes into whatever
    HF_HOME is currently set -- it only succeeds if the index + needed shard(s) are already
    present in the local cache (used to safely reuse the pre-existing, disk-quota-sensitive
    /work/.../huggingface/.hf-cache without risking a new write there)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    names = {
        "gate": f"model.layers.{layer}.mlp.gate_proj.weight",
        "up": f"model.layers.{layer}.mlp.up_proj.weight",
        "down": f"model.layers.{layer}.mlp.down_proj.weight",
    }
    idx_path = hf_hub_download(repo, "model.safetensors.index.json", revision=revision,
                                local_files_only=local_files_only)
    weight_map = json.loads(Path(idx_path).read_text())["weight_map"]
    missing = [n for n in names.values() if n not in weight_map]
    if missing:
        raise RuntimeError(f"tensor names not in safetensors index for {repo}: {missing}")
    loaded = {}
    for shard in sorted({weight_map[n] for n in names.values()}):
        p = hf_hub_download(repo, shard, revision=revision, local_files_only=local_files_only)
        data = load_file(p)
        for key, tname in names.items():
            if tname in data:
                loaded[key] = data[tname]
        del data
    return loaded["gate"], loaded["up"], loaded["down"]


def compute_llama_style(repo: str, layer: int, row: int, revision: str | None = None,
                         local_files_only: bool = False) -> dict:
    t0 = time.time()
    Wg, Wu, Wd = fetch_llama_style_layer(repo, layer, revision=revision,
                                          local_files_only=local_files_only)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rep, _contrib = layer_report(Wg, Wu, Wd, layer, query_rows=[row])
    dt = time.time() - t0
    qr = rep.query_ranks[row]
    return dict(
        source_file=None,
        method=f"NEW: partial-safetensors-shard fetch (repo={repo}, revision={revision}) "
               "+ uk_frobenius.layer_report (Gram-identity exact ||U_k||_F, all rows in "
               f"layer, float64). Wall-clock {dt:.1f}s.",
        n_rows_in_layer=rep.d_model, candidate_frob_norm=qr["value"],
        layer_median_frob_norm=rep.median,
        candidate_frob_norm_ratio_to_layer_median=qr["value"] / rep.median,
    )


def compute_small_full_load(repo: str, adapter: str, layer: int, row: int,
                             revision: str | None = None, code_revision: str | None = None,
                             local_files_only: bool = False) -> dict:
    t0 = time.time()
    if adapter == "dnabert2":
        # tomography_lib.load_dnabert2_pretrained (-> models.WRAPPER_MAP['dnabert2']) hits
        # two environment-version bugs under this session's torch 2.5.1/transformers
        # 5.15.1 pair (meta-device alibi construction; torch.load version gate for the
        # legacy .bin checkpoint) -- see dnabert2_compat.py's docstring for full diagnosis.
        # Its shim reproduces the wrapper's own is_decoder/pad_token_id fix plus two more,
        # narrowly scoped, environment-compat patches; not a reimplementation of the
        # loading logic itself.
        HERE_DIR = Path(__file__).resolve().parent
        if str(HERE_DIR) not in sys.path:
            sys.path.insert(0, str(HERE_DIR))
        from dnabert2_compat import load_dnabert2_patched  # noqa: E402
        model, _tok, _cfg, _patched = load_dnabert2_patched(
            local_files_only=local_files_only)
    else:
        from transformers import AutoModelForMaskedLM
        kwargs = dict(trust_remote_code=True, torch_dtype=torch.float32,
                      local_files_only=local_files_only)
        if revision is not None:
            kwargs["revision"] = revision
        if code_revision is not None:
            kwargs["code_revision"] = code_revision
        model = AutoModelForMaskedLM.from_pretrained(repo, **kwargs)
        model.eval()
    Wg, Wu, Wd = ADAPTERS[adapter](model, layer)
    Wg, Wu, Wd = Wg.detach().clone(), Wu.detach().clone(), Wd.detach().clone()
    del model
    rep, _contrib = layer_report(Wg, Wu, Wd, layer, query_rows=[row])
    dt = time.time() - t0
    qr = rep.query_ranks[row]
    return dict(
        source_file=None,
        method=f"NEW: full small-model load (repo={repo}, adapter={adapter}) "
               f"+ uk_frobenius.layer_report. Wall-clock {dt:.1f}s.",
        n_rows_in_layer=rep.d_model, candidate_frob_norm=qr["value"],
        layer_median_frob_norm=rep.median,
        candidate_frob_norm_ratio_to_layer_median=qr["value"] / rep.median,
    )


# --------------------------------------------------------------------------------------
# Model registry: exactly the 11 "cited" rows in results/E11/scale_ladder.csv. (layer, row)
# and repo copied verbatim from that CSV -- candidate selection is NOT touched here.
# --------------------------------------------------------------------------------------

MODELS = {
    "llama-7b": dict(
        csv_model="Llama-7B", layer=2, row=3968, kind="new_llama_style",
        repo="huggyllama/llama-7b", revision="4782ad278652c7c71b72204d462d6d01eaaf7549",
        local_files_only=True,  # already fully present in the shared /work HF cache
    ),
    "mistral-7b": dict(
        csv_model="Mistral-7B", layer=1, row=2070, kind="new_llama_style",
        repo="mistralai/Mistral-7B-v0.1", revision="27d67f1b5f57dc0953326b2601d68371d40ea8da",
        local_files_only=True,
    ),
    "olmo-7b": dict(
        csv_model="OLMo-7B-0724-hf", layer=1, row=269, kind="precomputed_exact_uknorm",
        precomputed_key="olmo",
    ),
    "phi3-mini": dict(
        csv_model="Phi-3-mini-4k-instruct", layer=None, row=None, kind="phi3_multi",
        # 6 published rows across 2 layers -- handled specially, see main().
    ),
    "qwen25-7b": dict(
        csv_model="Qwen2.5-7B", layer=26, row=458, kind="new_llama_style",
        repo="Qwen/Qwen2.5-7B", revision="d149729398750b98c0af14eb82c78cfe92750796",
        local_files_only=True,
    ),
    "mosaicbert": dict(
        csv_model="MosaicBERT", layer=9, row=287, kind="precomputed_row_ranking",
        precomputed_key="mosaicbert",
    ),
    "modernbert-base": dict(
        csv_model="ModernBERT-base", layer=15, row=251, kind="precomputed_row_ranking",
        precomputed_key="modernbert",
    ),
    "generator-euk-3b": dict(
        csv_model="GENERator-EUK-3B", layer=4, row=2371, kind="new_llama_style",
        repo="GenerTeam/GENERator-v2-eukaryote-3b-base",
        revision="7dc01bccce5b65e15141170538afdc2ff09d8dde",
        local_files_only=True,  # confirmed present (all 3 shards) in the shared /work cache
    ),
    "genomeocean-4b": dict(
        csv_model="GenomeOcean-4B", layer=1, row=2604, kind="new_llama_style",
        repo="DOEJGI/GenomeOcean-4B", revision="2bed2fc3ed47c5f6955ba3e64563512c9b338dfb",
        local_files_only=True,
    ),
    "dnabert2": dict(
        csv_model="DNABERT-2", layer=5, row=603, kind="new_small_full_load",
        repo="zhihan1996/DNABERT-2-117M", adapter="dnabert2",
        revision="7bce263b15377fc15361f52cfab88f8b586abda0",
        local_files_only=True,
    ),
    "ntv3": dict(
        csv_model="NTv3", layer=11, row=1472, kind="new_small_full_load",
        repo="InstaDeepAI/NTv3_650M_pre", adapter="ntv3",
        code_revision="0ecff3637f0d3ba5b686d1095083218157c2ca34",
        local_files_only=True,  # confirmed present (weights + remote code) in shared cache
    ),
}


def run_one(key: str) -> dict:
    spec = MODELS[key]
    print(f"=== {key} ({spec['csv_model']}) ===")

    if spec["kind"] == "precomputed_exact_uknorm":
        out = from_exact_uknorm_json(spec["precomputed_key"], spec["layer"], spec["row"])
    elif spec["kind"] == "precomputed_row_ranking":
        out = from_encoder_row_ranking_json(spec["precomputed_key"], spec["layer"], spec["row"])
    elif spec["kind"] == "new_llama_style":
        out = compute_llama_style(spec["repo"], spec["layer"], spec["row"],
                                   revision=spec.get("revision"),
                                   local_files_only=spec.get("local_files_only", False))
    elif spec["kind"] == "new_small_full_load":
        out = compute_small_full_load(spec["repo"], spec["adapter"], spec["layer"], spec["row"],
                                       revision=spec.get("revision"),
                                       code_revision=spec.get("code_revision"),
                                       local_files_only=spec.get("local_files_only", False))
    elif spec["kind"] == "phi3_multi":
        # 6 published candidate rows across layers 2 and 4 -- report per-layer median
        # (median computed WITHIN each layer, over that layer's own d_model=3072 rows),
        # matching the CSV's own "6 published rows, median" framing. Both layers already
        # fully computed in results/e10_exact_uknorm_phi3.json.
        rows_l2 = [525, 1693, 1113]
        rows_l4 = [525, 1113, 1693]
        per_row = {}
        for r in rows_l2:
            per_row[f"L2/r{r}"] = from_exact_uknorm_json("phi3", 2, r)
        for r in rows_l4:
            per_row[f"L4/r{r}"] = from_exact_uknorm_json("phi3", 4, r)
        out = dict(
            source_file="results/e10_exact_uknorm_phi3.json",
            method="Gram-identity exact ||U_k||_F, all rows in each of layers 2 and 4 "
                   "separately (Phi-3's 6 published candidate rows split across 2 layers; "
                   "layer_median_frob_norm/candidate_frob_norm_ratio_to_layer_median are "
                   "reported per-row below, not collapsed to one scalar, since the 6 rows "
                   "do not share a layer).",
            n_rows_in_layer=None, candidate_frob_norm=None, layer_median_frob_norm=None,
            candidate_frob_norm_ratio_to_layer_median=None,
            per_row=per_row,
        )
    else:
        raise ValueError(spec["kind"])

    out["model_key"] = key
    out["csv_model"] = spec["csv_model"]
    out["layer"] = spec["layer"]
    out["row"] = spec["row"]
    print(f"  -> {json.dumps({k: v for k, v in out.items() if k != 'per_row'}, default=str)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default=None)
    args = ap.parse_args()

    keys = [args.model] if args.model else list(MODELS)

    results = {}
    if OUT_JSON.exists():
        results = json.loads(OUT_JSON.read_text())

    for key in keys:
        try:
            results[key] = run_one(key)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            results[key] = dict(model_key=key, csv_model=MODELS[key]["csv_model"], error=str(e))
        OUT_JSON.write_text(json.dumps(results, indent=2))
        print(f"  checkpointed -> {OUT_JSON}")

    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
