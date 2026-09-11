"""
experiments/E11_scale_ladder/run_model.py

E11 scale-ladder measurement (trimmed scope: OLMo-2 family and 3 bonus data points
dropped by author instruction 2026-08-23, see E11_summary.md). One model per invocation.
Reuses E7's spectral_lib.py, E7/E8's DownProjRecorder + ratio/threshold detection pattern,
and E8/E10's SeedSequence(42)-based control-row draw, unmodified in method.

Fixed panel ordering (N=12, spawn index = position below) -- declared here and must match
results/experiments/E11/scale_ladder.csv's own header before any control row is drawn:

  0 qwen25-0.5b        6 generator-prok-1.2b
  1 qwen25-1.5b        7 generator-prok-3b
  2 qwen25-3b          8 eurobert-210m
  3 smollm2-135m       9 eurobert-610m
  4 smollm2-360m      10 eurobert-2.1b
  5 smollm2-1.7b      11 modernbert-large

Usage:
  python run_model.py --model qwen25-0.5b
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

SALVAGE = Path(__file__).resolve().parents[2]
ROOT = SALVAGE  # repo root; harness paths are written relative to it
sys.path.insert(0, str(SALVAGE / "src"))
sys.path.insert(0, str(SALVAGE / "experiments" / "E7_exact_dimensionality"))

from spectral_lib import row_spectral_metrics  # noqa: E402

PREREG = SALVAGE / "docs" / "prereg" / "PREREG_E11_scale_ladder.md"
RATIO_THRESHOLD = 5.0
N_CONTROL_ROWS = 5

PANEL_ORDER = [
    "qwen25-0.5b", "qwen25-1.5b", "qwen25-3b",
    "smollm2-135m", "smollm2-360m", "smollm2-1.7b",
    "generator-prok-1.2b", "generator-prok-3b",
    "eurobert-210m", "eurobert-610m", "eurobert-2.1b",
    "modernbert-large",
]
N_PANEL = len(PANEL_ORDER)

# repo, revision, family, domain, input_kind ("text"/"genomic"), model_class ("causal"/"mlm"),
# ffn_kind ("llama" separate gate/up/down | "packed_wi_wo" ModernBERT-style)
PANEL = {
    "qwen25-0.5b": dict(repo="Qwen/Qwen2.5-0.5B", revision="060db6499f32faf8b98477b0a26969ef7d8b9987",
                         family="Qwen2.5", domain="text", input_kind="text", model_class="causal", ffn_kind="llama"),
    "qwen25-1.5b": dict(repo="Qwen/Qwen2.5-1.5B", revision="8faed761d45a263340a0528343f099c05c9a4323",
                         family="Qwen2.5", domain="text", input_kind="text", model_class="causal", ffn_kind="llama"),
    "qwen25-3b": dict(repo="Qwen/Qwen2.5-3B", revision="3aab1f1954e9cc14eb9509a215f9e5ca08227a9b",
                       family="Qwen2.5", domain="text", input_kind="text", model_class="causal", ffn_kind="llama"),
    "smollm2-135m": dict(repo="HuggingFaceTB/SmolLM2-135M", revision="93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
                          family="SmolLM2", domain="text", input_kind="text", model_class="causal", ffn_kind="llama"),
    "smollm2-360m": dict(repo="HuggingFaceTB/SmolLM2-360M", revision="f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
                          family="SmolLM2", domain="text", input_kind="text", model_class="causal", ffn_kind="llama"),
    "smollm2-1.7b": dict(repo="HuggingFaceTB/SmolLM2-1.7B", revision="effd688a12921b4cc83e3312b6feb579f70f9c71",
                          family="SmolLM2", domain="text", input_kind="text", model_class="causal", ffn_kind="llama"),
    "generator-prok-1.2b": dict(repo="GenerTeam/GENERator-v2-prokaryote-1.2b-base",
                                 revision="8b2f768b0d293953518ff91d34600f9322ef1f94",
                                 family="GENERator-PROK", domain="genomic", input_kind="genomic",
                                 model_class="causal", ffn_kind="llama"),
    "generator-prok-3b": dict(repo="GenerTeam/GENERator-v2-prokaryote-3b-base",
                               revision="b18ac86df77359d894d7bc050cea78e2d0713021",
                               family="GENERator-PROK", domain="genomic", input_kind="genomic",
                               model_class="causal", ffn_kind="llama"),
    "eurobert-210m": dict(repo="EuroBERT/EuroBERT-210m", revision="39b51e15dd1f1a06f58b5cbf6a8a188cec60bd0e",
                           family="EuroBERT", domain="text", input_kind="text", model_class="mlm", ffn_kind="llama"),
    "eurobert-610m": dict(repo="EuroBERT/EuroBERT-610m", revision="d9af784ed20db6c2096e335ec6a67dd4a219924c",
                           family="EuroBERT", domain="text", input_kind="text", model_class="mlm", ffn_kind="llama"),
    "eurobert-2.1b": dict(repo="EuroBERT/EuroBERT-2.1B", revision="81245a4d71f43452badf5e04458e4ddb831ff109",
                           family="EuroBERT", domain="text", input_kind="text", model_class="mlm", ffn_kind="llama"),
    "modernbert-large": dict(repo="answerdotai/ModernBERT-large", revision="45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",
                              family="ModernBERT", domain="text", input_kind="text", model_class="mlm",
                              ffn_kind="packed_wi_wo"),
}

_ACTB_CDS = (
    "ATGGATGATGATATCGCCGCGCTCGTCGTCGACAACGGCTCCGGCATGTGCAAAGCCGGCTTCGCGGGCGACGATGCCCCGAGGGCC"
    "GTCTTCCCCTCCATCGTGGGGCGCCCCAGGCACCAGGGCGTGATGGTGGGCATGGGTCAGAAGGATTCCTATGTGGGCGACGAGGCC"
    "CAGAGCAAGAGAGGCATCCTCACCCTGAAGTACCCCATCGAGCACGGCATCGTCACCAACTGGGACGACATGGAGAAAATCTGGCAC"
    "CACACCTTCTACAATGAGCTGCGTGTGGCTCCCGAGGAGCACCCCGTGCTGCTCACCGAGGCCCCCCTGAACCCGAAGGCCAACCGC"
    "GAGAAGATGACCCAGATCATGTTTGAGACCTTCAATACCCCCGCCATGTACGTTGCTATCCAGGCTGTGCTATCCCTGTACGCCTCT"
    "GGCCGTACCACTGGCATCGTGATGGACTCCGGTGACGGGGTCACCCACACTGTGCCCATCTACGAGGGGTATGCCCTCCCCCATGCC"
    "ATCCTGCGTCTGGACCTGGCTGGCCGGGACCTGACTGACTACCTCATGAAGATCCTCACCGAGCGCGGCTACAGCTTCACCACCACG"
    "GCCGAGCGGGAAATCGTGCGTGACATTAAGGAGAAGCTGTGCTACGTCGCCCTGGACTTCGAGCAAGAGATGGCCACGGCTGCTTCC"
    "AGCTCCTCCCTGGAGAAGAGCTACGAGCTGCCTGACGGCCAGGTCATCACCATTGGCAATGAGCGGTTCCGCTGCCCTGAGGCACTC"
    "TTCCAGCCTTCCTTCCTGGGCATGGAGTCCTGTGGCATCCACGAAACTACCTTCAACTCCATCATGAAGTGTGACGTGGACATCCGC"
    "AAAGACCTGTACGCCAACACAGTGCTGTCTGGCGGCACCACCATGTACCCTGGCATTGCCGACAGGATGCAGAAGGAGATCACTGCC"
    "CTGGCACCCAGCACAATGAAGATCAAGATCATTGCTCCTCCTGAGCGCAAGTACTCCGTGTGGATCGGCGGCTCCATCCTGGCCTCG"
    "CTGTCCACCTTCCAGCAGATGTGGATCAGCAAGCAGGAGTATGACGAGTCCGGCCCCTCCATCGTCCACCGCAAATGCTTCTAA"
)
assert len(_ACTB_CDS) == 1128
_ACTB_500 = _ACTB_CDS[:504]
assert len(_ACTB_500) == 504


def verify_lock() -> None:
    r = subprocess.run(
        [sys.executable, str(SALVAGE / "src" / "prereg_lock.py"), "verify", str(PREREG)],
        capture_output=True, text=True,
    )
    print(r.stdout.strip())
    if r.returncode != 0 or "OK" not in r.stdout:
        print(r.stderr, file=sys.stderr)
        raise SystemExit("prereg_lock verify FAILED -- refusing to run any forward pass.")


def build_wikitext_input(tokenizer, max_tokens: int = 512) -> str:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    lines = []
    for row in ds:
        t = row["text"].strip()
        if t:
            lines.append(t)
        if len(lines) >= 20:
            break
    text = "\n".join(lines)
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    return tokenizer.decode(ids["input_ids"][0], skip_special_tokens=True)


class DownProjRecorder:
    def __init__(self):
        self.out_max = {}
        self.out_channel = {}
        self.median_channel_max = {}
        self._handles = []

    def register(self, modules: dict):
        for layer_idx, module in modules.items():
            self._handles.append(module.register_forward_hook(self._hook(layer_idx)))

    def _hook(self, layer_idx):
        def fn(module, args, output):
            y = output[0] if isinstance(output, tuple) else output
            flat = y.reshape(-1, y.shape[-1]).abs().float()
            channel_max = flat.max(dim=0).values
            self.out_max[layer_idx] = float(channel_max.max())
            self.out_channel[layer_idx] = int(channel_max.argmax())
            self.median_channel_max[layer_idx] = float(channel_max.median())
        return fn

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def get_decoder_stack(model, model_class: str):
    """Locate the list of transformer blocks. All 12 models here wrap it at
    model.model.layers (Llama-style causal decoders, EuroBertForMaskedLM, and
    ModernBertForMaskedLM all use this attribute path -- verified per-model below,
    not assumed)."""
    return model.model.layers


def down_proj_module(layer, ffn_kind: str):
    if ffn_kind == "llama":
        return layer.mlp.down_proj
    if ffn_kind == "packed_wi_wo":
        return layer.mlp.Wo
    raise ValueError(ffn_kind)


def gate_up_down_weights(layer, ffn_kind: str):
    if ffn_kind == "llama":
        return layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight, layer.mlp.down_proj.weight
    if ffn_kind == "packed_wi_wo":
        packed = layer.mlp.Wi.weight
        d_ffn = packed.shape[0] // 2
        return packed[:d_ffn], packed[d_ffn:], layer.mlp.Wo.weight
    raise ValueError(ffn_kind)


def verify_gated(layer, ffn_kind: str) -> str:
    """Per PREREG step 1-5: verify from actual named_modules(), not config/family name."""
    names = dict(layer.named_modules())
    if ffn_kind == "llama":
        for req in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            if req not in names:
                raise SystemExit(f"EXPECTED-GATED MODEL FAILED VERIFICATION: missing {req} "
                                  f"-- module tree: {list(names.keys())}")
        return "gate_proj/up_proj/down_proj (separate SwiGLU-style projections)"
    if ffn_kind == "packed_wi_wo":
        if "mlp.Wi" not in names or "mlp.Wo" not in names:
            raise SystemExit(f"EXPECTED-GATED MODEL FAILED VERIFICATION: missing mlp.Wi/mlp.Wo "
                              f"-- module tree: {list(names.keys())}")
        wi_out = names["mlp.Wi"].out_features
        wo_in = names["mlp.Wo"].in_features
        if wi_out != 2 * wo_in:
            raise SystemExit(f"EXPECTED-GATED MODEL FAILED VERIFICATION: packed Wi out_features "
                              f"{wi_out} != 2x Wo in_features {wo_in} -- not a gated packed FFN")
        return "packed Wi (chunked input/gate) -> elementwise product -> Wo (ModernBERT-style GeGLU)"
    raise ValueError(ffn_kind)


def load_model_and_tokenizer(spec: dict):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
    repo, rev = spec["repo"], spec["revision"]
    print(f"loading {repo}@{rev} ...")
    t0 = time.time()
    cfg = AutoConfig.from_pretrained(repo, revision=rev, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(repo, revision=rev, trust_remote_code=True)
    loader = AutoModelForCausalLM if spec["model_class"] == "causal" else AutoModelForMaskedLM
    dtype_used = "float32"
    try:
        model = loader.from_pretrained(repo, revision=rev, trust_remote_code=True,
                                        torch_dtype=torch.float32)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    except torch.cuda.OutOfMemoryError:
        print("  float32 OOM -- falling back to bfloat16 (per authorized dtype-fallback rule)")
        torch.cuda.empty_cache()
        dtype_used = "bfloat16"
        model = loader.from_pretrained(repo, revision=rev, trust_remote_code=True,
                                        torch_dtype=torch.bfloat16)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    print(f"  loaded in {time.time()-t0:.1f}s on {device}, dtype={dtype_used}")
    resolved_rev = getattr(model.config, "_commit_hash", None) or rev
    return model, tok, cfg, device, dtype_used, resolved_rev


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    embed = sum(p.numel() for n, p in model.named_parameters()
                if re.search(r"embed|lm_head|^cls\.|decoder\.weight|decoder\.bias", n))
    return total, embed, total - embed


def run_one(key: str):
    spec = PANEL[key]
    panel_idx = PANEL_ORDER.index(key)
    model, tok, cfg, device, dtype_used, resolved_rev = load_model_and_tokenizer(spec)

    n_layers = cfg.num_hidden_layers
    stack = get_decoder_stack(model, spec["model_class"])
    assert len(stack) == n_layers, f"stack length {len(stack)} != config layers {n_layers}"

    gate_class = verify_gated(stack[0], spec["ffn_kind"])
    print(f"  gated-FFN verification (layer 0): {gate_class}")

    if spec["input_kind"] == "text":
        text = build_wikitext_input(tok)
        inputs = tok(text, return_tensors="pt").to(device)
    else:
        bos = tok.bos_token or ""
        seq = bos + _ACTB_500
        inputs = tok(seq, return_tensors="pt", add_special_tokens=False).to(device)
    n_tokens = inputs["input_ids"].shape[1]
    print(f"  input token count: {n_tokens}")

    modules = {i: down_proj_module(stack[i], spec["ffn_kind"]) for i in range(n_layers)}
    rec = DownProjRecorder()
    rec.register(modules)
    with torch.no_grad():
        model(**inputs)
    rec.remove()

    per_layer = []
    for i in range(n_layers):
        med = rec.median_channel_max[i]
        ratio = rec.out_max[i] / med if med > 0 else float("inf")
        per_layer.append(dict(layer=i, out_max=rec.out_max[i], out_channel=rec.out_channel[i],
                               median_channel_max=med, ratio=ratio))
    best = max(per_layer, key=lambda r: r["out_max"])
    passes = best["ratio"] >= RATIO_THRESHOLD
    print(f"  candidate layer={best['layer']} row={best['out_channel']} "
          f"out_max={best['out_max']:.4g} ratio={best['ratio']:.2f} "
          f"{'ACCEPTED' if passes else 'NULL (below threshold)'}")

    d_model = cfg.hidden_size
    d_ffn = cfg.intermediate_size
    total_params, embed_params, non_embed_params = count_params(model)
    print(f"  total_params={total_params} embed_params={embed_params} non_embed={non_embed_params}")

    result = dict(
        key=key, model=spec["repo"], repo=spec["repo"], requested_revision=spec["revision"],
        resolved_revision=resolved_rev, family=spec["family"], domain=spec["domain"],
        architecture="decoder" if spec["model_class"] == "causal" else "encoder",
        panel_index=panel_idx, dtype=dtype_used,
        total_params=total_params, embed_params=embed_params, non_embed_params=non_embed_params,
        d_model=d_model, d_ffn=d_ffn, n_layers=n_layers,
        gated=True, gate_evidence=gate_class,
        ratio_threshold=RATIO_THRESHOLD, n_input_tokens=n_tokens,
        candidate=None, null_outcome=(not passes),
        null_reason=None if passes else f"global-max ratio {best['ratio']:.2f} < {RATIO_THRESHOLD}",
    )

    if passes:
        layer, row = best["layer"], best["out_channel"]
        Wg, Wu, Wd = gate_up_down_weights(stack[layer], spec["ffn_kind"])
        Wg, Wu, Wd = Wg.detach().float().cpu(), Wu.detach().float().cpu(), Wd.detach().float().cpu()

        metrics, sigmas = row_spectral_metrics(Wg, Wu, Wd[row], device=device)
        print(f"  candidate q1={metrics.q1:.6f} pr_spec={metrics.pr_spec:.4f} "
              f"frob_norm={metrics.frob_norm:.6g}")

        seed_seq = np.random.SeedSequence(42).spawn(N_PANEL)[panel_idx]
        rng = np.random.default_rng(seed_seq)
        pool = [r for r in range(d_model) if r != row]
        control_rows = sorted(rng.choice(pool, size=N_CONTROL_ROWS, replace=False).tolist())
        print(f"  control rows (seeded, layer {layer}): {control_rows}")

        control_results = []
        for crow in control_rows:
            cmetrics, _ = row_spectral_metrics(Wg, Wu, Wd[crow], device=device)
            control_results.append(dict(row=crow, q1=cmetrics.q1, pr_spec=cmetrics.pr_spec,
                                         frob_norm=cmetrics.frob_norm))
            print(f"    control row {crow}: q1={cmetrics.q1:.6f} pr_spec={cmetrics.pr_spec:.4f} "
                  f"frob_norm={cmetrics.frob_norm:.6g}")

        control_frobs = [c["frob_norm"] for c in control_results]
        layer_median_frob = float(np.median(control_frobs))
        candidate_frob_ratio = metrics.frob_norm / layer_median_frob if layer_median_frob > 0 else float("inf")

        result["candidate"] = dict(
            layer=layer, row=row, out_max=best["out_max"], detection_ratio=best["ratio"],
            relative_depth=layer / n_layers,
            q1=metrics.q1, pr_spec=metrics.pr_spec, stable_rank=metrics.stable_rank,
            frob_norm=metrics.frob_norm, n_singular_values=metrics.n_singular_values,
        )
        result["control_rows"] = control_results
        result["layer_median_frob_norm"] = layer_median_frob
        result["candidate_frob_norm_ratio_to_layer_median"] = candidate_frob_ratio

    out_path = ROOT / "results" / "experiments" / "E11" / "raw" / f"{key}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=PANEL_ORDER)
    args = p.parse_args()

    print("Verifying E11 prereg lock before any forward pass ...")
    verify_lock()

    run_one(args.model)


if __name__ == "__main__":
    main()
