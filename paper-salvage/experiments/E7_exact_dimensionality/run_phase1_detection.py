"""
experiments/E7_exact_dimensionality/run_phase1_detection.py

Phase 1: frozen prospective candidate-selection protocol (PREREG_exact_operator_dimensionality.md).
Runs on Qwen2.5-7B (NLP) and GenomeOcean-4B (genomic) -- standard transformers architectures,
no container needed. Evo 2 runs separately (run_phase1_detection_evo2.py, inside evo2.sif).

Verifies the E7 prereg lock before any forward pass. One forward pass per model, hooking
every eligible down_proj-equivalent module simultaneously, exactly the statistic/threshold/
tie-break/null rule the prereg fixes. This is NOT the spectral metric -- it only locates the
candidate row. No U_k, singular value, q1, or PR_spec is computed here.

Usage:
  python run_phase1_detection.py --model qwen25
  python run_phase1_detection.py --model genomeocean
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

SALVAGE = Path(__file__).resolve().parents[2]
ROOT = SALVAGE.parent
PREREG = SALVAGE / "docs" / "prereg" / "PREREG_exact_operator_dimensionality.md"

RATIO_THRESHOLD = 5.0

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
assert len(_ACTB_CDS) == 1128, f"ACTB_CDS length {len(_ACTB_CDS)} != 1128"
_ACTB_500 = _ACTB_CDS[:504]     # matches probes/dna_probes.py's actb_500 exactly (_ACTB_CDS[:504])
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
        self._handles = []

    def register(self, modules: dict):
        for layer_idx, module in modules.items():
            self._handles.append(module.register_forward_hook(self._hook(layer_idx)))

    def _hook(self, layer_idx):
        def fn(module, args, output):
            y = output[0] if isinstance(output, tuple) else output
            flat = y.reshape(-1, y.shape[-1]).abs().float()
            channel_max = flat.max(dim=0).values          # [d_model]
            self.out_max[layer_idx] = float(channel_max.max())
            self.out_channel[layer_idx] = int(channel_max.argmax())
            self._layer_channel_max = getattr(self, "_layer_channel_max", {})
            self._layer_channel_max[layer_idx] = channel_max.median().item()
        return fn

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def run_qwen25():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo = "Qwen/Qwen2.5-7B"
    print(f"loading {repo} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32, device_map="auto")
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")
    resolved_rev = model.config._commit_hash if hasattr(model.config, "_commit_hash") else None

    text = build_wikitext_input(tok)
    print(f"  input (first 200 chars): {text[:200]!r}")
    inputs = tok(text, return_tensors="pt").to(model.device)

    modules = {i: model.model.layers[i].mlp.down_proj for i in range(model.config.num_hidden_layers)}
    rec = DownProjRecorder()
    rec.register(modules)
    with torch.no_grad():
        model(**inputs)
    rec.remove()

    return finalize("Qwen2.5-7B", "nlp", repo, resolved_rev, rec, model.config.num_hidden_layers)


def run_genomeocean():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo = "DOEJGI/GenomeOcean-4B"
    print(f"loading {repo} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32, device_map="auto")
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")
    resolved_rev = model.config._commit_hash if hasattr(model.config, "_commit_hash") else None

    bos = tok.bos_token or ""
    seq = bos + _ACTB_500
    inputs = tok(seq, return_tensors="pt", add_special_tokens=False).to(model.device)
    print(f"  input token count: {inputs['input_ids'].shape[1]}")

    modules = {i: model.model.layers[i].mlp.down_proj for i in range(model.config.num_hidden_layers)}
    rec = DownProjRecorder()
    rec.register(modules)
    with torch.no_grad():
        model(**inputs)
    rec.remove()

    return finalize("GenomeOcean-4B", "genomic", repo, resolved_rev, rec, model.config.num_hidden_layers)


def finalize(name, group, repo, resolved_rev, rec: DownProjRecorder, n_layers):
    per_layer = []
    for i in range(n_layers):
        if i not in rec.out_max:
            continue
        ratio = rec.out_max[i] / rec._layer_channel_max[i] if rec._layer_channel_max[i] > 0 else float("inf")
        per_layer.append(dict(layer=i, out_max=rec.out_max[i], out_channel=rec.out_channel[i],
                               median_channel_max=rec._layer_channel_max[i], ratio=ratio))

    best = max(per_layer, key=lambda r: r["out_max"])
    passes = best["ratio"] >= RATIO_THRESHOLD

    result = dict(
        model=name, group=group, repo=repo, resolved_revision=resolved_rev,
        ratio_threshold=RATIO_THRESHOLD,
        candidate=dict(layer=best["layer"], row=best["out_channel"], out_max=best["out_max"],
                       ratio=best["ratio"]) if passes else None,
        null_outcome=(not passes),
        null_reason=None if passes else (
            f"global-max ratio {best['ratio']:.2f} < threshold {RATIO_THRESHOLD}"),
        per_layer=per_layer,
    )
    print(f"\n{name}: candidate layer={best['layer']} row={best['out_channel']} "
          f"out_max={best['out_max']:.4g} ratio={best['ratio']:.2f}  "
          f"{'ACCEPTED' if passes else 'NULL (below threshold)'}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["qwen25", "genomeocean"])
    args = p.parse_args()

    print("Verifying E7 prereg lock before any forward pass ...")
    verify_lock()

    if args.model == "qwen25":
        result = run_qwen25()
    else:
        result = run_genomeocean()

    out_path = ROOT / "results" / f"e7_phase1_detection_{args.model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
