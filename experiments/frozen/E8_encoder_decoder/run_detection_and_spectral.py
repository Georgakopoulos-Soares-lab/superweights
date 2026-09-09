"""
experiments/E8_encoder_decoder/run_detection_and_spectral.py

E8 Phase 2+3: frozen prospective detection (PREREG_encoder_decoder_dimensionality.md) plus,
for any qualifying candidate, the identical E7 spectral computation and a seeded set of
same-layer control rows. One model per invocation.

Usage:
  python run_detection_and_spectral.py --model mosaicbert
  python run_detection_and_spectral.py --model modernbert
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

SALVAGE = Path(__file__).resolve().parents[2]
ROOT = SALVAGE.parent
sys.path.insert(0, str(SALVAGE / "src"))
sys.path.insert(0, str(SALVAGE / "experiments" / "E7_exact_dimensionality"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from spectral_lib import row_spectral_metrics  # noqa: E402  (E7, reused unmodified)

PREREG = SALVAGE / "docs" / "prereg" / "PREREG_encoder_decoder_dimensionality.md"
RATIO_THRESHOLD = 5.0
N_CONTROL_ROWS = 5

PANEL_INDEX = {"mosaicbert": 0, "modernbert": 1}


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


def adapter_mosaicbert(model, layer: int):
    mlp = model.bert.encoder.layer[layer].mlp
    packed = mlp.gated_layers.weight
    d_ffn = packed.shape[0] // 2
    return packed[:d_ffn], packed[d_ffn:], mlp.wo.weight


def adapter_modernbert(model, layer: int):
    mlp = model.model.layers[layer].mlp
    packed = mlp.Wi.weight
    d_ffn = packed.shape[0] // 2
    return packed[:d_ffn], packed[d_ffn:], mlp.Wo.weight


def run_mosaicbert():
    from transformers import AutoModelForMaskedLM, AutoTokenizer, BertTokenizer
    repo = "mosaicml/mosaic-bert-base"
    print(f"loading {repo} ...")
    t0 = time.time()
    # MosaicBERT's own README: "the tokenizer for this model is simply the Hugging Face
    # bert-base-uncased tokenizer" -- the repo ships no tokenizer files of its own (verified
    # via the HF API file listing: only bert_layers.py/config.json/pytorch_model.bin/etc).
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained(repo, trust_remote_code=True, torch_dtype=torch.float32)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"  loaded in {time.time()-t0:.1f}s")
    resolved_rev = getattr(model.config, "_commit_hash", None)

    n_layers = model.config.num_hidden_layers
    text = build_wikitext_input(tok)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    print(f"  input token count: {inputs['input_ids'].shape[1]}")

    modules = {i: model.bert.encoder.layer[i].mlp.wo for i in range(n_layers)}
    rec = DownProjRecorder()
    rec.register(modules)
    with torch.no_grad():
        model(**inputs)
    rec.remove()

    return finalize("MosaicBERT", repo, resolved_rev, rec, n_layers, model, adapter_mosaicbert, device)


def run_modernbert():
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    repo = "answerdotai/ModernBERT-base"
    print(f"loading {repo} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForMaskedLM.from_pretrained(repo, torch_dtype=torch.float32)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"  loaded in {time.time()-t0:.1f}s")
    resolved_rev = getattr(model.config, "_commit_hash", None)

    n_layers = model.config.num_hidden_layers
    text = build_wikitext_input(tok)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    print(f"  input token count: {inputs['input_ids'].shape[1]}")

    modules = {i: model.model.layers[i].mlp.Wo for i in range(n_layers)}
    rec = DownProjRecorder()
    rec.register(modules)
    with torch.no_grad():
        model(**inputs)
    rec.remove()

    return finalize("ModernBERT", repo, resolved_rev, rec, n_layers, model, adapter_modernbert, device)


def finalize(name, repo, resolved_rev, rec: DownProjRecorder, n_layers, model, adapter_fn, device):
    per_layer = []
    for i in range(n_layers):
        if i not in rec.out_max:
            continue
        med = rec.median_channel_max[i]
        ratio = rec.out_max[i] / med if med > 0 else float("inf")
        per_layer.append(dict(layer=i, out_max=rec.out_max[i], out_channel=rec.out_channel[i],
                               median_channel_max=med, ratio=ratio))

    best = max(per_layer, key=lambda r: r["out_max"])
    passes = best["ratio"] >= RATIO_THRESHOLD

    result = dict(
        model=name, repo=repo, resolved_revision=resolved_rev, ratio_threshold=RATIO_THRESHOLD,
        candidate=dict(layer=best["layer"], row=best["out_channel"], out_max=best["out_max"],
                       ratio=best["ratio"]) if passes else None,
        null_outcome=(not passes),
        null_reason=None if passes else f"global-max ratio {best['ratio']:.2f} < {RATIO_THRESHOLD}",
        per_layer=per_layer,
    )
    print(f"\n{name}: candidate layer={best['layer']} row={best['out_channel']} "
          f"out_max={best['out_max']:.4g} ratio={best['ratio']:.2f}  "
          f"{'ACCEPTED' if passes else 'NULL (below threshold)'}")

    if passes:
        layer, row = best["layer"], best["out_channel"]
        Wg, Wu, Wd = adapter_fn(model, layer)
        Wg, Wu, Wd = Wg.detach().cpu(), Wu.detach().cpu(), Wd.detach().cpu()
        d_ffn, d_model = Wg.shape
        print(f"  spectral: gate{tuple(Wg.shape)} up{tuple(Wu.shape)} down{tuple(Wd.shape)}")

        metrics, sigmas = row_spectral_metrics(Wg, Wu, Wd[row], device=device)
        print(f"  candidate q1={metrics.q1:.6f} PR_spec={metrics.pr_spec:.4f} "
              f"frob={metrics.frob_norm:.6g}")
        result["spectral"] = dict(q1=metrics.q1, pr_spec=metrics.pr_spec,
                                   stable_rank=metrics.stable_rank, frob_norm=metrics.frob_norm,
                                   n_singular_values=metrics.n_singular_values)

        seed_seq = np.random.SeedSequence(42).spawn(2)[PANEL_INDEX[name.lower()]]
        rng = np.random.default_rng(seed_seq)
        candidates_pool = [r for r in range(d_model) if r != row]
        control_rows = sorted(rng.choice(candidates_pool, size=N_CONTROL_ROWS, replace=False).tolist())
        print(f"  control rows (seeded, layer {layer}): {control_rows}")

        control_results = []
        for crow in control_rows:
            cmetrics, _ = row_spectral_metrics(Wg, Wu, Wd[crow], device=device)
            control_results.append(dict(row=crow, q1=cmetrics.q1, pr_spec=cmetrics.pr_spec))
            print(f"    control row {crow}: q1={cmetrics.q1:.6f} PR_spec={cmetrics.pr_spec:.4f}")
        result["control_rows"] = control_results

    out_path = ROOT / "results" / f"e8_detection_{name.lower()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["mosaicbert", "modernbert"])
    args = p.parse_args()

    print("Verifying E8 prereg lock before any forward pass ...")
    verify_lock()

    if args.model == "mosaicbert":
        run_mosaicbert()
    else:
        run_modernbert()


if __name__ == "__main__":
    main()
