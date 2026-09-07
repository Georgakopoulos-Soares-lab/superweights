#!/usr/bin/env python3
"""One-model, restartable row-wise application of E11's activation detector.

The E11 detector's statistic is channel_max[k] / median(channel_max) within a
layer.  E11 retained only the global winner; this runner retains the complete
channel_max vector and applies the *unchanged* >=5 threshold to every row.
No causal endpoint is imported or evaluated.
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SALVAGE, ROOT = HERE.parents[1], HERE.parents[1].parent
sys.path[:0] = [str(HERE), str(SALVAGE / "experiments" / "E10_nlp_architecture_causal"),
                str(SALVAGE / "experiments" / "E7_exact_dimensionality"),
                str(SALVAGE / "experiments" / "E11_scale_ladder")]
import e10_lib as L
from dnabert2_compat import load_dnabert2_patched, load_mosaicbert_patched
from genomic_encoder_lib import load_ntv3_pretrained
from genomic_decoder_lib import load_genomeocean
from run_singleton_census import ALIASES, load_generic, load_structural_row, model_pattern
from run_model import _ACTB_500
from spectral_lib import row_spectral_metrics

THRESHOLD = 5.0
OUT = ROOT / "results" / "E13_multicandidate_structural"
FROZEN_WIKITEXT = Path(os.environ.get(
    "E13_FROZEN_WIKITEXT_TEST",
    "/scratch/11034/atzanakak/genomic-super-weights/frozen_inputs/wikitext_repo/wikitext-2-raw-v1/test-00000-of-00001.parquet",
))


def frozen_wikitext_input(tokenizer):
    """Reproduce E11's first-20-nonempty-lines input without datasets/HF resolution."""
    import pyarrow.parquet as pq
    table = pq.read_table(FROZEN_WIKITEXT, columns=["text"])
    lines = [str(x).strip() for x in table.column("text").to_pylist() if str(x).strip()]
    text = "\n".join(lines[:20])
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    return tokenizer.decode(ids["input_ids"][0], skip_special_tokens=True)


def ntv3_probe(tokenizer):
    """E7/E9 convention: right-pad the 504-bp ACTB probe with A to 256-token multiple."""
    seq = _ACTB_500
    while len(tokenizer(seq, add_special_tokens=False)["input_ids"]) % 256:
        seq += "A"
    return seq


class Recorder:
    def __init__(self): self.values, self.handles = {}, []
    def add(self, layer, module): self.handles.append(module.register_forward_hook(self.hook(layer)))
    def hook(self, layer):
        def f(_m, _a, output):
            y = output[0] if isinstance(output, tuple) else output
            self.values[layer] = y.reshape(-1, y.shape[-1]).abs().float().max(dim=0).values.cpu()
        return f
    def close(self):
        for h in self.handles: h.remove()


def load(name, structural):
    if name == "DNABERT-2":
        model, tok, cfg, _ = load_dnabert2_patched(local_files_only=True); return model, tok, cfg["down_proj_pattern"]
    if name == "MosaicBERT":
        rev = None if structural["revision"].startswith("unpinned") else structural["revision"]
        model, tok, _ = load_mosaicbert_patched(structural["repo"], revision=rev, local_files_only=True); return model, tok, model_pattern(name)
    if name == "NTv3":
        model, tok = load_ntv3_pretrained(local_files_only=True); return model, tok, model_pattern(name)
    if name == "GenomeOcean-4B":
        model, tok = load_genomeocean(local_files_only=True); return model, tok, model_pattern(name)
    model, tok, _ = load_generic({"model": name, "repo": structural["repo"], "revision": structural["revision"]}, structural["architecture"])
    return model, tok, model_pattern(name)


def weights(model, name, layer):
    if name in {"DNABERT-2", "MosaicBERT"}:
        m = model.bert.encoder.layer[layer].mlp; x = m.gated_layers.weight; n = x.shape[0] // 2; return x[:n], x[n:], m.wo.weight
    if name.startswith("ModernBERT") or name == "answerdotai/ModernBERT-large":
        m = model.model.layers[layer].mlp; x = m.Wi.weight; n = x.shape[0] // 2; return x[:n], x[n:], m.Wo.weight
    if name == "NTv3":
        m = model.core.transformer_blocks[layer]; x = m.fc1.weight; n = x.shape[0] // 2; return x[:n], x[n:], m.fc2.weight
    m = model.model.layers[layer].mlp; return m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=sorted(k for k in ALIASES if k != "phi3")); args = ap.parse_args()
    name = ALIASES[args.model]; structural = load_structural_row(name); out = OUT / "raw" / f"{args.model}.json"
    if out.exists(): print(f"SKIP {out}"); return
    model, tok, pattern = load(name, structural)
    rec = Recorder(); n_layers = int(structural["n_layers"])
    for layer in range(n_layers): rec.add(layer, L._resolve_module(model, pattern, layer))
    if structural["domain"] == "text":
        text = frozen_wikitext_input(tok); inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    else:
        probe = ntv3_probe(tok) if name == "NTv3" else _ACTB_500
        # The established NTv3 detector passes the padded nucleotide probe directly;
        # adding a literal BOS token after padding would change 512 -> 513 and break
        # its stride-256 U-Net alignment.
        sequence = probe if name == "NTv3" else (tok.bos_token or "") + probe
        inputs = tok(sequence, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad(): model(**inputs)
    rec.close(); rows, selected = [], []
    for layer, values in sorted(rec.values.items()):
        median = float(values.median())
        for row, value in enumerate(values.tolist()):
            score = value / median if median > 0 else float("inf")
            item = {"layer": layer, "row": row, "activation_max": value, "activation_median": median, "detector_score": score, "selected": score >= THRESHOLD}
            rows.append(item)
            if item["selected"]:
                Wg, Wu, Wd = weights(model, name, layer)
                metrics, _ = row_spectral_metrics(Wg.detach().cpu(), Wu.detach().cpu(), Wd[row].detach().cpu())
                item.update(q1=metrics.q1, pr_spec=metrics.pr_spec, frob_norm=metrics.frob_norm, stable_rank=metrics.stable_rank)
                selected.append(item)
    payload = {"model": name, "slug": args.model, "threshold": THRESHOLD, "input_tokens": int(inputs["input_ids"].shape[1]), "rows": rows, "selected": selected, "elapsed_seconds": time.time()}
    out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps(payload)); os.replace(tmp, out)
    print(f"WROTE {out} K={len(selected)}")

if __name__ == "__main__": main()
