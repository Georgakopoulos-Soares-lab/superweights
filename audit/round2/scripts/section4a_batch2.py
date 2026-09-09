#!/usr/bin/env python
"""Round-2 Section 4a batch 2 + Section 3 calibration check for the 3 text decoders.

Order (cheapest-loading first, per author instruction): ModernBERT-base, DNABERT-2,
GENERator-EUK-3B, Llama-7B, Mistral-7B, OLMo-7B-0724-hf. Reports running cohort summary
after EACH model, not just at the end.

For every model: top-5-by-norm + q1 comparison (4a), same method as batch 1.
For Llama-7B, Mistral-7B, OLMo-7B additionally: WikiText-2-probe ratio/activation
detector check, framed as CALIBRATION (these coordinates are literature-derived from
Yu et al., not detector-selected -- the question is only whether the current detector
independently recovers them, not a protocol-inconsistency finding).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
E5_DIR = ROOT / "experiments/frozen/E5_dimensionality"
E7_DIR = ROOT / "experiments/frozen/E7_exact_dimensionality"
E11_DIR = ROOT / "experiments/frozen/E11_scale_ladder"
E13_DIR = ROOT / "experiments/frozen/E13_full_cohort_causal_census"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(E5_DIR))
sys.path.insert(0, str(E7_DIR))
sys.path.insert(0, str(E11_DIR))
sys.path.insert(0, str(E13_DIR))

from dimensionality_lib import exact_uk_all_rows  # noqa: E402
from spectral_lib import row_spectral_metrics  # noqa: E402
from dnabert2_compat import load_dnabert2_patched  # noqa: E402
import run_model as rm  # noqa: E402

RATIO_THRESHOLD = 5.0
OUT_Q1 = ROOT / "audit/round2/topk_norm_q1_summary.csv"
OUT_MAIN = ROOT / "audit/round2/topk_norm_controls.csv"
OUT_DECODER_CAL = ROOT / "audit/round2/section3_text_decoder_calibration.json"

MODELS = [
    dict(key="ModernBERT-base", repo="answerdotai/ModernBERT-base",
         revision="8949b909ec900327062f0ebf497f51aef5e6f0c8",
         layer=15, row=251, ffn_kind="packed_wi_wo", loader="hf_mlm",
         pattern="model.layers.{i}.mlp.Wo"),
    dict(key="DNABERT-2", repo="zhihan1996/DNABERT-2-117M",
         revision="7bce263b15377fc15361f52cfab88f8b586abda0",
         layer=5, row=603, ffn_kind="gated_layers_bert", loader="dnabert2",
         pattern="bert.encoder.layer.{i}.mlp.wo"),
    dict(key="GENERator-EUK-3B", repo="GenerTeam/GENERator-v2-eukaryote-3b-base",
         revision="7dc01bccce5b65e15141170538afdc2ff09d8dde",
         layer=4, row=2371, ffn_kind="separate", loader="hf_causal",
         pattern="model.layers.{i}.mlp.down_proj"),
    dict(key="Llama-7B", repo="huggyllama/llama-7b",
         revision="4782ad278652c7c71b72204d462d6d01eaaf7549",
         layer=2, row=3968, ffn_kind="separate", loader="hf_causal",
         pattern="model.layers.{i}.mlp.down_proj", calibration=True),
    dict(key="Mistral-7B", repo="mistralai/Mistral-7B-v0.1",
         revision="27d67f1b5f57dc0953326b2601d68371d40ea8da",
         layer=1, row=2070, ffn_kind="separate", loader="hf_causal",
         pattern="model.layers.{i}.mlp.down_proj", calibration=True),
    dict(key="OLMo-7B-0724-hf", repo="allenai/OLMo-7B-0724-hf",
         revision="1ee306df318ee15bfe4a76ebd5c002b0105b1ab6",
         layer=1, row=269, ffn_kind="separate", loader="hf_causal",
         pattern="model.layers.{i}.mlp.down_proj", calibration=True),
    dict(key="MosaicBERT", repo="mosaicml/mosaic-bert-base",
         revision="c89bbadc24278928f22bcdd7de6b61a5a2d08553",
         layer=9, row=287, ffn_kind="gated_layers_bert", loader="mosaicbert",
         pattern="bert.encoder.layer.{i}.mlp.wo"),
    dict(key="GenomeOcean-4B", repo="DOEJGI/GenomeOcean-4B",
         revision="2bed2fc3ed47c5f6955ba3e64563512c9b338dfb",
         layer=1, row=2604, ffn_kind="separate", loader="hf_causal",
         pattern="model.layers.{i}.mlp.down_proj"),
    dict(key="Qwen2.5-7B", repo="Qwen/Qwen2.5-7B",
         revision="d149729398750b98c0af14eb82c78cfe92750796",
         layer=26, row=458, ffn_kind="separate", loader="hf_causal",
         pattern="model.layers.{i}.mlp.down_proj"),
    dict(key="NTv3", repo="InstaDeepAI/NTv3_650M_pre",
         revision="unpinned (E5/E6 original)",
         layer=11, row=1472, ffn_kind="packed_fc1_fc2", loader="ntv3",
         pattern="core.transformer_blocks.{i}.fc2"),
]


def gate_up_for(mods, pattern, layer, ffn_kind):
    if ffn_kind == "separate":
        base = pattern.format(i=layer)
        gate = mods[base.replace("down_proj", "gate_proj")].weight.detach().cpu()
        up = mods[base.replace("down_proj", "up_proj")].weight.detach().cpu()
        return gate, up
    if ffn_kind == "packed_wi_wo":
        base = pattern.format(i=layer)
        packed = mods[base.replace("Wo", "Wi")].weight.detach().cpu()
        d_ffn = packed.shape[0] // 2
        return packed[:d_ffn], packed[d_ffn:]
    if ffn_kind == "gated_layers_bert":
        base = pattern.format(i=layer)
        packed = mods[base.replace(".wo", ".gated_layers")].weight.detach().cpu()
        d_ffn = packed.shape[0] // 2
        return packed[:d_ffn], packed[d_ffn:]
    if ffn_kind == "packed_fc1_fc2":
        fc1_key = pattern.format(i=layer).replace("fc2", "fc1")
        packed = mods[fc1_key].weight.detach().cpu()
        d_ffn = packed.shape[0] // 2
        return packed[:d_ffn], packed[d_ffn:]
    raise ValueError(ffn_kind)


_NTV3_CODE_REV = "0ecff3637f0d3ba5b686d1095083218157c2ca34"


def load_model(spec):
    if spec["loader"] == "dnabert2":
        model, tok, cfg, patched = load_dnabert2_patched(device="cuda")
        return model.eval(), tok
    if spec["loader"] == "mosaicbert":
        from dnabert2_compat import load_mosaicbert_patched
        model, tok, resolved = load_mosaicbert_patched(spec["repo"], revision=spec["revision"],
                                                          device="cuda")
        return model.eval(), tok
    if spec["loader"] == "ntv3":
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec["repo"], trust_remote_code=True,
                                             code_revision=_NTV3_CODE_REV)
        model = AutoModelForMaskedLM.from_pretrained(spec["repo"], trust_remote_code=True,
                                                       code_revision=_NTV3_CODE_REV,
                                                       torch_dtype=torch.float32)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        return model, tok
    if spec["loader"] == "hf_mlm":
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"],
                                             trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(spec["repo"], revision=spec["revision"],
                                                       trust_remote_code=True, torch_dtype=torch.float32)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        return model, tok
    if spec["loader"] == "hf_causal":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"],
                                             trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(spec["repo"], revision=spec["revision"],
                                                       trust_remote_code=True, torch_dtype=torch.float32)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        return model, tok
    raise ValueError(spec["loader"])


def append_csv(path, fields, rows):
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def already_done(key):
    if not OUT_Q1.exists():
        return False
    for r in csv.DictReader(open(OUT_Q1)):
        if r["model"] == key:
            return True
    return False


class Recorder:
    def __init__(self):
        self.values = {}
        self.handles = []

    def add(self, layer, module):
        self.handles.append(module.register_forward_hook(self._hook(layer)))

    def _hook(self, layer):
        def fn(module, args, output):
            y = output[0] if isinstance(output, tuple) else output
            self.values[layer] = y.reshape(-1, y.shape[-1]).abs().max(dim=0).values.detach().cpu()
        return fn

    def close(self):
        for h in self.handles:
            h.remove()


def build_wikitext_probe(tok, max_tokens=512):
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
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    return tok.decode(ids["input_ids"][0], skip_special_tokens=True)


def run_decoder_calibration(model, tok, pattern, n_layers, frozen, dev):
    seq = build_wikitext_probe(tok)
    enc = tok(seq, return_tensors="pt").to(dev)
    rec = Recorder()
    mods = dict(model.named_modules())
    for layer in range(n_layers):
        rec.add(layer, mods[pattern.format(i=layer)])
    with torch.no_grad():
        model(**enc)
    rec.close()
    rows = []
    for layer, values in sorted(rec.values.items()):
        median = float(values.median())
        for row, activation_max in enumerate(values.tolist()):
            rows.append(dict(layer=layer, row=row, activation_max=activation_max,
                              detector_ratio=activation_max / median if median > 0 else float("inf")))
    by_ratio = sorted(rows, key=lambda x: (-x["detector_ratio"], x["layer"], x["row"]))
    by_act = sorted(rows, key=lambda x: (-x["activation_max"], x["layer"], x["row"]))
    for rank, r in enumerate(by_ratio, 1):
        r["ratio_rank"] = rank
    for rank, r in enumerate(by_act, 1):
        r["activation_rank"] = rank
    lookup = {(r["layer"], r["row"]): r for r in rows}
    frozen_entry = lookup[frozen]
    return dict(
        n_tokens=int(enc["input_ids"].shape[1]),
        global_ratio_argmax=(by_ratio[0]["layer"], by_ratio[0]["row"]),
        global_ratio_value=by_ratio[0]["detector_ratio"],
        global_activation_argmax=(by_act[0]["layer"], by_act[0]["row"]),
        global_activation_value=by_act[0]["activation_max"],
        frozen_activation=frozen_entry["activation_max"],
        frozen_ratio=frozen_entry["detector_ratio"],
        frozen_activation_rank=frozen_entry["activation_rank"],
        frozen_ratio_rank=frozen_entry["ratio_rank"],
        frozen_passes_ratio5=frozen_entry["detector_ratio"] >= RATIO_THRESHOLD,
        detector_independently_recovers_yu_et_al=(frozen_entry["ratio_rank"] == 1),
    )


def print_running_cohort_summary():
    if not OUT_Q1.exists():
        return
    from collections import defaultdict
    rows = list(csv.DictReader(open(OUT_Q1)))
    by_model = defaultdict(dict)
    for r in rows:
        by_model[r["model"]][r["coord_role"]] = float(r["q1"])
    gaps, ranks = [], []
    for model, d in by_model.items():
        cand = d["candidate"]
        topk = [v for k, v in d.items() if k != "candidate"]
        gap = cand - max(topk)
        rank = sorted([cand] + topk, reverse=True).index(cand) + 1
        gaps.append(gap)
        ranks.append(rank)
    print(f"\n--- RUNNING COHORT SUMMARY: n={len(gaps)} models ---")
    print(f"  candidate rank1/6: {sum(1 for r in ranks if r==1)}/{len(ranks)}   "
          f"rank<=2/6: {sum(1 for r in ranks if r<=2)}/{len(ranks)}")
    print(f"  median gap: {np.median(gaps):.4f}   mean gap: {np.mean(gaps):.4f}   "
          f"range: [{min(gaps):.4f}, {max(gaps):.4f}]")
    print("---\n")


def main():
    decoder_cal = {}
    if OUT_DECODER_CAL.exists():
        decoder_cal = json.loads(OUT_DECODER_CAL.read_text())

    for spec in MODELS:
        key = spec["key"]
        if already_done(key) and (not spec.get("calibration") or key in decoder_cal):
            print(f"skip {key} (checkpointed)")
            continue

        t0 = time.time()
        model, tok = load_model(spec)
        dev = next(model.parameters()).device
        try:
            n_layers = model.config.num_hidden_layers
        except AttributeError:
            n_layers = len(model.core.transformer_blocks)  # NTv3's custom config
        print(f"loaded {key} in {time.time()-t0:.1f}s, n_layers={n_layers}")

        mods = dict(model.named_modules())
        layer, cand_row = spec["layer"], spec["row"]

        if not already_done(key):
            gate, up = gate_up_for(mods, spec["pattern"], layer, spec["ffn_kind"])
            down_pattern_base = spec["pattern"].format(i=layer)
            down_mod = mods[down_pattern_base]
            norms = exact_uk_all_rows(gate, up, down_mod.weight.detach().cpu(), device="cpu").numpy()
            d_model = norms.shape[0]
            cand_norm = float(norms[cand_row])
            layer_median = float(np.median(norms))
            order = np.argsort(-norms)
            top_rows = [int(r) for r in order if r != cand_row][:5]
            top_norms = [float(norms[r]) for r in top_rows]

            row_main = dict(model=key, layer=layer, candidate_row=cand_row,
                             candidate_norm=cand_norm, layer_median_norm=layer_median,
                             **{f"rank{i+2}_row": top_rows[i] for i in range(5)},
                             **{f"rank{i+2}_norm": top_norms[i] for i in range(5)},
                             ratio_rank2_to_candidate=top_norms[0] / cand_norm if cand_norm else float("nan"),
                             existing_random_control_rows="", existing_control_norm_percentiles="")
            append_csv(OUT_MAIN, list(row_main.keys()), [row_main])

            q1_rows = []
            gate64, up64 = gate.double(), up.double()
            for role, row in [("candidate", cand_row)] + [(f"rank{i+2}_by_norm", r) for i, r in enumerate(top_rows)]:
                d_row = down_mod.weight[row].detach().cpu().double()
                metrics, _ = row_spectral_metrics(gate64, up64, d_row, device="cpu")
                q1_rows.append(dict(model=key, coord_role=role, layer=layer, row=row,
                                     norm=metrics.frob_norm, q1=metrics.q1, pr_spec=metrics.pr_spec))
            append_csv(OUT_Q1, ["model", "coord_role", "layer", "row", "norm", "q1", "pr_spec"], q1_rows)

            cand_q1 = q1_rows[0]["q1"]
            topk_q1s = [r["q1"] for r in q1_rows[1:]]
            print(f"{key}: candidate q1={cand_q1:.4f} (norm={cand_norm:.2f})  "
                  f"top5-by-norm q1s={[f'{q:.4f}' for q in topk_q1s]}  "
                  f"candidate_q1_rank_among_6={sorted([cand_q1]+topk_q1s, reverse=True).index(cand_q1)+1}")

        if spec.get("calibration") and key not in decoder_cal:
            cal = run_decoder_calibration(model, tok, spec["pattern"], n_layers, (layer, cand_row), dev)
            decoder_cal[key] = cal
            OUT_DECODER_CAL.write_text(json.dumps(decoder_cal, indent=2, default=str))
            print(f"  [calibration] {key}: frozen L{layer}/r{cand_row} activation_rank="
                  f"{cal['frozen_activation_rank']} ratio_rank={cal['frozen_ratio_rank']} "
                  f"ratio={cal['frozen_ratio']:.4f}  detector_independently_recovers="
                  f"{cal['detector_independently_recovers_yu_et_al']}")

        print(f"[{time.time()-t0:.1f}s total for {key}]")
        print_running_cohort_summary()

        # `mods` and `down_mod` hold live references to GPU nn.Module objects (only
        # individual rows were .cpu()'d, not the whole module) -- without deleting these
        # too, `del model` alone does not free GPU memory (confirmed this session: caused
        # a CUDA OOM loading Mistral-7B right after Llama-7B, since Llama's ~28GB fp32
        # weights were still referenced via the stale `mods` dict from the prior
        # iteration). locals() is a snapshot in function scope, so `del locals()[...]`
        # would silently no-op -- delete the actual names directly instead.
        del model, mods, tok
        try:
            del down_mod, gate, up, gate64, up64
        except NameError:
            pass  # not created when `already_done(key)` skipped the 4a block this pass
        import gc
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    print("=== BATCH 2 COMPLETE ===")


if __name__ == "__main__":
    main()
