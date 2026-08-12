"""
scripts/interpretability/run_sw_broadcast_impulse.py
----------------------------------------------------
Super-weight broadcast / impulse-response assay.

For each super-weight output row k (at its source MLP layer), we inject a
small perturbation ε·e_k into the residual stream at the SW token position
and measure how it propagates through all downstream layers. Three metrics
are computed at each later layer m:

  T_m  = mean_t ‖Δh_{m,t}‖₂ / ε        (total causal influence, normalised)
  C_m  = Δh_{m,t*,k} / ‖Δh_{m,t*}‖₂   (k-coordinate preservation)
  Y    = D_KL(p_clean ‖ p_patched)      (functional output KL)

where Δh = h'_{m,t} - h_{m,t} and t* = argmax_t ‖Δh_{m,t}‖₂.

Models (from super_weight_index.json):
  generator            — layer 4,  row 2371  (EUK,  transformer +)
  generator_prokaryote — layer 2,  row 1927  (PROK, transformer +)
  dnabert2             — layer 5,  row 603   (transformer +)
  ntv3                 — layer 11, row 1472  (transformer +)
  evo1                 — layer 11, row 3776  (SSM control)

Controls per SW:
  random_coord  : axis-aligned ε perturbation at a uniformly random coordinate
  neighbour_row : axis-aligned ε perturbation at row ± 1
  random_dir    : dense random-unit-direction perturbation of the same ε norm

Usage:
  python scripts/interpretability/run_sw_broadcast_impulse.py --model generator
  python scripts/interpretability/run_sw_broadcast_impulse.py --model all
  python scripts/interpretability/run_sw_broadcast_impulse.py --replot results/sw_broadcast_impulse.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

_NTV3_REV     = "0ecff3637f0d3ba5b686d1095083218157c2ca34"
_DNABERT2_REV = "7bce263b15377fc15361f52cfab88f8b586abda0"

# SW coordinates and probe mapping (primary SW per model)
SW_TARGETS: dict[str, dict] = {
    "generator":            {"sw_layer": 4,  "sw_row": 2371, "probe": "actb_500",
                             "causal": True,  "arch": "llama"},
    "generator_prokaryote": {"sw_layer": 2,  "sw_row": 1927, "probe": "pseudomonadota",
                             "causal": True,  "arch": "llama"},
    "dnabert2":             {"sw_layer": 5,  "sw_row": 603,  "probe": "actb_500",
                             "causal": False, "arch": "bert"},
    "ntv3":                 {"sw_layer": 11, "sw_row": 1472, "probe": "actb_500",
                             "causal": False, "arch": "ntv3"},
    "evo1":                 {"sw_layer": 11, "sw_row": 3776, "probe": "actb_500",
                             "causal": True,  "arch": "evo1"},
}

MODEL_COLORS = {
    "generator":            "#4878CF",
    "generator_prokaryote": "#6ACC65",
    "dnabert2":             "#D65F5F",
    "ntv3":                 "#B47CC7",
    "evo1":                 "#888888",
}


# ─── Model loading ────────────────────────────────────────────────────────────

def _load_model(model_name: str, device: str):
    """Return (model, tokenizer, input_dict_builder) for the named model."""
    arch = SW_TARGETS[model_name]["arch"]

    if arch == "llama":
        cfg = yaml.safe_load((ROOT / f"configs/{model_name}.yaml").read_text())
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"], trust_remote_code=True,
            torch_dtype=torch.float32, device_map="auto",
        )
        model.eval()
        return model, tok

    if arch == "bert":
        from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True, revision=_DNABERT2_REV)
        cfg = AutoConfig.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True, revision=_DNABERT2_REV)
        if not hasattr(cfg, "is_decoder") or cfg.is_decoder is None:
            cfg.is_decoder = False
        if not hasattr(cfg, "pad_token_id") or cfg.pad_token_id is None:
            cfg.pad_token_id = tok.pad_token_id or 0
        model = AutoModelForMaskedLM.from_pretrained(
            "zhihan1996/DNABERT-2-117M", config=cfg, trust_remote_code=True,
            revision=_DNABERT2_REV, device_map={"": "cpu"})
        model.eval()
        if device == "cuda":
            model = model.cuda()
        return model, tok

    if arch == "ntv3":
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("InstaDeepAI/NTv3_650M_pre", trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(
            "InstaDeepAI/NTv3_650M_pre", trust_remote_code=True,
            code_revision=_NTV3_REV, torch_dtype=torch.float32)
        model.eval()
        if device == "cuda":
            model = model.cuda()
        return model, tok

    if arch == "evo1":
        from evo import Evo
        evo = Evo("evo-1-8k-base", device=device)
        # StripedHyena's poles/residues must stay wide-range (fp32); everything
        # else goes to bf16 (same exponent range as fp32, so the layer-10/11
        # spike at the SW row doesn't overflow the way it does under fp16
        # autocast — see run_evo1_residual_attribution_fp32.py).
        for n, p in evo.model.named_parameters():
            if any(k in n for k in ("poles", "residues")):
                continue
            p.data = p.data.to(torch.bfloat16)
        return evo.model, evo.tokenizer

    raise ValueError(f"Unknown arch: {arch}")


def _get_blocks(model, arch: str) -> list:
    if arch == "llama":
        return list(model.model.layers)
    if arch == "bert":
        bert = model.bert if hasattr(model, "bert") else model
        return list(bert.encoder.layer)
    if arch == "ntv3":
        return list(model.core.transformer_blocks)
    if arch == "evo1":
        return list(model.blocks)
    raise ValueError(arch)


def _tokenize(tokenizer, seq: str, arch: str, device: str) -> dict:
    if arch == "llama":
        r = len(seq) % 6
        if r:
            seq = seq[r:]
        ids = tokenizer(seq, return_tensors="pt",
                        add_special_tokens=False)["input_ids"].to(device)
        return {"input_ids": ids}
    if arch == "bert":
        inp = tokenizer(seq, return_tensors="pt", truncation=True, max_length=512)
        return {k: v.to(device) for k, v in inp.items()}
    if arch == "ntv3":
        # NTv3 has a U-Net with 7 conv downsampling stages (128× total).
        # The tokenizer is character-level with no special tokens, so
        # n_tokens == seq_len and must be a multiple of 128.
        _NTV3_FACTOR = 128
        target_seq_len = (len(seq) // _NTV3_FACTOR) * _NTV3_FACTOR
        if target_seq_len == 0:
            target_seq_len = _NTV3_FACTOR
        seq = seq[:target_seq_len]
        inp = tokenizer(seq, return_tensors="pt")
        return {k: v.to(device) for k, v in inp.items()}
    if arch == "evo1":
        ids = torch.tensor(list(tokenizer.tokenize(seq)),
                           dtype=torch.long, device=device).unsqueeze(0)
        return {"__evo_ids__": ids}
    raise ValueError(arch)


def _forward(model, inp_dict: dict, arch: str):
    if arch == "evo1":
        return model(inp_dict["__evo_ids__"])
    return model(**inp_dict)


def _get_logits(out, arch: str) -> torch.Tensor:
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


# ─── Hook-based forward pass ──────────────────────────────────────────────────

def _hs_from_output(output):
    # NTv3 blocks return {"embeddings": tensor, "attention_weights": tensor}
    if isinstance(output, dict):
        hs = output["embeddings"]
    else:
        hs = output[0] if isinstance(output, tuple) else output
    return hs.float()


def _rebuild_output(output, new_hs):
    if isinstance(output, dict):
        result = dict(output)
        result["embeddings"] = new_hs.to(output["embeddings"].dtype)
        return result
    if isinstance(output, tuple):
        return (new_hs.to(output[0].dtype),) + output[1:]
    return new_hs.to(output.dtype)


def run_pass(model, inp_dict: dict, blocks: list, arch: str,
             inject_layer: int | None = None,
             inject_pos: int | None = None,
             inject_row: int | None = None,
             epsilon: float = 1.0,
             inject_dir: torch.Tensor | None = None):
    """
    Run forward pass, capturing residual-stream output at each block.
    If inject_layer is set, adds the perturbation at that block's output
    before downstream blocks see it.

    inject_dir: if provided, use this (d_model,) unit vector scaled by epsilon
                instead of the axis-aligned e_{inject_row} direction.

    Returns (captured: dict[int -> (T, D) float32 cpu], model_output).
    """
    captured: dict[int, torch.Tensor] = {}
    handles  = []

    def make_capture(li: int):
        def hook(mod, inp, out):
            hs = _hs_from_output(out)
            captured[li] = (hs[0] if hs.dim() == 3 else hs).detach().cpu()
        return hook

    def make_inject(li: int):
        def hook(mod, inp, out):
            hs = _hs_from_output(out).clone()
            if hs.dim() == 3:
                if inject_dir is not None:
                    hs[0, inject_pos, :] = hs[0, inject_pos, :] + \
                        epsilon * inject_dir.to(hs.device, dtype=hs.dtype)
                else:
                    hs[0, inject_pos, inject_row] = \
                        hs[0, inject_pos, inject_row] + epsilon
                captured[li] = hs[0].detach().cpu()
            else:
                if inject_dir is not None:
                    hs[inject_pos, :] = hs[inject_pos, :] + \
                        epsilon * inject_dir.to(hs.device, dtype=hs.dtype)
                else:
                    hs[inject_pos, inject_row] = hs[inject_pos, inject_row] + epsilon
                captured[li] = hs.detach().cpu()
            return _rebuild_output(out, hs)
        return hook

    for li, blk in enumerate(blocks):
        fn = make_inject(li) if li == inject_layer else make_capture(li)
        handles.append(blk.register_forward_hook(fn))

    try:
        with torch.no_grad():
            if arch == "evo1":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(inp_dict["__evo_ids__"])
            else:
                out = model(**inp_dict)
    finally:
        for h in handles:
            h.remove()

    return captured, out


# ─── Metrics ─────────────────────────────────────────────────────────────────

def broadcast_metrics(
    clean_hs: dict[int, torch.Tensor],
    pert_hs:  dict[int, torch.Tensor],
    sw_row: int,
    source_layer: int,
    epsilon: float,
) -> dict[int, dict]:
    """
    Compute per-layer broadcast metrics from clean vs perturbed hidden states.
    Returns dict: layer_idx -> {T_mean, T_max, C, t_star, dnorm_per_token}
    """
    results: dict[int, dict] = {}
    for li in sorted(clean_hs.keys()):
        if li <= source_layer:
            continue
        c = clean_hs.get(li)
        p = pert_hs.get(li)
        if c is None or p is None:
            continue
        L = min(c.shape[0], p.shape[0])
        delta  = (p[:L].float() - c[:L].float())    # (T, D)
        dnorm  = delta.norm(dim=-1) / abs(epsilon)   # (T,) normalised influence
        T_mean = float(dnorm.mean().item())
        T_max  = float(dnorm.max().item())
        t_star = int(dnorm.argmax().item())
        d_star = delta[t_star]                       # (D,)
        C = float(d_star[sw_row].item() / (d_star.norm().item() + 1e-12))
        results[li] = {
            "T_mean":          T_mean,
            "T_max":           T_max,
            "C":               C,
            "t_star":          t_star,
            "dnorm_per_token": dnorm.tolist(),
        }
    return results


def kl_divergence(logits_clean: torch.Tensor, logits_pert: torch.Tensor,
                  sw_pos: int, is_causal: bool) -> float:
    """KL(p_clean ‖ p_pert) at the functional output position."""
    if is_causal:
        t = min(sw_pos + 1, logits_clean.shape[1] - 1)
        p = torch.softmax(logits_clean[0, t].float(), dim=-1)
        q = torch.softmax(logits_pert[0, t].float(), dim=-1)
    else:
        # Mean KL over all token positions for bidirectional models
        p = torch.softmax(logits_clean[0].float(), dim=-1)
        q = torch.softmax(logits_pert[0].float(), dim=-1)
        kl = (p * ((p + 1e-10).log() - (q + 1e-10).log())).sum(dim=-1).mean().item()
        return float(kl)
    kl = (p * ((p + 1e-10).log() - (q + 1e-10).log())).sum().item()
    return float(max(kl, 0.0))


# ─── Per-model assay ──────────────────────────────────────────────────────────

def run_model_assay(model_name: str, epsilon: float, n_controls: int,
                    seed: int, device: str) -> dict:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    tgt = SW_TARGETS[model_name]
    sw_layer  = tgt["sw_layer"]
    sw_row    = tgt["sw_row"]
    is_causal = tgt["causal"]
    arch      = tgt["arch"]

    print(f"\n[broadcast:{model_name}] sw_layer={sw_layer}  sw_row={sw_row}")

    from probes.dna_probes import get_probe
    seq = get_probe(tgt["probe"])

    print(f"[broadcast:{model_name}] loading model …")
    model, tokenizer = _load_model(model_name, device)
    blocks = _get_blocks(model, arch)
    n_layers = len(blocks)

    inp = _tokenize(tokenizer, seq, arch, device)
    n_tokens = inp.get("input_ids", inp.get("__evo_ids__")).shape[1]
    print(f"[broadcast:{model_name}] n_tokens={n_tokens}  n_layers={n_layers}")

    # ── Clean pass ──────────────────────────────────────────────────────────
    clean_hs, clean_out = run_pass(model, inp, blocks, arch)
    logits_clean = _get_logits(clean_out, arch)

    # SW token position: max |h[sw_row]| at source layer output
    src_hs = clean_hs.get(sw_layer)
    sw_pos = (int(src_hs[:, sw_row].abs().argmax().item())
              if src_hs is not None else n_tokens // 2)
    d_model = src_hs.shape[-1] if src_hs is not None else clean_hs[0].shape[-1]
    print(f"[broadcast:{model_name}] sw_pos={sw_pos}  d_model={d_model}")

    # ── SW perturbed pass ───────────────────────────────────────────────────
    pert_hs, pert_out = run_pass(model, inp, blocks, arch,
                                 inject_layer=sw_layer, inject_pos=sw_pos,
                                 inject_row=sw_row, epsilon=epsilon)
    logits_pert = _get_logits(pert_out, arch)

    sw_metrics = broadcast_metrics(clean_hs, pert_hs, sw_row, sw_layer, epsilon)
    kl_sw = kl_divergence(logits_clean, logits_pert, sw_pos, is_causal)

    print(f"[broadcast:{model_name}] KL_sw={kl_sw:.4e}")
    for li, m in sorted(sw_metrics.items()):
        print(f"  layer {li:3d}  T_mean={m['T_mean']:.4f}  C={m['C']:+.4f}")

    # ── Control passes ───────────────────────────────────────────────────────
    non_sw = [r for r in range(d_model) if r != sw_row]
    controls = {}

    # Control A: random coordinate (same ε, axis-aligned)
    rand_rows = [rng.choice(non_sw) for _ in range(n_controls)]
    rand_results = []
    for rr in rand_rows:
        rh, ro = run_pass(model, inp, blocks, arch,
                          inject_layer=sw_layer, inject_pos=sw_pos,
                          inject_row=rr, epsilon=epsilon)
        rm = broadcast_metrics(clean_hs, rh, rr, sw_layer, epsilon)
        kl_r = kl_divergence(logits_clean, _get_logits(ro, arch), sw_pos, is_causal)
        rand_results.append({"row": rr, "kl": kl_r,
                             "metrics": {str(k): v for k, v in rm.items()}})
    controls["random_coord"] = rand_results

    # Control B: neighbour row (sw_row ± 1)
    for delta_r, label in [(-1, "neighbour_minus"), (+1, "neighbour_plus")]:
        nr = sw_row + delta_r
        if 0 <= nr < d_model:
            nh, no_ = run_pass(model, inp, blocks, arch,
                               inject_layer=sw_layer, inject_pos=sw_pos,
                               inject_row=nr, epsilon=epsilon)
            nm = broadcast_metrics(clean_hs, nh, nr, sw_layer, epsilon)
            kl_n = kl_divergence(logits_clean, _get_logits(no_, arch), sw_pos, is_causal)
            controls[label] = {"row": nr, "kl": kl_n,
                               "metrics": {str(k): v for k, v in nm.items()}}

    # Control C: equal-norm random dense direction
    rand_dir_results = []
    for _ in range(n_controls):
        v = torch.randn(d_model)
        v = v / (v.norm() + 1e-12)           # unit vector
        rh2, ro2 = run_pass(model, inp, blocks, arch,
                             inject_layer=sw_layer, inject_pos=sw_pos,
                             inject_row=None, epsilon=epsilon, inject_dir=v)
        # For C metric, use a random reference row
        ref_row = rng.choice(non_sw)
        rm2 = broadcast_metrics(clean_hs, rh2, ref_row, sw_layer, epsilon)
        kl_r2 = kl_divergence(logits_clean, _get_logits(ro2, arch), sw_pos, is_causal)
        rand_dir_results.append({"kl": kl_r2,
                                 "metrics": {str(k): v_m for k, v_m in rm2.items()}})
    controls["random_dir"] = rand_dir_results

    return {
        "model":      model_name,
        "arch":       arch,
        "sw_layer":   sw_layer,
        "sw_row":     sw_row,
        "sw_pos":     sw_pos,
        "epsilon":    epsilon,
        "n_tokens":   n_tokens,
        "n_layers":   n_layers,
        "d_model":    d_model,
        "is_causal":  is_causal,
        "sw_metrics": {str(k): v for k, v in sw_metrics.items()},
        "kl_sw":      kl_sw,
        "controls":   controls,
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(all_results: dict, out_png: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip plot] matplotlib not available")
        return

    sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
    try:
        from _figstyle import apply_style, panel_label
        apply_style()
    except ImportError:
        pass

    # Panel layout: rows = models with data; cols = T_mean curve, C curve
    models_with_data = [m for m in SW_TARGETS if m in all_results]
    if not models_with_data:
        return

    fig, axes = plt.subplots(len(models_with_data), 2,
                              figsize=(9, 2.8 * len(models_with_data)))
    if len(models_with_data) == 1:
        axes = [axes]

    CTRL_COLOR = "#AAAAAA"
    SW_COLOR   = "#D62728"

    for row_i, mname in enumerate(models_with_data):
        res = all_results[mname]
        sw_m = res["sw_metrics"]
        sw_layer = res["sw_layer"]
        n_layers = res["n_layers"]

        layers     = sorted(int(k) for k in sw_m.keys())
        T_sw       = [sw_m[str(li)]["T_mean"] for li in layers]
        C_sw       = [sw_m[str(li)]["C"]      for li in layers]

        # Aggregate random controls
        ctrl_T = {}
        ctrl_C = {}
        for ctrl_name, ctrl_list in res.get("controls", {}).items():
            if ctrl_name.startswith("neighbour"):
                continue
            if isinstance(ctrl_list, list):
                for entry in ctrl_list:
                    em = entry.get("metrics", {})
                    for li_str, lm in em.items():
                        li = int(li_str)
                        ctrl_T.setdefault(li, []).append(lm["T_mean"])
                        ctrl_C.setdefault(li, []).append(lm["C"])

        ax_T = axes[row_i][0]
        ax_C = axes[row_i][1]

        # T_mean curve
        ax_T.plot(layers, T_sw, color=SW_COLOR, lw=2, label="SW row")
        if ctrl_T:
            ctrl_layers = sorted(ctrl_T.keys())
            ctrl_T_mean = [np.mean(ctrl_T[li]) for li in ctrl_layers]
            ctrl_T_std  = [np.std(ctrl_T[li])  for li in ctrl_layers]
            ax_T.plot(ctrl_layers, ctrl_T_mean, color=CTRL_COLOR, lw=1.5,
                      linestyle="--", label="random controls")
            ax_T.fill_between(ctrl_layers,
                               np.array(ctrl_T_mean) - np.array(ctrl_T_std),
                               np.array(ctrl_T_mean) + np.array(ctrl_T_std),
                               color=CTRL_COLOR, alpha=0.25)
        ax_T.axvline(sw_layer, color="k", linewidth=0.8, linestyle=":")
        ax_T.set_xlabel("Layer")
        ax_T.set_ylabel("Broadcast gain  T / ε")
        ax_T.set_title(f"{mname}", fontsize=9, fontweight="bold")
        ax_T.legend(fontsize=7)

        # C curve (coordinate preservation)
        ax_C.plot(layers, C_sw, color=SW_COLOR, lw=2, label="SW row")
        ax_C.axhline(0, color="k", linewidth=0.6, linestyle="--")
        ax_C.axvline(sw_layer, color="k", linewidth=0.8, linestyle=":")
        ax_C.set_xlabel("Layer")
        ax_C.set_ylabel("Coord. preservation  C")
        ax_C.set_title(f"{mname} — k-preservation", fontsize=9)
        ax_C.set_ylim(-1.05, 1.05)

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"  Plot saved → {out_png}")
    plt.close(fig)


def plot_heatmap(result: dict, out_png: str) -> None:
    """Layer × token-offset heatmap of total influence for a single model."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    sw_pos    = result["sw_pos"]
    sw_layer  = result["sw_layer"]
    n_layers  = result["n_layers"]
    sw_metrics = result["sw_metrics"]

    layer_ids = sorted(int(k) for k in sw_metrics.keys())
    if not layer_ids:
        return

    # Build matrix: rows = downstream layers, cols = token positions
    max_t = max(len(sw_metrics[str(li)]["dnorm_per_token"]) for li in layer_ids)
    mat   = np.zeros((len(layer_ids), max_t))
    for ri, li in enumerate(layer_ids):
        row = np.array(sw_metrics[str(li)]["dnorm_per_token"])
        mat[ri, :len(row)] = row

    offsets = np.arange(max_t) - sw_pos   # token offset from injection site

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(mat, aspect="auto", origin="lower",
                   extent=[offsets[0] - 0.5, offsets[-1] + 0.5,
                           layer_ids[0] - 0.5, layer_ids[-1] + 0.5],
                   cmap="YlOrRd")
    ax.axvline(0, color="white", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Token offset from injection site")
    ax.set_ylabel("Layer index")
    ax.set_title(f"{result['model']}  layer×token influence heatmap (T / ε)", fontsize=10)
    plt.colorbar(im, ax=ax, label="T / ε")
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"  Heatmap saved → {out_png}")
    plt.close(fig)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="generator",
                   choices=list(SW_TARGETS) + ["all"],
                   help="Model to analyse (or 'all' to run every model sequentially).")
    p.add_argument("--epsilon",    type=float, default=1.0,
                   help="Perturbation magnitude (default 1.0).")
    p.add_argument("--n_controls", type=int, default=5,
                   help="Number of random-control draws per type.")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--out_dir",    default="results")
    p.add_argument("--replot",     default=None, metavar="JSON",
                   help="Skip model runs; replot from a saved JSON file.")
    return p.parse_args()


def main():
    args = parse_args()
    out_json = ROOT / args.out_dir / "sw_broadcast_impulse.json"
    out_png  = ROOT / args.out_dir / "sw_broadcast_impulse.png"

    if args.replot:
        all_results = json.loads(Path(args.replot).read_text())
    else:
        models_to_run = list(SW_TARGETS) if args.model == "all" else [args.model]
        # Load previously accumulated results so per-model runs merge rather than overwrite
        all_results: dict = json.loads(out_json.read_text()) if out_json.exists() else {}
        for mname in models_to_run:
            try:
                all_results[mname] = run_model_assay(
                    mname, args.epsilon, args.n_controls, args.seed, args.device)
            except Exception as exc:
                print(f"[broadcast:{mname}] FAILED — {exc}")
                import traceback; traceback.print_exc()

        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(all_results, indent=2))
        print(f"\n  Results saved → {out_json}")

    plot_results(all_results, str(out_png))

    # Per-model heatmaps
    for mname, res in all_results.items():
        hmap_png = ROOT / args.out_dir / f"sw_broadcast_heatmap_{mname}.png"
        plot_heatmap(res, str(hmap_png))


if __name__ == "__main__":
    main()
