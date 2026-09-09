"""
scripts/interpretability/run_sw_relay_heads.py
----------------------------------------------
Relay-head identification and ablation for GENERator transformer models.

For each attention head h at every layer downstream of the super-weight (SW)
source layer, we compute two quantities from the OV circuit (W^O_h W^V_h):

  g_h(k) = ‖W^O_h W^V_h e_k‖₂ / median_r ‖W^O_h W^V_h e_r‖₂

    Amplification gain — how much the head preferentially projects the SW
    coordinate k compared with a typical coordinate.

  p_h(k) = cos(W^O_h W^V_h e_k, e_k)

    Directional preservation — does the head re-emit the SW signal in the
    same residual-stream coordinate after OV multiplication?

Combined relay score:  s_h(k) = g_h(k) * max(p_h(k), 0)

The top-K relay heads (highest s) are ablated by zeroing their contribution
to the attention output projection (W^O_h columns for head h).  We compare
the resulting SW-dependent next-token KL divergence and task degradation to
ablating an equal number of randomly selected, layer-matched control heads.

Models: generator (EUK) and generator_prokaryote (PROK).

Output:
  results/sw_relay_heads_{model}.json
  results/sw_relay_heads_{model}.png

Usage:
  python scripts/interpretability/run_sw_relay_heads.py --model generator
  python scripts/interpretability/run_sw_relay_heads.py --model generator_prokaryote
  python scripts/interpretability/run_sw_relay_heads.py --replot results/sw_relay_heads_generator.json
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

SW_DEFAULTS = {
    "generator":            {"sw_layer": 4,  "sw_row": 2371, "probe": "actb_500"},
    "generator_prokaryote": {"sw_layer": 2,  "sw_row": 1927, "probe": "pseudomonadota"},
}

N_RELAY_SAMPLE = 256   # random coordinate samples for denominator estimation
TOP_K_RELAY    = 10    # heads to ablate in each condition


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(model_name: str):
    """Return (model, tokenizer, config_dict)."""
    cfg = yaml.safe_load((ROOT / f"configs/{model_name}.yaml").read_text())
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], trust_remote_code=True,
        torch_dtype=torch.float32, device_map="auto")
    model.eval()
    return model, tok, cfg


def _prepare_ids(tokenizer, seq: str, device):
    r = len(seq) % 6
    if r:
        seq = seq[r:]
    return tokenizer(seq, return_tensors="pt",
                     add_special_tokens=False)["input_ids"].to(device)


# ─── OV circuit analysis (weight-space) ──────────────────────────────────────

def _get_attn_weights(layer):
    """Return (v_proj_weight, o_proj_weight) for a LlamaAttention layer."""
    attn = layer.self_attn
    return (
        attn.v_proj.weight.detach().float().cpu(),  # (n_kv_heads*hd, d_model)
        attn.o_proj.weight.detach().float().cpu(),  # (d_model, n_q_heads*hd)
    )


def _head_params(model_cfg) -> tuple[int, int, int, int]:
    """Return (d_model, n_q_heads, n_kv_heads, head_dim)."""
    hf_cfg = model_cfg
    d_model     = hf_cfg.hidden_size
    n_q_heads   = hf_cfg.num_attention_heads
    n_kv_heads  = getattr(hf_cfg, "num_key_value_heads", n_q_heads)
    head_dim    = d_model // n_q_heads
    return d_model, n_q_heads, n_kv_heads, head_dim


def compute_ov_relay_scores(
    V: torch.Tensor,          # (n_kv_heads * head_dim, d_model) — v_proj
    O: torch.Tensor,          # (d_model, n_q_heads * head_dim) — o_proj
    sw_row: int,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    n_sample: int = N_RELAY_SAMPLE,
    rng: random.Random = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-query-head gain g_h(k) and preservation p_h(k) for the SW coordinate k.

    Returns:
        gains  : (n_q_heads,) float32
        presrv : (n_q_heads,) float32
    """
    d_model = O.shape[0]
    gains   = np.zeros(n_q_heads, dtype=np.float32)
    presrv  = np.zeros(n_q_heads, dtype=np.float32)

    kv_ratio = n_q_heads // n_kv_heads   # how many Q heads share one KV head

    # For denominator: sample n_sample random residual-stream coordinates
    all_cols = list(range(d_model))
    if rng is not None:
        sample_cols = rng.sample(all_cols, min(n_sample, d_model))
    else:
        sample_cols = np.random.choice(d_model, min(n_sample, d_model), replace=False).tolist()
    sample_cols = torch.tensor(sample_cols, dtype=torch.long)

    for h in range(n_q_heads):
        kv_h = h // kv_ratio           # KV head index for this query head
        # V slice for this KV head: (head_dim, d_model)
        Vh = V[kv_h * head_dim : (kv_h + 1) * head_dim, :]
        # O slice for this Q head: (d_model, head_dim)
        Oh = O[:, h * head_dim : (h + 1) * head_dim]

        # OV column for SW coordinate k: W^O_h @ W^V_h[:, k]
        v_k  = Vh[:, sw_row]              # (head_dim,)
        ov_k = Oh @ v_k                   # (d_model,)
        gain_k = float(ov_k.norm().item())

        # Preservation: fraction of ov_k in direction e_k
        pres_k = float(ov_k[sw_row].item() / (gain_k + 1e-12))

        # Denominator: median gain over sample_cols
        v_sample  = Vh[:, sample_cols]    # (head_dim, n_sample)
        ov_sample = Oh @ v_sample         # (d_model, n_sample)
        sample_norms = ov_sample.norm(dim=0)  # (n_sample,)
        median_norm  = float(sample_norms.median().item())

        gains[h]  = gain_k / (median_norm + 1e-12)
        presrv[h] = pres_k

    return gains, presrv


def analyse_relay_heads(model, sw_layer: int, sw_row: int,
                         n_q_heads: int, n_kv_heads: int, head_dim: int,
                         n_layers: int, seed: int) -> list[dict]:
    """
    For each layer downstream of sw_layer, compute gain and preservation
    for every attention head.
    Returns list of {layer, head, gain, preservation, score} sorted by score desc.
    """
    rng = random.Random(seed)
    all_entries = []
    layers = model.model.layers

    for li in range(sw_layer + 1, n_layers):
        V, O = _get_attn_weights(layers[li])
        gains, presrv = compute_ov_relay_scores(
            V, O, sw_row, n_q_heads, n_kv_heads, head_dim,
            n_sample=N_RELAY_SAMPLE, rng=rng)
        for h in range(n_q_heads):
            score = float(gains[h] * max(presrv[h], 0.0))
            all_entries.append({
                "layer":        li,
                "head":         h,
                "gain":         float(gains[h]),
                "preservation": float(presrv[h]),
                "score":        score,
            })

    all_entries.sort(key=lambda x: -x["score"])
    return all_entries


# ─── Ablation ─────────────────────────────────────────────────────────────────

def _make_o_proj_ablation_hook(heads: list[int], head_dim: int):
    """
    Hook on o_proj (nn.Linear) that zeroes the contribution of the given heads.
    o_proj: Linear(n_q_heads*head_dim, d_model)
      input  : (B, T, n_q_heads * head_dim)
      output : (B, T, d_model)
    """
    def hook(mod, inp, out):
        result = out.clone()
        in_hs  = inp[0]                           # (B, T, n_q_heads * head_dim)
        W      = mod.weight                       # (d_model, n_q_heads * head_dim)
        for h in heads:
            h_inp = in_hs[..., h * head_dim : (h + 1) * head_dim]  # (B, T, hd)
            h_w   = W[:, h * head_dim : (h + 1) * head_dim]        # (d_model, hd)
            result = result - (h_inp @ h_w.T)    # subtract head h contribution
        return result
    return hook


def run_forward_kl(model, input_ids: torch.Tensor, sw_pos: int,
                   ablate_spec: dict[int, list[int]] | None,
                   head_dim: int) -> tuple[float, float]:
    """
    Run a forward pass (optionally ablating specific heads at specific layers).
    Returns (CE_loss, KL_vs_clean_at_sw_pos).
    If ablate_spec is None this is the clean pass (KL = 0 by definition).
    """
    handles = []
    if ablate_spec:
        for li, heads in ablate_spec.items():
            o_proj = model.model.layers[li].self_attn.o_proj
            h = o_proj.register_forward_hook(
                _make_o_proj_ablation_hook(heads, head_dim))
            handles.append(h)

    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)
        logits = out.logits
        ce_loss = out.loss.item()

    for h in handles:
        h.remove()

    return ce_loss, logits


def measure_ablation_effect(model, input_ids: torch.Tensor, sw_pos: int,
                             ablate_spec: dict[int, list[int]], head_dim: int,
                             logits_clean: torch.Tensor) -> dict:
    """
    Ablate heads, measure CE loss and KL at sw_pos+1 vs clean.
    """
    ce_abl, logits_abl = run_forward_kl(
        model, input_ids, sw_pos, ablate_spec, head_dim)

    t = min(sw_pos + 1, logits_clean.shape[1] - 1)
    p = torch.softmax(logits_clean[0, t].float(), dim=-1)
    q = torch.softmax(logits_abl[0, t].float(), dim=-1)
    kl = float((p * ((p + 1e-10).log() - (q + 1e-10).log())).sum().clamp(min=0).item())

    return {"ce_loss": ce_abl, "kl_vs_clean": kl}


# ─── Main assay ──────────────────────────────────────────────────────────────

def run_relay_assay(model_name: str, top_k: int, n_rand_ablations: int,
                    seed: int) -> dict:
    rng = random.Random(seed)

    defaults = SW_DEFAULTS[model_name]
    sw_layer = defaults["sw_layer"]
    sw_row   = defaults["sw_row"]
    probe    = defaults["probe"]

    print(f"\n[relay:{model_name}] sw_layer={sw_layer}  sw_row={sw_row}")

    from src.dna_probes import get_probe
    seq = get_probe(probe)

    print(f"[relay:{model_name}] loading model …")
    model, tokenizer, yaml_cfg = load_model(model_name)
    hf_cfg = model.config
    n_layers = yaml_cfg["num_layers"]
    d_model, n_q_heads, n_kv_heads, head_dim = _head_params(hf_cfg)
    print(f"[relay:{model_name}] d_model={d_model}  n_q_heads={n_q_heads}  "
          f"n_kv_heads={n_kv_heads}  head_dim={head_dim}  n_layers={n_layers}")

    device = next(model.parameters()).device
    input_ids = _prepare_ids(tokenizer, seq, device)
    n_tokens  = input_ids.shape[1]

    # ── Clean forward pass ──────────────────────────────────────────────────
    ce_clean, logits_clean = run_forward_kl(model, input_ids, 0, None, head_dim)

    # Find SW token position from MLP down_proj output
    _store_sw: dict = {}
    def _sw_hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        _store_sw["act"] = hs[..., sw_row].detach().float().cpu()

    target_mlp = model.model.layers[sw_layer].mlp.down_proj
    h_sw = target_mlp.register_forward_hook(_sw_hook)
    with torch.no_grad():
        model(input_ids=input_ids)
    h_sw.remove()
    sw_pos = int(_store_sw["act"][0].abs().argmax().item())
    print(f"[relay:{model_name}] sw_pos={sw_pos}  n_tokens={n_tokens}")
    print(f"[relay:{model_name}] clean CE loss = {ce_clean:.4f}")

    # ── OV circuit analysis ─────────────────────────────────────────────────
    print(f"[relay:{model_name}] computing OV relay scores …")
    all_heads = analyse_relay_heads(
        model, sw_layer, sw_row, n_q_heads, n_kv_heads, head_dim, n_layers, seed)

    # Top-K relay heads
    top_relay = all_heads[:top_k]
    print(f"[relay:{model_name}] top {top_k} relay heads:")
    for e in top_relay:
        print(f"  layer={e['layer']:2d}  head={e['head']:2d}  "
              f"gain={e['gain']:.3f}  pres={e['preservation']:+.3f}  "
              f"score={e['score']:.3f}")

    # Build ablation spec: which heads to ablate per layer
    def make_ablate_spec(head_entries: list[dict]) -> dict[int, list[int]]:
        spec: dict[int, list[int]] = {}
        for e in head_entries:
            spec.setdefault(e["layer"], []).append(e["head"])
        return spec

    relay_spec = make_ablate_spec(top_relay)

    # ── Ablation: relay heads ────────────────────────────────────────────────
    relay_result = measure_ablation_effect(
        model, input_ids, sw_pos, relay_spec, head_dim, logits_clean)
    print(f"[relay:{model_name}] relay ablation:  "
          f"CE={relay_result['ce_loss']:.4f}  KL={relay_result['kl_vs_clean']:.4e}")

    # ── Ablation: random layer-matched controls ──────────────────────────────
    # Build pool of available (layer, head) pairs at the same layers as relay heads
    relay_layers = sorted(set(e["layer"] for e in top_relay))
    all_available = {li: list(range(n_q_heads)) for li in relay_layers}
    for e in top_relay:
        if e["head"] in all_available.get(e["layer"], []):
            all_available[e["layer"]].remove(e["head"])

    # Match count per layer
    relay_count_per_layer: dict[int, int] = {}
    for e in top_relay:
        relay_count_per_layer[e["layer"]] = relay_count_per_layer.get(e["layer"], 0) + 1

    rand_results = []
    for _ in range(n_rand_ablations):
        rand_spec: dict[int, list[int]] = {}
        for li, cnt in relay_count_per_layer.items():
            pool = all_available.get(li, [])
            chosen = rng.sample(pool, min(cnt, len(pool)))
            if chosen:
                rand_spec[li] = chosen
        r = measure_ablation_effect(
            model, input_ids, sw_pos, rand_spec, head_dim, logits_clean)
        rand_results.append(r)
    rand_ce_mean   = float(np.mean([r["ce_loss"]     for r in rand_results]))
    rand_kl_mean   = float(np.mean([r["kl_vs_clean"] for r in rand_results]))
    print(f"[relay:{model_name}] random ablation: "
          f"CE={rand_ce_mean:.4f}  KL={rand_kl_mean:.4e}  (mean over {n_rand_ablations})")

    return {
        "model":          model_name,
        "sw_layer":       sw_layer,
        "sw_row":         sw_row,
        "sw_pos":         sw_pos,
        "n_tokens":       n_tokens,
        "n_layers":       n_layers,
        "d_model":        d_model,
        "n_q_heads":      n_q_heads,
        "n_kv_heads":     n_kv_heads,
        "head_dim":       head_dim,
        "ce_clean":       ce_clean,
        "top_relay_heads": top_relay,
        "relay_ablation":  relay_result,
        "random_ablations": rand_results,
        "random_ablation_mean": {"ce_loss": rand_ce_mean, "kl_vs_clean": rand_kl_mean},
        "all_heads_ranked": all_heads,
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_relay(result: dict, out_png: str) -> None:
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: head relay scores by layer (heatmap)
    ax_a = axes[0]
    n_q_heads = result["n_q_heads"]
    sw_layer  = result["sw_layer"]
    n_layers  = result["n_layers"]
    all_heads = result["all_heads_ranked"]

    score_mat = np.zeros((n_layers, n_q_heads))
    for e in all_heads:
        score_mat[e["layer"], e["head"]] = e["score"]

    # Only show downstream layers
    downstream_layers = list(range(sw_layer + 1, n_layers))
    sub_mat = score_mat[downstream_layers, :]
    im = ax_a.imshow(sub_mat, aspect="auto", origin="lower",
                     extent=[-0.5, n_q_heads - 0.5,
                              downstream_layers[0] - 0.5, downstream_layers[-1] + 0.5],
                     cmap="hot_r")
    ax_a.set_xlabel("Head index")
    ax_a.set_ylabel("Layer index")
    ax_a.set_title(f"{result['model']}\nOV relay score  g·max(p,0)", fontsize=9)
    plt.colorbar(im, ax=ax_a)

    # Mark top relay heads
    top_heads = result["top_relay_heads"]
    ax_a.scatter([e["head"] for e in top_heads],
                 [e["layer"] for e in top_heads],
                 c="cyan", s=20, marker="*", zorder=5, label="top relay")
    ax_a.legend(fontsize=7, loc="upper right")

    # Panel B: ablation comparison bar chart
    ax_b = axes[1]
    relay_kl = result["relay_ablation"]["kl_vs_clean"]
    rand_kl  = result["random_ablation_mean"]["kl_vs_clean"]
    rand_kl_std = float(np.std([r["kl_vs_clean"]
                                 for r in result.get("random_ablations", [])]))

    bars = ax_b.bar(["Relay heads", "Random heads"],
                    [relay_kl, rand_kl], color=["#D62728", "#AAAAAA"], width=0.5)
    ax_b.errorbar([1], [rand_kl], yerr=[rand_kl_std], fmt="none",
                  color="k", capsize=5, linewidth=1.5)
    ax_b.set_ylabel("KL(clean ‖ ablated)  at SW next-token")
    ax_b.set_title(f"Ablation effect (top {len(top_heads)} heads)", fontsize=9)
    ax_b.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"  Plot saved → {out_png}")
    plt.close(fig)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     default="generator",
                   choices=list(SW_DEFAULTS))
    p.add_argument("--top_k",     type=int, default=TOP_K_RELAY,
                   help="Number of relay heads to ablate.")
    p.add_argument("--n_rand",    type=int, default=10,
                   help="Number of random control ablation draws.")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--out_dir",   default="results")
    p.add_argument("--replot",    default=None, metavar="JSON",
                   help="Replot from existing JSON without running the model.")
    return p.parse_args()


def main():
    args  = parse_args()
    model = args.model
    out_j = ROOT / args.out_dir / f"sw_relay_heads_{model}.json"
    out_p = ROOT / args.out_dir / f"sw_relay_heads_{model}.png"

    if args.replot:
        result = json.loads(Path(args.replot).read_text())
    else:
        result = run_relay_assay(
            model, args.top_k, args.n_rand, args.seed)
        Path(out_j).parent.mkdir(parents=True, exist_ok=True)
        out_j.write_text(json.dumps(result, indent=2))
        print(f"\n  Results saved → {out_j}")

    plot_relay(result, str(out_p))


if __name__ == "__main__":
    main()
