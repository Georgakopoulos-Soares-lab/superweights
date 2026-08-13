#!/usr/bin/env python
"""
Evaluate SAE features on REAL genomic sequence, reporting n_active with every stat.

Why this exists
---------------
`sae/analyze.py` probes features with 4,096 synthetic single-hexamer 11-token contexts.
On the standardised EUK SAE that probe finds only 51/12,288 features alive (0.4%) even
though the same dictionary is 99.5% alive on real hg38 (frac_dead = 0.005). The features
are context-dependent, so the synthetic probe cannot see them.

Worse, the probe manufactures artifacts: `analyze.py` reports SW-r = -0.9949 for features
with n_active = 1 -- a Pearson correlation over a vector that is zero at 4,095 of 4,096
positions is fixed by a single point. That degenerate signature appears even when the
underlying dictionary is healthy, so it indicts the measurement, not the model.

This script instead:
  * streams real activation shards (the same ones the SAE trained on),
  * applies the checkpoint's saved data_scale (mandatory -- mismatched preprocessing
    reproduces the degenerate signature),
  * reports, for EVERY feature: firing rate, n_active (raw token count), and the
    correlation of its activation with the super-weight channel magnitude,
  * and refuses to report a correlation for features below --min_active, because those
    are exactly the degenerate cases.

Output
------
  <out_dir>/real_sequence_features.json
     per-feature: freq, n_active, r_with_sw, mean_act
     summary: alive counts, correlation distribution restricted to well-sampled features
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sae.model import BatchTopKSAE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_ckpt", required=True)
    ap.add_argument("--acts_dir", required=True)
    ap.add_argument("--sw_channel", type=int, default=2371,
                    help="residual channel index of the super weight")
    ap.add_argument("--max_tokens", type=int, default=400_000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--min_active", type=int, default=100,
                    help="minimum firing count before a correlation is reported")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sae = BatchTopKSAE.load(args.sae_ckpt, device=str(dev)).eval()
    scale = getattr(sae, "data_scale", None)
    if scale is None:
        print("[warn] checkpoint has NO data_scale — assuming raw-activation training. "
              "If this SAE was trained with --standardize, results are invalid.")
    else:
        print(f"[ok] applying saved per-channel scale (max/median = "
              f"{(scale.max()/scale.median()).item():.0f}x)")

    shards = sorted(glob.glob(str(Path(args.acts_dir) / "shard_*.npy")))
    nF = sae.n_features
    fire_count = np.zeros(nF, dtype=np.int64)
    act_sum = np.zeros(nF, dtype=np.float64)
    # streaming co-moments vs the SW channel
    n_tot = 0
    sw_sum = 0.0; sw_sq = 0.0
    prod_sum = np.zeros(nF, dtype=np.float64)
    act_sq = np.zeros(nF, dtype=np.float64)

    for sp in shards:
        A = np.load(sp, mmap_mode="r")
        for i in range(0, A.shape[0], args.batch):
            if n_tot >= args.max_tokens:
                break
            blk = np.asarray(A[i:i + args.batch], dtype=np.float32)
            sw_vals = blk[:, args.sw_channel].astype(np.float64)
            x = torch.from_numpy(blk).to(dev)
            if scale is not None:
                x = x / scale
            with torch.no_grad():
                acts = sae.encode(x)["acts"].detach().cpu().numpy().astype(np.float64)
            fire_count += (acts > 0).sum(0)
            act_sum += acts.sum(0)
            act_sq += (acts ** 2).sum(0)
            prod_sum += (acts * sw_vals[:, None]).sum(0)
            sw_sum += sw_vals.sum(); sw_sq += (sw_vals ** 2).sum()
            n_tot += blk.shape[0]
        if n_tot >= args.max_tokens:
            break

    print(f"[ok] streamed {n_tot:,} real tokens")
    sw_mean = sw_sum / n_tot
    sw_var = max(sw_sq / n_tot - sw_mean ** 2, 0.0)
    a_mean = act_sum / n_tot
    a_var = np.maximum(act_sq / n_tot - a_mean ** 2, 0.0)
    cov = prod_sum / n_tot - a_mean * sw_mean
    denom = np.sqrt(a_var * sw_var)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(denom > 1e-12, cov / denom, np.nan)

    freq = fire_count / n_tot
    alive = fire_count > 0
    well = fire_count >= args.min_active

    print(f"  alive (fire at least once): {alive.sum()}/{nF} ({100*alive.mean():.1f}%)")
    print(f"  well-sampled (>= {args.min_active}): {well.sum()} ({100*well.mean():.1f}%)")
    print(f"  median firing rate among alive: {np.median(freq[alive]):.5f}")

    rw = r[well]
    rw = rw[~np.isnan(rw)]
    if rw.size:
        print(f"  |r vs SW channel| over WELL-SAMPLED features: "
              f"median={np.median(np.abs(rw)):.3f}  max={np.abs(rw).max():.3f}")
        top = np.argsort(-np.abs(np.where(well & ~np.isnan(r), r, 0)))[:15]
        print("  top features by |r| (n_active reported alongside — the check "
              "analyze.py omitted):")
        for f in top:
            print(f"    feat {f:6d}  r={r[f]:+.4f}  n_active={fire_count[f]:>8d}  "
                  f"freq={freq[f]:.5f}")
    else:
        print("  no well-sampled features — cannot report correlations honestly")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "real_sequence_features.json").write_text(json.dumps({
        "sae_ckpt": args.sae_ckpt, "n_tokens": int(n_tot),
        "sw_channel": args.sw_channel, "min_active": args.min_active,
        "n_features": int(nF),
        "n_alive": int(alive.sum()), "n_well_sampled": int(well.sum()),
        "median_freq_alive": float(np.median(freq[alive])) if alive.any() else 0.0,
        "abs_r_median_well": float(np.median(np.abs(rw))) if rw.size else None,
        "abs_r_max_well": float(np.abs(rw).max()) if rw.size else None,
        "per_feature": [
            {"feature": int(f), "n_active": int(fire_count[f]), "freq": float(freq[f]),
             "r_with_sw": (None if np.isnan(r[f]) or not well[f] else float(r[f]))}
            for f in range(nF) if alive[f]
        ],
    }, indent=2))
    print(f"saved → {out/'real_sequence_features.json'}")


if __name__ == "__main__":
    main()
