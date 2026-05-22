# sae/train.py
"""
Train a Batch-TopK SAE on pre-collected GENERator activation shards.

After training, runs two sanity checks:
  (a) Histogram of feature activation frequencies (should be highly sparse)
  (b) Reconstruction MSE vs a random-decoder baseline

Usage
-----
  python sae/train.py \\
      --acts_dir  data/sae_acts/generator_euk_layer4  \\
      --out_dir   results/sae/generator_euk_layer4    \\
      --dict_mult 4                                   \\   # n_features = dict_mult × d_in
      --k         64                                  \\   # TopK budget per token
      --lr        2e-4                                \\
      --steps     200_000                             \\
      --batch     2048                                \\
      --log_every 1000                                \\
      --save_every 20000

Checkpoints are saved as:
  <out_dir>/sae_step_<N>.pt
  <out_dir>/sae_final.pt
  <out_dir>/training_log.json
  <out_dir>/sanity_check.png
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from sae.model import BatchTopKSAE


# ──────────────────────────────────────────────────────────────────────────────
# Data loader — streams float16 shards
# ──────────────────────────────────────────────────────────────────────────────

class ShardStreamer:
    """
    Infinite stream of batches sampled from float16 .npy shard files.
    Shards are shuffled, loaded one at a time, and recycled.
    """

    def __init__(self, acts_dir: str, batch_size: int, seed: int = 42):
        self.acts_dir   = Path(acts_dir)
        self.batch_size = batch_size
        self.rng        = random.Random(seed)
        self.np_rng     = np.random.default_rng(seed)

        self.shard_paths = sorted(self.acts_dir.glob("shard_*.npy"))
        if not self.shard_paths:
            raise FileNotFoundError(
                f"No shard files found in {acts_dir}.  "
                "Run sae/collect.py first."
            )
        print(f"[ShardStreamer] {len(self.shard_paths)} shards in {acts_dir}")

        meta_path = self.acts_dir / "meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.hidden_size = self.meta.get("hidden_size", None)

        self._buf    = None   # current shard in memory
        self._buf_idx = 0
        self._queue  = []

    def _refill_queue(self):
        order = list(self.shard_paths)
        self.rng.shuffle(order)
        self._queue = order

    def _load_next_shard(self):
        if not self._queue:
            self._refill_queue()
        path = self._queue.pop()
        arr  = np.load(str(path)).astype(np.float32)   # [N, D]
        self.np_rng.shuffle(arr)
        self._buf     = arr
        self._buf_idx = 0
        if self.hidden_size is None:
            self.hidden_size = arr.shape[1]

    def next_batch(self) -> torch.Tensor:
        """Return a [batch_size, hidden_size] float32 tensor."""
        if self._buf is None or self._buf_idx + self.batch_size > len(self._buf):
            self._load_next_shard()
        batch = self._buf[self._buf_idx : self._buf_idx + self.batch_size]
        self._buf_idx += self.batch_size
        t = torch.from_numpy(batch)
        # Guard: super-weight channels can overflow float16 → Inf → NaN loss.
        # Clip any residual Inf/NaN values before they reach the SAE.
        return torch.nan_to_num(t, nan=0.0, posinf=60000.0, neginf=-60000.0)


# ──────────────────────────────────────────────────────────────────────────────
# Online mean / std estimator (Welford)
# ──────────────────────────────────────────────────────────────────────────────

class RunningNorm:
    """
    Tracks a running estimate of the per-dimension mean and std of the
    residual stream.  Used to initialise SAE pre_bias and to sanity-check
    the data.
    """

    def __init__(self, d: int):
        self.n    = 0
        self.mean = np.zeros(d, dtype=np.float64)
        self.M2   = np.zeros(d, dtype=np.float64)

    def update(self, x: np.ndarray):
        """x: [B, D]"""
        for row in x:
            self.n   += 1
            delta     = row - self.mean
            self.mean += delta / self.n
            self.M2   += delta * (row - self.mean)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.M2 / max(self.n - 1, 1))


# ──────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ──────────────────────────────────────────────────────────────────────────────

def run_sanity_checks(sae: BatchTopKSAE, streamer: ShardStreamer, out_path: str,
                      n_eval_batches: int = 500):
    """
    Two sanity checks:
      (a) Feature activation frequency histogram
      (b) Reconstruction MSE vs random-decoder baseline
    Saves a two-panel PNG.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[sanity] matplotlib not available — skipping plot.")
        return

    device = next(sae.parameters()).device
    sae.eval()

    freq      = np.zeros(sae.n_features, dtype=np.float32)
    recon_mse = 0.0
    rand_mse  = 0.0
    n_tokens  = 0

    # Random decoder baseline: unit-norm columns, random orientation
    W_rand = torch.nn.functional.normalize(
        torch.randn(sae.d_in, sae.n_features, device=device), dim=0
    )

    with torch.no_grad():
        for _ in range(n_eval_batches):
            x = streamer.next_batch().to(device)

            out = sae(x, use_aux_loss=False)
            acts = out["acts"]                           # [B, n_features]

            freq  += (acts > 0).float().cpu().numpy().sum(axis=0)
            recon_mse += out["recon_loss"].item() * x.shape[0]

            # Random baseline reconstruction
            z_rand  = sae._topk_sparse(torch.relu(x @ W_rand), sae.k)
            x_rand  = z_rand @ W_rand.T
            rand_mse += torch.nn.functional.mse_loss(x_rand, x).item() * x.shape[0]

            n_tokens += x.shape[0]

    freq      /= n_tokens          # activation frequency per feature
    recon_mse /= n_tokens
    rand_mse  /= n_tokens

    print(f"\n[sanity] recon_MSE={recon_mse:.6f}  rand_MSE={rand_mse:.6f}")
    print(f"[sanity] median_freq={np.median(freq):.5f}  "
          f"frac_dead={( freq == 0 ).mean():.3f}  "
          f"frac_rare=(freq<1e-4)={(freq < 1e-4).mean():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # (a) Feature activation frequency histogram
    ax = axes[0]
    ax.hist(freq, bins=100, log=True, color="steelblue", edgecolor="none")
    ax.axvline(sae.k / sae.n_features, color="red", linestyle="--",
               label=f"k/n = {sae.k}/{sae.n_features}")
    ax.set_xlabel("Feature activation frequency")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Feature activation frequency\n(sparse firing → long left tail)")
    ax.legend(fontsize=8)
    ax.text(0.98, 0.98,
            f"dead: {(freq == 0).mean()*100:.1f}%\n"
            f"rare (<1e-4): {(freq < 1e-4).mean()*100:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # (b) MSE comparison bar chart
    ax = axes[1]
    bars = ax.bar(["SAE recon", "Random decoder"], [recon_mse, rand_mse],
                  color=["steelblue", "salmon"])
    ax.set_ylabel("MSE")
    ax.set_title("Reconstruction MSE vs random baseline")
    for bar, val in zip(bars, [recon_mse, rand_mse]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.5f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[sanity] Plot saved to {out_path}")

    return {"recon_mse": recon_mse, "rand_mse": rand_mse, "feature_freq": freq.tolist()}


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Data ──────────────────────────────────────────────────────────────────
    streamer = ShardStreamer(args.acts_dir, args.batch)
    d_in     = streamer.hidden_size
    if d_in is None:
        # Probe one batch to detect hidden size
        x    = streamer.next_batch()
        d_in = x.shape[1]

    n_features = d_in * args.dict_mult
    print(f"\n[train] d_in={d_in}  n_features={n_features}  k={args.k}")
    print(f"[train] steps={args.steps}  batch={args.batch}  lr={args.lr}")

    # ── Estimate data mean for pre_bias initialisation ─────────────────────────
    print("[train] Estimating data mean (1000 batches)...", flush=True)
    norm = RunningNorm(d_in)
    for _ in range(min(1000, args.steps // 4)):
        norm.update(streamer.next_batch().numpy())
    data_mean = torch.tensor(norm.mean, dtype=torch.float32)

    # ── SAE ───────────────────────────────────────────────────────────────────
    sae = BatchTopKSAE(d_in=d_in, n_features=n_features, k=args.k).to(device)
    with torch.no_grad():
        sae.pre_bias.data = data_mean.to(device)

    # Initialise decoder columns to unit norm (already done in __init__)
    # Tie encoder weights to decoder transpose initially (optional warm start)
    with torch.no_grad():
        sae.W_enc.weight.data = sae.W_dec.data.T.clone()

    optimizer = optim.Adam(sae.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    # LR warmup: linear ramp over first 1000 steps
    warmup_steps = min(1000, args.steps // 20)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )

    # ── Training ──────────────────────────────────────────────────────────────
    log = []
    sae.train()

    print(f"\n[train] Starting training on {device}...\n", flush=True)

    for step in range(1, args.steps + 1):
        x = streamer.next_batch().to(device)

        out = sae(x, use_aux_loss=True, lambda_aux=args.lambda_aux)

        optimizer.zero_grad()
        out["loss"].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)

        optimizer.step()
        if step <= warmup_steps:
            scheduler.step()

        # Renormalise decoder columns after each update
        sae.normalise_decoder()

        # Update dead-feature tracker
        with torch.no_grad():
            sae.update_dead_mask(out["acts"].detach())

        if step % args.log_every == 0:
            entry = {
                "step":       step,
                "loss":       out["loss"].item(),
                "recon_loss": out["recon_loss"].item(),
                "aux_loss":   out["aux_loss"].item(),
                "l0":         out["l0"].item(),
                "frac_dead":  out["frac_dead"].item(),
                "lr":         optimizer.param_groups[0]["lr"],
            }
            log.append(entry)
            print(
                f"step {step:>7d} | loss={entry['loss']:.5f} "
                f"recon={entry['recon_loss']:.5f} "
                f"aux={entry['aux_loss']:.5f} "
                f"L0={entry['l0']:.1f}/{n_features} "
                f"dead={entry['frac_dead']*100:.1f}%",
                flush=True,
            )

        if step % args.save_every == 0:
            ckpt_path = out_dir / f"sae_step_{step:07d}.pt"
            sae.save(str(ckpt_path))
            print(f"  checkpoint → {ckpt_path}", flush=True)

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = out_dir / "sae_final.pt"
    sae.save(str(final_path))
    print(f"\n[train] Final model → {final_path}")

    log_path = out_dir / "training_log.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"[train] Log → {log_path}")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print("\n[train] Running sanity checks...", flush=True)
    sae.eval()
    sanity = run_sanity_checks(
        sae, streamer,
        out_path=str(out_dir / "sanity_check.png"),
    )

    if sanity:
        (out_dir / "sanity_check.json").write_text(json.dumps(sanity, indent=2))

    # Save training config alongside outputs
    config_out = {
        "acts_dir":    args.acts_dir,
        "d_in":        d_in,
        "n_features":  n_features,
        "k":           args.k,
        "dict_mult":   args.dict_mult,
        "lr":          args.lr,
        "steps":       args.steps,
        "batch":       args.batch,
        "lambda_aux":  args.lambda_aux,
    }
    (out_dir / "train_config.json").write_text(json.dumps(config_out, indent=2))

    print("\n[train] Done.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Batch-TopK SAE on GENERator activations.")
    parser.add_argument("--acts_dir",   required=True,
                        help="Directory with shard_*.npy files (from sae/collect.py)")
    parser.add_argument("--out_dir",    required=True,
                        help="Directory for checkpoints, logs, and sanity plots")
    parser.add_argument("--dict_mult",  type=int,   default=4,
                        help="Dictionary size = dict_mult × d_in (default: 4)")
    parser.add_argument("--k",          type=int,   default=64,
                        help="TopK budget per token / target L0 (default: 64)")
    parser.add_argument("--lr",         type=float, default=2e-4,
                        help="Adam learning rate (default: 2e-4)")
    parser.add_argument("--steps",      type=int,   default=200_000,
                        help="Total gradient steps (default: 200 000)")
    parser.add_argument("--batch",      type=int,   default=2048,
                        help="Batch size in tokens (default: 2048)")
    parser.add_argument("--lambda_aux", type=float, default=1/32,
                        help="Auxiliary dead-feature loss weight (default: 1/32)")
    parser.add_argument("--log_every",  type=int,   default=1_000,
                        help="Log interval in steps (default: 1000)")
    parser.add_argument("--save_every", type=int,   default=20_000,
                        help="Checkpoint interval in steps (default: 20 000)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
