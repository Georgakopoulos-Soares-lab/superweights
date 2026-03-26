#!/usr/bin/env python
from __future__ import annotations

"""General, task-free superactivations scan for Borzoi-PyTorch.

This script identifies channels with heavy-tailed / extreme internal activations across
generic genomic windows (promoters/enhancers/random controls), analogous to "superweights"
style analyses but without using any downstream task labels.

It streams sequences from a genome FASTA and BED region sets, runs Borzoi forward passes,
captures per-window per-channel reduced activations at chosen layers, and computes per-channel
tail metrics and top-activating windows.

Example
-------

/scratch/10906/arisk/envs/superweights-borzoi/bin/python -u scripts/general_superactivations_scan.py \
  --model_name_or_path johahi/borzoi-replicate-0 \
  --genome_fasta data/hg38.fa \
  --regions_promoters_bed data/regions/promoters_262kb.bed \
  --regions_enhancers_bed data/regions/enhancers_262kb.bed \
  --regions_random_bed data/regions/random_262kb.bed \
  --num_windows_per_set 10000 \
  --batch_size 1 \
  --layers horizontal_conv1.conv_layer,separable0.conv_layer.1,final_joined_convs.0.conv_layer \
  --reduce abs_max \
  --quantiles 0.99,0.999 \
  --tail_z 4.0 \
  --top_windows_per_channel 200 \
  --out_dir results/general_superactivations/run1 \
  --seed 1

Outputs
-------
- run_manifest.json
- summary_by_layer.tsv
- per-layer:
  - <layer_sanitized>.channel_metrics.(parquet|tsv)
  - <layer_sanitized>.top_channels.tsv
  - <layer_sanitized>.tail_metric_hist.png
  - <layer_sanitized>.topk_windows_per_channel.parquet
  - <layer_sanitized>.topk_windows_per_channel.bed
- combined:
  - all_layers_top_channels.tsv
  - all_layers_top_windows.parquet

Notes
-----
- Activation tensors are expected to be shaped like [B, C, L] (Conv1d-like).
- Default reduction is abs_max over L (recommended).
- Uses deterministic region subsampling given --seed.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Allow running as `python scripts/...py` from any cwd.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Avoid transformers importing TensorFlow/JAX/Flax if present (can be slow/hang on HPC nodes).
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from superweights_borzoi.regions import Region, iter_region_batches, prepare_region_set
from superweights_borzoi.genome import Genome


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="General superactivations scan (task-free).")

    p.add_argument("--model_name_or_path", type=str, default="johahi/borzoi-replicate-0")
    p.add_argument("--output_key", type=str, default=None)

    p.add_argument("--genome_fasta", type=str, required=True)

    # New, general interface: repeatable region sets.
    p.add_argument(
        "--region_set",
        type=str,
        action="append",
        default=None,
        help="Region set spec like name=path/to/file.bed(.gz). Repeatable.",
    )

    # Backward-compatible interface (deprecated): fixed 3 sets.
    p.add_argument("--regions_promoters_bed", type=str, default=None)
    p.add_argument("--regions_enhancers_bed", type=str, default=None)
    p.add_argument("--regions_random_bed", type=str, default=None)

    p.add_argument("--seq_len", type=int, default=None, help="Override input window length (bp)")
    p.add_argument("--seq_len_bp", type=int, default=None, help="Alias for --seq_len")
    p.add_argument("--num_windows_per_set", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=1)

    p.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated module names to hook. Default: auto-pick 3 conv layers.",
    )
    p.add_argument(
        "--reduce",
        type=str,
        default="absmax",
        choices=["absmax", "abs_max", "max", "mean", "absmean", "abs_mean"],
        help="Reduce activations over length dimension.",
    )
    p.add_argument("--quantiles", type=str, default="0.99,0.999")
    p.add_argument("--tail_z", type=float, default=4.0)
    p.add_argument("--top_channels", type=int, default=50, help="Export top windows only for these channels")
    p.add_argument("--top_windows_per_channel", type=int, default=200)
    p.add_argument("--compute_kurtosis", action="store_true", help="Also compute kurtosis_approx (slower)")
    p.add_argument(
        "--use_memmap",
        action="store_true",
        help="Store reduced activations on disk via numpy.memmap (recommended for large scans)",
    )

    p.add_argument(
        "--checkpoint_every_batches",
        type=int,
        default=50,
        help="Write progress checkpoints every N batches (0 disables).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partially completed run in --out_dir (requires --use_memmap).",
    )

    p.add_argument("--device", type=str, default="auto", help="cuda, cpu, or auto")
    p.add_argument("--seed", type=int, default=1)

    p.add_argument(
        "--respect_strand",
        action="store_true",
        help="If set, reverse-complement sequences for BED regions with strand='-'.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, only sample/validate windows and write windows.{tsv,parquet} + run_manifest.json; skip model forward.",
    )

    p.add_argument("--out_dir", type=str, required=True)

    return p.parse_args()


_RC_TRANS = str.maketrans({"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"})


def _revcomp(seq: str) -> str:
    s = str(seq).upper()
    # Unknown bases -> N
    s = "".join((c if c in {"A", "C", "G", "T", "N"} else "N") for c in s)
    return s.translate(_RC_TRANS)[::-1]


def _parse_csv_str(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    return [p.strip() for p in str(s).split(",") if p.strip()]


def _parse_csv_float(s: Optional[str]) -> List[float]:
    out: List[float] = []
    for part in _parse_csv_str(s):
        out.append(float(part))
    return out


def _sanitize_layer_name(layer: str) -> str:
    return str(layer).replace("/", "_").replace(".", "_")


def _parse_region_set_specs(specs: Optional[Sequence[str]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for s in specs or []:
        if "=" in s:
            name, path = s.split("=", 1)
        elif ":" in s:
            name, path = s.split(":", 1)
        else:
            raise ValueError(f"Invalid --region_set spec (expected name=path): {s}")
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --region_set spec: {s}")
        out.append((name, path))
    return out


def _stable_hash32(s: str) -> int:
    """Stable 32-bit hash for reproducible per-set RNG seeds."""
    return int(zlib.crc32(str(s).encode("utf-8")) & 0xFFFFFFFF)


def _git_commit() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_PROJECT_ROOT),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return proc.stdout.decode("utf-8").strip()
    except Exception:
        return None


def _compute_kurtosis_approx(a: np.ndarray, mean: np.ndarray, std: np.ndarray, chunk_rows: int = 2048) -> np.ndarray:
    """Compute per-channel kurtosis in a memory-safe way.

    kurtosis = E[(x-mu)^4] / std^4
    """
    N, C = a.shape
    mu = mean.astype(np.float64)
    sd = std.astype(np.float64)
    denom = np.where(sd > 0, sd**4, np.nan)

    m4 = np.zeros((C,), dtype=np.float64)
    for lo in range(0, N, int(chunk_rows)):
        hi = min(N, lo + int(chunk_rows))
        x = a[lo:hi].astype(np.float64, copy=False)
        d = x - mu[None, :]
        m4 += np.sum(d**4, axis=0)

    m4 = m4 / float(N)
    out = m4 / denom
    return out.astype(np.float64)


def _write_hist_png(values: np.ndarray, title: str, out_png: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4))
        v = np.asarray(values, dtype=np.float64)
        v = v[np.isfinite(v)]
        plt.hist(v, bins=60)
        plt.title(title)
        plt.xlabel("super_score")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
    except Exception as e:
        logger.warning(f"Plotting failed: {e}")


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    _atomic_write_text(Path(path), json.dumps(payload, indent=2))


def _atomic_write_npy(path: Path, arr: np.ndarray) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.save(tmp, arr)
    # np.save adds .npy if not present; ensure we replace the target path.
    tmp_on_disk = tmp
    if not str(tmp).endswith(".npy") and (tmp.parent / (tmp.name + ".npy")).exists():
        tmp_on_disk = tmp.parent / (tmp.name + ".npy")
    os.replace(tmp_on_disk, path)


def main() -> None:
    args = parse_args()

    np.random.seed(int(args.seed))

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if bool(args.resume) and not bool(args.use_memmap):
        raise SystemExit("--resume requires --use_memmap")

    quantiles = _parse_csv_float(args.quantiles)
    if not quantiles:
        raise SystemExit("--quantiles parsed empty")
    for q in quantiles:
        if not (0.0 < q < 1.0):
            raise SystemExit(f"Invalid quantile: {q}")

    genome = Genome(Path(args.genome_fasta))

    # seq_len and layers: for --dry_run we must be explicit to avoid loading torch/model code.
    seq_len_arg = args.seq_len if args.seq_len is not None else args.seq_len_bp
    layers_arg = _parse_csv_str(args.layers)

    device = None
    wrapper = None
    layers: List[str] = []

    if bool(args.dry_run):
        if seq_len_arg is None:
            raise SystemExit("--dry_run requires --seq_len (or --seq_len_bp)")
        seq_len = int(seq_len_arg)
        layers = list(layers_arg)
        logger.info("Dry run: skipping model load and forward passes")
    else:
        import torch

        from superweights_borzoi.encoding import one_hot_encode_batch
        from superweights_borzoi.models.borzoi_pt import (
            detect_seq_len,
            detect_seq_len_from_crop,
            get_module_by_name,
            load_borzoi,
        )

        torch.manual_seed(int(args.seed))

        device_s = str(args.device).strip().lower()
        device_auto = device_s == "auto"
        if device_s == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_s)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--device cuda but CUDA is not available")

        logger.info(f"Device: {device}")
        if device.type == "cuda":
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

        # Hook helpers (torch-dependent)
        def _as_tensor(x):
            if torch.is_tensor(x):
                return x
            if isinstance(x, (tuple, list)) and x and torch.is_tensor(x[0]):
                return x[0]
            return None

        class _LayerReducer:
            def __init__(self, model: torch.nn.Module, layer_names: Sequence[str], reduce: str) -> None:
                self.model = model
                self.layer_names = list(layer_names)
                self.reduce = str(reduce)
                self._handles: List[torch.utils.hooks.RemovableHandle] = []
                self._current: Dict[str, np.ndarray] = {}

            def _hook(self, layer_name: str):
                def fn(_module, _inp, out):
                    t = _as_tensor(out)
                    if t is None:
                        logger.warning(f"Layer {layer_name} hook: non-tensor output")
                        return
                    if t.ndim < 3:
                        logger.warning(f"Layer {layer_name} hook: expected [B,C,L], got shape {tuple(t.shape)}")
                        return

                    x = t
                    if self.reduce in {"absmax", "abs_max"}:
                        red = x.abs().amax(dim=-1)
                    elif self.reduce == "max":
                        red = x.amax(dim=-1)
                    elif self.reduce == "mean":
                        red = x.mean(dim=-1)
                    elif self.reduce in {"absmean", "abs_mean"}:
                        red = x.abs().mean(dim=-1)
                    else:
                        raise ValueError(f"Unknown reduce: {self.reduce}")

                    self._current[layer_name] = red.detach().to("cpu", dtype=torch.float32).numpy()

                return fn

            def register(self) -> None:
                self.close()
                self._current = {}
                for ln in self.layer_names:
                    module = get_module_by_name(self.model, str(ln))
                    self._handles.append(module.register_forward_hook(self._hook(str(ln))))
                logger.info(f"Registered {len(self._handles)} activation hooks")

            def pop(self) -> Dict[str, np.ndarray]:
                out = self._current
                self._current = {}
                return out

            def close(self) -> None:
                for h in self._handles:
                    try:
                        h.remove()
                    except Exception:
                        pass
                self._handles = []

            def __enter__(self):
                self.register()
                return self

            def __exit__(self, exc_type, exc, tb):
                self.close()
                return False

        def _auto_pick_layers(model: torch.nn.Module, n: int = 3) -> List[str]:
            """Pick some reasonable conv layers without hard-coding a specific architecture."""

            preferred_early = "horizontal_conv1.conv_layer"
            preferred_late = "final_joined_convs.0.conv_layer"

            conv_candidates = [name for name, _m in model.named_modules() if name.endswith("conv_layer")]
            conv_candidates = [c for c in conv_candidates if c]

            picked: List[str] = []

            def _maybe_add(name: str) -> None:
                if not name or name in picked:
                    return
                try:
                    get_module_by_name(model, name)
                except Exception:
                    return
                picked.append(name)

            _maybe_add(preferred_early)
            if conv_candidates:
                _maybe_add(conv_candidates[len(conv_candidates) // 2])
            _maybe_add(preferred_late)
            for name in conv_candidates:
                if len(picked) >= int(n):
                    break
                _maybe_add(name)
            if not picked:
                raise ValueError("Failed to auto-pick layers; provide --layers explicitly")
            return picked[: int(n)]

        # Load model early so we can auto-detect seq_len and layers.
        try:
            wrapper = load_borzoi(str(args.model_name_or_path), device=device, output_key=args.output_key)
        except RuntimeError as e:
            msg = str(e).lower()
            if device_auto and device.type == "cuda" and ("busy" in msg or "unavailable" in msg):
                logger.warning(f"CUDA appears busy/unavailable; falling back to CPU (error was: {e})")
                device = torch.device("cpu")
                wrapper = load_borzoi(str(args.model_name_or_path), device=device, output_key=args.output_key)
            else:
                raise

        seq_len = int(seq_len_arg) if seq_len_arg is not None else None
        if seq_len is None:
            seq_len = detect_seq_len(wrapper.model)
        if seq_len is None:
            seq_len = detect_seq_len_from_crop(wrapper.model)
        if seq_len is None:
            raise SystemExit("Could not detect sequence length; provide --seq_len")
        seq_len = int(seq_len)
        logger.info(f"seq_len={seq_len}")

        layers = list(layers_arg)
        if not layers:
            layers = _auto_pick_layers(wrapper.model, n=3)
        logger.info(f"Layers: {layers}")

    # Prepare region sets
    region_sets = _parse_region_set_specs(args.region_set)
    if not region_sets:
        # Backward-compatible (deprecated) mode
        if not args.regions_promoters_bed:
            raise SystemExit("Provide either --region_set name=bed (repeatable) or --regions_promoters_bed")
        region_sets = [("promoters", str(args.regions_promoters_bed))]
        if args.regions_enhancers_bed:
            region_sets.append(("enhancers", str(args.regions_enhancers_bed)))
        if args.regions_random_bed:
            region_sets.append(("random", str(args.regions_random_bed)))

    all_windows: List[Region] = []
    for name, bed_path in region_sets:
        win = prepare_region_set(
            bed_path,
            genome=genome,
            region_set=name,
            width=int(seq_len),
            n=int(args.num_windows_per_set),
            seed=int(args.seed) + (_stable_hash32(name) % 10_000_000),
        )
        if not win:
            raise RuntimeError(f"No valid windows for region_set={name} from BED: {bed_path}")
        win = sorted(win, key=lambda r: (r.region_set, r.chrom, int(r.start0), int(r.end0)))
        all_windows.extend(win)

    if not all_windows:
        raise RuntimeError("No windows selected")

    # Stable ordering for reproducibility
    all_windows = list(all_windows)
    logger.info(f"Total windows: {len(all_windows)} across sets={len(region_sets)}")

    N_total = int(len(all_windows))

    # Persist window metadata
    df_meta = pd.DataFrame(
        {
            "idx": np.arange(len(all_windows), dtype=np.int32),
            "region_set": [w.region_set for w in all_windows],
            "chrom": [w.chrom for w in all_windows],
            "start0": [int(w.start0) for w in all_windows],
            "end0": [int(w.end0) for w in all_windows],
            "strand": [w.strand for w in all_windows],
        }
    )
    meta_path = outdir / "windows.parquet"
    df_meta.to_parquet(meta_path, index=False)
    df_meta.to_csv(outdir / "windows.tsv", sep="\t", index=False)

    # Resume: load prior processed mask if present.
    processed_mask_path = outdir / "processed_mask.npy"
    checkpoint_path = outdir / "scan_checkpoint.json"

    if bool(args.dry_run):
        manifest = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": _git_commit(),
            "args": vars(args),
            "layers": list(layers),
            "quantiles": list(quantiles),
            "region_sets": [{"name": n, "bed": p} for n, p in region_sets],
            "seq_len": int(seq_len),
            "n_total_windows": int(N_total),
            "n_processed_windows": 0,
            "skipped_fetch": 0,
            "windows_metadata": str(meta_path),
            "activations_memmap": {},
            "outputs": {
                "summary_by_layer": None,
                "all_layers_top_channels": None,
                "all_layers_top_windows": None,
            },
        }
        _atomic_write_json(outdir / "run_manifest.json", manifest)
        logger.info(f"Dry run complete. Wrote windows to: {outdir}")
        return

    # Forward + capture reduced activations
    acts_by_layer: Dict[str, np.ndarray] = {}
    acts_shape: Dict[str, Tuple[int, int]] = {}
    acts_paths: Dict[str, str] = {}
    processed = np.zeros((N_total,), dtype=bool)
    if bool(args.resume) and processed_mask_path.exists():
        try:
            processed = np.load(processed_mask_path).astype(bool)
            if processed.shape != (N_total,):
                raise ValueError(f"processed_mask shape mismatch: {processed.shape} vs ({N_total},)")
            logger.info(f"Resuming: loaded processed mask with {int(processed.sum())}/{N_total} done")
        except Exception as e:
            raise SystemExit(f"Failed to load processed mask for resume: {e}")

    # Resume: if memmaps already exist, open them now so we can compute metrics even
    # in the case where all windows are already processed.
    if bool(args.resume) and bool(args.use_memmap):
        ck_shapes: Dict[str, Tuple[int, int]] = {}
        if checkpoint_path.exists():
            try:
                ck = json.loads(checkpoint_path.read_text())
                shapes = ck.get("acts_shape", {})
                for ln, shp in shapes.items():
                    if isinstance(shp, (list, tuple)) and len(shp) == 2:
                        ck_shapes[str(ln)] = (int(shp[0]), int(shp[1]))
            except Exception as e:
                logger.warning(f"Could not parse checkpoint shapes (will infer from file sizes): {e}")

        opened = 0
        for layer_name in layers:
            safe = _sanitize_layer_name(layer_name)
            p = outdir / f"{safe}.acts.float32.memmap"
            if not p.exists():
                continue
            if layer_name in acts_by_layer:
                continue

            C = None
            if layer_name in ck_shapes and ck_shapes[layer_name][0] == int(N_total):
                C = int(ck_shapes[layer_name][1])
            else:
                # Infer C from file size.
                bytes_ = int(p.stat().st_size)
                denom = int(N_total) * 4
                if denom > 0 and bytes_ % denom == 0:
                    C = int(bytes_ // denom)
            if C is None or C <= 0:
                raise RuntimeError(f"Could not infer channel count for existing memmap: {p}")

            mm = np.memmap(p, dtype=np.float32, mode="r+", shape=(N_total, int(C)))
            acts_by_layer[layer_name] = mm
            acts_shape[layer_name] = (int(N_total), int(C))
            acts_paths[layer_name] = str(p)
            opened += 1

        if opened:
            logger.info(f"Resuming: opened {opened} existing activation memmaps")

    stopping = {"flag": False}

    def _checkpoint(reason: str, *, skipped_fetch: int) -> None:
        try:
            if bool(args.use_memmap):
                for _ln, mm in acts_by_layer.items():
                    try:
                        mm.flush()  # type: ignore[attr-defined]
                    except Exception:
                        pass
            _atomic_write_npy(processed_mask_path, processed)
            payload = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "reason": str(reason),
                "n_total_windows": int(N_total),
                "n_processed_windows": int(processed.sum()),
                "skipped_fetch": int(skipped_fetch),
                "layers": list(layers),
                "acts_paths": dict(acts_paths),
                "acts_shape": {k: [int(v[0]), int(v[1])] for k, v in acts_shape.items()},
            }
            _atomic_write_json(checkpoint_path, payload)
        except Exception as e:
            logger.warning(f"Checkpoint failed: {e}")

    def _handle_signal(signum, _frame):
        logger.warning(f"Received signal {signum}; checkpointing then exiting")
        stopping["flag"] = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    def _ensure_store(layer_name: str, C: int) -> np.ndarray:
        if layer_name in acts_by_layer:
            return acts_by_layer[layer_name]
        safe = _sanitize_layer_name(layer_name)
        if bool(args.use_memmap):
            p = outdir / f"{safe}.acts.float32.memmap"
            mode = "w+"
            if p.exists() and bool(args.resume):
                mode = "r+"
                # Validate file size matches expected shape.
                expected_bytes = int(N_total) * int(C) * 4
                actual_bytes = int(p.stat().st_size)
                if actual_bytes != expected_bytes:
                    # Try to infer C from file size for a clearer error.
                    if actual_bytes % (int(N_total) * 4) == 0:
                        inferred_c = actual_bytes // (int(N_total) * 4)
                        raise RuntimeError(
                            f"Existing memmap size mismatch for {layer_name}: file implies C={inferred_c} but current C={C}"
                        )
                    raise RuntimeError(f"Existing memmap size mismatch for {layer_name}: bytes={actual_bytes} expected={expected_bytes}")
            mm = np.memmap(p, dtype=np.float32, mode=mode, shape=(N_total, int(C)))
            acts_paths[layer_name] = str(p)
        else:
            mm = np.empty((N_total, int(C)), dtype=np.float32)
            acts_paths[layer_name] = ""
        acts_by_layer[layer_name] = mm
        acts_shape[layer_name] = (N_total, int(C))
        if bool(args.use_memmap):
            logger.info(f"Allocated memmap for {layer_name}: shape=({N_total},{C})")
        else:
            logger.info(f"Allocated in-memory store for {layer_name}: shape=({N_total},{C})")
        return mm

    skipped_fetch = 0

    # Write an initial manifest early (useful for preemption debugging).
    init_manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _git_commit(),
        "args": vars(args),
        "layers": list(layers),
        "quantiles": list(quantiles),
        "region_sets": [{"name": n, "bed": p} for n, p in region_sets],
        "seq_len": int(seq_len),
        "n_total_windows": int(N_total),
        "n_processed_windows": int(processed.sum()),
        "skipped_fetch": int(skipped_fetch),
        "windows_metadata": str(meta_path),
        "activations_memmap": acts_paths,
        "outputs": {
            "summary_by_layer": None,
            "all_layers_top_channels": None,
            "all_layers_top_windows": None,
        },
    }
    _atomic_write_json(outdir / "run_manifest.json", init_manifest)

    with _LayerReducer(wrapper.model, layers, reduce=str(args.reduce)) as cap:
        bs = int(args.batch_size)
        for start in tqdm(range(0, N_total, bs), desc="windows"):
            if stopping["flag"]:
                _checkpoint("signal", skipped_fetch=skipped_fetch)
                raise SystemExit("Stopped by signal")

            batch = all_windows[start : start + bs]

            seqs: List[str] = []
            keep_idx: List[int] = []
            for i, w in enumerate(batch):
                global_idx = start + i
                if processed[int(global_idx)]:
                    continue
                try:
                    seq = genome.fetch_sequence(chrom=w.chrom, start0=int(w.start0), end0=int(w.end0))
                except Exception:
                    skipped_fetch += 1
                    continue
                if len(seq) != int(seq_len):
                    skipped_fetch += 1
                    continue
                if bool(args.respect_strand) and str(w.strand) == "-":
                    seq = _revcomp(seq)
                seqs.append(seq)
                keep_idx.append(int(global_idx))

            if not seqs:
                continue

            x = one_hot_encode_batch(seqs, device=device)
            with torch.no_grad():
                _ = wrapper.model(x)

            reduced = cap.pop()  # layer -> [B,C]

            for layer_name, a in reduced.items():
                mm = _ensure_store(layer_name, int(a.shape[1]))
                for j, global_idx in enumerate(keep_idx):
                    mm[int(global_idx), :] = a[int(j), :]

            for global_idx in keep_idx:
                processed[int(global_idx)] = True

            if (start // bs) % 50 == 0:
                logger.info(f"Progress: {min(start+bs, N_total)}/{N_total} windows")

            ck_every = int(args.checkpoint_every_batches)
            if ck_every > 0 and ((start // bs) % ck_every == 0):
                _checkpoint("periodic", skipped_fetch=skipped_fetch)

    if skipped_fetch:
        logger.warning(f"Skipped windows due to FASTA fetch/length errors: {skipped_fetch}")

    idx_proc = np.where(processed)[0]
    if idx_proc.size == 0:
        raise RuntimeError("No windows were successfully processed")
    if idx_proc.size != N_total:
        logger.warning(f"Processed windows: {int(idx_proc.size)}/{N_total} (some windows missing/zeroed)")

    # Compute metrics per layer
    summary_rows: List[dict] = []
    all_top_channels_rows: List[dict] = []
    all_top_windows_rows: List[pd.DataFrame] = []

    for layer_name in layers:
        if layer_name not in acts_by_layer:
            raise RuntimeError(f"No activations captured for layer: {layer_name}")

        mm = acts_by_layer[layer_name]
        a_full = np.asarray(mm)
        a = a_full[idx_proc, :]
        N, C = a.shape

        mean = a.mean(axis=0).astype(np.float64)
        std = a.std(axis=0).astype(np.float64)

        qs = np.quantile(a, q=quantiles, axis=0).astype(np.float64)  # [Q,C]
        qcols: List[str] = []
        for q in quantiles:
            if abs(q - 0.99) < 1e-12:
                qcols.append("q99")
            elif abs(q - 0.999) < 1e-12:
                qcols.append("q999")
            else:
                qcols.append(f"q{int(round(q*1000))}")

        # Quantile-based tail ratios (often helpful as a simple heavy-tail score)
        # Use the highest and lowest provided quantiles if there are >=2.
        q_high = qs[-1, :].astype(np.float64)
        q_low = qs[0, :].astype(np.float64)
        tail_ratio_qhigh_qlow = q_high / np.where(q_low != 0, q_low, np.nan)

        thresh = mean + float(args.tail_z) * std
        tail_freq_z = (a > thresh[None, :]).mean(axis=0).astype(np.float64)

        # tail ratio: mean(top 1%) / median (vectorized)
        k_top = max(1, int(np.ceil(0.01 * float(N))))
        med = np.median(a, axis=0).astype(np.float64)
        topk = np.partition(a, kth=N - k_top, axis=0)[N - k_top :, :]
        top_mean = topk.mean(axis=0).astype(np.float64)
        tail_ratio_top1pct = top_mean / np.where(med != 0, med, np.nan)

        tail_ratio_qhigh_median = q_high / np.where(med != 0, med, np.nan)

        kurt = None
        if bool(args.compute_kurtosis):
            kurt = _compute_kurtosis_approx(a=a, mean=mean, std=std, chunk_rows=2048)

        # super_score default: highest quantile
        super_score = qs[-1, :].astype(np.float64)

        df_m = pd.DataFrame({"channel": np.arange(C, dtype=np.int32)})
        df_m["mean"] = mean
        df_m["std"] = std
        for i, col in enumerate(qcols):
            df_m[col] = qs[i, :]
        df_m["tail_freq_z"] = tail_freq_z
        df_m["tail_ratio_top1pct"] = tail_ratio_top1pct
        df_m["tail_ratio_qhigh_qlow"] = tail_ratio_qhigh_qlow
        df_m["tail_ratio_qhigh_median"] = tail_ratio_qhigh_median
        if kurt is not None:
            df_m["kurtosis_approx"] = kurt
        df_m["super_score"] = super_score

        df_m = df_m.sort_values("super_score", ascending=False).reset_index(drop=True)

        safe = _sanitize_layer_name(layer_name)
        out_parq = outdir / f"{safe}.channel_metrics.parquet"
        out_tsv = outdir / f"{safe}.channel_metrics.tsv"
        df_m.to_parquet(out_parq, index=False)
        df_m.to_csv(out_tsv, sep="\t", index=False)

        df_top = df_m.head(int(args.top_channels)).copy()
        out_top = outdir / f"{safe}.top_channels.tsv"
        df_top.to_csv(out_top, sep="\t", index=False)

        out_png = outdir / f"{safe}.tail_metric_hist.png"
        _write_hist_png(values=df_m["super_score"].values, title=f"Super-score histogram: {layer_name}", out_png=out_png)

        summary_rows.append(
            {
                "layer": str(layer_name),
                "reduce": str(args.reduce),
                "n_windows": int(N),
                "channels": int(C),
                "quantiles": str(args.quantiles),
                "tail_z": float(args.tail_z),
                "top_channels": int(args.top_channels),
                "top_windows_per_channel": int(args.top_windows_per_channel),
            }
        )

        # Export topK windows per channel
        K = int(args.top_windows_per_channel)
        top_rows: List[dict] = []
        top_channels = [int(x) for x in df_top["channel"].astype(int).tolist()]
        for ch in top_channels:
            col = a[:, ch]
            if K >= N:
                local_idx = np.argsort(col)[::-1]
            else:
                local_idx = np.argpartition(col, -K)[-K:]
                local_idx = local_idx[np.argsort(col[local_idx])[::-1]]

            for rank, li in enumerate(local_idx[:K], start=1):
                global_wi = int(idx_proc[int(li)])
                w = all_windows[global_wi]
                top_rows.append(
                    {
                        "layer": str(layer_name),
                        "channel": int(ch),
                        "rank": int(rank),
                        "region_set": str(w.region_set),
                        "chrom": str(w.chrom),
                        "start0": int(w.start0),
                        "end0": int(w.end0),
                        "strand": str(w.strand),
                        "activation_value": float(col[int(li)]),
                    }
                )

        df_topw = pd.DataFrame(top_rows)
        out_topw = outdir / f"{safe}.topk_windows_per_channel.parquet"
        df_topw.to_parquet(out_topw, index=False)

        # Optional BED export
        out_bed = outdir / f"{safe}.topk_windows_per_channel.bed"
        with out_bed.open("w") as f:
            for r in top_rows:
                name = f"{safe}|ch{r['channel']}|rank{r['rank']}|set{r['region_set']}"
                # BED score must be 0..1000. We just emit 0; the exact activation is in the parquet.
                f.write(
                    "\t".join(
                        [
                            str(r["chrom"]),
                            str(int(r["start0"])),
                            str(int(r["end0"])),
                            name,
                            "0",
                            str(r["strand"]),
                        ]
                    )
                    + "\n"
                )

        # Combined top-channels
        df_top_layer = df_m.head(int(args.top_channels)).copy()
        df_top_layer.insert(0, "layer", str(layer_name))
        all_top_channels_rows.append(df_top_layer)

        # Combined top windows for those top channels
        top_layer_channels = set(int(x) for x in df_top_layer["channel"].astype(int).tolist())
        df_topw_small = df_topw[df_topw["channel"].astype(int).isin(top_layer_channels)].copy()
        all_top_windows_rows.append(df_topw_small)

    df_sum = pd.DataFrame(summary_rows)
    (outdir / "summary_by_layer.tsv").write_text(df_sum.to_csv(sep="\t", index=False))

    if all_top_channels_rows:
        df_all_top = pd.concat(all_top_channels_rows, axis=0, ignore_index=True)
        df_all_top = df_all_top.sort_values("super_score", ascending=False).reset_index(drop=True)
        df_all_top.to_csv(outdir / "all_layers_top_channels.tsv", sep="\t", index=False)
        df_all_top.to_parquet(outdir / "all_layers_top_channels.parquet", index=False)

    if all_top_windows_rows:
        df_all_w = pd.concat(all_top_windows_rows, axis=0, ignore_index=True)
        df_all_w.to_parquet(outdir / "all_layers_top_windows.parquet", index=False)

    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _git_commit(),
        "args": vars(args),
        "layers": list(layers),
        "quantiles": list(quantiles),
        "region_sets": [{"name": n, "bed": p} for n, p in region_sets],
        "seq_len": int(seq_len),
        "n_total_windows": int(N_total),
        "n_processed_windows": int(idx_proc.size),
        "skipped_fetch": int(skipped_fetch),
        "windows_metadata": str(meta_path),
        "activations_memmap": acts_paths,
        "outputs": {
            "summary_by_layer": str(outdir / "summary_by_layer.tsv"),
            "all_layers_top_channels": str(outdir / "all_layers_top_channels.tsv"),
            "all_layers_top_windows": str(outdir / "all_layers_top_windows.parquet"),
        },
    }
    _atomic_write_json(outdir / "run_manifest.json", manifest)

    # Final checkpoint artifacts.
    _checkpoint("final", skipped_fetch=skipped_fetch)

    logger.info(f"Done. Wrote results to: {outdir}")


if __name__ == "__main__":
    main()
