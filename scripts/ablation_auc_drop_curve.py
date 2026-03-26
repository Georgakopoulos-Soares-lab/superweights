#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Allow running as `python scripts/...py` from any cwd.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Avoid transformers importing TensorFlow/JAX/Flax if present.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from superweights_borzoi.data import canonicalize_variant_id, parse_canonical_variant_id, sample_posneg_ids
from superweights_borzoi.encoding import one_hot_encode_batch
from superweights_borzoi.genome import Genome, make_ref_alt_sequence
from superweights_borzoi.interpret.ablation import ChannelAblator
from superweights_borzoi.models.borzoi_pt import BorzoiWrapper, detect_seq_len, detect_seq_len_from_crop, load_borzoi


@dataclass(frozen=True)
class TaskWindow:
    out_lo: int
    out_hi: int
    out_bin_bp: float
    win_start_bp: float
    win_end_bp: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute AUC drop vs K for channel ablations tied to a task head defined by track_indices + (pad_bins,shift_bins). "
            "Uses abs(delta_task_readout) by default. Reads top channels from a variant_channel_rank.py run_dir."
        )
    )

    p.add_argument("--rank_run_dir", required=True, type=str, help="variant_channel_rank.py output directory")

    p.add_argument("--pos_ids", required=True, type=str, help="Pos variant_id list")
    p.add_argument("--neg_ids", required=True, type=str, help="Neg variant_id list")
    p.add_argument("--fasta", required=True, type=str, help="Reference genome FASTA")

    p.add_argument("--model_name", type=str, default=None, help="Override model name (default: from rank run report)")
    p.add_argument("--output_key", type=str, default=None, help="Override output_key (default: from rank run report)")
    p.add_argument("--seq_len", type=int, default=None, help="Override seq_len (default: from report/model)")

    p.add_argument(
        "--track_indices",
        type=str,
        default=None,
        help="Comma-separated track indices for the head (default: from rank run report.json)",
    )
    p.add_argument("--pad_bins", type=int, default=None, help="Pad bins for the head window")
    p.add_argument("--shift_bins", type=int, default=None, help="Shift bins for the head window")

    p.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layers to evaluate. Default: layers from report.json",
    )
    p.add_argument("--ks", type=str, default="1,5,10", help="Comma-separated K values")

    p.add_argument("--n_pos", type=str, default="500")
    p.add_argument("--n_neg", type=str, default="500")
    p.add_argument("--batch_size", type=int, default=4)

    p.add_argument("--n_null", type=int, default=10, help="Random channel-set replicates per (layer,K)")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--outdir", type=str, default="results/ablation/auc_drop_curve")

    return p.parse_args()


def _parse_n(s: str) -> Optional[int]:
    if s is None:
        return None
    s = str(s).strip().lower()
    if s in {"all", "none", ""}:
        return None
    return int(s)


def _parse_int_list_csv(s: Optional[str]) -> List[int]:
    if s is None:
        return []
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_str_list_csv(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    return [p.strip() for p in str(s).split(",") if p.strip()]


def _load_id_list(path: str | Path) -> set[str]:
    ids: set[str] = set()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.add(canonicalize_variant_id(line))
    return ids


def _select_output_tensor(output, output_key: Optional[str]) -> torch.Tensor:
    if isinstance(output, dict):
        if output_key is not None:
            if output_key not in output:
                raise KeyError(f"output_key not in output dict: {output_key}")
            output = output[output_key]
        else:
            for v in output.values():
                output = v
                break

    if isinstance(output, (tuple, list)):
        output = output[0]

    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported output type: {type(output)}")

    y = output.float()
    if y.ndim == 2:
        y = y[:, :, None]
    if y.ndim != 3:
        raise RuntimeError(f"Unexpected model output shape: {tuple(y.shape)}")
    return y


def _compute_task_window(seq_len: int, out_bins: int, pad_bins: int, shift_bins: int) -> TaskWindow:
    out_bin_bp = float(seq_len) / float(out_bins)
    center_bp = float(seq_len // 2)

    win_start_bp = center_bp + float(shift_bins - pad_bins) * out_bin_bp
    win_end_bp = center_bp + float(shift_bins + pad_bins + 1) * out_bin_bp

    out_lo = int(np.floor(win_start_bp / out_bin_bp))
    out_hi = int(np.ceil(win_end_bp / out_bin_bp))

    out_lo = max(0, min(out_lo, out_bins))
    out_hi = max(0, min(out_hi, out_bins))
    if out_hi <= out_lo:
        mid = max(0, min(int(np.floor(center_bp / out_bin_bp)), out_bins - 1))
        out_lo, out_hi = mid, mid + 1

    return TaskWindow(out_lo=out_lo, out_hi=out_hi, out_bin_bp=out_bin_bp, win_start_bp=win_start_bp, win_end_bp=win_end_bp)


def _task_readout(y: torch.Tensor, track_idx: torch.Tensor, out_lo: int, out_hi: int) -> torch.Tensor:
    y_sel = y.index_select(dim=1, index=track_idx)
    y_win = y_sel[:, :, out_lo:out_hi]
    return y_win.mean(dim=(1, 2))


@torch.no_grad()
def _score_abs_delta_task(
    df_run: pd.DataFrame,
    wrapper: BorzoiWrapper,
    genome: Genome,
    seq_len: int,
    track_idx_t: torch.Tensor,
    tw: TaskWindow,
    batch_size: int,
    device: torch.device,
    ablate: Optional[Tuple[str, Sequence[int]]] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    skipped_indel = 0
    skipped_ref_mismatch = 0
    skipped_fasta_missing = 0

    y_list: List[np.ndarray] = []
    score_list: List[np.ndarray] = []

    ablator = None
    if ablate is not None:
        layer, channels = ablate
        ablator = ChannelAblator(wrapper.model, layer_name=str(layer), channels=[int(c) for c in channels])
        ablator.register()

    try:
        for start in tqdm(range(0, len(df_run), int(batch_size)), desc="batches", leave=False):
            batch = df_run.iloc[start : start + int(batch_size)]

            ref_seqs: List[str] = []
            alt_seqs: List[str] = []
            keep_rows: List[pd.Series] = []

            for _idx, row in batch.iterrows():
                ref = str(row.get("ref"))
                alt = str(row.get("alt"))
                if len(ref) != 1 or len(alt) != 1:
                    skipped_indel += 1
                    continue

                try:
                    ref_seq, alt_seq = make_ref_alt_sequence(
                        genome=genome,
                        chrom=str(row.get("chrom")),
                        pos1=int(row.get("pos")),
                        ref=ref,
                        alt=alt,
                        seq_len=seq_len,
                        verify_ref=True,
                    )
                except KeyError:
                    skipped_fasta_missing += 1
                    continue
                except ValueError as e:
                    if "Reference mismatch" in str(e):
                        skipped_ref_mismatch += 1
                        continue
                    raise

                ref_seqs.append(ref_seq)
                alt_seqs.append(alt_seq)
                keep_rows.append(row)

            if not keep_rows:
                continue

            y_batch = np.asarray([int(r.get("label")) for r in keep_rows], dtype=np.int32)

            x_ref = one_hot_encode_batch(ref_seqs, device=device)
            y_ref = wrapper.model(x_ref)
            y_ref_t = _select_output_tensor(y_ref, output_key=wrapper.output_key)

            x_alt = one_hot_encode_batch(alt_seqs, device=device)
            y_alt = wrapper.model(x_alt)
            y_alt_t = _select_output_tensor(y_alt, output_key=wrapper.output_key)

            ref_task = _task_readout(y_ref_t, track_idx_t, tw.out_lo, tw.out_hi)
            alt_task = _task_readout(y_alt_t, track_idx_t, tw.out_lo, tw.out_hi)
            delta = (alt_task - ref_task).detach().cpu().numpy().astype(np.float64)
            score = np.abs(delta)

            y_list.append(y_batch)
            score_list.append(score)

    finally:
        if ablator is not None:
            ablator.close()

    y_all = np.concatenate(y_list, axis=0) if y_list else np.asarray([], dtype=np.int32)
    score_all = np.concatenate(score_list, axis=0) if score_list else np.asarray([], dtype=np.float64)

    stats = {
        "n_requested": int(len(df_run)),
        "n_scored": int(len(y_all)),
        "n_pos_scored": int(y_all.sum()) if y_all.size else 0,
        "n_neg_scored": int((y_all == 0).sum()) if y_all.size else 0,
        "skipped_indel": int(skipped_indel),
        "skipped_ref_mismatch": int(skipped_ref_mismatch),
        "skipped_fasta_missing": int(skipped_fasta_missing),
    }
    return y_all, score_all, stats


def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(args.rank_run_dir)
    report_path = run_dir / "report.json"
    if not report_path.exists():
        raise SystemExit(f"Missing report.json in {run_dir}")
    report = json.loads(report_path.read_text())

    model_name = str(args.model_name or report.get("model_name") or "johahi/borzoi-replicate-0")
    output_key = args.output_key if args.output_key is not None else report.get("output_key")

    track_list = _parse_int_list_csv(args.track_indices) if args.track_indices is not None else [int(x) for x in report.get("tracks", [])]
    if not track_list:
        raise SystemExit("Empty track list; provide --track_indices or ensure report.json contains tracks")

    settings = report.get("settings") or []
    pad_bins = args.pad_bins
    shift_bins = args.shift_bins
    if pad_bins is None or shift_bins is None:
        if isinstance(settings, list) and len(settings) == 1 and (pad_bins is None and shift_bins is None):
            pad_bins = int(settings[0]["pad_bins"]) if isinstance(settings[0], dict) else int(settings[0][0])
            shift_bins = int(settings[0]["shift_bins"]) if isinstance(settings[0], dict) else int(settings[0][1])
        else:
            raise SystemExit("Provide --pad_bins and --shift_bins (or use a rank_run_dir with exactly one setting)")

    layers = _parse_str_list_csv(args.layers) if args.layers is not None else [str(x) for x in report.get("layers", [])]
    if not layers:
        raise SystemExit("No layers specified; provide --layers or ensure report.json contains layers")

    ks = _parse_int_list_csv(args.ks)
    if not ks:
        raise SystemExit("--ks parsed empty")

    # Load variant IDs and sample
    pos_ids = _load_id_list(args.pos_ids)
    neg_ids = _load_id_list(args.neg_ids)

    n_pos = _parse_n(args.n_pos)
    n_neg = _parse_n(args.n_neg)

    df_run = sample_posneg_ids(pos_ids, neg_ids, n_pos=n_pos, n_neg=n_neg, seed=int(args.seed))

    chroms, poss, refs, alts = [], [], [], []
    for vid in df_run["variant_id"].tolist():
        c, p1, r, a = parse_canonical_variant_id(str(vid))
        chroms.append(c)
        poss.append(p1)
        refs.append(r)
        alts.append(a)
    df_run["chrom"] = chroms
    df_run["pos"] = poss
    df_run["ref"] = refs
    df_run["alt"] = alts

    logger.info(f"Run variants: n={len(df_run)} pos={int(df_run['label'].sum())} neg={int((df_run['label']==0).sum())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    wrapper: BorzoiWrapper = load_borzoi(model_name, device=device, output_key=output_key)

    seq_len = args.seq_len
    if seq_len is None:
        seq_len = int(report.get("seq_len") or 0) or (detect_seq_len_from_crop(wrapper.model) or detect_seq_len(wrapper.model) or 262144)
    seq_len = int(seq_len)

    # Determine output bins
    dummy = torch.zeros((1, 4, seq_len), device=device, dtype=torch.float32)
    y0t = _select_output_tensor(wrapper.model(dummy), output_key=wrapper.output_key)
    out_bins = int(y0t.shape[-1])

    tw = _compute_task_window(seq_len=seq_len, out_bins=out_bins, pad_bins=int(pad_bins), shift_bins=int(shift_bins))
    track_idx_t = torch.tensor(track_list, dtype=torch.long, device=device)

    genome = Genome(Path(args.fasta))

    # Baseline AUC
    y_base, s_base, stats_base = _score_abs_delta_task(
        df_run=df_run,
        wrapper=wrapper,
        genome=genome,
        seq_len=seq_len,
        track_idx_t=track_idx_t,
        tw=tw,
        batch_size=int(args.batch_size),
        device=device,
        ablate=None,
    )
    if y_base.size == 0 or np.unique(y_base).size < 2:
        raise RuntimeError("Baseline scoring produced no usable labels")

    auc_base = float(roc_auc_score(y_base, s_base))
    logger.info(f"Baseline AUROC(abs Δ(head)) = {auc_base:.4f} (scored n={int(y_base.size)})")

    # Evaluate each layer x K
    rng = np.random.default_rng(int(args.seed))
    rows: List[dict] = []

    for layer in layers:
        layer_key = layer.replace(".", "_")
        best_path = run_dir / f"best_{layer_key}_across_settings.parquet"
        rank_path = run_dir / f"rank_{layer_key}_by_pad_shift.parquet"
        if not best_path.exists():
            raise SystemExit(f"Missing: {best_path}")
        if not rank_path.exists():
            raise SystemExit(f"Missing: {rank_path}")

        df_best = pd.read_parquet(best_path).sort_values("abs_rho", ascending=False).reset_index(drop=True)
        top_channels_all = [int(x) for x in df_best["channel"].tolist()]

        df_rank = pd.read_parquet(rank_path)
        C = int(df_rank["channel"].max()) + 1

        logger.info(f"Layer {layer}: channels={C} top_available={len(top_channels_all)}")

        for k in ks:
            k = int(k)
            topk_channels = top_channels_all[:k]
            if len(topk_channels) < k:
                raise RuntimeError(f"Not enough channels in best file for layer {layer}: need {k}")

            y_ab, s_ab, stats_ab = _score_abs_delta_task(
                df_run=df_run,
                wrapper=wrapper,
                genome=genome,
                seq_len=seq_len,
                track_idx_t=track_idx_t,
                tw=tw,
                batch_size=int(args.batch_size),
                device=device,
                ablate=(layer, topk_channels),
            )
            auc_ab = float(roc_auc_score(y_ab, s_ab))
            delta_auc = float(auc_ab - auc_base)

            # Null distribution: random channels in same layer
            null_deltas: List[float] = []
            universe = np.arange(C, dtype=np.int32)
            exclude = set(top_channels_all[: max(ks)])
            universe = np.asarray([c for c in universe.tolist() if c not in exclude], dtype=np.int32)
            if len(universe) < k:
                universe = np.arange(C, dtype=np.int32)

            for ni in range(int(args.n_null)):
                rand_set = rng.choice(universe, size=k, replace=False).tolist()
                y_n, s_n, _stats_n = _score_abs_delta_task(
                    df_run=df_run,
                    wrapper=wrapper,
                    genome=genome,
                    seq_len=seq_len,
                    track_idx_t=track_idx_t,
                    tw=tw,
                    batch_size=int(args.batch_size),
                    device=device,
                    ablate=(layer, rand_set),
                )
                auc_n = float(roc_auc_score(y_n, s_n))
                null_deltas.append(float(auc_n - auc_base))

            null_arr = np.asarray(null_deltas, dtype=np.float64)
            rows.append(
                {
                    "layer": layer,
                    "k": k,
                    "auc_base": auc_base,
                    "auc_ablate_topk": auc_ab,
                    "delta_auc_topk": delta_auc,
                    "null_mean": float(np.mean(null_arr)) if null_arr.size else float("nan"),
                    "null_p05": float(np.quantile(null_arr, 0.05)) if null_arr.size else float("nan"),
                    "null_p95": float(np.quantile(null_arr, 0.95)) if null_arr.size else float("nan"),
                    "n_null": int(len(null_deltas)),
                    "pad_bins": int(pad_bins),
                    "shift_bins": int(shift_bins),
                    "tracks": ",".join(map(str, track_list)),
                    "topk_channels": ",".join(map(str, topk_channels)),
                    "stats_base": stats_base,
                    "stats_ablate": stats_ab,
                }
            )

    df_out = pd.DataFrame(rows)
    out_tsv = outdir / "auc_drop_curve.tsv"
    df_out.to_csv(out_tsv, sep="\t", index=False)
    logger.info(f"Wrote: {out_tsv}")

    # Simple plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4 + 1.0 * max(1, len(layers))))
        for i, layer in enumerate(layers):
            sub = df_out[df_out["layer"] == layer].sort_values("k")
            if sub.empty:
                continue
            x = sub["k"].astype(int).values
            y = -sub["delta_auc_topk"].astype(float).values
            lo = -sub["null_p95"].astype(float).values
            hi = -sub["null_p05"].astype(float).values
            plt.plot(x, y, marker="o", label=f"topK drop: {layer}")
            plt.fill_between(x, lo, hi, alpha=0.2)

        plt.axhline(0.0, color="black", linewidth=1)
        plt.xlabel("K channels ablated")
        plt.ylabel("AUROC drop (baseline - ablated)")
        plt.title("AUC drop vs K with matched-layer random null band (5–95%)")
        plt.legend(loc="best")
        plt.tight_layout()
        out_png = outdir / "auc_drop_curve.png"
        plt.savefig(out_png, dpi=200)
        plt.close()
        logger.info(f"Wrote: {out_png}")
    except Exception as e:
        logger.warning(f"Plotting failed: {e}")

    manifest = {
        "rank_run_dir": str(run_dir),
        "pos_ids": str(args.pos_ids),
        "neg_ids": str(args.neg_ids),
        "fasta": str(args.fasta),
        "model_name": str(model_name),
        "output_key": output_key,
        "seq_len": int(seq_len),
        "tracks": track_list,
        "pad_bins": int(pad_bins),
        "shift_bins": int(shift_bins),
        "layers": layers,
        "ks": ks,
        "n_pos": args.n_pos,
        "n_neg": args.n_neg,
        "n_null": int(args.n_null),
        "seed": int(args.seed),
        "outputs": {
            "auc_drop_curve_tsv": str(out_tsv),
            "auc_drop_curve_png": str(outdir / "auc_drop_curve.png"),
        },
    }
    (outdir / "report.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
