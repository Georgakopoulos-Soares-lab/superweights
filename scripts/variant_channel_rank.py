#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow running as `python scripts/variant_channel_rank.py` from any cwd.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Avoid transformers importing TensorFlow/JAX/Flax if present (can be slow/hang on HPC nodes).
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from borzoi.data import ensure_variant_id, load_eqtl_parquet, load_posneg, parse_canonical_variant_id, sample_posneg_ids
from borzoi.encode import one_hot_encode_batch
from borzoi.genome import Genome, make_ref_alt_sequence
from borzoi.activations import ActivationMultiWindowCapturer
from borzoi.model import BorzoiWrapper, detect_seq_len, detect_seq_len_from_crop, load_borzoi


DEFAULT_LAYERS = [
    # early
    "horizontal_conv1.conv_layer",
    # pre-head
    "final_joined_convs.0.conv_layer",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Rank task-aligned superweight channels for a variant dataset, using a task-specific output readout "
            "(subset of tracks + local bins) and localized activation summaries aligned to the same window."
        )
    )

    p.add_argument(
        "--parquet",
        required=False,
        default=None,
        type=str,
        help=(
            "Optional eQTL parquet to attach metadata columns (e.g. beta/pval). "
            "If omitted, ranking uses only variant_id lists + model predictions."
        ),
    )
    p.add_argument("--pos_ids", required=True, type=str, help="Pos variant_id list")
    p.add_argument("--neg_ids", required=True, type=str, help="Neg variant_id list")
    p.add_argument("--fasta", required=True, type=str, help="Reference genome FASTA (hg38.fa)")

    p.add_argument("--seq_len", type=int, default=None, help="Override sequence length")
    p.add_argument(
        "--window_bp",
        type=int,
        default=None,
        help="Alias for --seq_len (window length). If both set, they must match.",
    )

    p.add_argument("--n_pos", type=str, default="all")
    p.add_argument("--n_neg", type=str, default="all")
    p.add_argument("--batch_size", type=int, default=4)

    p.add_argument("--model_name", type=str, default="johahi/borzoi-replicate-0")
    p.add_argument("--output_key", type=str, default=None)

    p.add_argument(
        "--target",
        type=str,
        default="signed",
        choices=["signed", "abs"],
        help=(
            "Target used for channel ranking and AUC diagnostics. "
            "signed: delta_task_readout = (alt-ref). abs: |alt-ref|."
        ),
    )

    p.add_argument(
        "--layers",
        type=str,
        default=",".join(DEFAULT_LAYERS),
        help="Comma-separated layer names to rank.",
    )

    p.add_argument(
        "--track_indices",
        type=str,
        default="260,262,784",
        help="Comma-separated track indices used for the task readout.",
    )
    p.add_argument(
        "--pad_bins",
        type=int,
        default=4,
        help="Pad in OUTPUT bins for the task readout window (ignored if --pad_bins_grid is set)",
    )
    p.add_argument(
        "--shift_bins",
        type=int,
        default=-1,
        help="Shift in OUTPUT bins for the task readout window (ignored if --shift_bins_grid is set)",
    )
    p.add_argument(
        "--pad_bins_grid",
        type=str,
        default="2,4,10",
        help="Comma-separated pad sizes in output bins to evaluate (e.g. 2,4,10)",
    )
    p.add_argument(
        "--shift_bins_grid",
        type=str,
        default="-4,-2,-1,0,1,2,4",
        help="Comma-separated shifts in output bins to evaluate (e.g. -4,-2,-1,0,1,2,4)",
    )

    p.add_argument(
        "--outdir",
        type=str,
        default="results/variants/wholebloodCAGE_task_rank",
        help="Output directory.",
    )
    p.add_argument("--seed", type=int, default=1337)

    return p.parse_args()


def _parse_n(s: str) -> Optional[int]:
    if s is None:
        return None
    s = str(s).strip().lower()
    if s in {"all", "none", ""}:
        return None
    return int(s)


def _parse_int_list_csv(s: str) -> List[int]:
    if s is None:
        return []
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


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
        # [B, T] -> [B, T, 1]
        y = y[:, :, None]
    if y.ndim != 3:
        raise RuntimeError(f"Unexpected model output shape: {tuple(y.shape)}")
    return y


@dataclass
class TaskWindow:
    out_lo: int
    out_hi: int
    out_bin_bp: float
    win_start_bp: float
    win_end_bp: float


def _compute_task_window(seq_len: int, out_bins: int, pad_bins: int, shift_bins: int) -> TaskWindow:
    if out_bins <= 0:
        raise ValueError("out_bins must be positive")

    out_bin_bp = float(seq_len) / float(out_bins)
    center_bp = float(seq_len // 2)

    win_start_bp = center_bp + float(shift_bins - pad_bins) * out_bin_bp
    win_end_bp = center_bp + float(shift_bins + pad_bins + 1) * out_bin_bp

    out_lo = int(np.floor(win_start_bp / out_bin_bp))
    out_hi = int(np.ceil(win_end_bp / out_bin_bp))

    out_lo = max(0, min(out_lo, out_bins))
    out_hi = max(0, min(out_hi, out_bins))
    if out_hi <= out_lo:
        # ensure at least one bin
        mid = max(0, min(int(np.floor(center_bp / out_bin_bp)), out_bins - 1))
        out_lo, out_hi = mid, mid + 1

    return TaskWindow(
        out_lo=int(out_lo),
        out_hi=int(out_hi),
        out_bin_bp=float(out_bin_bp),
        win_start_bp=float(win_start_bp),
        win_end_bp=float(win_end_bp),
    )


def _iter_settings(args: argparse.Namespace) -> List[Tuple[int, int]]:
    pad_grid = _parse_int_list_csv(args.pad_bins_grid)
    shift_grid = _parse_int_list_csv(args.shift_bins_grid)
    if not pad_grid:
        pad_grid = [int(args.pad_bins)]
    if not shift_grid:
        shift_grid = [int(args.shift_bins)]

    settings: List[Tuple[int, int]] = []
    for p in pad_grid:
        for s in shift_grid:
            settings.append((int(p), int(s)))
    # stable deterministic ordering
    settings = sorted(settings, key=lambda x: (x[0], x[1]))
    return settings


def _task_readout(y: torch.Tensor, track_idx: torch.Tensor, out_lo: int, out_hi: int) -> torch.Tensor:
    # y: [B, T, L]
    y_sel = y.index_select(dim=1, index=track_idx)
    y_win = y_sel[:, :, out_lo:out_hi]
    return y_win.mean(dim=(1, 2))


def _rank_channels(delta_act: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    # delta_act: [N, C], y: [N]
    if delta_act.ndim != 2:
        raise ValueError("delta_act must be [N,C]")
    if y.ndim != 1:
        raise ValueError("y must be [N]")

    N, C = delta_act.shape
    rhos = np.full((C,), np.nan, dtype=np.float64)
    ps = np.full((C,), np.nan, dtype=np.float64)

    for c in range(C):
        xc = delta_act[:, c]
        m = ~(np.isnan(xc) | np.isnan(y))
        if int(m.sum()) < 10:
            continue
        rho, p = spearmanr(xc[m], y[m])
        rhos[c] = float(rho)
        ps[c] = float(p)

    # BH-FDR across channels
    valid = ~np.isnan(ps)
    qvals = np.full_like(ps, np.nan)
    if int(valid.sum()) > 0:
        qvals[valid] = multipletests(ps[valid], method="fdr_bh")[1]

    abs_rho = np.abs(rhos)

    # empirical p using the across-all-channels distribution (two-sided)
    # p_emp(c) = (#{|rho| >= |rho_c|}+1)/(#valid+1)
    valid_rho = valid & ~np.isnan(rhos)
    denom = int(valid_rho.sum())
    p_emp = np.full_like(abs_rho, np.nan)
    if denom > 0:
        vals = abs_rho[valid_rho]
        for c in range(C):
            if not valid_rho[c]:
                continue
            p_emp[c] = float((int((vals >= abs_rho[c]).sum()) + 1) / float(denom + 1))

    df = pd.DataFrame(
        {
            "channel": np.arange(C, dtype=np.int32),
            "rho": rhos.astype(np.float64),
            "abs_rho": abs_rho.astype(np.float64),
            "p": ps.astype(np.float64),
            "q": qvals.astype(np.float64),
            "p_emp": p_emp.astype(np.float64),
            "n_used": np.full((C,), int(N), dtype=np.int32),
        }
    )

    df = df.sort_values("abs_rho", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1, dtype=np.int32)
    return df


def main() -> None:
    args = parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device available: cuda={torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    if args.seq_len is not None and args.window_bp is not None and int(args.seq_len) != int(args.window_bp):
        raise ValueError("--seq_len and --window_bp must match if both are set")

    split = load_posneg(args.pos_ids, args.neg_ids)
    df_parquet = None
    if args.parquet is not None:
        df_parquet = load_eqtl_parquet(args.parquet)
        df_parquet = ensure_variant_id(df_parquet)

    n_pos = _parse_n(args.n_pos)
    n_neg = _parse_n(args.n_neg)
    df_run = sample_posneg_ids(split.pos_ids, split.neg_ids, n_pos=n_pos, n_neg=n_neg, seed=int(args.seed))

    # bring metadata when available
    if df_parquet is not None:
        df_run = df_run.merge(df_parquet.drop_duplicates("variant_id"), on="variant_id", how="left", suffixes=("", "_parquet"))

    # parse variant fields
    chroms, poss, refs, alts = [], [], [], []
    for vid in df_run["variant_id"].tolist():
        c, p1, r, a = parse_canonical_variant_id(vid)
        chroms.append(c)
        poss.append(p1)
        refs.append(r)
        alts.append(a)
    df_run["chrom"] = chroms
    df_run["pos"] = poss
    df_run["ref"] = refs
    df_run["alt"] = alts

    df_run["label"] = df_run["variant_id"].isin(split.pos_ids).astype(np.int32)
    logger.info(f"Running variants (pos+neg): n={len(df_run)}")

    wrapper: BorzoiWrapper = load_borzoi(str(args.model_name), device=device, output_key=args.output_key)

    seq_len = args.seq_len if args.seq_len is not None else args.window_bp
    if seq_len is None:
        seq_len = detect_seq_len_from_crop(wrapper.model) or detect_seq_len(wrapper.model)
        if seq_len is None:
            seq_len = 262144
            logger.warning("Could not detect seq_len; defaulting to 262144")
    seq_len = int(seq_len)
    logger.info(f"Using seq_len={seq_len}")

    # acceptance: dummy forward + determine output bins
    dummy = torch.zeros((1, 4, seq_len), device=device, dtype=torch.float32)
    with torch.no_grad():
        y0 = wrapper.model(dummy)
    y0t = _select_output_tensor(y0, output_key=wrapper.output_key)
    out_bins = int(y0t.shape[-1])
    out_tracks = int(y0t.shape[1])
    logger.info(f"Model output: tracks={out_tracks} bins={out_bins}")

    track_list = _parse_int_list_csv(args.track_indices)
    if not track_list:
        raise SystemExit("--track_indices parsed to empty list")
    track_idx_t = torch.tensor(track_list, dtype=torch.long, device=device)

    settings = _iter_settings(args)
    task_windows: List[TaskWindow] = []
    windows_bp: List[Tuple[float, float]] = []
    for pad_bins, shift_bins in settings:
        tw = _compute_task_window(seq_len=seq_len, out_bins=out_bins, pad_bins=int(pad_bins), shift_bins=int(shift_bins))
        task_windows.append(tw)
        windows_bp.append((tw.win_start_bp, tw.win_end_bp))

    logger.info(
        "Task readout grid: tracks=%s pads=%s shifts=%s (settings=%d)"
        % (
            "{" + ",".join(map(str, track_list)) + "}",
            ",".join(map(str, sorted(set(p for p, _ in settings)))),
            ",".join(map(str, sorted(set(s for _, s in settings)))),
            int(len(settings)),
        )
    )

    layers = [s.strip() for s in str(args.layers).split(",") if s.strip()]
    genome = Genome(Path(args.fasta))

    # We will accumulate per-layer delta activation tensors as [B, C, W] chunks
    delta_act_by_layer: Dict[str, List[np.ndarray]] = {layer: [] for layer in layers}

    per_variant_rows: List[dict] = []

    skipped_indel = 0
    skipped_ref_mismatch = 0
    skipped_fasta_missing = 0

    # windows_bp aligned to each task output window, used for activation summaries
    with ActivationMultiWindowCapturer(
        wrapper.model,
        layers,
        seq_len=seq_len,
        windows_bp=windows_bp,
    ) as cap:
        for start in tqdm(range(0, len(df_run), int(args.batch_size)), desc="batches"):
            batch = df_run.iloc[start : start + int(args.batch_size)]

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

            x_ref = one_hot_encode_batch(ref_seqs, device=device)
            cap.clear()
            with torch.no_grad():
                y_ref = wrapper.model(x_ref)
            ref_act = cap.pop().by_layer
            y_ref_t = _select_output_tensor(y_ref, output_key=wrapper.output_key)
            ref_task_list: List[np.ndarray] = []
            for tw in task_windows:
                ref_task_list.append(_task_readout(y_ref_t, track_idx_t, tw.out_lo, tw.out_hi).detach().cpu().numpy())
            ref_task_mat = np.stack(ref_task_list, axis=1).astype(np.float64)  # [B, W]

            x_alt = one_hot_encode_batch(alt_seqs, device=device)
            cap.clear()
            with torch.no_grad():
                y_alt = wrapper.model(x_alt)
            alt_act = cap.pop().by_layer
            y_alt_t = _select_output_tensor(y_alt, output_key=wrapper.output_key)
            alt_task_list: List[np.ndarray] = []
            for tw in task_windows:
                alt_task_list.append(_task_readout(y_alt_t, track_idx_t, tw.out_lo, tw.out_hi).detach().cpu().numpy())
            alt_task_mat = np.stack(alt_task_list, axis=1).astype(np.float64)  # [B, W]

            delta_task_mat = (alt_task_mat - ref_task_mat).astype(np.float64)  # [B, W]

            # per-layer delta activation: alt - ref
            for layer in layers:
                if layer not in ref_act or layer not in alt_act:
                    raise RuntimeError(f"Missing activations for layer {layer}")
                ref_mean = ref_act[layer]["mean_win"].astype(np.float32)  # [B, C, W]
                alt_mean = alt_act[layer]["mean_win"].astype(np.float32)
                delta = alt_mean - ref_mean
                delta_act_by_layer[layer].append(delta)

            for i, row in enumerate(keep_rows):
                for wi, (pad_bins, shift_bins) in enumerate(settings):
                    per_variant_rows.append(
                        {
                            "variant_id": str(row.get("variant_id")),
                            "label": int(row.get("label")),
                            "pad_bins": int(pad_bins),
                            "shift_bins": int(shift_bins),
                            "delta_expr_task": float(delta_task_mat[i, wi]),
                            "chrom": str(row.get("chrom")),
                            "pos": int(row.get("pos")),
                            "ref": str(row.get("ref")),
                            "alt": str(row.get("alt")),
                            "beta": float(row.get("beta")) if pd.notna(row.get("beta")) else np.nan,
                            "pval": float(row.get("pval")) if pd.notna(row.get("pval")) else np.nan,
                        }
                    )

    df_per = pd.DataFrame(per_variant_rows)
    if str(args.target).lower() == "abs":
        df_per["target_expr_task"] = df_per["delta_expr_task"].abs()
    else:
        df_per["target_expr_task"] = df_per["delta_expr_task"].astype(float)
    df_per["target_mode"] = str(args.target).lower()
    df_per_path = outdir / "per_variant_task_readout_by_pad_shift.parquet"
    df_per.to_parquet(df_per_path, index=False)

    # AUC for task readout (sanity) per setting
    # Keep both signed and abs to help diagnose weak readouts.
    auc_signed_by_setting: Dict[str, float] = {}
    auc_abs_by_setting: Dict[str, float] = {}
    auc_target_by_setting: Dict[str, float] = {}
    for pad_bins, shift_bins in settings:
        df_s = df_per[(df_per["pad_bins"] == int(pad_bins)) & (df_per["shift_bins"] == int(shift_bins))]
        auc_signed = float("nan")
        auc_abs = float("nan")
        auc_target = float("nan")
        try:
            if df_s["label"].nunique() >= 2:
                ylab = df_s["label"].astype(int).values
                signed = df_s["delta_expr_task"].values
                target = df_s["target_expr_task"].values
                auc_signed = float(roc_auc_score(ylab, signed))
                auc_abs = float(roc_auc_score(ylab, np.abs(signed)))
                auc_target = float(roc_auc_score(ylab, target))
        except Exception:
            pass
        key = f"pad{int(pad_bins)}_shift{int(shift_bins)}"
        auc_signed_by_setting[key] = auc_signed
        auc_abs_by_setting[key] = auc_abs
        auc_target_by_setting[key] = auc_target

    # Rank channels per layer x setting
    rank_paths: List[str] = []
    for layer in layers:
        X3 = np.concatenate(delta_act_by_layer[layer], axis=0)  # [N, C, W]
        if X3.ndim != 3:
            raise RuntimeError(f"Expected [N,C,W] for layer {layer}, got {X3.shape}")

        N, C, W = X3.shape
        if int(W) != int(len(settings)):
            raise RuntimeError(f"Mismatch W for layer {layer}: X3.W={W} settings={len(settings)}")

        df_rank_all: List[pd.DataFrame] = []
        for wi, (pad_bins, shift_bins) in enumerate(settings):
            df_s = df_per[(df_per["pad_bins"] == int(pad_bins)) & (df_per["shift_bins"] == int(shift_bins))]
            y = df_s["target_expr_task"].astype(float).values
            if len(y) != N:
                # df_per is long; each setting should have N rows.
                raise RuntimeError(
                    f"Unexpected row count for setting pad={pad_bins} shift={shift_bins}: {len(y)} vs N={N}"
                )
            X = X3[:, :, wi].astype(np.float64)
            df_rank = _rank_channels(X, y)
            df_rank.insert(0, "layer", layer)
            df_rank.insert(1, "pad_bins", int(pad_bins))
            df_rank.insert(2, "shift_bins", int(shift_bins))
            df_rank_all.append(df_rank)

            # small per-setting top-k
            topk = df_rank.head(25)[["channel", "rho", "p", "q", "p_emp"]]
            (outdir / f"top25_{layer.replace('.', '_')}_pad{int(pad_bins)}_shift{int(shift_bins)}.tsv").write_text(
                topk.to_csv(sep="\t", index=False)
            )

        df_rank_all2 = pd.concat(df_rank_all, axis=0, ignore_index=True)
        out_pq = outdir / f"rank_{layer.replace('.', '_')}_by_pad_shift.parquet"
        out_tsv = outdir / f"rank_{layer.replace('.', '_')}_by_pad_shift.tsv"
        df_rank_all2.to_parquet(out_pq, index=False)
        df_rank_all2.to_csv(out_tsv, sep="\t", index=False)
        rank_paths.append(str(out_pq))

        # consensus: best |rho| per channel across settings
        best = (
            df_rank_all2.assign(abs_rho=lambda d: d["rho"].abs())
            .sort_values(["channel", "abs_rho"], ascending=[True, False])
            .groupby(["layer", "channel"], as_index=False)
            .head(1)
            .sort_values("abs_rho", ascending=False)
        )
        best.to_parquet(outdir / f"best_{layer.replace('.', '_')}_across_settings.parquet", index=False)
        best.head(100).to_csv(outdir / f"best_{layer.replace('.', '_')}_across_settings_top100.tsv", sep="\t", index=False)

    meta = {
        "n_variants": int(len(df_per)),
        "n_pos": int(df_per["label"].sum()),
        "n_neg": int((df_per["label"] == 0).sum()),
        "target": str(args.target),
        "auc_label_from_delta_expr_task_by_setting": auc_signed_by_setting,
        "auc_label_from_abs_delta_expr_task_by_setting": auc_abs_by_setting,
        "auc_label_from_target_expr_task_by_setting": auc_target_by_setting,
        "tracks": track_list,
        "settings": [{"pad_bins": int(p), "shift_bins": int(s)} for p, s in settings],
        "out_bins": int(out_bins),
        "out_bin_bp": float(task_windows[0].out_bin_bp),
        "seq_len": int(seq_len),
        "layers": layers,
        "model_name": str(args.model_name),
        "output_key": args.output_key,
        "skipped_indel": int(skipped_indel),
        "skipped_ref_mismatch": int(skipped_ref_mismatch),
        "skipped_fasta_missing": int(skipped_fasta_missing),
        "rank_parquets": rank_paths,
    }

    (outdir / "report.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    logger.info(f"Wrote per-variant readout: {df_per_path}")
    logger.info(f"Wrote ranking report: {outdir / 'report.json'}")


if __name__ == "__main__":
    main()
