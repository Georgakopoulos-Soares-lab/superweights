#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow running as `python scripts/export_top_channel_deltas.py` from any cwd.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from superweights_borzoi.encoding import one_hot_encode_batch
from superweights_borzoi.genome import Genome, make_ref_alt_sequence
from superweights_borzoi.interpret.activations import ActivationMultiWindowCapturer
from superweights_borzoi.models.borzoi_pt import BorzoiWrapper, load_borzoi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export per-variant delta activations for the top-K channels (from best_*_across_settings) "
            "at each channel's best pad/shift setting. Intended to enable fast resampling stability without "
            "re-running full channel ranking."
        )
    )

    p.add_argument(
        "--run_dir",
        required=True,
        type=str,
        help="Existing variant_channel_rank.py output directory (contains report.json, best_* files).",
    )
    p.add_argument("--fasta", required=True, type=str, help="Reference genome FASTA (hg38.fa)")

    p.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layers to export. Default: use layers from report.json.",
    )
    p.add_argument(
        "--target",
        type=str,
        default=None,
        choices=["signed", "abs"],
        help="Target mode. Default: use report.json target (or 'signed' if absent).",
    )
    p.add_argument("--topk", type=int, default=50, help="Number of top channels per layer to export")
    p.add_argument("--batch_size", type=int, default=4)

    p.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Override model_name (default: from report.json)",
    )
    p.add_argument(
        "--output_key",
        type=str,
        default=None,
        help="Override output_key (default: from report.json)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string (e.g. cuda, cpu). Default: auto.",
    )

    return p.parse_args()


def _parse_layer_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    layers = [p.strip() for p in str(s).split(",") if p.strip()]
    return layers or None


@dataclass(frozen=True)
class TaskWindow:
    pad_bins: int
    shift_bins: int
    win_start_bp: float
    win_end_bp: float


def _compute_task_window_bp(*, seq_len: int, out_bins: int, pad_bins: int, shift_bins: int) -> TaskWindow:
    out_bin_bp = float(seq_len) / float(out_bins)
    center_bp = float(seq_len // 2)

    win_start_bp = center_bp + float(shift_bins - pad_bins) * out_bin_bp
    win_end_bp = center_bp + float(shift_bins + pad_bins + 1) * out_bin_bp

    return TaskWindow(
        pad_bins=int(pad_bins),
        shift_bins=int(shift_bins),
        win_start_bp=float(win_start_bp),
        win_end_bp=float(win_end_bp),
    )


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


def _task_readout(y: torch.Tensor, track_idx: torch.Tensor, out_lo: int, out_hi: int) -> torch.Tensor:
    y_sel = y.index_select(dim=1, index=track_idx)
    y_win = y_sel[:, :, out_lo:out_hi]
    return y_win.mean(dim=(1, 2))


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir)
    report_path = run_dir / "report.json"
    if not report_path.exists():
        raise SystemExit(f"Missing report.json in {run_dir}")

    report = json.loads(report_path.read_text())

    target_mode = args.target
    if target_mode is None:
        target_mode = str(report.get("target") or report.get("target_mode") or "signed")
    target_mode = str(target_mode).lower()
    if target_mode not in {"signed", "abs"}:
        raise SystemExit(f"Unsupported --target: {target_mode}")

    device_str = args.device
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(device_str))

    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    model_name = args.model_name if args.model_name is not None else str(report.get("model_name"))
    output_key = args.output_key if args.output_key is not None else report.get("output_key")

    seq_len = int(report["seq_len"])
    out_bins = int(report["out_bins"])
    track_list = [int(x) for x in report["tracks"]]
    track_idx_t = torch.tensor(track_list, dtype=torch.long, device=device)

    layers = _parse_layer_list(args.layers)
    if layers is None:
        layers = [str(x) for x in report["layers"]]

    df_per = pd.read_parquet(run_dir / "per_variant_task_readout_by_pad_shift.parquet")

    # unique variants to run
    keep_cols = ["variant_id", "label", "chrom", "pos", "ref", "alt"]
    df_var = df_per[keep_cols].drop_duplicates("variant_id").reset_index(drop=True)
    logger.info(f"Unique variants to export: n={len(df_var)}")

    # sanity: ensure no missing key fields
    if df_var["chrom"].isna().any() or df_var["pos"].isna().any():
        raise RuntimeError("Variant metadata missing chrom/pos in per_variant parquet")

    genome = Genome(Path(args.fasta))
    wrapper: BorzoiWrapper = load_borzoi(str(model_name), device=device, output_key=output_key)

    # determine output tensor dims
    dummy = torch.zeros((1, 4, seq_len), device=device, dtype=torch.float32)
    with torch.no_grad():
        y0 = wrapper.model(dummy)
    y0t = _select_output_tensor(y0, output_key=wrapper.output_key)
    if int(y0t.shape[-1]) != int(out_bins):
        logger.warning(f"out_bins mismatch report={out_bins} model={int(y0t.shape[-1])}; using model")
        out_bins = int(y0t.shape[-1])

    # Build a lookup for delta_task_readout per variant per (pad,shift)
    df_per_key = df_per[["variant_id", "pad_bins", "shift_bins", "delta_expr_task"]].copy()
    df_per_key["pad_bins"] = df_per_key["pad_bins"].astype(int)
    df_per_key["shift_bins"] = df_per_key["shift_bins"].astype(int)

    for layer in layers:
        best_path = run_dir / f"best_{layer.replace('.', '_')}_across_settings.parquet"
        if not best_path.exists():
            raise SystemExit(f"Missing best file for layer {layer}: {best_path}")

        df_best = pd.read_parquet(best_path)
        df_best = df_best.sort_values("abs_rho", ascending=False).head(int(args.topk)).reset_index(drop=True)

        # per-channel setting
        df_best["pad_bins"] = df_best["pad_bins"].astype(int)
        df_best["shift_bins"] = df_best["shift_bins"].astype(int)
        channel_list = [int(x) for x in df_best["channel"].tolist()]

        # unique settings needed
        settings = sorted({(int(r.pad_bins), int(r.shift_bins)) for r in df_best.itertuples(index=False)})
        windows = [_compute_task_window_bp(seq_len=seq_len, out_bins=out_bins, pad_bins=p, shift_bins=s) for p, s in settings]
        windows_bp = [(w.win_start_bp, w.win_end_bp) for w in windows]
        setting_to_wi: Dict[Tuple[int, int], int] = {ps: i for i, ps in enumerate(settings)}

        logger.info(
            f"Layer {layer}: exporting topk={len(channel_list)} channels across {len(settings)} unique settings"
        )

        rows: List[dict] = []

        with ActivationMultiWindowCapturer(
            wrapper.model,
            [layer],
            seq_len=seq_len,
            windows_bp=windows_bp,
            channels=channel_list,
        ) as cap:
            for start in tqdm(range(0, len(df_var), int(args.batch_size)), desc=f"export_{layer}"):
                batch = df_var.iloc[start : start + int(args.batch_size)]

                ref_seqs: List[str] = []
                alt_seqs: List[str] = []
                keep_rows: List[pd.Series] = []

                for _idx, row in batch.iterrows():
                    ref = str(row.get("ref"))
                    alt = str(row.get("alt"))
                    if len(ref) != 1 or len(alt) != 1:
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
                    except Exception:
                        continue

                    ref_seqs.append(ref_seq)
                    alt_seqs.append(alt_seq)
                    keep_rows.append(row)

                if not keep_rows:
                    continue

                x_ref = one_hot_encode_batch(ref_seqs, device=device)
                cap.clear()
                with torch.no_grad():
                    y_ref = wrapper.model(x_ref)
                ref_act = cap.pop().by_layer[layer]["mean_win"].astype(np.float32)  # [B, K, W]
                y_ref_t = _select_output_tensor(y_ref, output_key=wrapper.output_key)

                x_alt = one_hot_encode_batch(alt_seqs, device=device)
                cap.clear()
                with torch.no_grad():
                    y_alt = wrapper.model(x_alt)
                alt_act = cap.pop().by_layer[layer]["mean_win"].astype(np.float32)
                y_alt_t = _select_output_tensor(y_alt, output_key=wrapper.output_key)

                delta_act = (alt_act - ref_act).astype(np.float32)  # [B, K, W]

                # compute task readout for all needed settings
                ref_task = []
                alt_task = []
                for (pad_bins, shift_bins), w in zip(settings, windows):
                    # map bp window to output bins
                    out_bin_bp = float(seq_len) / float(out_bins)
                    out_lo = int(np.floor(w.win_start_bp / out_bin_bp))
                    out_hi = int(np.ceil(w.win_end_bp / out_bin_bp))
                    out_lo = max(0, min(out_lo, out_bins))
                    out_hi = max(0, min(out_hi, out_bins))
                    if out_hi <= out_lo:
                        out_lo, out_hi = max(0, min(out_bins - 1, out_lo)), max(1, min(out_bins, out_lo + 1))

                    ref_task.append(_task_readout(y_ref_t, track_idx_t, out_lo, out_hi).detach().cpu().numpy())
                    alt_task.append(_task_readout(y_alt_t, track_idx_t, out_lo, out_hi).detach().cpu().numpy())

                ref_task_mat = np.stack(ref_task, axis=1).astype(np.float32)  # [B, W]
                alt_task_mat = np.stack(alt_task, axis=1).astype(np.float32)
                delta_task_mat = (alt_task_mat - ref_task_mat).astype(np.float32)

                for bi, row in enumerate(keep_rows):
                    vid = str(row.get("variant_id"))
                    lab = int(row.get("label"))

                    for ci, ch in enumerate(channel_list):
                        r = df_best[df_best["channel"] == int(ch)].iloc[0]
                        pad_bins = int(r["pad_bins"])
                        shift_bins = int(r["shift_bins"])
                        wi = int(setting_to_wi[(pad_bins, shift_bins)])

                        # task readout for this variant at this setting (from this forward pass)
                        dtask_signed = float(delta_task_mat[bi, wi])
                        dtask_target = float(abs(dtask_signed)) if target_mode == "abs" else float(dtask_signed)

                        rows.append(
                            {
                                "variant_id": vid,
                                "label": lab,
                                "layer": layer,
                                "channel": int(ch),
                                "pad_bins": pad_bins,
                                "shift_bins": shift_bins,
                                "delta_act": float(delta_act[bi, ci, wi]),
                                "delta_task_readout": dtask_signed,
                                "target_task_readout": dtask_target,
                                "target_mode": target_mode,
                                "rho_full": float(r["rho"]),
                                "abs_rho_full": float(r["abs_rho"]),
                                "rank_full": int(r["rank"]),
                            }
                        )

        df_out = pd.DataFrame(rows)
        out_path = run_dir / f"per_variant_delta_activation_top{int(args.topk)}_{layer.replace('.', '_')}.parquet"
        df_out.to_parquet(out_path, index=False)
        logger.info(f"Wrote export: {out_path}")


if __name__ == "__main__":
    main()
