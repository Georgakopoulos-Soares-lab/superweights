#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow running as `python scripts/track_group_auc_scan.py` from any cwd.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from borzoi.data import (
    canonicalize_variant_id,
    ensure_variant_id,
    load_eqtl_parquet,
    normalize_chrom_str,
    parse_canonical_variant_id,
    sample_posneg_ids,
)
from borzoi.encode import one_hot_encode_batch
from borzoi.genome import Genome, make_ref_alt_sequence
from borzoi.model import BorzoiWrapper, detect_seq_len, detect_seq_len_from_crop, load_borzoi


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
            "Scan curated track groups and report the best AUROC(pos vs neg) over a pad/shift grid using |Δ(group)|, "
            "where Δ(group) is the mean (alt-ref) over group tracks and output bins in the task window."
        )
    )

    p.add_argument("--parquet", type=str, default=None, help="Optional eQTL parquet to attach metadata (beta/pval).")

    p.add_argument("--pos_ids", type=str, default=None, help="Pos variant_id list (one per line).")
    p.add_argument("--neg_ids", type=str, default=None, help="Neg variant_id list (one per line).")
    p.add_argument("--pos_vcf", type=str, default=None, help="Pos VCF(.gz) to derive variant_id list.")
    p.add_argument("--neg_vcf", type=str, default=None, help="Neg VCF(.gz) to derive variant_id list.")

    p.add_argument("--fasta", required=True, type=str, help="Reference genome FASTA (hg38.fa)")
    p.add_argument(
        "--targets",
        type=str,
        default="/work/10906/arisk/borzoi_hg38/targets.txt",
        help="Borzoi targets metadata TSV (borzoi_hg38/targets.txt)",
    )

    p.add_argument("--model_name", type=str, default="johahi/borzoi-replicate-0")
    p.add_argument("--output_key", type=str, default=None)
    p.add_argument("--seq_len", type=int, default=None, help="Override sequence length")
    p.add_argument("--window_bp", type=int, default=None, help="Alias for --seq_len")

    p.add_argument("--n_pos", type=str, default="all")
    p.add_argument("--n_neg", type=str, default="all")
    p.add_argument("--batch_size", type=int, default=4)

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
        "--groups",
        type=str,
        default=None,
        help=(
            "Optional comma-separated subset of group names to evaluate (e.g. "
            "cage_wholeblood_ribopure,cage_blood_other). Default: evaluate all default groups."
        ),
    )

    p.add_argument(
        "--bootstrap",
        type=int,
        default=300,
        help="Bootstrap resamples for best-setting AUROC confidence intervals (per group).",
    )
    p.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap CIs (e.g. 0.95).",
    )
    p.add_argument(
        "--heatmap_topk",
        type=int,
        default=2,
        help="Write pad×shift AUROC heatmaps for the top-K groups by best AUC (0 disables).",
    )

    p.add_argument(
        "--outdir",
        type=str,
        default="results/track_group_scan",
        help="Output directory (relative to current working dir).",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--write_per_variant",
        action="store_true",
        help=(
            "Also write per-variant group deltas for the selected groups/settings (can be large). "
            "For big runs prefer leaving this off."
        ),
    )

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


def _parse_str_list_csv(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    out: List[str] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(part)
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
        y = y[:, :, None]
    if y.ndim != 3:
        raise RuntimeError(f"Unexpected model output shape: {tuple(y.shape)}")
    return y


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
        mid = max(0, min(int(np.floor(center_bp / out_bin_bp)), out_bins - 1))
        out_lo, out_hi = mid, mid + 1

    return TaskWindow(
        out_lo=int(out_lo),
        out_hi=int(out_hi),
        out_bin_bp=float(out_bin_bp),
        win_start_bp=float(win_start_bp),
        win_end_bp=float(win_end_bp),
    )


def _iter_settings(pad_bins_grid: str, shift_bins_grid: str) -> List[Tuple[int, int]]:
    pads = _parse_int_list_csv(pad_bins_grid)
    shifts = _parse_int_list_csv(shift_bins_grid)
    settings = sorted([(int(p), int(s)) for p in pads for s in shifts], key=lambda x: (x[0], x[1]))
    return settings


def _open_text_maybe_gzip(path: str | Path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def _load_variant_ids_from_list(path: str | Path) -> set[str]:
    ids: set[str] = set()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.add(canonicalize_variant_id(line))
    return ids


def _load_variant_ids_from_vcf(path: str | Path) -> set[str]:
    ids: set[str] = set()
    with _open_text_maybe_gzip(path) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            chrom, pos, vid, ref, alt = parts[0], parts[1], parts[2], parts[3], parts[4]
            try:
                pos1 = int(pos)
            except Exception:
                continue

            # Prefer parsing ref/alt from the ID field when it encodes alleles.
            # Some of our pos/neg VCF-like files include an ID like
            #   chr1_959193_G_A_b38
            # but may have REF/ALT columns swapped relative to the true reference.
            if vid:
                cand = canonicalize_variant_id(vid)
                try:
                    c2, p2, r2, a2 = parse_canonical_variant_id(cand)
                    if int(p2) == int(pos1) and str(c2) == normalize_chrom_str(chrom) and len(r2) == 1 and len(a2) == 1:
                        ids.add(cand)
                        continue
                except Exception:
                    pass

            # Fallback: parse from VCF columns; keep only biallelic SNVs.
            alts = alt.split(",")
            if len(alts) != 1:
                continue
            alt1 = alts[0]
            if len(ref) != 1 or len(alt1) != 1:
                continue
            ids.add(canonicalize_variant_id(f"{chrom}:{pos1}:{ref}:{alt1}"))
    return ids


@dataclass(frozen=True)
class TrackGroup:
    name: str
    assay: str
    track_indices: Tuple[int, ...]


def _prefer_plus_for_stranded(df: pd.DataFrame) -> pd.DataFrame:
    ident = df["identifier"].astype(str)
    stranded = ident.str.endswith("+") | ident.str.endswith("-")
    keep = (~stranded) | ident.str.endswith("+")
    return df.loc[keep].copy()


def _targets_df(targets_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(targets_path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    if "description" not in df.columns or "identifier" not in df.columns:
        raise ValueError(f"Unexpected targets file format: columns={df.columns.tolist()}")

    # Track index is stored as the first column (often 'Unnamed: 0')
    if "Unnamed: 0" in df.columns:
        df["track_index"] = df["Unnamed: 0"].astype(int)
    else:
        first = df.columns[0]
        df["track_index"] = df[first].astype(int)

    df["description"] = df["description"].astype(str)
    df["identifier"] = df["identifier"].astype(str)
    return df


def _build_default_groups(df_t: pd.DataFrame) -> List[TrackGroup]:
    desc = df_t["description"].astype(str)

    defs: List[Tuple[str, str, pd.Series]] = []

    defs.append(
        (
            "cage_wholeblood_ribopure",
            "CAGE",
            desc.str.contains("CAGE:Whole blood (ribopure)", case=False, regex=False),
        )
    )

    defs.append(
        (
            "cage_blood_other",
            "CAGE",
            desc.str.startswith("CAGE:")
            & desc.str.contains("blood", case=False, regex=False)
            & ~desc.str.contains("CAGE:Whole blood (ribopure)", case=False, regex=False),
        )
    )

    defs.append(
        (
            "rna_pbmc",
            "RNA",
            desc.str.startswith("RNA:") & desc.str.contains("peripheral blood mononuclear", case=False, regex=False),
        )
    )

    # Exclude blood_vessel by using a word boundary after 'blood'
    defs.append(("rna_gtex_blood", "RNA", desc.str.contains(r"^RNA:blood\b", case=False, regex=True)))

    defs.append(("atac_blood_cd34_bm", "ATAC", desc.str.startswith("ATAC:") & desc.str.contains(r"Bone Marrow|CD34", case=False, regex=True)))

    defs.append(("dnase_blood_cell_lines", "DNASE", desc.str.startswith("DNASE:") & desc.str.contains(r"GM12878|K562", case=False, regex=True)))

    defs.append(("chip_h3k27ac", "CHIP", desc.str.contains("CHIP:H3K27ac", case=False, regex=False)))
    defs.append(("chip_ctcf", "CHIP", desc.str.contains("CHIP:CTCF", case=False, regex=False)))

    defs.append(("rna_gtex_liver", "RNA", desc.str.startswith("RNA:liver")))
    defs.append(("rna_gtex_brain", "RNA", desc.str.startswith("RNA:brain")))

    groups: List[TrackGroup] = []
    for name, assay, m in defs:
        sub = df_t.loc[m].copy()
        sub = _prefer_plus_for_stranded(sub)
        tracks = tuple(int(x) for x in sub["track_index"].tolist())
        if len(tracks) == 0:
            logger.warning(f"Group empty, skipping: {name}")
            continue
        groups.append(TrackGroup(name=name, assay=assay, track_indices=tracks))

    # Deduplicate by name in case of accidental collisions
    seen: set[str] = set()
    uniq: List[TrackGroup] = []
    for g in groups:
        if g.name in seen:
            continue
        seen.add(g.name)
        uniq.append(g)
    return uniq


def _bootstrap_auc_ci(
    y: np.ndarray,
    score: np.ndarray,
    n_resamples: int,
    ci: float,
    seed: int,
) -> Tuple[float, float]:
    """Stratified bootstrap CI for AUROC.

    Resamples positives and negatives separately to preserve class balance.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    if y.ndim != 1 or score.ndim != 1 or len(y) != len(score):
        raise ValueError("y and score must be 1D arrays with same length")
    if np.unique(y).size < 2:
        return float("nan"), float("nan")

    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) < 5 or len(neg) < 5:
        return float("nan"), float("nan")

    rng = np.random.default_rng(int(seed))
    aucs = np.empty((int(n_resamples),), dtype=np.float64)
    for i in range(int(n_resamples)):
        ip = rng.choice(pos, size=len(pos), replace=True)
        ineg = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([ip, ineg], axis=0)
        aucs[i] = float(roc_auc_score(y[idx], score[idx]))

    alpha = 1.0 - float(ci)
    lo = float(np.quantile(aucs, alpha / 2.0))
    hi = float(np.quantile(aucs, 1.0 - alpha / 2.0))
    return lo, hi


def _try_write_plots(outdir: Path, df_best_ci: pd.DataFrame, df_auc: pd.DataFrame, heatmap_topk: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning(f"Plotting unavailable (matplotlib import failed): {e}")
        return

    # Bar plot with CIs
    dfp = df_best_ci.sort_values("best_auc_abs", ascending=False).reset_index(drop=True)
    x = np.arange(len(dfp))
    y = dfp["best_auc_abs"].astype(float).values
    ylo = dfp["auc_CI_low"].astype(float).values
    yhi = dfp["auc_CI_high"].astype(float).values
    yerr = np.vstack([y - ylo, yhi - y])

    plt.figure(figsize=(10, 4 + 0.25 * len(dfp)))
    plt.bar(x, y, yerr=yerr, capsize=4)
    plt.xticks(x, dfp["group"].tolist(), rotation=30, ha="right")
    plt.ylabel("Best AUROC (abs Δ(group))")
    plt.title("Track-group AUROC (best pad/shift) with bootstrap CI")
    plt.tight_layout()
    out_png = outdir / "best_auc_barplot.png"
    plt.savefig(out_png, dpi=200)
    plt.close()
    logger.info(f"Wrote: {out_png}")

    # Heatmaps for top-K groups
    if int(heatmap_topk) <= 0:
        return
    top_groups = dfp["group"].tolist()[: int(heatmap_topk)]
    for g in top_groups:
        sub = df_auc[df_auc["group"] == g].copy()
        if len(sub) == 0:
            continue
        piv = sub.pivot(index="pad_bins", columns="shift_bins", values="auc_abs")
        plt.figure(figsize=(8, 4))
        plt.imshow(piv.values, aspect="auto", origin="lower")
        plt.colorbar(label="AUROC")
        plt.yticks(np.arange(len(piv.index)), [str(v) for v in piv.index.tolist()])
        plt.xticks(np.arange(len(piv.columns)), [str(v) for v in piv.columns.tolist()])
        plt.xlabel("shift_bins")
        plt.ylabel("pad_bins")
        plt.title(f"AUROC heatmap: {g}")
        plt.tight_layout()
        out_hm = outdir / f"heatmap_{g}.png"
        plt.savefig(out_hm, dpi=200)
        plt.close()
        logger.info(f"Wrote: {out_hm}")


def main() -> None:
    args = parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.seq_len is not None and args.window_bp is not None and int(args.seq_len) != int(args.window_bp):
        raise ValueError("--seq_len and --window_bp must match if both are set")

    if (args.pos_ids is None) == (args.pos_vcf is None):
        raise SystemExit("Provide exactly one of --pos_ids or --pos_vcf")
    if (args.neg_ids is None) == (args.neg_vcf is None):
        raise SystemExit("Provide exactly one of --neg_ids or --neg_vcf")

    if args.pos_ids is not None:
        pos_ids = _load_variant_ids_from_list(args.pos_ids)
    else:
        pos_ids = _load_variant_ids_from_vcf(args.pos_vcf)

    if args.neg_ids is not None:
        neg_ids = _load_variant_ids_from_list(args.neg_ids)
    else:
        neg_ids = _load_variant_ids_from_vcf(args.neg_vcf)

    overlap = len(pos_ids & neg_ids)
    if overlap:
        logger.warning(f"pos/neg overlap: {overlap}")

    n_pos = _parse_n(args.n_pos)
    n_neg = _parse_n(args.n_neg)

    df_run = sample_posneg_ids(pos_ids, neg_ids, n_pos=n_pos, n_neg=n_neg, seed=int(args.seed))

    if args.parquet is not None:
        df_pq = load_eqtl_parquet(args.parquet)
        df_pq = ensure_variant_id(df_pq)
        df_run = df_run.merge(df_pq.drop_duplicates("variant_id"), on="variant_id", how="left", suffixes=("", "_parquet"))

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

    logger.info(
        f"Sampled variants: n={len(df_run)} (pos={int(df_run['label'].sum())}, neg={int((df_run['label']==0).sum())})"
    )

    logger.info(f"Device available: cuda={torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    wrapper: BorzoiWrapper = load_borzoi(str(args.model_name), device=device, output_key=args.output_key)

    seq_len = args.seq_len if args.seq_len is not None else args.window_bp
    if seq_len is None:
        seq_len = detect_seq_len_from_crop(wrapper.model) or detect_seq_len(wrapper.model)
        if seq_len is None:
            seq_len = 262144
            logger.warning("Could not detect seq_len; defaulting to 262144")
    seq_len = int(seq_len)
    logger.info(f"Using seq_len={seq_len}")

    dummy = torch.zeros((1, 4, seq_len), device=device, dtype=torch.float32)
    with torch.no_grad():
        y0 = wrapper.model(dummy)
    y0t = _select_output_tensor(y0, output_key=wrapper.output_key)
    out_bins = int(y0t.shape[-1])
    out_tracks = int(y0t.shape[1])
    logger.info(f"Model output: tracks={out_tracks} bins={out_bins}")

    settings = _iter_settings(args.pad_bins_grid, args.shift_bins_grid)
    task_windows = [_compute_task_window(seq_len=seq_len, out_bins=out_bins, pad_bins=p, shift_bins=s) for p, s in settings]

    df_t = _targets_df(args.targets)
    groups = _build_default_groups(df_t)

    sel = set(_parse_str_list_csv(args.groups))
    if sel:
        groups = [g for g in groups if g.name in sel]
        missing = sorted(sel - {g.name for g in groups})
        if missing:
            raise SystemExit(f"Requested unknown groups: {missing}")

    logger.info("Track groups:")
    for g in groups:
        logger.info(f"  - {g.name}: assay={g.assay} n_tracks={len(g.track_indices)}")

    group_idx_t: Dict[str, torch.Tensor] = {}
    for g in groups:
        idx = torch.tensor(list(g.track_indices), dtype=torch.long, device=device)
        if int(idx.max()) >= out_tracks:
            raise ValueError(f"Group {g.name} has track index beyond model output tracks")
        group_idx_t[g.name] = idx

    genome = Genome(Path(args.fasta))

    # Accumulate arrays per group×setting to avoid huge per-variant tables.
    # key: (group_name, pad_bins, shift_bins) -> list of np arrays (signed deltas)
    signed_by_key: Dict[Tuple[str, int, int], List[np.ndarray]] = {}
    labels_list: List[np.ndarray] = []
    if bool(args.write_per_variant):
        per_variant_rows: List[dict] = []

    skipped_indel = 0
    skipped_ref_mismatch = 0
    skipped_fasta_missing = 0

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

        y_batch = np.asarray([int(r.get("label")) for r in keep_rows], dtype=np.int32)
        labels_list.append(y_batch)

        x_ref = one_hot_encode_batch(ref_seqs, device=device)
        with torch.no_grad():
            y_ref = wrapper.model(x_ref)
        y_ref_t = _select_output_tensor(y_ref, output_key=wrapper.output_key)

        x_alt = one_hot_encode_batch(alt_seqs, device=device)
        with torch.no_grad():
            y_alt = wrapper.model(x_alt)
        y_alt_t = _select_output_tensor(y_alt, output_key=wrapper.output_key)

        # For each pad/shift setting, compute [B, T] delta means over output bins.
        for (pad_bins, shift_bins), tw in zip(settings, task_windows):
            ref_binmean = y_ref_t[:, :, tw.out_lo : tw.out_hi].mean(dim=2)
            alt_binmean = y_alt_t[:, :, tw.out_lo : tw.out_hi].mean(dim=2)
            delta_binmean = alt_binmean - ref_binmean  # [B, T]

            for g in groups:
                idx = group_idx_t[g.name]
                delta_group = delta_binmean.index_select(dim=1, index=idx).mean(dim=1)  # [B]

                d_np = delta_group.detach().cpu().numpy().astype(np.float64)
                key = (g.name, int(pad_bins), int(shift_bins))
                signed_by_key.setdefault(key, []).append(d_np)

                if bool(args.write_per_variant):
                    a_np = np.abs(d_np)
                    for i, row in enumerate(keep_rows):
                        per_variant_rows.append(
                            {
                                "variant_id": str(row.get("variant_id")),
                                "label": int(row.get("label")),
                                "group": g.name,
                                "assay": g.assay,
                                "n_tracks": int(len(g.track_indices)),
                                "pad_bins": int(pad_bins),
                                "shift_bins": int(shift_bins),
                                "delta_group": float(d_np[i]),
                                "abs_delta_group": float(a_np[i]),
                            }
                        )

    y_all = np.concatenate(labels_list, axis=0) if labels_list else np.asarray([], dtype=np.int32)
    if y_all.size == 0:
        raise RuntimeError("No variants were processed (all skipped?)")

    logger.info(
        f"Processed variants: n={int(y_all.size)} (pos={int(y_all.sum())}, neg={int((y_all==0).sum())}); "
        f"skipped_indel={int(skipped_indel)} skipped_ref_mismatch={int(skipped_ref_mismatch)} skipped_fasta_missing={int(skipped_fasta_missing)}"
    )

    # Compute AUC per group x setting
    auc_rows: List[dict] = []
    for (pad_bins, shift_bins) in settings:
        for g in groups:
            key = (g.name, int(pad_bins), int(shift_bins))
            s_list = signed_by_key.get(key, [])
            if not s_list:
                continue
            signed = np.concatenate(s_list, axis=0)
            if signed.shape[0] != y_all.shape[0]:
                raise RuntimeError(
                    f"Mismatched lengths for {key}: scores={signed.shape[0]} labels={y_all.shape[0]}"
                )
            auc_abs = float("nan")
            auc_signed = float("nan")
            try:
                if np.unique(y_all).size >= 2:
                    auc_abs = float(roc_auc_score(y_all, np.abs(signed)))
                    auc_signed = float(roc_auc_score(y_all, signed))
            except Exception:
                pass
            auc_rows.append(
                {
                    "group": g.name,
                    "assay": g.assay,
                    "n_tracks": int(len(g.track_indices)),
                    "pad_bins": int(pad_bins),
                    "shift_bins": int(shift_bins),
                    "auc_abs": auc_abs,
                    "auc_signed": auc_signed,
                }
            )

    df_auc = pd.DataFrame(auc_rows)
    df_auc_path = outdir / "group_auc_by_pad_shift.tsv"
    df_auc.to_csv(df_auc_path, sep="\t", index=False)

    best_rows: List[dict] = []
    for g in groups:
        sub = df_auc[df_auc["group"] == g.name].copy()
        sub = sub.sort_values(["auc_abs", "auc_signed"], ascending=False).reset_index(drop=True)
        if len(sub) == 0:
            continue
        best = sub.iloc[0].to_dict()
        best_rows.append(best)

    df_best = pd.DataFrame(best_rows)
    df_best = df_best.sort_values("auc_abs", ascending=False).reset_index(drop=True)
    df_best_path = outdir / "best_by_group.tsv"
    df_best.to_csv(df_best_path, sep="\t", index=False)

    # Bootstrap CI for best abs-AUC per group
    best_ci_rows: List[dict] = []
    for _, row in df_best.iterrows():
        gname = str(row["group"])
        pad_bins = int(row["pad_bins"])
        shift_bins = int(row["shift_bins"])
        key = (gname, pad_bins, shift_bins)
        signed = np.concatenate(signed_by_key[key], axis=0)
        score_abs = np.abs(signed)
        ci_low, ci_high = _bootstrap_auc_ci(
            y=y_all,
            score=score_abs,
            n_resamples=int(args.bootstrap),
            ci=float(args.ci),
            seed=int(args.seed) + (hash(gname) % 10_000_000),
        )
        best_ci_rows.append(
            {
                "group": gname,
                "assay": str(row["assay"]),
                "n_tracks": int(row["n_tracks"]),
                "best_auc_abs": float(row["auc_abs"]),
                "best_auc_signed": float(row["auc_signed"]),
                "best_pad": pad_bins,
                "best_shift": shift_bins,
                "auc_CI_low": float(ci_low),
                "auc_CI_high": float(ci_high),
                "bootstrap_n": int(args.bootstrap),
                "ci": float(args.ci),
            }
        )

    df_best_ci = pd.DataFrame(best_ci_rows).sort_values("best_auc_abs", ascending=False).reset_index(drop=True)
    df_best_ci_path = outdir / "best_by_group_with_ci.tsv"
    df_best_ci.to_csv(df_best_ci_path, sep="\t", index=False)

    _try_write_plots(outdir=outdir, df_best_ci=df_best_ci, df_auc=df_auc, heatmap_topk=int(args.heatmap_topk))

    manifest = {
        "model_name": str(args.model_name),
        "output_key": args.output_key,
        "seq_len": int(seq_len),
        "out_bins": int(out_bins),
        "out_tracks": int(out_tracks),
        "settings": [{"pad_bins": int(p), "shift_bins": int(s)} for p, s in settings],
        "groups": [
            {"name": g.name, "assay": g.assay, "n_tracks": int(len(g.track_indices)), "tracks": list(g.track_indices)}
            for g in groups
        ],
        "n_variants_sampled": int(len(df_run)),
        "n_pos_sampled": int(df_run["label"].sum()),
        "n_neg_sampled": int((df_run["label"] == 0).sum()),
        "n_variants_processed": int(y_all.size),
        "n_pos_processed": int(y_all.sum()),
        "n_neg_processed": int((y_all == 0).sum()),
        "skipped_indel": int(skipped_indel),
        "skipped_ref_mismatch": int(skipped_ref_mismatch),
        "skipped_fasta_missing": int(skipped_fasta_missing),
        "outputs": {
            "group_auc_by_pad_shift": str(df_auc_path),
            "best_by_group": str(df_best_path),
            "best_by_group_with_ci": str(df_best_ci_path),
            "best_auc_barplot": str(outdir / "best_auc_barplot.png"),
        },
    }

    manifest_path = outdir / "report.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if bool(args.write_per_variant):
        df = pd.DataFrame(per_variant_rows)
        df_path = outdir / "per_variant_group_deltas.parquet"
        df.to_parquet(df_path, index=False)
        logger.info(f"Wrote per-variant deltas: {df_path}")

    logger.info(f"Wrote: {df_auc_path}")
    logger.info(f"Wrote: {df_best_path}")
    logger.info(f"Wrote: {df_best_ci_path}")
    logger.info(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()
