#!/usr/bin/env python
from __future__ import annotations

"""Ablation → AUC-drop experiment for Borzoi superweights (abs-Δ benchmark).

This script measures causal importance of CNN channels by ablating (zeroing) selected
channels at an internal layer during forward pass, then recomputing AUROC on a fixed
pos/neg variant benchmark using a fixed task readout.

Example (primary Whole Blood abs-Δ benchmark):

  PYTHONPATH=. python scripts/ablation_auc_drop.py \
    --preset wholeblood_cage_abs_primary \
    --pos_vcf data/eqtl/Whole_Blood_pos.vcf.gz \
    --neg_vcf data/eqtl/Whole_Blood_neg.vcf.gz \
    --genome_fasta data/hg38.fa \
    --rank_tsv results/variants/<run>/best_final_joined_convs_0_conv_layer_across_settings_top100.tsv \
    --out_dir results/ablation/wholeblood_cage_abs_primary_prehead

You can override early layer:

  --layer horizontal_conv1.conv_layer

Notes
- Score is always abs(delta_group) where delta_group = task(alt) - task(ref).
- Ref/alt allele parsing for VCF-like files prefers allele-encoded IDs when present
  (e.g. chr1_959193_G_A_b38) to avoid REF/ALT swaps.
"""

import argparse
import gzip
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Allow running as `python scripts/...py` from any cwd.
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
from sklearn.metrics import roc_auc_score

from superweights_borzoi.data import (
    canonicalize_variant_id,
    normalize_chrom_str,
    parse_canonical_variant_id,
    sample_posneg_ids,
)
from superweights_borzoi.encoding import one_hot_encode_batch
from superweights_borzoi.genome import Genome, make_ref_alt_sequence
from superweights_borzoi.interpret.ablation import ChannelAblator
from superweights_borzoi.models.borzoi_pt import (
    BorzoiWrapper,
    detect_seq_len,
    detect_seq_len_from_crop,
    get_module_by_name,
    load_borzoi,
)


@dataclass(frozen=True)
class TaskWindow:
    out_lo: int
    out_hi: int
    out_bin_bp: float
    win_start_bp: float
    win_end_bp: float


@dataclass(frozen=True)
class TrackGroup:
    name: str
    assay: str
    track_indices: Tuple[int, ...]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Ablation study: compute baseline AUROC on abs(delta_group) for a fixed track-group readout, "
            "then ablate channels at a chosen layer and measure AUROC drop vs topK and randomK channel sets."
        )
    )

    p.add_argument("--preset", type=str, default=None, help="Convenience preset (e.g. wholeblood_cage_abs_primary)")

    p.add_argument("--model_name_or_path", type=str, default="johahi/borzoi-replicate-0")
    p.add_argument("--output_key", type=str, default=None)
    p.add_argument("--seq_len", type=int, default=None)

    p.add_argument("--genome_fasta", required=True, type=str, help="Reference genome FASTA (hg38.fa)")

    p.add_argument("--pos_ids", type=str, default=None, help="Pos variant_id list (one per line)")
    p.add_argument("--neg_ids", type=str, default=None, help="Neg variant_id list (one per line)")
    p.add_argument("--pos_vcf", type=str, default=None, help="Pos VCF(.gz) to derive variant_id list")
    p.add_argument("--neg_vcf", type=str, default=None, help="Neg VCF(.gz) to derive variant_id list")

    p.add_argument("--n_pos", type=str, default="all")
    p.add_argument("--n_neg", type=str, default="all")

    p.add_argument(
        "--targets_tsv",
        type=str,
        default="/work/10906/arisk/borzoi_hg38/targets.txt",
        help="Borzoi targets metadata TSV (targets.txt)",
    )

    p.add_argument("--track_group", type=str, default="cage_wholeblood_ribopure")
    p.add_argument(
        "--track_group_config",
        type=str,
        default=None,
        help="Optional JSON defining groups. If omitted, uses curated groups from targets.tsv",
    )

    p.add_argument("--pad_bins", type=int, default=4)
    p.add_argument("--shift_bins", type=int, default=4)

    p.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Layer/module name to ablate (comma-separated allowed).",
    )
    p.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Alias for --layer (comma-separated).",
    )

    p.add_argument(
        "--channels",
        type=str,
        default=None,
        help=(
            "Optional single ablation spec: comma-separated ints, or topK:<rank_tsv>:K, or randomK:K:seed. "
            "If provided, runs *only* that ablation set (plus baseline) and writes ablation_results.tsv."
        ),
    )

    p.add_argument(
        "--rank_tsv",
        type=str,
        default=None,
        help=(
            "Channel ranking TSV used to select top-K channels. Must contain a 'channel' column. "
            "If omitted, you must provide explicit --channels for single-set mode."
        ),
    )

    p.add_argument("--topk_list", type=str, default="1,5,10,20,50")
    p.add_argument("--random_draws", type=int, default=30)
    p.add_argument(
        "--random_exclude_top",
        action="store_true",
        help="If set, randomK sampling excludes the top max(K) channels from the ranking file (when available).",
    )

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--out_dir", type=str, required=True)

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
    """Read variant IDs from a (possibly headerless) VCF-like file.

    Prefer parsing alleles from the ID field when it looks like chr_pos_ref_alt(_b38),
    because some files can have REF/ALT columns swapped relative to the true reference.
    """

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

            if vid:
                cand = canonicalize_variant_id(vid)
                try:
                    c2, p2, r2, a2 = parse_canonical_variant_id(cand)
                    if int(p2) == int(pos1) and str(c2) == normalize_chrom_str(chrom) and len(r2) == 1 and len(a2) == 1:
                        ids.add(cand)
                        continue
                except Exception:
                    pass

            # Fallback: parse from REF/ALT columns; keep only biallelic SNVs.
            alts = alt.split(",")
            if len(alts) != 1:
                continue
            alt1 = alts[0]
            if len(ref) != 1 or len(alt1) != 1:
                continue
            ids.add(canonicalize_variant_id(f"{chrom}:{pos1}:{ref}:{alt1}"))

    return ids


def _targets_df(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    if "description" not in df.columns or "identifier" not in df.columns:
        raise ValueError(f"Unexpected targets format: columns={df.columns.tolist()}")

    if "Unnamed: 0" in df.columns:
        df["track_index"] = df["Unnamed: 0"].astype(int)
    else:
        df["track_index"] = df[df.columns[0]].astype(int)

    df["description"] = df["description"].astype(str)
    df["identifier"] = df["identifier"].astype(str)
    return df


def _prefer_plus_for_stranded(df: pd.DataFrame) -> pd.DataFrame:
    ident = df["identifier"].astype(str)
    stranded = ident.str.endswith("+") | ident.str.endswith("-")
    keep = (~stranded) | ident.str.endswith("+")
    return df.loc[keep].copy()


def _build_default_groups(df_t: pd.DataFrame) -> List[TrackGroup]:
    desc = df_t["description"].astype(str)

    defs: List[Tuple[str, str, pd.Series]] = []

    defs.append(("cage_wholeblood_ribopure", "CAGE", desc.str.contains("CAGE:Whole blood (ribopure)", case=False, regex=False)))

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
            "rna_gtex_blood",
            "RNA",
            desc.str.contains(r"^RNA:blood\b", case=False, regex=True),
        )
    )

    defs.append(("atac_blood_cd34_bm", "ATAC", desc.str.startswith("ATAC:") & desc.str.contains(r"Bone Marrow|CD34", case=False, regex=True)))

    defs.append(("dnase_blood_cell_lines", "DNASE", desc.str.startswith("DNASE:") & desc.str.contains(r"GM12878|K562", case=False, regex=True)))

    groups: List[TrackGroup] = []
    for name, assay, m in defs:
        sub = df_t.loc[m].copy()
        sub = _prefer_plus_for_stranded(sub)
        tracks = tuple(int(x) for x in sub["track_index"].tolist())
        if not tracks:
            continue
        groups.append(TrackGroup(name=name, assay=assay, track_indices=tracks))

    seen: set[str] = set()
    uniq: List[TrackGroup] = []
    for g in groups:
        if g.name in seen:
            continue
        seen.add(g.name)
        uniq.append(g)
    return uniq


def _load_groups(targets_tsv: str, group_config_json: Optional[str]) -> List[TrackGroup]:
    if group_config_json is None:
        return _build_default_groups(_targets_df(targets_tsv))

    cfg = json.loads(Path(group_config_json).read_text())
    if not isinstance(cfg, dict) or "groups" not in cfg:
        raise ValueError("track_group_config JSON must be an object with key 'groups'")

    out: List[TrackGroup] = []
    for g in cfg["groups"]:
        name = str(g["name"])
        assay = str(g.get("assay", ""))
        tracks = tuple(int(x) for x in g["tracks"])
        out.append(TrackGroup(name=name, assay=assay, track_indices=tracks))
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

    return TaskWindow(out_lo=int(out_lo), out_hi=int(out_hi), out_bin_bp=float(out_bin_bp), win_start_bp=float(win_start_bp), win_end_bp=float(win_end_bp))


def _task_readout(y: torch.Tensor, track_idx_t: torch.Tensor, out_lo: int, out_hi: int) -> torch.Tensor:
    # y: [B, T, L]
    y_sel = y.index_select(dim=1, index=track_idx_t)
    y_win = y_sel[:, :, out_lo:out_hi]
    return y_win.mean(dim=(1, 2))


def _infer_layer_channels(model: torch.nn.Module, layer_name: str) -> int:
    m = get_module_by_name(model, layer_name)
    for attr in ("out_channels", "channels"):
        v = getattr(m, attr, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    w = getattr(m, "weight", None)
    if torch.is_tensor(w) and w.ndim >= 1:
        return int(w.shape[0])
    raise RuntimeError(f"Could not infer channel count for layer {layer_name}")


@torch.no_grad()
def _score_abs_delta_group(
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

    ablator: Optional[ChannelAblator] = None
    if ablate is not None:
        layer, channels = ablate
        ablator = ChannelAblator(wrapper.model, layer_name=str(layer), channels=[int(c) for c in channels])
        ablator.register()

    try:
        for start in range(0, len(df_run), int(batch_size)):
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
            y_ref = _select_output_tensor(wrapper.model(x_ref), output_key=wrapper.output_key)

            x_alt = one_hot_encode_batch(alt_seqs, device=device)
            y_alt = _select_output_tensor(wrapper.model(x_alt), output_key=wrapper.output_key)

            ref_task = _task_readout(y_ref, track_idx_t, tw.out_lo, tw.out_hi)
            alt_task = _task_readout(y_alt, track_idx_t, tw.out_lo, tw.out_hi)

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


def _read_rank_tsv(rank_tsv: str | Path) -> List[int]:
    path = Path(rank_tsv)
    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        df = pd.read_csv(path, sep=sep)

    # Common formats in this repo: variant_channel_rank outputs 'channel'
    if "channel" in df.columns:
        ch = df["channel"].astype(int).tolist()
    else:
        # fallback heuristics
        for cand in ("channel_idx", "unit", "feature", "neuron"):
            if cand in df.columns:
                ch = df[cand].astype(int).tolist()
                break
        else:
            raise ValueError(f"Ranking file missing channel column: {rank_tsv} columns={df.columns.tolist()}")

    # Prefer explicit ordering columns when present.
    if "rank" in df.columns:
        df = df.sort_values("rank", ascending=True)
    elif "abs_rho" in df.columns:
        df = df.sort_values("abs_rho", ascending=False)
    elif "score" in df.columns:
        df = df.sort_values("score", ascending=False)

    if "channel" in df.columns:
        ch = df["channel"].astype(int).tolist()

    # Drop duplicates while preserving order
    seen: set[int] = set()
    out: List[int] = []
    for c in ch:
        c = int(c)
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def _parse_channels_spec(spec: str, layer_channels: int, seed: int) -> Tuple[str, List[int]]:
    """Parse --channels spec into (set_type, channels)."""
    spec = str(spec).strip()
    if spec.startswith("randomK:"):
        # randomK:K:seed
        parts = spec.split(":")
        if len(parts) not in {2, 3}:
            raise ValueError("randomK spec must be randomK:K or randomK:K:seed")
        k = int(parts[1])
        s = int(parts[2]) if len(parts) == 3 else int(seed)
        rng = np.random.default_rng(int(s))
        ch = rng.choice(np.arange(int(layer_channels)), size=int(k), replace=False).astype(int).tolist()
        return "random", sorted([int(x) for x in ch])

    if spec.startswith("topK:"):
        # topK:<path_to_rank_table>:K
        rest = spec[len("topK:") :]
        # allow colons in path by splitting from the end
        if ":" not in rest:
            raise ValueError("topK spec must be topK:<rank_tsv>:K")
        path_str, k_str = rest.rsplit(":", 1)
        k = int(k_str)
        top = _read_rank_tsv(path_str)[:k]
        return "top", sorted([int(x) for x in top])

    # explicit CSV
    ch = _parse_int_list_csv(spec)
    if not ch:
        raise ValueError("Parsed empty channel list from --channels")
    return "explicit", sorted([int(x) for x in ch])


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


def _apply_preset(args: argparse.Namespace) -> None:
    if args.preset is None:
        return
    if args.preset == "wholeblood_cage_abs_primary":
        args.track_group = "cage_wholeblood_ribopure"
        args.pad_bins = 4
        args.shift_bins = 4
        if args.layer is None and args.layers is None:
            args.layer = "final_joined_convs.0.conv_layer"
        if args.topk_list is None:
            args.topk_list = "1,5,10,20,50"
        if args.random_draws is None:
            args.random_draws = 30
        return
    raise SystemExit(f"Unknown preset: {args.preset}")


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def _atomic_write_tsv(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, sep="\t", index=False)
    tmp.replace(path)


def _summarize_curve(df_res: pd.DataFrame) -> pd.DataFrame:
    if df_res.empty:
        return pd.DataFrame(
            columns=[
                "layer",
                "K",
                "mean_auc_drop_topK",
                "mean_auc_drop_randomK",
                "std_auc_drop_randomK",
                "random_draws",
                "empirical_p",
                "effect_size",
            ]
        )

    summary_rows: List[dict] = []
    for (layer, k), sub in df_res.groupby(["layer", "K"], dropna=False):
        top = sub[sub["set_type"] == "top"]
        rnd = sub[sub["set_type"] == "random"]
        if len(top) != 1 or len(rnd) < 1:
            continue
        top_drop = float(top.iloc[0]["auc_drop"])
        rnd_drops = rnd["auc_drop"].astype(float).values

        p_emp = float((1.0 + float(np.sum(rnd_drops >= top_drop))) / (1.0 + float(len(rnd_drops))))
        mu = float(np.mean(rnd_drops))
        sd = float(np.std(rnd_drops, ddof=1)) if len(rnd_drops) > 1 else float("nan")
        eff = float((top_drop - mu) / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan")

        summary_rows.append(
            {
                "layer": str(layer),
                "K": int(k),
                "mean_auc_drop_topK": float(top_drop),
                "mean_auc_drop_randomK": float(mu),
                "std_auc_drop_randomK": float(sd),
                "random_draws": int(len(rnd_drops)),
                "empirical_p": float(p_emp),
                "effect_size": float(eff),
            }
        )

    df_sum = pd.DataFrame(summary_rows)
    if not df_sum.empty:
        df_sum = df_sum.sort_values(["layer", "K"]).reset_index(drop=True)
    return df_sum


def _write_curve_checkpoint(
    outdir: Path,
    args: argparse.Namespace,
    baseline: dict,
    rank_path: str,
    rows: List[dict],
) -> None:
    """Write intermediate results so preemption/disconnect still leaves usable outputs."""
    df_res = pd.DataFrame(rows)
    out_res = outdir / "ablation_results.tsv"
    _atomic_write_tsv(out_res, df_res)

    df_sum = _summarize_curve(df_res)
    out_sum = outdir / "summary.tsv"
    _atomic_write_tsv(out_sum, df_sum)

    manifest = {
        "timestamp": baseline.get("timestamp"),
        "git_commit": _git_commit(),
        "mode": "curve",
        "args": vars(args),
        "baseline": str(outdir / "baseline.json"),
        "rank_tsv": str(rank_path),
        "progress": {
            "n_rows": int(len(rows)),
            "n_layers": int(len(set([r.get("layer") for r in rows]))),
            "last_row": (rows[-1] if rows else None),
        },
        "outputs": {
            "ablation_results": str(out_res),
            "summary": str(out_sum),
        },
    }
    _atomic_write_json(outdir / "run_manifest.json", manifest)



def main() -> None:
    args = parse_args()
    _apply_preset(args)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Input validation: choose exactly one of (pos_ids,pos_vcf) and (neg_ids,neg_vcf)
    if (args.pos_ids is None) == (args.pos_vcf is None):
        raise SystemExit("Provide exactly one of --pos_ids or --pos_vcf")
    if (args.neg_ids is None) == (args.neg_vcf is None):
        raise SystemExit("Provide exactly one of --neg_ids or --neg_vcf")

    if args.layer is not None and args.layers is not None and str(args.layer) != str(args.layers):
        raise SystemExit("Provide only one of --layer or --layers")
    layers_s = args.layer if args.layer is not None else args.layers
    layers = _parse_str_list_csv(layers_s)
    if not layers:
        raise SystemExit("No layers specified; pass --layer")

    # Load variant IDs
    if args.pos_ids is not None:
        pos_ids = _load_variant_ids_from_list(args.pos_ids)
    else:
        pos_ids = _load_variant_ids_from_vcf(args.pos_vcf)

    if args.neg_ids is not None:
        neg_ids = _load_variant_ids_from_list(args.neg_ids)
    else:
        neg_ids = _load_variant_ids_from_vcf(args.neg_vcf)

    n_pos = _parse_n(args.n_pos)
    n_neg = _parse_n(args.n_neg)

    df_run = sample_posneg_ids(pos_ids, neg_ids, n_pos=n_pos, n_neg=n_neg, seed=int(args.seed))

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

    logger.info(f"Sampled variants: n={len(df_run)} (pos={int(df_run['label'].sum())}, neg={int((df_run['label']==0).sum())})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    wrapper: BorzoiWrapper = load_borzoi(str(args.model_name_or_path), device=device, output_key=args.output_key)

    seq_len = args.seq_len
    if seq_len is None:
        seq_len = detect_seq_len_from_crop(wrapper.model) or detect_seq_len(wrapper.model) or 262144
    seq_len = int(seq_len)

    # Determine output bins/tracks
    dummy = torch.zeros((1, 4, seq_len), device=device, dtype=torch.float32)
    y0 = _select_output_tensor(wrapper.model(dummy), output_key=wrapper.output_key)
    out_tracks = int(y0.shape[1])
    out_bins = int(y0.shape[2])

    groups = _load_groups(targets_tsv=str(args.targets_tsv), group_config_json=args.track_group_config)
    gmap: Dict[str, TrackGroup] = {g.name: g for g in groups}
    if args.track_group not in gmap:
        raise SystemExit(f"Unknown track_group: {args.track_group}. Available: {sorted(gmap.keys())}")
    group = gmap[args.track_group]

    track_idx = torch.tensor(list(group.track_indices), dtype=torch.long, device=device)
    if int(track_idx.max()) >= out_tracks:
        raise RuntimeError(f"Group track index out of range for model outputs: max_track={int(track_idx.max())} out_tracks={out_tracks}")

    tw = _compute_task_window(seq_len=seq_len, out_bins=out_bins, pad_bins=int(args.pad_bins), shift_bins=int(args.shift_bins))

    genome = Genome(Path(args.genome_fasta))

    # Baseline
    y_base, s_base, stats_base = _score_abs_delta_group(
        df_run=df_run,
        wrapper=wrapper,
        genome=genome,
        seq_len=seq_len,
        track_idx_t=track_idx,
        tw=tw,
        batch_size=int(args.batch_size),
        device=device,
        ablate=None,
    )
    if y_base.size == 0 or np.unique(y_base).size < 2:
        raise RuntimeError("Baseline produced no usable labels")
    auc_base = float(roc_auc_score(y_base, s_base))
    logger.info(f"Baseline AUROC(abs Δ(group)) = {auc_base:.6f} (scored n={int(y_base.size)})")

    baseline = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name_or_path": str(args.model_name_or_path),
        "output_key": args.output_key,
        "seq_len": int(seq_len),
        "out_tracks": int(out_tracks),
        "out_bins": int(out_bins),
        "track_group": {"name": group.name, "assay": group.assay, "tracks": list(group.track_indices)},
        "pad_bins": int(args.pad_bins),
        "shift_bins": int(args.shift_bins),
        "auc": float(auc_base),
        "stats": stats_base,
        "n_pos": int(y_base.sum()),
        "n_neg": int((y_base == 0).sum()),
    }
    (outdir / "baseline.json").write_text(json.dumps(baseline, indent=2))

    # Main experiment mode selection
    single_spec = args.channels is not None

    rows: List[dict] = []

    def run_one(layer: str, channels: Sequence[int], set_type: str, k: int, draw_idx: int) -> None:
        y_ab, s_ab, stats_ab = _score_abs_delta_group(
            df_run=df_run,
            wrapper=wrapper,
            genome=genome,
            seq_len=seq_len,
            track_idx_t=track_idx,
            tw=tw,
            batch_size=int(args.batch_size),
            device=device,
            ablate=(layer, list(channels)),
        )
        auc = float(roc_auc_score(y_ab, s_ab))
        drop = float(auc_base - auc)
        rows.append(
            {
                "layer": str(layer),
                "K": int(k),
                "set_type": str(set_type),
                "draw_idx": int(draw_idx),
                "channels": ",".join(map(str, sorted([int(x) for x in channels]))),
                "auc": float(auc),
                "auc_drop": float(drop),
                "baseline_auc": float(auc_base),
                "n_scored": int(stats_ab.get("n_scored", 0)),
                "skipped_ref_mismatch": int(stats_ab.get("skipped_ref_mismatch", 0)),
            }
        )

    if single_spec:
        for layer in layers:
            C = _infer_layer_channels(wrapper.model, layer)
            set_type, ch = _parse_channels_spec(args.channels, layer_channels=C, seed=int(args.seed))
            run_one(layer=layer, channels=ch, set_type=set_type, k=len(ch), draw_idx=0)

        df_res = pd.DataFrame(rows)
        out_res = outdir / "ablation_results.tsv"
        df_res.to_csv(out_res, sep="\t", index=False)

        manifest = {
            "timestamp": baseline["timestamp"],
            "git_commit": _git_commit(),
            "mode": "single",
            "args": vars(args),
            "baseline": str(outdir / "baseline.json"),
            "outputs": {"ablation_results": str(out_res)},
        }
        (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        logger.info(f"Wrote: {out_res}")
        return

    # Curve mode: topK vs randomK for each K
    ks = _parse_int_list_csv(args.topk_list)
    if not ks:
        raise SystemExit("--topk_list parsed empty")

    if args.rank_tsv is None and (args.channels is None or not str(args.channels).startswith("topK:")):
        raise SystemExit("Provide --rank_tsv (or --channels topK:<rank_tsv>:K) for curve mode")

    rank_path = args.rank_tsv
    if rank_path is None:
        # derive from channels spec
        rest = str(args.channels)[len("topK:") :]
        rank_path, _k = rest.rsplit(":", 1)

    # Rank file can either be layer-specific (no layer column) or combined (layer column).
    df_rank = None
    path_rank = Path(rank_path)
    if path_rank.suffix.lower() == ".parquet":
        df_rank = pd.read_parquet(path_rank)
    else:
        sep = "\t" if path_rank.suffix.lower() in {".tsv", ".txt"} else ","
        df_rank = pd.read_csv(path_rank, sep=sep)

    has_layer_col = "layer" in df_rank.columns
    if len(layers) > 1 and not has_layer_col:
        raise SystemExit("Multiple layers requested but rank file has no 'layer' column; run per-layer or provide a combined rank file")

    rng = np.random.default_rng(int(args.seed))

    # Ensure we still leave intermediate results if the job is preempted.
    def _on_signal(signum, _frame):
        try:
            logger.warning(f"Received signal {signum}; checkpointing partial results then exiting")
            _write_curve_checkpoint(outdir=outdir, args=args, baseline=baseline, rank_path=str(rank_path), rows=rows)
        finally:
            raise SystemExit(128 + int(signum))

    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _on_signal)
        except Exception:
            pass

    for layer in layers:
        C = _infer_layer_channels(wrapper.model, layer)
        logger.info(f"Layer {layer}: inferred channels={C}")

        if has_layer_col:
            df_l = df_rank[df_rank["layer"].astype(str) == str(layer)].copy()
            if df_l.empty:
                raise SystemExit(f"Rank file missing rows for layer={layer}")
            # write to a temporary in-memory selection order
            if "rank" in df_l.columns:
                df_l = df_l.sort_values("rank", ascending=True)
            elif "abs_rho" in df_l.columns:
                df_l = df_l.sort_values("abs_rho", ascending=False)
            elif "score" in df_l.columns:
                df_l = df_l.sort_values("score", ascending=False)
            top_all = [int(x) for x in df_l["channel"].astype(int).tolist()]
        else:
            top_all = _read_rank_tsv(rank_path)

        # Determine universe for random sampling
        universe = np.arange(C, dtype=np.int32)
        exclude: set[int] = set()
        if bool(args.random_exclude_top):
            exclude = set(top_all[: max(ks)])
        if exclude:
            universe = np.asarray([int(c) for c in universe.tolist() if int(c) not in exclude], dtype=np.int32)
            if len(universe) < int(max(ks)):
                universe = np.arange(C, dtype=np.int32)

        for k in ks:
            k = int(k)
            topk = top_all[:k]
            if len(topk) < k:
                raise RuntimeError(f"Ranking file has insufficient channels: need {k}, have {len(top_all)}")

            run_one(layer=layer, channels=topk, set_type="top", k=k, draw_idx=0)
            _write_curve_checkpoint(outdir=outdir, args=args, baseline=baseline, rank_path=str(rank_path), rows=rows)

            for di in range(int(args.random_draws)):
                ch = rng.choice(universe, size=k, replace=False).astype(int).tolist()
                run_one(layer=layer, channels=ch, set_type="random", k=k, draw_idx=int(di))
                _write_curve_checkpoint(outdir=outdir, args=args, baseline=baseline, rank_path=str(rank_path), rows=rows)

    df_res = pd.DataFrame(rows)
    out_res = outdir / "ablation_results.tsv"
    df_res.to_csv(out_res, sep="\t", index=False)
    logger.info(f"Wrote: {out_res}")

    df_sum = _summarize_curve(df_res)
    out_sum = outdir / "summary.tsv"
    df_sum.to_csv(out_sum, sep="\t", index=False)
    logger.info(f"Wrote: {out_sum}")

    # Plot: AUROC drop vs K (top vs random mean ± std)
    plot_paths: Dict[str, str] = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Combined plot (multiple layers)
        plt.figure(figsize=(10, 5))
        for layer in sorted(df_sum["layer"].unique().tolist()):
            s = df_sum[df_sum["layer"] == layer].sort_values("K")
            if s.empty:
                continue
            x = s["K"].astype(int).values
            y_top = s["mean_auc_drop_topK"].astype(float).values
            y_mu = s["mean_auc_drop_randomK"].astype(float).values
            y_sd = s["std_auc_drop_randomK"].astype(float).values
            plt.plot(x, y_top, marker="o", label=f"topK drop: {layer}")
            plt.plot(x, y_mu, linestyle="--", alpha=0.8, label=f"random mean: {layer}")
            plt.fill_between(x, y_mu - y_sd, y_mu + y_sd, alpha=0.15)

        plt.axhline(0.0, color="black", linewidth=1)
        plt.xlabel("K channels ablated")
        plt.ylabel("AUROC drop (baseline - ablated)")
        plt.title("AUC drop vs K (topK vs randomK)")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        out_png = outdir / "auc_drop_vs_k.png"
        plt.savefig(out_png, dpi=200)
        plt.close()
        plot_paths["auc_drop_vs_k"] = str(out_png)
        logger.info(f"Wrote: {out_png}")

        # Per-layer plots
        for layer in sorted(df_sum["layer"].unique().tolist()):
            s = df_sum[df_sum["layer"] == layer].sort_values("K")
            if s.empty:
                continue
            x = s["K"].astype(int).values
            y_top = s["mean_auc_drop_topK"].astype(float).values
            y_mu = s["mean_auc_drop_randomK"].astype(float).values
            y_sd = s["std_auc_drop_randomK"].astype(float).values

            plt.figure(figsize=(7, 4))
            plt.plot(x, y_top, marker="o", label="topK")
            plt.plot(x, y_mu, linestyle="--", label="random mean")
            plt.fill_between(x, y_mu - y_sd, y_mu + y_sd, alpha=0.2, label="random ±1sd")
            plt.axhline(0.0, color="black", linewidth=1)
            plt.xlabel("K")
            plt.ylabel("AUROC drop")
            plt.title(f"AUC drop vs K ({layer})")
            plt.legend(loc="best")
            plt.tight_layout()
            safe = layer.replace(".", "_")
            pth = outdir / f"auc_drop_vs_k_{safe}.png"
            plt.savefig(pth, dpi=200)
            plt.close()
            plot_paths[f"auc_drop_vs_k_{safe}"] = str(pth)
            logger.info(f"Wrote: {pth}")

    except Exception as e:
        logger.warning(f"Plotting failed: {e}")

    manifest = {
        "timestamp": baseline["timestamp"],
        "git_commit": _git_commit(),
        "mode": "curve",
        "args": vars(args),
        "baseline": str(outdir / "baseline.json"),
        "rank_tsv": str(rank_path),
        "outputs": {
            "ablation_results": str(out_res),
            "summary": str(out_sum),
            **plot_paths,
        },
    }
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
