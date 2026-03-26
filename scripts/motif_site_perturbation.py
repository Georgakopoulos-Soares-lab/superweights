#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import hashlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Make local imports work when running this as a script.
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Avoid TensorFlow auto-import (and reduce noisy logs) when using HuggingFace.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy.stats import wilcoxon
from scipy.stats import spearmanr

from borzoi.activations import ActivationMapCapturer
from borzoi.model import default_device, detect_seq_len, detect_seq_len_from_crop, load_borzoi, score_expression
from borzoi.genome import Genome, bin_size_bp, window_start0
from borzoi.io import ensure_dir, write_parquet, write_text
from borzoi.data import parse_variant_id_any
from borzoi.encode import one_hot_encode_batch
from borzoi.genome import make_ref_alt_sequence


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Motif-site perturbation experiment using sanitized FIMO hits: WT vs motif-mutation vs control-mutation, "
            "scoring Borzoi output and a specific channel activation."
        )
    )

    p.add_argument("--fimo_tsv", required=True, help="Sanitized FIMO TSV (motif instances)")
    p.add_argument("--map_tsv", required=True, help="Sanitized FASTA map TSV (id -> original_header)")
    p.add_argument("--genome_fasta", required=True, help="Genome FASTA (hg38.fa)")

    p.add_argument("--motif_id", default="6-CGGAAG", help="Motif ID to perturb (must match motif_id column in fimo.tsv)")

    p.add_argument("--layer", default="horizontal_conv1.conv_layer")
    p.add_argument("--channel", type=int, default=130, help="Single channel index (ignored if --channels is set)")
    p.add_argument(
        "--channels",
        default=None,
        help="Comma-separated channel indices to measure in one run (e.g. 130,12,77)",
    )

    p.add_argument("--model_name", default="johahi/borzoi-replicate-0")
    p.add_argument("--output_key", default=None)
    p.add_argument("--seq_len", type=int, default=None)

    p.add_argument("--flank_bp", type=int, default=None, help="Flank used in the (near_variant.*.<flank>) FASTA")
    p.add_argument("--max_hits", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument(
        "--mut_mode",
        choices=["shuffle", "substitute", "baseflip", "dinuc_shuffle", "pwm_minimize"],
        default="shuffle",
        help=(
            "Mutation scheme for motif/control segments: "
            "shuffle=mono-nucleotide shuffle; substitute=fill with --sub_base; "
            "baseflip=random base != original (A/C/G/T only); dinuc_shuffle=preserve dinucleotide counts."
        ),
    )
    p.add_argument("--sub_base", default="N", help="Used only for mut_mode=substitute")
    p.add_argument(
        "--baseflip_preserve_gc",
        action="store_true",
        help="For mut_mode=baseflip, bias replacements to preserve GC vs AT when possible.",
    )

    # PWM-minimizing mutation (guaranteed disruption w.r.t. a chosen PWM).
    p.add_argument(
        "--pwm_meme",
        default="/scratch/10906/arisk/envs/superweights-borzoi/share/meme-5.5.9/doc/examples/example-datasets/JASPAR2018_CORE_non-redundant.meme",
        help="MEME motif database used for PWM-minimizing mode and auditing.",
    )
    p.add_argument(
        "--pwm_id",
        default="MA0062.2",
        help="Motif ID in the MEME DB for PWM-minimizing mode (default: Gabpa MA0062.2).",
    )
    p.add_argument(
        "--pwm_topk",
        type=int,
        default=3,
        help="How many high-information positions to mutate in PWM-minimize mode.",
    )
    p.add_argument(
        "--pwm_scan_pad",
        type=int,
        default=8,
        help="When scoring a longer PWM, scan within +/- this many bp around the motif hit region.",
    )
    p.add_argument(
        "--audit_pwm",
        action="store_true",
        help="Record PWM strength WT vs mutated (both STREME motif and JASPAR PWM, if available).",
    )
    p.add_argument(
        "--streme_txt",
        default="results/motifs/ch130_streme_211/streme.txt",
        help="STREME output text (used to load the query PWM for motif_id like 6-CGGAAG).",
    )

    p.add_argument("--control_min_gap", type=int, default=20, help="Min bp gap between motif and control region")
    p.add_argument("--control_max_tries", type=int, default=200)

    # Output readout sweep (track-level, local-bin delta-delta output).
    p.add_argument(
        "--output_sweep",
        action="store_true",
        help=(
            "Compute per-track, local-bin delta-delta output under motif mutation (mut - WT), "
            "and write a track table to the outdir. This avoids global averaging over all bins/tracks."
        ),
    )
    p.add_argument(
        "--targets_tsv",
        default="../../borzoi_hg38/targets.txt",
        help="Borzoi targets metadata TSV to map track indices -> identifiers/descriptions.",
    )
    p.add_argument(
        "--track_regex",
        default=None,
        help=(
            "Optional case-insensitive regex to select a subset of tracks based on targets TSV (identifier/description). "
            "Example: 'blood|PBMC|monocyte|lymph'"
        ),
    )
    p.add_argument(
        "--track_field",
        choices=["identifier", "description", "both"],
        default="both",
        help="Which targets TSV field(s) to match against --track_regex.",
    )
    p.add_argument(
        "--output_sweep_pad_bp",
        type=int,
        default=0,
        help="Pad bp added on each side of the motif interval when computing local-bin output readouts.",
    )
    p.add_argument(
        "--output_sweep_pad_bins",
        type=int,
        default=None,
        help=(
            "Pad in OUTPUT bins (added on each side of the motif interval) when computing local-bin readouts. "
            "If set, overrides --output_sweep_pad_bp."
        ),
    )
    p.add_argument(
        "--output_sweep_pad_bins_grid",
        default=None,
        help=(
            "Optional comma-separated pad sizes (in OUTPUT bins) to evaluate in one run. "
            "Example: 0,1,2,4. If set, overrides --output_sweep_pad_bins/--output_sweep_pad_bp."
        ),
    )
    p.add_argument(
        "--output_sweep_shift_bins",
        default="0",
        help=(
            "Comma-separated integer shifts (in OUTPUT bins) to apply to the motif-centered bin window. "
            "Example: -4,-2,-1,0,1,2,4. Note: if the value starts with '-', pass it as --output_sweep_shift_bins=-4,-2,... "
            "(argparse may otherwise interpret it as flags)."
        ),
    )
    p.add_argument(
        "--output_sweep_track_indices",
        default=None,
        help=(
            "Optional comma-separated explicit track indices to use for output sweep (overrides --track_regex). "
            "Example: 123,456,789"
        ),
    )
    p.add_argument(
        "--output_sweep_topk_tracks",
        type=int,
        default=5,
        help="How many top tracks (by |median delta-delta|) to report/emit correlations for.",
    )
    p.add_argument(
        "--output_sweep_corr_all_selected",
        action="store_true",
        help=(
            "If activation parquets are provided, compute rho/p correlations for ALL selected tracks "
            "(and for ALL --output_sweep_shift_bins) rather than only top-K tracks."
        ),
    )
    p.add_argument(
        "--act_early_parquet",
        default=None,
        help="Optional parquet with activation effects to correlate against outputs (early).",
    )
    p.add_argument("--act_early_layer", default="horizontal_conv1.conv_layer")
    p.add_argument("--act_early_channel", type=int, default=130)
    p.add_argument(
        "--act_receiver_parquet",
        default=None,
        help="Optional parquet with activation effects to correlate against outputs (receiver/pre-head).",
    )
    p.add_argument("--act_receiver_layer", default="final_joined_convs.0.conv_layer")
    p.add_argument("--act_receiver_channel", type=int, default=1181)

    p.add_argument("--batch_hits", type=int, default=1, help="How many motif instances per forward (each hit expands to 6 sequences)")
    p.add_argument("--log_every", type=int, default=50, help="Log progress every N motif hits")
    p.add_argument("--outdir", default="results/motifs/perturb")

    return p.parse_args()


def _infer_flank_bp(args: argparse.Namespace) -> int:
    if args.flank_bp is not None:
        return int(args.flank_bp)

    # Try to infer from filenames (e.g. near_variant.pos.211.sanitized.*)
    for s in [str(args.map_tsv), str(args.fimo_tsv)]:
        m = re.search(r"\.(\d+)\.sanitized\.", s)
        if m:
            return int(m.group(1))
        m = re.search(r"_(\d+)_sanitized", s)
        if m:
            return int(m.group(1))
        m = re.search(r"_(\d+)(?:\b|\.)", s)
        if m and int(m.group(1)) in {100, 150, 200, 211, 250, 300, 400, 500}:
            return int(m.group(1))

    raise ValueError("Could not infer flank_bp; pass --flank_bp explicitly")


def _load_map_tsv(path: Path) -> Dict[str, str]:
    df = pd.read_csv(path, sep="\t")
    if "id" not in df.columns or "original_header" not in df.columns:
        raise ValueError("map TSV must have columns: id, original_header")
    return dict(zip(df["id"].astype(str), df["original_header"].astype(str)))


_site_center_re = re.compile(r"(?:^|\|)site_center1=(\d+)(?:\||$)")


def _parse_original_header(original_header: str) -> Tuple[str, int, str]:
    # original header format begins with variant id, then |key=value fields.
    # Example:
    #   10:101032154:C:A|label=1|site_center1=101032058|...
    variant_id = original_header.split("|")[0]
    m = _site_center_re.search(original_header)
    if not m:
        raise ValueError(f"Could not parse site_center1 from header: {original_header}")
    site_center1 = int(m.group(1))
    return variant_id, site_center1, original_header


def _mutate_seq(
    seq: str,
    start: int,
    end: int,
    rng: np.random.Generator,
    mode: str,
    sub_base: str,
    baseflip_preserve_gc: bool,
) -> str:
    if start < 0 or end > len(seq) or end <= start:
        raise ValueError("Invalid mutation bounds")

    motif = seq[start:end]
    if mode == "shuffle":
        arr = list(motif)
        rng.shuffle(arr)
        motif2 = "".join(arr)
    elif mode == "dinuc_shuffle":
        motif2 = _dinuc_shuffle(motif, rng)
    elif mode == "baseflip":
        motif2 = _baseflip(motif, rng, preserve_gc=baseflip_preserve_gc)
    else:
        motif2 = (sub_base.upper() * (end - start))

    return seq[:start] + motif2 + seq[end:]


def _select_output_tensor(output: object, output_key: Optional[str]) -> torch.Tensor:
    """Select the model output tensor without reducing it.

    Mirrors score_expression's output_key / tuple handling, but returns the full tensor.
    Expected shape: [B, tracks, bins] (or [B, tracks]).
    """
    out = output
    if isinstance(out, dict):
        if output_key is not None:
            if output_key not in out:
                raise KeyError(f"output_key not in output dict: {output_key}")
            out = out[output_key]
        else:
            for v in out.values():
                out = v
                break
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not torch.is_tensor(out):
        raise TypeError(f"Unsupported output type: {type(out)}")
    return out


def _load_targets_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    # tolerate optional leading unnamed index column
    if "identifier" not in df.columns:
        raise ValueError(f"targets TSV missing identifier column: {path}")
    if "description" not in df.columns:
        df["description"] = ""
    return df


def _select_track_indices(targets: pd.DataFrame, regex: Optional[str], field: str) -> np.ndarray:
    if regex is None or str(regex).strip() == "":
        return np.arange(len(targets), dtype=np.int64)
    pat = re.compile(str(regex), flags=re.IGNORECASE)

    def _match_row(r: pd.Series) -> bool:
        ident = str(r.get("identifier", ""))
        desc = str(r.get("description", ""))
        if field == "identifier":
            return pat.search(ident) is not None
        if field == "description":
            return pat.search(desc) is not None
        return (pat.search(ident) is not None) or (pat.search(desc) is not None)

    mask = targets.apply(_match_row, axis=1).astype(bool).to_numpy()
    idx = np.where(mask)[0].astype(np.int64)
    return idx


def _parse_int_list_csv(s: Optional[str]) -> List[int]:
    if s is None:
        return []
    s = str(s).strip()
    if s == "":
        return []
    out: List[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        out.append(int(tok))
    return out


def _extract_act_effect_series(
    parquet_path: Path, *, layer: str, channel: int, metric: str = "effect_act_ref_wt_minus_motif"
) -> pd.Series:
    df = pd.read_parquet(parquet_path)
    need = {"layer", "channel", "fimo_sequence_name", metric}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {parquet_path}: {sorted(missing)}")
    df = df[(df["layer"].astype(str) == str(layer)) & (df["channel"].astype(int) == int(channel))].copy()
    if df.empty:
        raise ValueError(f"No rows for layer={layer} channel={channel} in {parquet_path}")
    # dedup by sequence_name just in case
    df = df.sort_values(["fimo_sequence_name"]).drop_duplicates(subset=["fimo_sequence_name"], keep="first")
    s = pd.Series(df[metric].astype(float).values, index=df["fimo_sequence_name"].astype(str).values)
    s.name = f"{Path(parquet_path).name}:{layer}:ch{int(channel)}"
    return s


def _baseflip(motif: str, rng: np.random.Generator, preserve_gc: bool) -> str:
    """Replace each A/C/G/T with a different base; leave non-ACGT unchanged."""
    out: List[str] = []
    for ch in motif.upper():
        if ch not in {"A", "C", "G", "T"}:
            out.append(ch)
            continue
        choices = [b for b in ["A", "C", "G", "T"] if b != ch]
        if preserve_gc:
            if ch in {"G", "C"}:
                choices = [b for b in choices if b in {"G", "C"}] or choices
            else:
                choices = [b for b in choices if b in {"A", "T"}] or choices
        out.append(str(rng.choice(choices)))
    return "".join(out)


def _dinuc_shuffle(motif: str, rng: np.random.Generator) -> str:
    """Dinucleotide-preserving shuffle for A/C/G/T-only strings.

    Falls back to mono-nucleotide shuffle if the motif contains non-ACGT or is too short.
    """
    s = motif.upper()
    if len(s) < 3 or any(ch not in {"A", "C", "G", "T"} for ch in s):
        arr = list(s)
        rng.shuffle(arr)
        return "".join(arr)

    outgoing: Dict[str, List[str]] = {"A": [], "C": [], "G": [], "T": []}
    for a, b in zip(s[:-1], s[1:]):
        outgoing[a].append(b)
    for a in outgoing:
        rng.shuffle(outgoing[a])

    # Eulerian trail via Hierholzer's algorithm.
    stack = [s[0]]
    path: List[str] = []
    while stack:
        v = stack[-1]
        if outgoing[v]:
            stack.append(outgoing[v].pop())
        else:
            path.append(stack.pop())
    path = path[::-1]
    if len(path) != len(s):
        arr = list(s)
        rng.shuffle(arr)
        return "".join(arr)
    return "".join(path)


_COMP = str.maketrans({"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"})


def _revcomp(seq: str) -> str:
    return seq.upper().translate(_COMP)[::-1]


def _load_meme_pwm(meme_path: Path, motif_id: str) -> np.ndarray:
    """Load a single PWM from a MEME-format motif DB. Returns shape [W, 4] in A,C,G,T order."""
    motif_id = str(motif_id).strip()
    with open(meme_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("MOTIF "):
            parts = line.split()
            if len(parts) >= 2 and parts[1] == motif_id:
                # find next letter-probability matrix
                j = i + 1
                while j < len(lines) and "letter-probability matrix" not in lines[j]:
                    j += 1
                if j >= len(lines):
                    raise ValueError(f"No PWM found for motif {motif_id} in {meme_path}")
                header = lines[j]
                m = re.search(r"\bw\s*=\s*(\d+)", header)
                if not m:
                    raise ValueError(f"Could not parse PWM width from: {header.strip()}")
                w = int(m.group(1))
                mat = []
                for k in range(w):
                    row = lines[j + 1 + k].strip().split()
                    if len(row) < 4:
                        raise ValueError(f"Bad PWM row for motif {motif_id}: {lines[j+1+k]}")
                    mat.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
                return np.array(mat, dtype=np.float64)
        i += 1

    raise ValueError(f"Motif {motif_id} not found in {meme_path}")


def _load_streme_pwm(streme_txt: Path, motif_id: str) -> np.ndarray:
    """Load STREME motif PWM from streme.txt. Returns shape [W, 4] in A,C,G,T order."""
    motif_id = str(motif_id).strip()
    with open(streme_txt, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    # Look for section starting with 'MOTIF <id>' (STREME may include extra tokens after the ID)
    for i, line in enumerate(lines):
        if line.strip().startswith(f"MOTIF {motif_id} ") or line.strip() == f"MOTIF {motif_id}":
            j = i + 1
            while j < len(lines) and "letter-probability matrix" not in lines[j]:
                j += 1
            if j >= len(lines):
                raise ValueError(f"No PWM matrix for {motif_id} in {streme_txt}")
            header = lines[j]
            m = re.search(r"\bw\s*=\s*(\d+)", header)
            if not m:
                raise ValueError(f"Could not parse PWM width from: {header.strip()}")
            w = int(m.group(1))
            mat = []
            for k in range(w):
                row = lines[j + 1 + k].strip().split()
                if len(row) < 4:
                    raise ValueError(f"Bad STREME PWM row: {lines[j+1+k]}")
                mat.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
            return np.array(mat, dtype=np.float64)
    raise ValueError(f"Motif {motif_id} not found in {streme_txt}")


def _pwm_log_odds_score(seq: str, pwm: np.ndarray, bg: float = 0.25, eps: float = 1e-6) -> float:
    """Score an A/C/G/T string of length W against PWM with log-odds; returns -inf if non-ACGT present."""
    s = seq.upper()
    if len(s) != int(pwm.shape[0]):
        raise ValueError("Sequence length must equal PWM width")
    idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    score = 0.0
    for i, ch in enumerate(s):
        if ch not in idx:
            return float("-inf")
        p = float(pwm[i, idx[ch]])
        score += float(np.log((p + eps) / bg))
    return float(score)


def _pwm_best_score(seq: str, pwm: np.ndarray) -> Tuple[float, str]:
    """Best strand score (forward vs revcomp) for a PWM on a fixed-length seq."""
    s = seq.upper()
    f = _pwm_log_odds_score(s, pwm)
    rc = _pwm_log_odds_score(_revcomp(s), pwm)
    if rc > f:
        return rc, "-"
    return f, "+"


def _pwm_scan_best(seq: str, pwm: np.ndarray) -> Tuple[float, int, str]:
    """Scan best PWM score across all offsets in seq (both strands). Returns (best_score, offset0, strand)."""
    w = int(pwm.shape[0])
    if len(seq) < w:
        return float("-inf"), -1, "+"
    best = float("-inf")
    best_off = -1
    best_strand = "+"
    for off in range(0, len(seq) - w + 1):
        window = seq[off : off + w]
        sc, strand = _pwm_best_score(window, pwm)
        if sc > best:
            best = sc
            best_off = off
            best_strand = strand
    return best, best_off, best_strand


def _pwm_info(pwm: np.ndarray, bg: float = 0.25, eps: float = 1e-6) -> np.ndarray:
    """Information content per position (nats)."""
    p = np.clip(pwm.astype(np.float64), eps, 1.0)
    return (p * (np.log(p) - np.log(bg))).sum(axis=1)


def _mutate_pwm_minimize(
    seq: str,
    start: int,
    end: int,
    pwm: np.ndarray,
    scan_pad: int,
    topk: int,
) -> str:
    """Deterministic mutation that reduces PWM score near the motif hit.

    Finds the best-scoring alignment of the PWM in a window around [start,end), then mutates up to topk high-info
    positions that overlap [start,end) to the lowest-probability base under the PWM (and != current base).
    """
    w = int(pwm.shape[0])
    L = len(seq)
    lo = max(0, int(start) - int(scan_pad))
    hi = min(L, int(end) + int(scan_pad) + w)
    local = seq[lo:hi]
    best, best_off, best_strand = _pwm_scan_best(local, pwm)
    if best_off < 0:
        return seq

    # Determine alignment start in global coordinates.
    align0 = lo + best_off
    # Use PWM orientation that gave best score.
    pwm_use = pwm
    if best_strand == "-":
        pwm_use = pwm[::-1, ::-1]

    info = _pwm_info(pwm_use)
    # candidate PWM positions that overlap mutation segment.
    cand = []
    for i in range(w):
        gpos = align0 + i
        if gpos < start or gpos >= end:
            continue
        cand.append((float(info[i]), i, gpos))
    if not cand:
        return seq
    cand.sort(key=lambda t: (-t[0], t[1]))
    cand = cand[: max(1, int(topk))]

    seq_list = list(seq)
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    idx_to_base = ["A", "C", "G", "T"]
    for _ic, i, gpos in cand:
        cur = seq_list[gpos].upper()
        if cur not in base_to_idx:
            continue
        probs = pwm_use[i]
        order = np.argsort(probs)  # low to high
        # choose lowest-prob base that changes the nucleotide
        for bidx in order:
            b = idx_to_base[int(bidx)]
            if b != cur:
                seq_list[gpos] = b
                break
    return "".join(seq_list)


def _mutate_pwm_minimize_fixed_segment(
    seq: str,
    start: int,
    end: int,
    pwm: np.ndarray,
    topk: int,
    strand: str,
) -> str:
    """Deterministic mutation that reduces PWM score for the *fixed* segment [start,end).

    This is intended for when the PWM width matches the motif hit width (e.g. STREME query motif).
    It is strand-aware: if strand == '-', it minimizes the PWM score on the reverse-complement binding
    sequence (so the genomic segment is mutated accordingly).
    """
    w = int(pwm.shape[0])
    if (end - start) != w:
        raise ValueError("Fixed-segment PWM minimize requires (end-start)==PWM width")

    strand = str(strand)
    if strand not in {"+", "-"}:
        strand = "+"

    info = _pwm_info(pwm)
    order = np.argsort(-info)
    k = max(1, int(topk))
    order = order[:k]

    seq_list = list(seq)
    idx_to_base = ["A", "C", "G", "T"]
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    comp = {"A": "T", "C": "G", "G": "C", "T": "A"}

    for i in order:
        i = int(i)
        if strand == "+":
            gpos = start + i
            cur = seq_list[gpos].upper()
            if cur not in base_to_idx:
                continue
            probs = pwm[i]
            for bidx in np.argsort(probs):
                b = idx_to_base[int(bidx)]
                if b != cur:
                    seq_list[gpos] = b
                    break
        else:
            # binding sequence is revcomp of genomic segment;
            # PWM position i corresponds to genomic position (end-1-i)
            gpos = end - 1 - i
            cur_g = seq_list[gpos].upper()
            if cur_g not in base_to_idx:
                continue
            probs = pwm[i]
            for bidx in np.argsort(probs):
                b_bind = idx_to_base[int(bidx)]
                b_gen = comp[b_bind]
                if b_gen != cur_g:
                    seq_list[gpos] = b_gen
                    break

    return "".join(seq_list)


def _parse_channels(args: argparse.Namespace) -> List[int]:
    if args.channels is None or str(args.channels).strip() == "":
        return [int(args.channel)]
    chs: List[int] = []
    for part in str(args.channels).split(","):
        part = part.strip()
        if not part:
            continue
        chs.append(int(part))
    return chs if chs else [int(args.channel)]


def _channels_tag(channels: List[int]) -> str:
    """Compact tag for filenames when many channels are requested.

    Prevents Linux ENAMETOOLONG errors when users pass long `--channels` lists.
    """
    ch = [int(c) for c in channels]
    if len(ch) <= 6:
        return "chs" + "-".join(map(str, ch))
    s = ",".join(map(str, ch)).encode("utf-8")
    h = hashlib.md5(s).hexdigest()[:10]
    return f"chs{len(ch)}_{h}"


def _pick_control_region(
    seq: str,
    motif_start: int,
    motif_end: int,
    width: int,
    rng: np.random.Generator,
    min_gap: int,
    max_tries: int,
) -> Optional[Tuple[int, int]]:
    L = len(seq)
    if width <= 0 or width > L:
        return None

    forbidden_lo = max(0, motif_start - min_gap - width)
    forbidden_hi = min(L, motif_end + min_gap + width)

    # sample uniformly; reject overlap and low-quality regions.
    for _ in range(int(max_tries)):
        s = int(rng.integers(0, L - width + 1))
        e = s + width
        # reject if overlaps motif or within forbidden band
        if not (e <= motif_start - min_gap or s >= motif_end + min_gap):
            continue
        if forbidden_lo <= s <= forbidden_hi:
            # extra cushion, keeps away from the motif
            continue
        # avoid mutating into Ns-heavy region
        if seq[s:e].count("N") > 0:
            continue
        return s, e

    return None


@dataclass
class _BatchHit:
    variant_id: str
    chrom: str
    pos1: int
    ref: str
    alt: str
    site_center1: int
    motif_start0: int
    motif_end0: int
    motif_off_start: int
    motif_off_end: int
    ctrl_off_start: int
    ctrl_off_end: int
    fimo_row: dict


def main() -> None:
    args = parse_args()

    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))

    outdir = ensure_dir(args.outdir)
    flank_bp = _infer_flank_bp(args)

    fimo_tsv = Path(args.fimo_tsv)
    map_tsv = Path(args.map_tsv)

    logger.info(f"Loading FIMO: {fimo_tsv}")
    df_fimo = pd.read_csv(fimo_tsv, sep="\t", comment="#")
    if df_fimo.empty:
        raise SystemExit(f"Empty FIMO TSV: {fimo_tsv}")

    if "motif_id" not in df_fimo.columns:
        raise ValueError("FIMO TSV missing motif_id column")

    df_fimo = df_fimo[df_fimo["motif_id"].astype(str) == str(args.motif_id)].copy()
    if df_fimo.empty:
        raise SystemExit(f"No hits for motif_id={args.motif_id} in {fimo_tsv}")

    # Deduplicate per sequence (keep best p-value), then take top-K by p-value.
    df_fimo = df_fimo.sort_values(["sequence_name", "p-value", "q-value"], ascending=[True, True, True])
    df_fimo = df_fimo.groupby("sequence_name", as_index=False).first()
    df_fimo = df_fimo.sort_values(["p-value", "q-value"], ascending=[True, True]).head(int(args.max_hits)).copy()

    logger.info(f"Using hits={len(df_fimo)} motif_id={args.motif_id} (deduped by sequence_name), flank_bp={flank_bp}")

    id_to_header = _load_map_tsv(map_tsv)

    streme_pwm: Optional[np.ndarray] = None
    jaspar_pwm: Optional[np.ndarray] = None
    if bool(args.audit_pwm) or str(args.mut_mode) == "pwm_minimize":
        # Load STREME query motif PWM if available.
        try:
            streme_path = Path(str(args.streme_txt))
            if streme_path.exists():
                streme_pwm = _load_streme_pwm(streme_path, str(args.motif_id))
        except Exception as e:
            logger.warning(f"Could not load STREME PWM for {args.motif_id}: {e}")

        # Load chosen JASPAR PWM.
        try:
            jaspar_path = Path(str(args.pwm_meme))
            if jaspar_path.exists():
                jaspar_pwm = _load_meme_pwm(jaspar_path, str(args.pwm_id))
        except Exception as e:
            logger.warning(f"Could not load JASPAR PWM {args.pwm_id} from {args.pwm_meme}: {e}")

    device = default_device()
    wrapper = load_borzoi(args.model_name, device=device, output_key=args.output_key)

    seq_len = args.seq_len
    if seq_len is None:
        seq_len = detect_seq_len_from_crop(wrapper.model) or detect_seq_len(wrapper.model) or 262144
    seq_len = int(seq_len)

    # Warm-up forward for shape acceptance.
    with torch.no_grad():
        _ = wrapper.model(torch.zeros((1, 4, seq_len), device=device, dtype=torch.float32))

    genome_fa = Path(args.genome_fasta)
    if not genome_fa.exists():
        raise SystemExit(f"Genome FASTA not found: {genome_fa}")
    genome = Genome(genome_fa)
    # Force file open early so we fail fast (otherwise errors can be swallowed in per-hit try/except).
    _ = genome.fasta

    # Optional: output readout sweep (track-level, local bins).
    sweep_enabled = bool(args.output_sweep)
    sweep_targets: Optional[pd.DataFrame] = None
    sweep_track_idx: Optional[np.ndarray] = None
    sweep_track_idx_t: Optional[torch.Tensor] = None
    sweep_hit_ids: List[str] = []
    sweep_shifts_bins: List[int] = []
    sweep_pad_bins_list: List[int] = []
    sweep_dd_by_pad_shift: Dict[Tuple[int, int], List[np.ndarray]] = {}
    sweep_dwt_by_pad_shift: Dict[Tuple[int, int], List[np.ndarray]] = {}
    sweep_dmut_by_pad_shift: Dict[Tuple[int, int], List[np.ndarray]] = {}
    if sweep_enabled:
        targets_path = Path(str(args.targets_tsv))
        if not targets_path.exists():
            raise SystemExit(f"targets_tsv not found: {targets_path}")
        sweep_targets = _load_targets_table(targets_path)

        explicit_idx = _parse_int_list_csv(args.output_sweep_track_indices)
        if explicit_idx:
            sweep_track_idx = np.asarray(explicit_idx, dtype=np.int64)
        else:
            sweep_track_idx = _select_track_indices(sweep_targets, args.track_regex, str(args.track_field))
        if len(sweep_track_idx) == 0:
            raise SystemExit(f"No tracks matched track_regex={args.track_regex!r} in {targets_path}")

        # shifts in output bins
        sweep_shifts_bins = _parse_int_list_csv(args.output_sweep_shift_bins)
        if not sweep_shifts_bins:
            sweep_shifts_bins = [0]

        pad_grid = _parse_int_list_csv(args.output_sweep_pad_bins_grid)
        if pad_grid:
            sweep_pad_bins_list = [max(0, int(v)) for v in pad_grid]
        else:
            if args.output_sweep_pad_bins is not None:
                sweep_pad_bins_list = [max(0, int(args.output_sweep_pad_bins))]
            else:
                sweep_pad_bins_list = []  # will be inferred per-hit from --output_sweep_pad_bp

        # Pre-init storage if we know pads; otherwise we will lazily create per (pad,shift).
        for pad_bins in (sweep_pad_bins_list or [0]):
            for shift in sweep_shifts_bins:
                sweep_dd_by_pad_shift[(int(pad_bins), int(shift))] = []
                sweep_dwt_by_pad_shift[(int(pad_bins), int(shift))] = []
                sweep_dmut_by_pad_shift[(int(pad_bins), int(shift))] = []

        sweep_track_idx_t = torch.tensor(sweep_track_idx, dtype=torch.long, device=device)
        logger.info(
            f"Output sweep enabled: tracks_selected={len(sweep_track_idx)}/{len(sweep_targets)} "
            f"regex={args.track_regex!r} explicit_tracks={bool(explicit_idx)} "
            f"pad_bp={int(args.output_sweep_pad_bp)} pad_bins_grid={(','.join(map(str, sweep_pad_bins_list)) if sweep_pad_bins_list else 'infer')} "
            f"shift_bins={','.join(map(str, sweep_shifts_bins))}"
        )

    layer = str(args.layer)
    channels = _parse_channels(args)
    layer_to_channels = {layer: channels}

    results: List[dict] = []
    n_done_hits = 0

    def act_region_mean(maps_1ch: torch.Tensor, off_start: int, off_end: int) -> float:
        # maps_1ch: [L]
        L = int(maps_1ch.shape[-1])
        bin_bp = float(bin_size_bp(seq_len, L))
        lo = int(np.floor(off_start / bin_bp))
        hi = int(np.ceil(off_end / bin_bp))
        lo = max(0, lo)
        hi = min(L, hi)
        if hi <= lo:
            return float("nan")
        return float(maps_1ch[lo:hi].mean().item())

    with ActivationMapCapturer(wrapper.model, layer_to_channels) as cap:
        for start_i in range(0, len(df_fimo), int(args.batch_hits)):
            batch_df = df_fimo.iloc[start_i : start_i + int(args.batch_hits)]

            batch_hits: List[_BatchHit] = []
            seqs: List[str] = []

            for _idx, r in batch_df.iterrows():
                seq_id = str(r["sequence_name"])
                if seq_id not in id_to_header:
                    continue

                original_header = id_to_header[seq_id]
                variant_id, site_center1, _ = _parse_original_header(original_header)
                chrom, pos1, ref, alt = parse_variant_id_any(variant_id)

                # Map FIMO hit (relative to the small FASTA window) into genomic coords.
                # FIMO start/stop are 1-based inclusive on the FASTA sequence.
                win_start0 = (int(site_center1) - 1) - int(flank_bp)
                motif_start0 = int(win_start0 + (int(r["start"]) - 1))
                motif_end0 = int(win_start0 + int(r["stop"]))

                # Build allele-specific Borzoi input sequences centered on the variant.
                try:
                    ref_seq, alt_seq = make_ref_alt_sequence(
                        genome=genome,
                        chrom=str(chrom),
                        pos1=int(pos1),
                        ref=str(ref),
                        alt=str(alt),
                        seq_len=seq_len,
                        verify_ref=True,
                    )
                except Exception:
                    continue

                w_start0 = window_start0(int(pos1), seq_len)
                off_start = int(motif_start0 - w_start0)
                off_end = int(motif_end0 - w_start0)
                if off_end <= 0 or off_start >= seq_len:
                    continue
                off_start = max(0, off_start)
                off_end = min(seq_len, off_end)
                if off_end <= off_start:
                    continue

                width = off_end - off_start
                ctrl = _pick_control_region(
                    ref_seq,
                    motif_start=off_start,
                    motif_end=off_end,
                    width=width,
                    rng=rng,
                    min_gap=int(args.control_min_gap),
                    max_tries=int(args.control_max_tries),
                )
                if ctrl is None:
                    continue
                ctrl_s, ctrl_e = ctrl

                # Build sequence set: 6 per hit.
                if str(args.mut_mode) == "pwm_minimize":
                    hit_strand = str(r.get("strand", "+"))

                    # Prefer the STREME query PWM if it matches the hit width: this guarantees disruption
                    # at the exact coordinates of the FIMO instance (strand-aware).
                    if streme_pwm is not None and int(streme_pwm.shape[0]) == int(width):
                        ref_motif = _mutate_pwm_minimize_fixed_segment(
                            ref_seq,
                            start=off_start,
                            end=off_end,
                            pwm=streme_pwm,
                            topk=int(args.pwm_topk),
                            strand=hit_strand,
                        )
                        alt_motif = _mutate_pwm_minimize_fixed_segment(
                            alt_seq,
                            start=off_start,
                            end=off_end,
                            pwm=streme_pwm,
                            topk=int(args.pwm_topk),
                            strand=hit_strand,
                        )
                    else:
                        # Fallback: minimize a chosen (possibly longer) PWM by scanning near the hit.
                        if jaspar_pwm is None:
                            raise SystemExit("mut_mode=pwm_minimize requires a loadable STREME PWM or --pwm_meme/--pwm_id")
                        if int(jaspar_pwm.shape[0]) != int(width):
                            logger.warning(
                                "pwm_minimize fallback: JASPAR PWM width differs from hit width; disruption may not be guaranteed"
                            )
                        ref_motif = _mutate_pwm_minimize(
                            ref_seq,
                            start=off_start,
                            end=off_end,
                            pwm=jaspar_pwm,
                            scan_pad=int(args.pwm_scan_pad),
                            topk=int(args.pwm_topk),
                        )
                        alt_motif = _mutate_pwm_minimize(
                            alt_seq,
                            start=off_start,
                            end=off_end,
                            pwm=jaspar_pwm,
                            scan_pad=int(args.pwm_scan_pad),
                            topk=int(args.pwm_topk),
                        )
                else:
                    ref_motif = _mutate_seq(
                        ref_seq,
                        off_start,
                        off_end,
                        rng=rng,
                        mode=args.mut_mode,
                        sub_base=args.sub_base,
                        baseflip_preserve_gc=bool(args.baseflip_preserve_gc),
                    )
                    alt_motif = _mutate_seq(
                        alt_seq,
                        off_start,
                        off_end,
                        rng=rng,
                        mode=args.mut_mode,
                        sub_base=args.sub_base,
                        baseflip_preserve_gc=bool(args.baseflip_preserve_gc),
                    )
                ref_ctrl = _mutate_seq(
                    ref_seq,
                    ctrl_s,
                    ctrl_e,
                    rng=rng,
                    mode=("baseflip" if str(args.mut_mode) == "pwm_minimize" else args.mut_mode),
                    sub_base=args.sub_base,
                    baseflip_preserve_gc=bool(args.baseflip_preserve_gc),
                )
                alt_ctrl = _mutate_seq(
                    alt_seq,
                    ctrl_s,
                    ctrl_e,
                    rng=rng,
                    mode=("baseflip" if str(args.mut_mode) == "pwm_minimize" else args.mut_mode),
                    sub_base=args.sub_base,
                    baseflip_preserve_gc=bool(args.baseflip_preserve_gc),
                )

                seqs.extend([ref_seq, alt_seq, ref_motif, alt_motif, ref_ctrl, alt_ctrl])

                batch_hits.append(
                    _BatchHit(
                        variant_id=variant_id,
                        chrom=str(chrom),
                        pos1=int(pos1),
                        ref=str(ref),
                        alt=str(alt),
                        site_center1=int(site_center1),
                        motif_start0=motif_start0,
                        motif_end0=motif_end0,
                        motif_off_start=off_start,
                        motif_off_end=off_end,
                        ctrl_off_start=ctrl_s,
                        ctrl_off_end=ctrl_e,
                        fimo_row=r.to_dict(),
                    )
                )

            if not batch_hits:
                continue

            n_done_hits += len(batch_hits)
            if int(args.log_every) > 0 and (n_done_hits % int(args.log_every) == 0):
                logger.info(f"Progress: processed_hits={n_done_hits}/{len(df_fimo)}")

            x = one_hot_encode_batch(seqs, device=device)
            cap.clear()
            with torch.no_grad():
                out = wrapper.model(x)
            scores = score_expression(out, output_key=wrapper.output_key).detach().cpu().numpy().reshape(-1)
            y_sel: Optional[torch.Tensor] = None
            if sweep_enabled:
                y_full = _select_output_tensor(out, output_key=wrapper.output_key).float()
                if y_full.ndim == 2:
                    y_full = y_full[:, :, None]
                if y_full.ndim != 3:
                    raise RuntimeError(f"Unexpected model output shape for sweep: {tuple(y_full.shape)}")
                assert sweep_track_idx_t is not None
                y_sel = y_full.index_select(dim=1, index=sweep_track_idx_t)
            maps = cap.pop().maps[layer]  # [6B, C, L]

            for bi, hit in enumerate(batch_hits):
                i0 = bi * 6
                score_ref, score_alt, score_ref_m, score_alt_m, score_ref_c, score_alt_c = map(float, scores[i0 : i0 + 6])

                delta_expr_wt = score_alt - score_ref
                delta_expr_motif = score_alt_m - score_ref_m
                delta_expr_ctrl = score_alt_c - score_ref_c

                abs_delta_expr_wt = float(abs(delta_expr_wt))
                abs_delta_expr_motif = float(abs(delta_expr_motif))
                abs_delta_expr_ctrl = float(abs(delta_expr_ctrl))

                for cj, channel in enumerate(channels):
                    # activation: mean over motif region bins
                    m_ref = maps[i0 + 0, cj]
                    m_alt = maps[i0 + 1, cj]
                    m_ref_m = maps[i0 + 2, cj]
                    m_alt_m = maps[i0 + 3, cj]
                    m_ref_c = maps[i0 + 4, cj]
                    m_alt_c = maps[i0 + 5, cj]

                    act_ref = act_region_mean(m_ref, hit.motif_off_start, hit.motif_off_end)
                    act_alt = act_region_mean(m_alt, hit.motif_off_start, hit.motif_off_end)
                    act_ref_m = act_region_mean(m_ref_m, hit.motif_off_start, hit.motif_off_end)
                    act_alt_m = act_region_mean(m_alt_m, hit.motif_off_start, hit.motif_off_end)
                    act_ref_c = act_region_mean(m_ref_c, hit.motif_off_start, hit.motif_off_end)
                    act_alt_c = act_region_mean(m_alt_c, hit.motif_off_start, hit.motif_off_end)

                    # control-region activations (sanity check for local effects)
                    act_ctrl_ref = act_region_mean(m_ref, hit.ctrl_off_start, hit.ctrl_off_end)
                    act_ctrl_alt = act_region_mean(m_alt, hit.ctrl_off_start, hit.ctrl_off_end)
                    act_ctrl_ref_m = act_region_mean(m_ref_m, hit.ctrl_off_start, hit.ctrl_off_end)
                    act_ctrl_alt_m = act_region_mean(m_alt_m, hit.ctrl_off_start, hit.ctrl_off_end)
                    act_ctrl_ref_c = act_region_mean(m_ref_c, hit.ctrl_off_start, hit.ctrl_off_end)
                    act_ctrl_alt_c = act_region_mean(m_alt_c, hit.ctrl_off_start, hit.ctrl_off_end)

                    # Paper-friendly signs: WT - mutated (positive => mutation reduced activation / reduced delta_expr)
                    eff_act_ref_motif = float(act_ref - act_ref_m)
                    eff_act_alt_motif = float(act_alt - act_alt_m)
                    eff_act_ref_ctrl = float(act_ref - act_ref_c)
                    eff_act_alt_ctrl = float(act_alt - act_alt_c)
                    eff_act_ref_contrast = float(eff_act_ref_motif - eff_act_ref_ctrl)
                    eff_act_alt_contrast = float(eff_act_alt_motif - eff_act_alt_ctrl)

                    eff_dd_expr_motif = float(delta_expr_wt - delta_expr_motif)
                    eff_dd_expr_ctrl = float(delta_expr_wt - delta_expr_ctrl)
                    eff_dd_expr_contrast = float(eff_dd_expr_motif - eff_dd_expr_ctrl)

                    eff_abs_dd_expr_motif = float(abs_delta_expr_wt - abs_delta_expr_motif)
                    eff_abs_dd_expr_ctrl = float(abs_delta_expr_wt - abs_delta_expr_ctrl)
                    eff_abs_dd_expr_contrast = float(eff_abs_dd_expr_motif - eff_abs_dd_expr_ctrl)

                    results.append(
                        {
                            "motif_id": str(args.motif_id),
                            "layer": layer,
                            "channel": int(channel),
                            "variant_id": hit.variant_id,
                            "chrom": hit.chrom,
                            "pos1": hit.pos1,
                            "ref": hit.ref,
                            "alt": hit.alt,
                            "site_center1": hit.site_center1,
                            "motif_start0": int(hit.motif_start0),
                            "motif_end0": int(hit.motif_end0),
                            "motif_width": int(hit.motif_off_end - hit.motif_off_start),
                            "ctrl_off_start": int(hit.ctrl_off_start),
                            "ctrl_off_end": int(hit.ctrl_off_end),
                            "fimo_sequence_name": str(hit.fimo_row.get("sequence_name")),
                            "fimo_start": int(hit.fimo_row.get("start")),
                            "fimo_stop": int(hit.fimo_row.get("stop")),
                            "fimo_strand": str(hit.fimo_row.get("strand")),
                            "fimo_score": float(hit.fimo_row.get("score")),
                            "fimo_p": float(hit.fimo_row.get("p-value")),
                            "fimo_q": float(hit.fimo_row.get("q-value")),
                            "fimo_matched_sequence": str(hit.fimo_row.get("matched_sequence")),
                            "score_ref_wt": score_ref,
                            "score_alt_wt": score_alt,
                            "score_ref_motif": score_ref_m,
                            "score_alt_motif": score_alt_m,
                            "score_ref_ctrl": score_ref_c,
                            "score_alt_ctrl": score_alt_c,
                            "delta_expr_wt": float(delta_expr_wt),
                            "delta_expr_motif": float(delta_expr_motif),
                            "delta_expr_ctrl": float(delta_expr_ctrl),
                            "abs_delta_expr_wt": abs_delta_expr_wt,
                            "abs_delta_expr_motif": abs_delta_expr_motif,
                            "abs_delta_expr_ctrl": abs_delta_expr_ctrl,
                            "delta_delta_expr_motif": float(delta_expr_motif - delta_expr_wt),
                            "delta_delta_expr_ctrl": float(delta_expr_ctrl - delta_expr_wt),
                            "effect_dd_expr_wt_minus_motif": eff_dd_expr_motif,
                            "effect_dd_expr_wt_minus_ctrl": eff_dd_expr_ctrl,
                            "effect_dd_expr_contrast_motif_minus_ctrl": eff_dd_expr_contrast,
                            "effect_abs_dd_expr_wt_minus_motif": eff_abs_dd_expr_motif,
                            "effect_abs_dd_expr_wt_minus_ctrl": eff_abs_dd_expr_ctrl,
                            "effect_abs_dd_expr_contrast_motif_minus_ctrl": eff_abs_dd_expr_contrast,
                            "act_mean_motif_ref_wt": act_ref,
                            "act_mean_motif_alt_wt": act_alt,
                            "act_mean_motif_ref_motif": act_ref_m,
                            "act_mean_motif_alt_motif": act_alt_m,
                            "act_mean_motif_ref_ctrl": act_ref_c,
                            "act_mean_motif_alt_ctrl": act_alt_c,
                            "act_mean_ctrl_ref_wt": act_ctrl_ref,
                            "act_mean_ctrl_alt_wt": act_ctrl_alt,
                            "act_mean_ctrl_ref_motif": act_ctrl_ref_m,
                            "act_mean_ctrl_alt_motif": act_ctrl_alt_m,
                            "act_mean_ctrl_ref_ctrl": act_ctrl_ref_c,
                            "act_mean_ctrl_alt_ctrl": act_ctrl_alt_c,
                            "delta_act_ref_motif": float(act_ref_m - act_ref),
                            "delta_act_alt_motif": float(act_alt_m - act_alt),
                            "delta_act_ref_ctrl": float(act_ref_c - act_ref),
                            "delta_act_alt_ctrl": float(act_alt_c - act_alt),
                            "effect_act_ref_wt_minus_motif": eff_act_ref_motif,
                            "effect_act_alt_wt_minus_motif": eff_act_alt_motif,
                            "effect_act_ref_wt_minus_ctrl": eff_act_ref_ctrl,
                            "effect_act_alt_wt_minus_ctrl": eff_act_alt_ctrl,
                            "effect_act_ref_contrast_motif_minus_ctrl": eff_act_ref_contrast,
                            "effect_act_alt_contrast_motif_minus_ctrl": eff_act_alt_contrast,
                            "delta_act_ctrl_region_ref_motif": float(act_ctrl_ref_m - act_ctrl_ref),
                            "delta_act_ctrl_region_alt_motif": float(act_ctrl_alt_m - act_ctrl_alt),
                            "delta_act_ctrl_region_ref_ctrl": float(act_ctrl_ref_c - act_ctrl_ref),
                            "delta_act_ctrl_region_alt_ctrl": float(act_ctrl_alt_c - act_ctrl_alt),
                            "mut_mode": str(args.mut_mode),
                            "baseflip_preserve_gc": bool(args.baseflip_preserve_gc),
                            "pwm_id": str(args.pwm_id),
                            "pwm_topk": int(args.pwm_topk),
                            "pwm_scan_pad": int(args.pwm_scan_pad),
                            "pwm_width_jaspar": int(jaspar_pwm.shape[0]) if jaspar_pwm is not None else None,
                            "pwm_width_streme": int(streme_pwm.shape[0]) if streme_pwm is not None else None,
                            # PWM audit fields (record only for the first channel to avoid duplication in multi-channel runs)
                            "motif_seg_ref_wt": ref_seq[hit.motif_off_start : hit.motif_off_end] if (bool(args.audit_pwm) and cj == 0) else None,
                            "motif_seg_ref_mut": ref_motif[hit.motif_off_start : hit.motif_off_end] if (bool(args.audit_pwm) and cj == 0) else None,
                            "motif_seg_alt_wt": alt_seq[hit.motif_off_start : hit.motif_off_end] if (bool(args.audit_pwm) and cj == 0) else None,
                            "motif_seg_alt_mut": alt_motif[hit.motif_off_start : hit.motif_off_end] if (bool(args.audit_pwm) and cj == 0) else None,
                        }
                    )

                    if bool(args.audit_pwm) and cj == 0:
                        # STREME PWM score at the mutated segment (best strand, fixed alignment)
                        try:
                            if streme_pwm is not None:
                                seg_ref_wt = ref_seq[hit.motif_off_start : hit.motif_off_end]
                                seg_ref_mut = ref_motif[hit.motif_off_start : hit.motif_off_end]
                                sc_wt, st_wt = _pwm_best_score(seg_ref_wt, streme_pwm)
                                sc_mut, st_mut = _pwm_best_score(seg_ref_mut, streme_pwm)
                                results[-1]["streme_pwm_ref_wt"] = float(sc_wt)
                                results[-1]["streme_pwm_ref_mut"] = float(sc_mut)
                                results[-1]["streme_pwm_ref_delta_wt_minus_mut"] = float(sc_wt - sc_mut)
                                results[-1]["streme_pwm_ref_strand_wt"] = str(st_wt)
                                results[-1]["streme_pwm_ref_strand_mut"] = str(st_mut)

                                # Strand-aware score using the FIMO-reported strand (more interpretable for site disruption).
                                try:
                                    fimo_strand = str(hit.fimo_row.get("strand", "+"))
                                    if fimo_strand == "-":
                                        seg_ref_wt_use = _revcomp(seg_ref_wt)
                                        seg_ref_mut_use = _revcomp(seg_ref_mut)
                                    else:
                                        seg_ref_wt_use = seg_ref_wt
                                        seg_ref_mut_use = seg_ref_mut
                                    sc_wt_fimo = _pwm_log_odds_score(seg_ref_wt_use, streme_pwm)
                                    sc_mut_fimo = _pwm_log_odds_score(seg_ref_mut_use, streme_pwm)
                                    results[-1]["streme_pwm_ref_fimo_wt"] = float(sc_wt_fimo)
                                    results[-1]["streme_pwm_ref_fimo_mut"] = float(sc_mut_fimo)
                                    results[-1]["streme_pwm_ref_fimo_delta_wt_minus_mut"] = float(sc_wt_fimo - sc_mut_fimo)
                                    results[-1]["streme_pwm_ref_fimo_strand"] = str(fimo_strand)
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        # JASPAR PWM best scan near the motif hit region.
                        try:
                            if jaspar_pwm is not None:
                                w = int(jaspar_pwm.shape[0])
                                pad = int(args.pwm_scan_pad)
                                lo = max(0, hit.motif_off_start - pad)
                                hi = min(seq_len, hit.motif_off_end + pad + w)
                                local_wt = ref_seq[lo:hi]
                                local_mut = ref_motif[lo:hi]
                                sc_wt, off_wt, st_wt = _pwm_scan_best(local_wt, jaspar_pwm)
                                sc_mut, off_mut, st_mut = _pwm_scan_best(local_mut, jaspar_pwm)
                                results[-1]["jaspar_pwm_ref_best_wt"] = float(sc_wt)
                                results[-1]["jaspar_pwm_ref_best_mut"] = float(sc_mut)
                                results[-1]["jaspar_pwm_ref_best_delta_wt_minus_mut"] = float(sc_wt - sc_mut)
                                results[-1]["jaspar_pwm_ref_best_off0_wt"] = int(off_wt)
                                results[-1]["jaspar_pwm_ref_best_off0_mut"] = int(off_mut)
                                results[-1]["jaspar_pwm_ref_best_strand_wt"] = str(st_wt)
                                results[-1]["jaspar_pwm_ref_best_strand_mut"] = str(st_mut)
                        except Exception:
                            pass

                    # Output sweep: compute once per hit (guarded by cj==0).
                    if sweep_enabled and cj == 0:
                        assert y_sel is not None
                        _, n_tracks_sel, L_out = y_sel.shape
                        bin_bp_out = float(seq_len) / float(L_out)

                        base_lo = int(np.floor(float(hit.motif_off_start) / bin_bp_out))
                        base_hi = int(np.ceil(float(hit.motif_off_end) / bin_bp_out))
                        if base_hi <= base_lo:
                            base_lo = min(max(0, int(np.floor(float(hit.motif_off_start) / bin_bp_out))), int(L_out) - 1)
                            base_hi = min(base_lo + 1, int(L_out))

                        # If pad grid wasn't specified, infer a single pad_bins from pad_bp (or explicit pad_bins).
                        pad_bins_list = list(sweep_pad_bins_list)
                        if not pad_bins_list:
                            if args.output_sweep_pad_bins is not None:
                                pad_bins_list = [max(0, int(args.output_sweep_pad_bins))]
                            else:
                                pad_bp = int(args.output_sweep_pad_bp)
                                pad_bins_list = [max(0, int(np.round(float(pad_bp) / float(bin_bp_out))))]

                        for pad_bins in pad_bins_list:
                            lo0 = max(0, base_lo - int(pad_bins))
                            hi0 = min(int(L_out), base_hi + int(pad_bins))
                            if hi0 <= lo0:
                                continue

                            for shift in sweep_shifts_bins:
                                lo = lo0 + int(shift)
                                hi = hi0 + int(shift)
                                if hi <= lo:
                                    continue
                                if hi <= 0 or lo >= int(L_out):
                                    continue
                                lo = max(0, lo)
                                hi = min(int(L_out), hi)
                                if hi <= lo:
                                    continue

                                ref_wt = y_sel[i0 + 0, :, lo:hi].mean(dim=-1)
                                alt_wt = y_sel[i0 + 1, :, lo:hi].mean(dim=-1)
                                ref_mut = y_sel[i0 + 2, :, lo:hi].mean(dim=-1)
                                alt_mut = y_sel[i0 + 3, :, lo:hi].mean(dim=-1)
                                d_wt = (alt_wt - ref_wt)
                                d_mut = (alt_mut - ref_mut)
                                dd = d_mut - d_wt
                                key = (int(pad_bins), int(shift))
                                if key not in sweep_dd_by_pad_shift:
                                    sweep_dd_by_pad_shift[key] = []
                                if key not in sweep_dwt_by_pad_shift:
                                    sweep_dwt_by_pad_shift[key] = []
                                if key not in sweep_dmut_by_pad_shift:
                                    sweep_dmut_by_pad_shift[key] = []
                                sweep_dd_by_pad_shift[key].append(
                                    dd.detach().cpu().numpy().astype(np.float32, copy=False).reshape(n_tracks_sel)
                                )
                                sweep_dwt_by_pad_shift[key].append(
                                    d_wt.detach().cpu().numpy().astype(np.float32, copy=False).reshape(n_tracks_sel)
                                )
                                sweep_dmut_by_pad_shift[key].append(
                                    d_mut.detach().cpu().numpy().astype(np.float32, copy=False).reshape(n_tracks_sel)
                                )

                        sweep_hit_ids.append(str(hit.fimo_row.get("sequence_name")))

    df_out = pd.DataFrame(results)
    ch_tag = _channels_tag(channels)
    out_parquet = Path(outdir) / f"perturb_{re.sub(r'[^A-Za-z0-9_.-]+','_',str(args.motif_id))}_{layer}_{ch_tag}.parquet"
    write_parquet(df_out, out_parquet)

    summary: dict = {
        "motif_id": str(args.motif_id),
        "layer": layer,
        "channels": [int(c) for c in channels],
        "n_rows": int(len(df_out)),
        "n_hits": int(df_out["fimo_sequence_name"].nunique()) if "fimo_sequence_name" in df_out.columns else int(len(df_out)),
        "mut_mode": str(args.mut_mode),
        "baseflip_preserve_gc": bool(args.baseflip_preserve_gc),
        "control_min_gap": int(args.control_min_gap),
        "control_max_tries": int(args.control_max_tries),
        "seq_len": int(seq_len),
        "flank_bp": int(flank_bp),
    }

    def _mean_sem(x: pd.Series) -> Tuple[float, float]:
        x2 = x.dropna().astype(float)
        if len(x2) == 0:
            return float("nan"), float("nan")
        mean = float(x2.mean())
        sem = float(x2.std(ddof=1) / np.sqrt(len(x2))) if len(x2) > 1 else float("nan")
        return mean, sem

    def _median_iqr(x: pd.Series) -> Tuple[float, float, float]:
        x2 = x.dropna().astype(float)
        if len(x2) == 0:
            return float("nan"), float("nan"), float("nan")
        q1 = float(x2.quantile(0.25))
        med = float(x2.quantile(0.50))
        q3 = float(x2.quantile(0.75))
        return med, q1, q3

    def _wilcoxon_p_vs0(x: pd.Series) -> float:
        x2 = x.dropna().astype(float)
        if len(x2) < 5:
            return float("nan")
        try:
            return float(wilcoxon(x2.values, zero_method="wilcox", correction=False, alternative="two-sided").pvalue)
        except Exception:
            return float("nan")

    def _summarize_series(x: pd.Series) -> dict:
        x2 = x.dropna().astype(float)
        m, sem = _mean_sem(x2)
        med, q1, q3 = _median_iqr(x2)
        return {
            "n": int(len(x2)),
            "mean": m,
            "sem": sem,
            "median": med,
            "q1": q1,
            "q3": q3,
            "frac_pos": float((x2 > 0).mean()) if len(x2) else float("nan"),
            "frac_neg": float((x2 < 0).mean()) if len(x2) else float("nan"),
            "wilcoxon_p_vs0": _wilcoxon_p_vs0(x2),
        }

    # Keep legacy mean/sem for quick sanity checks.
    legacy_cols = [
        "delta_delta_expr_motif",
        "delta_delta_expr_ctrl",
        "delta_act_ref_motif",
        "delta_act_ref_ctrl",
        "delta_act_alt_motif",
        "delta_act_alt_ctrl",
        "delta_act_ctrl_region_ref_ctrl",
        "delta_act_ctrl_region_alt_ctrl",
    ]
    summary["legacy_mean_sem"] = {c: {"mean": _mean_sem(df_out[c])[0], "sem": _mean_sem(df_out[c])[1]} for c in legacy_cols if c in df_out.columns}

    metric_cols = [
        "effect_act_ref_wt_minus_motif",
        "effect_act_ref_wt_minus_ctrl",
        "effect_act_ref_contrast_motif_minus_ctrl",
        "effect_act_alt_wt_minus_motif",
        "effect_act_alt_wt_minus_ctrl",
        "effect_act_alt_contrast_motif_minus_ctrl",
        "effect_dd_expr_wt_minus_motif",
        "effect_dd_expr_wt_minus_ctrl",
        "effect_dd_expr_contrast_motif_minus_ctrl",
        "effect_abs_dd_expr_wt_minus_motif",
        "effect_abs_dd_expr_wt_minus_ctrl",
        "effect_abs_dd_expr_contrast_motif_minus_ctrl",
    ]

    per_channel: dict = {}
    for ch in channels:
        df_ch = df_out[df_out["channel"] == int(ch)]
        per_channel[str(int(ch))] = {c: _summarize_series(df_ch[c]) for c in metric_cols if c in df_ch.columns}
        # Add paired test: motif vs control via contrast.
        if "effect_act_ref_contrast_motif_minus_ctrl" in df_ch.columns:
            per_channel[str(int(ch))]["effect_act_ref_contrast_motif_minus_ctrl"]["wilcoxon_p_vs0"] = _wilcoxon_p_vs0(
                df_ch["effect_act_ref_contrast_motif_minus_ctrl"]
            )
        if "effect_dd_expr_contrast_motif_minus_ctrl" in df_ch.columns:
            per_channel[str(int(ch))]["effect_dd_expr_contrast_motif_minus_ctrl"]["wilcoxon_p_vs0"] = _wilcoxon_p_vs0(
                df_ch["effect_dd_expr_contrast_motif_minus_ctrl"]
            )

    summary["per_channel"] = per_channel

    out_json = Path(outdir) / f"perturb_{re.sub(r'[^A-Za-z0-9_.-]+','_',str(args.motif_id))}_{layer}_{ch_tag}.summary.json"
    write_text(out_json, json.dumps(summary, indent=2) + "\n")

    logger.info(f"Wrote: {out_parquet} rows={len(df_out)}")
    logger.info(f"Wrote: {out_json}")

    if sweep_enabled:
        assert sweep_targets is not None and sweep_track_idx is not None
        if not sweep_dd_by_pad_shift or all(len(v) == 0 for v in sweep_dd_by_pad_shift.values()):
            raise SystemExit("output_sweep produced no hits; check inputs")

        hit_ids = np.array(sweep_hit_ids, dtype=object)
        meta = sweep_targets.iloc[sweep_track_idx].copy().reset_index(drop=True)

        per_shift_tables: List[pd.DataFrame] = []
        for (pad_bins, shift), dd_list in sorted(sweep_dd_by_pad_shift.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            if len(dd_list) == 0:
                continue
            dd_mat = np.stack(dd_list, axis=0)  # [n_hits, n_tracks_sel]
            med = np.nanmedian(dd_mat, axis=0)
            mean = np.nanmean(dd_mat, axis=0)
            frac_pos = np.nanmean(dd_mat > 0, axis=0)
            frac_neg = np.nanmean(dd_mat < 0, axis=0)
            abs_med = np.abs(med)
            tab = pd.DataFrame(
                {
                    "pad_bins": int(pad_bins),
                    "shift_bins": int(shift),
                    "track_index": sweep_track_idx.astype(int),
                    "identifier": meta["identifier"].astype(str).values,
                    "description": meta.get("description", "").astype(str).values,
                    "n_hits": int(dd_mat.shape[0]),
                    "median_delta_delta": med.astype(float),
                    "mean_delta_delta": mean.astype(float),
                    "abs_median_delta_delta": abs_med.astype(float),
                    "frac_pos": frac_pos.astype(float),
                    "frac_neg": frac_neg.astype(float),
                }
            )
            tab = tab.sort_values("abs_median_delta_delta", ascending=False).reset_index(drop=True)
            tab["rank_abs_median"] = np.arange(1, len(tab) + 1)
            per_shift_tables.append(tab)

            # Back-compat: if this is the canonical (pad=0, shift=0) table, also write the legacy filename.
            if int(pad_bins) == 0 and int(shift) == 0:
                out_tracks0 = Path(outdir) / "output_sweep_tracks.parquet"
                write_parquet(tab.drop(columns=["pad_bins", "shift_bins"]), out_tracks0)
                logger.info(f"Wrote: {out_tracks0} tracks={len(tab)}")

        track_table_all = pd.concat(per_shift_tables, axis=0, ignore_index=True) if per_shift_tables else pd.DataFrame()
        out_tracks_all = Path(outdir) / "output_sweep_tracks_by_pad_shift.parquet"
        write_parquet(track_table_all, out_tracks_all)
        logger.info(f"Wrote: {out_tracks_all} rows={len(track_table_all)}")

        # Long-format per-hit table (optional but useful for downstream analysis).
        # Includes delta_expr_wt (alt-ref), delta_expr_mut, and delta_delta (=mut-wt).
        try:
            per_hit_rows: List[pd.DataFrame] = []
            for (pad_bins, shift), dd_list in sorted(sweep_dd_by_pad_shift.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                if len(dd_list) == 0:
                    continue
                dwt_list = sweep_dwt_by_pad_shift.get((int(pad_bins), int(shift)), [])
                dmut_list = sweep_dmut_by_pad_shift.get((int(pad_bins), int(shift)), [])
                if len(dwt_list) != len(dd_list) or len(dmut_list) != len(dd_list):
                    continue
                dd_mat = np.stack(dd_list, axis=0)
                dwt_mat = np.stack(dwt_list, axis=0)
                dmut_mat = np.stack(dmut_list, axis=0)

                n_hits, n_tr = dd_mat.shape
                base = pd.DataFrame(
                    {
                        "fimo_sequence_name": np.repeat(hit_ids, n_tr),
                        "pad_bins": int(pad_bins),
                        "shift_bins": int(shift),
                        "track_index": np.tile(sweep_track_idx.astype(int), n_hits),
                    }
                )
                base["delta_expr_wt"] = dwt_mat.reshape(-1).astype(float)
                base["delta_expr_mut"] = dmut_mat.reshape(-1).astype(float)
                base["delta_delta"] = dd_mat.reshape(-1).astype(float)
                per_hit_rows.append(base)

            per_hit_df = pd.concat(per_hit_rows, axis=0, ignore_index=True) if per_hit_rows else pd.DataFrame()
            if not per_hit_df.empty:
                out_per_hit = Path(outdir) / "output_sweep_per_hit_long_by_pad_shift.parquet"
                write_parquet(per_hit_df, out_per_hit)
                logger.info(f"Wrote: {out_per_hit} rows={len(per_hit_df)}")
        except Exception as e:
            logger.warning(f"Failed to write per-hit output sweep table: {e}")

        # Optional: correlate selected-track readouts with early/receiver activation effects.
        if args.act_early_parquet and args.act_receiver_parquet and bool(args.output_sweep_corr_all_selected):
            early_s = _extract_act_effect_series(
                Path(str(args.act_early_parquet)),
                layer=str(args.act_early_layer),
                channel=int(args.act_early_channel),
            )
            recv_s = _extract_act_effect_series(
                Path(str(args.act_receiver_parquet)),
                layer=str(args.act_receiver_layer),
                channel=int(args.act_receiver_channel),
            )
            early = np.array([float(early_s.get(k, np.nan)) for k in hit_ids], dtype=float)
            recv = np.array([float(recv_s.get(k, np.nan)) for k in hit_ids], dtype=float)

            def _iqr(y: np.ndarray) -> float:
                y2 = y[np.isfinite(y)]
                if y2.size == 0:
                    return float("nan")
                q1 = float(np.quantile(y2, 0.25))
                q3 = float(np.quantile(y2, 0.75))
                return float(q3 - q1)

            def _wilcoxon_p(y: np.ndarray) -> float:
                y2 = y[np.isfinite(y)]
                if y2.size < 5:
                    return float("nan")
                try:
                    return float(wilcoxon(y2, zero_method="wilcox", correction=False, alternative="two-sided").pvalue)
                except Exception:
                    return float("nan")

            corr_rows: List[dict] = []
            for (pad_bins, shift), dd_list in sorted(sweep_dd_by_pad_shift.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                if len(dd_list) == 0:
                    continue
                dd_mat = np.stack(dd_list, axis=0)  # [n_hits, n_tracks_sel]
                for j, trk in enumerate(sweep_track_idx.astype(int).tolist()):
                    y = dd_mat[:, j].astype(float)
                    m_e = ~(np.isnan(y) | np.isnan(early))
                    m_r = ~(np.isnan(y) | np.isnan(recv))
                    rho_e, p_e = spearmanr(y[m_e], early[m_e]) if int(m_e.sum()) >= 10 else (float("nan"), float("nan"))
                    rho_r, p_r = spearmanr(y[m_r], recv[m_r]) if int(m_r.sum()) >= 10 else (float("nan"), float("nan"))
                    corr_rows.append(
                        {
                            "pad_bins": int(pad_bins),
                            "shift_bins": int(shift),
                            "track_index": int(trk),
                            "identifier": str(meta.iloc[j]["identifier"]),
                            "description": str(meta.iloc[j].get("description", "")),
                            "median_delta_delta": float(np.nanmedian(y)),
                            "iqr_delta_delta": _iqr(y),
                            "wilcoxon_p": _wilcoxon_p(y),
                            "rho_vs_early": float(rho_e),
                            "p_rho_early": float(p_e),
                            "rho_vs_receiver": float(rho_r),
                            "p_rho_receiver": float(p_r),
                            "n_early": int(m_e.sum()),
                            "n_receiver": int(m_r.sum()),
                        }
                    )

                # Meta readout: max over 3 tracks (max |delta_delta| per hit).
                # Useful when different hits affect different donors.
                try:
                    y_max3 = np.nanmax(np.abs(dd_mat.astype(float)), axis=1)
                    m_e = ~(np.isnan(y_max3) | np.isnan(early))
                    m_r = ~(np.isnan(y_max3) | np.isnan(recv))
                    rho_e, p_e = spearmanr(y_max3[m_e], early[m_e]) if int(m_e.sum()) >= 10 else (float("nan"), float("nan"))
                    rho_r, p_r = spearmanr(y_max3[m_r], recv[m_r]) if int(m_r.sum()) >= 10 else (float("nan"), float("nan"))
                    corr_rows.append(
                        {
                            "pad_bins": int(pad_bins),
                            "shift_bins": int(shift),
                            "track_index": -1,
                            "identifier": "max3_abs",
                            "description": "meta: max over selected tracks of |delta_delta|",
                            "median_delta_delta": float(np.nanmedian(y_max3)),
                            "iqr_delta_delta": _iqr(y_max3),
                            "wilcoxon_p": _wilcoxon_p(y_max3),
                            "rho_vs_early": float(rho_e),
                            "p_rho_early": float(p_e),
                            "rho_vs_receiver": float(rho_r),
                            "p_rho_receiver": float(p_r),
                            "n_early": int(m_e.sum()),
                            "n_receiver": int(m_r.sum()),
                        }
                    )
                except Exception:
                    pass

            corr_df = pd.DataFrame(corr_rows)
            out_corr = Path(outdir) / "output_sweep_selected_tracks_correlations_by_pad_shift.parquet"
            write_parquet(corr_df, out_corr)
            logger.info(f"Wrote: {out_corr} rows={len(corr_df)}")

            # Deliverable: a concise long-format summary table with the requested columns.
            deliver_cols = [
                "track_index",
                "identifier",
                "pad_bins",
                "shift_bins",
                "median_delta_delta",
                "iqr_delta_delta",
                "wilcoxon_p",
                "rho_vs_early",
                "p_rho_early",
                "rho_vs_receiver",
                "p_rho_receiver",
                "n_early",
                "n_receiver",
            ]
            out_deliv = Path(outdir) / "output_sweep_summary_by_pad_shift.parquet"
            try:
                write_parquet(corr_df[deliver_cols].copy(), out_deliv)
                logger.info(f"Wrote: {out_deliv} rows={len(corr_df)}")
            except Exception as e:
                logger.warning(f"Failed to write output_sweep_summary_by_pad_shift.parquet: {e}")

        # Still report top-K tracks for shift=0 for quick sanity.
        if not track_table_all.empty:
            topk = max(1, int(args.output_sweep_topk_tracks))
            top0 = track_table_all[(track_table_all["pad_bins"] == 0) & (track_table_all["shift_bins"] == 0)].head(topk)
            logger.info("Output sweep top tracks (shift=0, by |median delta-delta|):")
            for i, r in enumerate(top0.itertuples(index=False), start=1):
                logger.info(
                    f"  {i:2d}. track={int(r.track_index)} id={str(r.identifier)} |median|={float(r.abs_median_delta_delta):.4g}"
                )


if __name__ == "__main__":
    main()
