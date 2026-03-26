#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

# Allow running as `python scripts/eval_baseline.py` without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate baseline AUC/correlation from an existing per-variant scores parquet")

    p.add_argument(
        "--scores_parquet",
        type=str,
        default="results/full_v1/per_variant_scores.parquet",
        help="Parquet that already contains both positives and negatives with at least: variant_id, label, delta_expr",
    )

    p.add_argument("--n_pos", type=str, default="all", help="Number of positives to sample (or 'all')")
    p.add_argument("--n_neg", type=str, default="all", help="Number of negatives to sample (or 'all')")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument(
        "--score",
        type=str,
        default="abs_delta_expr",
        help="Score to use for AUC. Supported: abs_delta_expr, delta_expr",
    )

    p.add_argument(
        "--beta_col",
        type=str,
        default="beta",
        help="Column to use for beta correlation on positives (if present)",
    )

    p.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="Number of bootstrap resamples for AUC CI (0 disables)",
    )

    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output json path (default: results/ablation/baseline_eval_{n_pos}_{n_neg}.json)",
    )

    return p.parse_args()


def _parse_n(s: str) -> Optional[int]:
    s = str(s).strip().lower()
    if s in {"all", "", "none"}:
        return None
    return int(s)


def _get_score(df: pd.DataFrame, score_name: str) -> np.ndarray:
    score_name = score_name.strip().lower()

    if score_name in {"abs_delta_expr", "abs(delta_expr)", "abs"}:
        if "delta_expr" not in df.columns:
            raise ValueError("scores parquet must contain column 'delta_expr' for abs_delta_expr")
        return np.abs(df["delta_expr"].to_numpy(dtype=np.float32))

    if score_name in {"delta_expr", "signed_delta_expr"}:
        if "delta_expr" not in df.columns:
            raise ValueError("scores parquet must contain column 'delta_expr'")
        return df["delta_expr"].to_numpy(dtype=np.float32)

    raise ValueError(f"Unsupported --score: {score_name} (supported: abs_delta_expr, delta_expr)")


def _bootstrap_auc(y: np.ndarray, score: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    if n_boot <= 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n, endpoint=False)
        yb = y[idx]
        sb = score[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(yb, sb)))
        except Exception:
            continue

    if not aucs:
        return float("nan"), float("nan")

    lo, hi = np.percentile(np.asarray(aucs, dtype=np.float64), [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    args = parse_args()

    n_pos = _parse_n(args.n_pos)
    n_neg = _parse_n(args.n_neg)

    scores_path = Path(args.scores_parquet)
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing scores parquet: {scores_path}")

    df = pd.read_parquet(scores_path)
    if "label" not in df.columns:
        raise ValueError("scores parquet must contain column 'label'")

    # Sample a fixed subset (so repeated runs are comparable)
    rng = np.random.default_rng(int(args.seed))

    df_pos = df[df["label"].astype(int) == 1]
    df_neg = df[df["label"].astype(int) == 0]

    if n_pos is not None and n_pos < len(df_pos):
        df_pos = df_pos.sample(n=n_pos, random_state=int(args.seed))
    if n_neg is not None and n_neg < len(df_neg):
        df_neg = df_neg.sample(n=n_neg, random_state=int(args.seed) + 1)

    df_sub = pd.concat([df_pos, df_neg], axis=0, ignore_index=True)
    df_sub = df_sub.sample(frac=1.0, random_state=int(args.seed) + 2).reset_index(drop=True)

    y = df_sub["label"].astype(int).to_numpy()
    score = _get_score(df_sub, args.score)

    auc = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")
    ci_lo, ci_hi = _bootstrap_auc(y, score, n_boot=int(args.bootstrap), seed=int(args.seed) + 3)

    # Correlation on positives: always use signed delta_expr (even if AUC uses abs)
    spearman_beta_pos = float("nan")
    beta_n = 0
    if args.beta_col in df_sub.columns and "delta_expr" in df_sub.columns:
        pos = df_sub["label"].astype(int) == 1
        beta = df_sub.loc[pos, args.beta_col].to_numpy(dtype=np.float64)
        delta = df_sub.loc[pos, "delta_expr"].to_numpy(dtype=np.float64)
        mask = np.isfinite(beta) & np.isfinite(delta)
        beta_n = int(mask.sum())
        if beta_n >= 3:
            spearman_beta_pos = float(spearmanr(delta[mask], beta[mask]).correlation)

    out = {
        "inputs": {
            "scores_parquet": str(scores_path),
            "n_pos": int(len(df_pos)),
            "n_neg": int(len(df_neg)),
            "seed": int(args.seed),
            "score": str(args.score),
            "beta_col": str(args.beta_col),
            "bootstrap": int(args.bootstrap),
        },
        "baseline": {
            "auc_label": auc,
            "auc_ci95": [ci_lo, ci_hi],
            "spearman_beta_pos": spearman_beta_pos,
            "spearman_beta_pos_n": beta_n,
        },
    }

    out_path = Path(args.out) if args.out else Path(f"results/ablation/baseline_eval_{len(df_pos)}_{len(df_neg)}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    logger.info(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
