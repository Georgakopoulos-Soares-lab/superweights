from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.metrics import roc_auc_score


def _pearson_with_p(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Pearson r and p over columns of x.

    x: [N, C], y: [N]
    Returns: (r[C], p[C])
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(y)
    if not mask.any():
        r = np.full(x.shape[1], np.nan)
        p = np.full(x.shape[1], np.nan)
        return r, p

    x = x[mask]
    y = y[mask]
    n = x.shape[0]
    if n < 3:
        r = np.full(x.shape[1], np.nan)
        p = np.full(x.shape[1], np.nan)
        return r, p

    y0 = y - y.mean()
    x0 = x - x.mean(axis=0, keepdims=True)
    cov = (x0 * y0[:, None]).sum(axis=0)

    denom = np.sqrt((x0 * x0).sum(axis=0) * (y0 * y0).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        r = cov / denom

    # p-values from t distribution
    df = n - 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r * np.sqrt(df / (1.0 - r * r))
        p = 2.0 * stats.t.sf(np.abs(t), df=df)

    # handle constant columns
    r = np.where(np.isfinite(r), r, np.nan)
    p = np.where(np.isfinite(p), p, np.nan)
    return r.astype(np.float32), p.astype(np.float32)


def _spearman_with_p(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Spearman via rank transform then Pearson t-test approximation."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(y)
    if not mask.any():
        r = np.full(x.shape[1], np.nan)
        p = np.full(x.shape[1], np.nan)
        return r, p

    x = x[mask]
    y = y[mask]
    n = x.shape[0]
    if n < 3:
        r = np.full(x.shape[1], np.nan)
        p = np.full(x.shape[1], np.nan)
        return r, p

    # rank y
    y_rank = stats.rankdata(y)

    # rank each column; loop over channels but keep fast
    r_list = np.empty(x.shape[1], dtype=np.float64)
    for c in range(x.shape[1]):
        xc = x[:, c]
        if not np.isfinite(xc).any():
            r_list[c] = np.nan
            continue
        x_rank = stats.rankdata(xc)
        rr, _pp = stats.pearsonr(x_rank, y_rank)
        r_list[c] = rr

    r = r_list

    df = n - 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r * np.sqrt(df / (1.0 - r * r))
        p = 2.0 * stats.t.sf(np.abs(t), df=df)

    r = np.where(np.isfinite(r), r, np.nan)
    p = np.where(np.isfinite(p), p, np.nan)
    return r.astype(np.float32), p.astype(np.float32)


def _auc_per_channel(x: np.ndarray, y_label01: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y_label01, dtype=np.int64)

    auc = np.full(x.shape[1], np.nan, dtype=np.float32)
    if len(np.unique(y)) < 2:
        return auc

    for c in range(x.shape[1]):
        xc = x[:, c]
        if not np.isfinite(xc).any():
            continue
        if np.allclose(xc, xc[0]):
            continue
        try:
            auc[c] = float(roc_auc_score(y, xc))
        except Exception:
            auc[c] = np.nan
    return auc


def compute_channel_correlations(
    per_variant: pd.DataFrame,
    feature_mats: Dict[Tuple[str, str], np.ndarray],
) -> pd.DataFrame:
    """Compute per-layer/channel stats.

    feature_mats maps (layer, feature_type) -> np.ndarray [N, C]
    """

    y_delta = per_variant["delta_expr"].to_numpy(dtype=np.float64)

    beta_col = None
    for cand in ("beta", "posterior_beta", "post_beta"):
        if cand in per_variant.columns:
            beta_col = cand
            break

    y_beta = per_variant[beta_col].to_numpy(dtype=np.float64) if beta_col is not None else None

    label = per_variant["label"].to_numpy(dtype=np.int64)

    rows: List[dict] = []
    for (layer, feature_type), X in feature_mats.items():
        if X.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix for {(layer, feature_type)}")

        r_p, p_p = _pearson_with_p(X, y_delta)
        r_s, p_s = _spearman_with_p(X, y_delta)

        if y_beta is not None:
            rb_p, pb_p = _pearson_with_p(X, y_beta)
            rb_s, pb_s = _spearman_with_p(X, y_beta)
        else:
            rb_p = pb_p = rb_s = pb_s = np.full(X.shape[1], np.nan, dtype=np.float32)

        auc = _auc_per_channel(X, label)

        n = int(X.shape[0])
        for ch in range(X.shape[1]):
            rows.append(
                {
                    "layer": layer,
                    "channel": int(ch),
                    "feature_type": feature_type,
                    "pearson_delta": float(r_p[ch]) if np.isfinite(r_p[ch]) else np.nan,
                    "p_pearson_delta": float(p_p[ch]) if np.isfinite(p_p[ch]) else np.nan,
                    "spearman_delta": float(r_s[ch]) if np.isfinite(r_s[ch]) else np.nan,
                    "p_spearman_delta": float(p_s[ch]) if np.isfinite(p_s[ch]) else np.nan,
                    "pearson_beta": float(rb_p[ch]) if np.isfinite(rb_p[ch]) else np.nan,
                    "p_pearson_beta": float(pb_p[ch]) if np.isfinite(pb_p[ch]) else np.nan,
                    "spearman_beta": float(rb_s[ch]) if np.isfinite(rb_s[ch]) else np.nan,
                    "p_spearman_beta": float(pb_s[ch]) if np.isfinite(pb_s[ch]) else np.nan,
                    "auc_label": float(auc[ch]) if np.isfinite(auc[ch]) else np.nan,
                    "n": n,
                }
            )

    out = pd.DataFrame(rows)
    return out
