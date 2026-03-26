from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def ensure_parent_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    p = ensure_parent_dir(path)
    df.to_parquet(p, index=False)
    return p


def safe_slug(s: str) -> str:
    """File-name-safe slug (keeps alnum, dot, dash, underscore)."""
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in {".", "-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


@dataclass(frozen=True)
class MarkdownTable:
    header: list[str]
    rows: list[list[str]]

    def render(self) -> str:
        head = "| " + " | ".join(self.header) + " |"
        sep = "| " + " | ".join(["---"] * len(self.header)) + " |"
        body = "\n".join("| " + " | ".join(r) + " |" for r in self.rows)
        return "\n".join([head, sep, body]) + "\n"


def df_to_markdown_table(df: pd.DataFrame, columns: Iterable[str], max_rows: int = 20) -> MarkdownTable:
    cols = list(columns)
    df2 = df.loc[:, cols].head(max_rows)
    rows: list[list[str]] = []
    for _, r in df2.iterrows():
        rows.append(["" if pd.isna(r[c]) else str(r[c]) for c in cols])
    return MarkdownTable(header=cols, rows=rows)


def write_text(path: str | Path, text: str) -> Path:
    p = ensure_parent_dir(path)
    p.write_text(text)
    return p


__all__ = [
    "ensure_parent_dir",
    "ensure_dir",
    "read_parquet",
    "write_parquet",
    "safe_slug",
    "MarkdownTable",
    "df_to_markdown_table",
    "write_text",
]
