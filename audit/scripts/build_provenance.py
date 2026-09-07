#!/usr/bin/env python
"""Build/append audit/provenance.json: file -> {git commit, mtime, sha256}.

Reads the list of files touched by the audit from audit/scripts/provenance_files.txt
(one repo-relative path per line, '#' comments allowed). Rerunnable/deterministic:
re-running regenerates the same output for unchanged files.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILELIST = Path(__file__).resolve().parent / "provenance_files.txt"
OUT = ROOT / "audit" / "provenance.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_last_commit(rel: str) -> str | None:
    out = subprocess.run(["git", "log", "-1", "--format=%H %ci", "--", rel],
                          cwd=ROOT, capture_output=True, text=True)
    s = out.stdout.strip()
    return s if s else None


def git_tracked_dirty(rel: str) -> str:
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel],
                              cwd=ROOT, capture_output=True, text=True).returncode == 0
    if not tracked:
        return "untracked"
    dirty = subprocess.run(["git", "diff", "--quiet", "--", rel], cwd=ROOT).returncode != 0
    return "dirty" if dirty else "clean"


def main():
    entries = {}
    if OUT.exists():
        entries = json.loads(OUT.read_text())
    lines = [l.strip() for l in FILELIST.read_text().splitlines()
             if l.strip() and not l.strip().startswith("#")]
    for rel in lines:
        p = ROOT / rel
        if not p.exists():
            entries[rel] = {"error": "file not found at audit time"}
            continue
        st = p.stat()
        entries[rel] = {
            "git_last_commit": git_last_commit(rel),
            "git_status": git_tracked_dirty(rel),
            "mtime": __import__("datetime").datetime.fromtimestamp(st.st_mtime).isoformat(),
            "size_bytes": st.st_size,
            "sha256": sha256_of(p),
        }
    OUT.write_text(json.dumps(entries, indent=2, sort_keys=True))
    print(f"wrote {OUT} ({len(entries)} files)")


if __name__ == "__main__":
    main()
