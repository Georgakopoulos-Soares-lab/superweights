"""
Lock a preregistration so a later claim of "we predicted this beforehand" is checkable.

Records sha256 of the file, the git commit, and a UTC timestamp into an append-only
ledger. `verify` re-hashes and reports any file that changed after locking.

    python src/prereg_lock.py lock docs/prereg/PREREG_evo1_broadcast.md
    python src/prereg_lock.py verify --all
    python src/prereg_lock.py show

The ledger is append-only. Editing a locked prereg is not forbidden -- it is *recorded*,
which is the point. If you must revise, lock again and disclose both entries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

LEDGER = Path(__file__).resolve().parent.parent / "docs" / "prereg" / "LOCKS.jsonl"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def entries() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]


def cmd_lock(path_str: str) -> int:
    path = Path(path_str).resolve()
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 1

    digest = sha256(path)
    prior = [e for e in entries() if e["path"] == str(path)]
    if prior and prior[-1]["sha256"] == digest:
        print(f"already locked, unchanged: {path.name}\n  {digest[:16]}  {prior[-1]['utc']}")
        return 0

    entry = {
        "path": str(path),
        "name": path.name,
        "sha256": digest,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "relock_of": prior[-1]["sha256"] if prior else None,
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(entry) + "\n")

    if prior:
        print(f"WARNING: re-locking {path.name} -- it changed after a previous lock.")
        print("         Both entries stay in the ledger. Disclose this if the prereg is cited.")
    if entry["git_dirty"]:
        print("WARNING: working tree is dirty; the commit reference is approximate.")
    print(f"locked {path.name}\n  sha256 {digest}\n  utc    {entry['utc']}\n  commit {entry['git_commit']}")
    return 0


def cmd_verify(all_: bool, path_str: str | None) -> int:
    es = entries()
    if not es:
        print("no locks recorded")
        return 0
    latest: dict[str, dict] = {}
    for e in es:
        latest[e["path"]] = e
    targets = latest.values() if all_ else [latest[str(Path(path_str).resolve())]]

    bad = 0
    for e in targets:
        p = Path(e["path"])
        if not p.exists():
            print(f"MISSING  {e['name']}")
            bad += 1
            continue
        now = sha256(p)
        if now == e["sha256"]:
            print(f"OK       {e['name']}  locked {e['utc']}")
        else:
            print(f"CHANGED  {e['name']}  locked {e['utc']}")
            print(f"           was {e['sha256'][:16]}  now {now[:16]}")
            bad += 1
    return 1 if bad else 0


def cmd_show() -> int:
    for e in entries():
        flag = "  (relock)" if e.get("relock_of") else ""
        print(f"{e['utc']}  {e['sha256'][:12]}  {e['name']}{flag}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("lock"); s.add_argument("path")
    s = sub.add_parser("verify")
    s.add_argument("path", nargs="?"); s.add_argument("--all", action="store_true")
    sub.add_parser("show")

    a = p.parse_args()
    if a.cmd == "lock":
        sys.exit(cmd_lock(a.path))
    if a.cmd == "verify":
        if not a.all and not a.path:
            p.error("give a path or --all")
        sys.exit(cmd_verify(a.all, a.path))
    sys.exit(cmd_show())
