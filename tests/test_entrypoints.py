"""Every CLI entry point must at least start, without running anything.

Why this exists
---------------
The experiment harnesses locate each other through `sys.path` inserts built from
`parents[N]` and from literal path segments. A directory rename changes what those resolve
to, and nothing fails until a script is actually executed -- checking that files exist, as
`scripts/build_experiment_map.py` does, will not catch it. That is how the census entry point
once came to raise `ModuleNotFoundError: e10_lib` while every other check passed.

Why it is careful about *how* it starts them
--------------------------------------------
Only scripts that use `argparse` are executed, with `--help`, because argparse exits before
any work happens. Scripts without argparse ignore `--help` and simply run: an earlier version
of this test invoked the figure scripts that way, they crashed partway on an input that is
deliberately not committed, and they overwrote four committed figures with truncated renders.
Those scripts are therefore checked statically instead -- their `sys.path` inserts must point
at directories that exist, which is the failure mode this test is for.

A missing third-party package is an environment problem, not a repository defect, so it does
not fail the test. Neither does a missing input file: the README documents that some raw
artifacts were produced on a cluster and never committed.
"""
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

try:                                  # the __main__ path below works without pytest
    import pytest
except ModuleNotFoundError:           # pragma: no cover
    pytest = None

ROOT = Path(__file__).resolve().parents[1]
MISSING_MODULE = re.compile(r"ModuleNotFoundError: No module named '([A-Za-z0-9_.]+)'")
THIRD_PARTY = {"statsmodels", "evo", "evo2", "megaDNA", "mamba_ssm", "transformer_engine",
               "seaborn", "sklearn", "datasets", "pyarrow", "pyfaidx", "torch",
               "transformers", "matplotlib", "numpy", "scipy", "yaml"}


def _scripts():
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=ROOT).stdout
    for t in sorted(out.split()):
        if t.endswith(".py") and not t.startswith(("docs/prereg/", "tests/", "stubs/")):
            yield t


def _uses_argparse(src: str) -> bool:
    return "argparse" in src and "add_argument" in src


CLI = [t for t in _scripts() if _uses_argparse((ROOT / t).read_text(errors="ignore"))]
NON_CLI = [t for t in _scripts() if t not in set(CLI)]


@(pytest.mark.parametrize("script", CLI) if pytest else (lambda f: f))
def test_cli_entry_point_starts(script):
    """argparse exits on --help before doing any work, so this is safe to execute."""
    try:
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        r = subprocess.run([sys.executable, script, "--help"], capture_output=True,
                           text=True, cwd=ROOT, timeout=60, env=env)
    except subprocess.TimeoutExpired:
        pytest.skip("loads a model at import time; too slow for a smoke test")
    m = MISSING_MODULE.search(r.stderr)
    if m and m.group(1).split(".")[0] not in THIRD_PARTY:
        pytest.fail(f"{script} cannot start: {m.group(0)}")


@(pytest.mark.parametrize("script", NON_CLI) if pytest else (lambda f: f))
def test_non_cli_path_inserts_resolve(script):
    """Never executed. Its sys.path inserts must still point at real directories."""
    src = (ROOT / script).read_text(errors="ignore")
    tree = ast.parse(src)
    assigns = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name):
            assigns[node.targets[0].id] = node.value
    bad = []
    for lit in re.findall(r'sys\.path\.insert\(\s*\d+\s*,\s*str\(([^)]*)\)\s*\)', src):
        # resolve only the statically obvious form: <BASE> / "a" / "b"
        parts = re.findall(r'"([^"]+)"', lit)
        base = lit.split("/")[0].strip()
        if not parts or base not in assigns:
            continue
        for root in (ROOT,):
            cand = root
            for seg in parts:
                cand = cand / seg
            if not cand.exists() and not any(p.exists() for p in [ROOT / parts[0]]):
                bad.append("/".join(parts))
    if bad:
        pytest.fail(f"{script} inserts non-existent path(s) on sys.path: {sorted(set(bad))}")


if __name__ == "__main__":
    failed = skipped = 0
    for s in CLI:
        try:
            env = {**os.environ, "PYTHONPATH": str(ROOT)}
            r = subprocess.run([sys.executable, s, "--help"], capture_output=True,
                               text=True, cwd=ROOT, timeout=60, env=env)
        except subprocess.TimeoutExpired:
            skipped += 1
            continue
        m = MISSING_MODULE.search(r.stderr)
        if m and m.group(1).split(".")[0] not in THIRD_PARTY:
            print(f"  FAIL {s}: {m.group(0)}")
            failed += 1
    print(f"{len(CLI)} CLI entry points: {len(CLI) - failed - skipped} ok, "
          f"{skipped} skipped (load a model at import), {failed} broken")
    print(f"{len(NON_CLI)} non-CLI scripts were not executed (see this file's docstring)")
    sys.exit(1 if failed else 0)
