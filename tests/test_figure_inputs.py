"""Every figure script's input artifacts must exist in the tree.

Why this exists
---------------
`tests/test_entrypoints.py` verifies that imports resolve. It does not look at data paths, and
it never executes the figure scripts at all (they have no argparse, so running them would do
real work -- an earlier version of that test did exactly that and overwrote four committed
figures with truncated renders).

That left a blind spot, and a real bug sat in it: `fig4_generator.py` read
`results/mechanism/attention_sink_implicit_bias.json` for months after the artifact moved to
`results/analyses/mechanism_generator/`, so Figure 4 could not render from a clean checkout.
For TMLR this is not hygiene -- a figure that cannot be produced from the committed tree is a
claim unsupported by available evidence.

This test statically extracts the `results/...` and `audit/...` paths each figure script reads
and checks they exist. Paths built from variables are not resolvable this way and are skipped;
the check is deliberately conservative, catching the literal-path case that actually broke.
"""
import re
from pathlib import Path

try:
    import pytest
except ModuleNotFoundError:                      # pragma: no cover
    pytest = None

ROOT = Path(__file__).resolve().parents[1]
FIGDIRS = [ROOT / "experiments" / "figures", ROOT / "experiments" / "figures" / "supplement"]
# `<BASE> / "a" / "b.json"` where BASE is a module-level Path variable, and bare literals.
SEGMENTED = re.compile(r'\b([A-Z][A-Z0-9_]*)\s*/\s*((?:"[^"]+"\s*/\s*)*"[^"]+\.(?:json|csv|tsv)")')
LITERAL = re.compile(r'["\']((?:results|audit)/[^"\']+\.(?:json|csv|tsv))["\']')
# BASE assignments such as: RESULTS = ROOT / "results"   /   AUDIT2 = REPO_ROOT / "audit" / "rederivations"
BASE_DEF = re.compile(r'^\s*([A-Z][A-Z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*/\s*'
                      r'((?:"[^"]+"\s*/?\s*)*"[^"]+")\s*$', re.M)


def _scripts():
    for d in FIGDIRS:
        if d.is_dir():
            for p in sorted(d.glob("*.py")):
                if p.name != "_figstyle.py" and not p.name.startswith("_"):
                    yield p


def _declared_inputs(src: str):
    """Repo-relative paths the script names as inputs, resolving its base variables."""
    # Collect `NAME = PARENT / "a" / "b"` then resolve chains (E12 = RESULTS/"experiments"/"E12"
    # only means results/experiments/E12 once RESULTS itself is known). ROOT is the repo root,
    # so it contributes no prefix.
    raw = {m.group(1): (m.group(2), "/".join(re.findall(r'"([^"]+)"', m.group(3))))
           for m in BASE_DEF.finditer(src)}
    bases = {}
    for _ in range(len(raw) + 1):                    # iterate to a fixed point
        for name, (parent, tail) in raw.items():
            if name in bases:
                continue
            if parent in {"ROOT", "REPO_ROOT", "HERE"}:
                bases[name] = tail
            elif parent in bases:
                bases[name] = f"{bases[parent]}/{tail}"
    out = set()
    for m in LITERAL.finditer(src):
        out.add(m.group(1))
    for m in SEGMENTED.finditer(src):
        base, rest = m.group(1), m.group(2)
        if base not in bases:            # unknown base -> cannot resolve statically; skip
            continue
        segs = re.findall(r'"([^"]+)"', rest)
        if segs:
            out.add(bases[base] + "/" + "/".join(segs))
    return out


def _check(path: Path):
    src = path.read_text(errors="ignore")
    missing = []
    for rel in sorted(_declared_inputs(src)):
        if not (ROOT / rel).exists():
            missing.append(rel)
    return missing


if pytest:
    @pytest.mark.parametrize("script", sorted(_scripts(), key=str), ids=lambda p: p.name)
    def test_figure_inputs_exist(script):
        missing = _check(script)
        if missing:
            pytest.fail(f"{script.relative_to(ROOT)} reads artifacts that are not in the tree: "
                        + ", ".join(missing))


if __name__ == "__main__":
    import sys
    bad = 0
    for p in _scripts():
        m = _check(p)
        status = "OK " if not m else "MISSING"
        print(f"  {status:8s} {p.relative_to(ROOT)}")
        for x in m:
            print(f"           -> {x}")
            bad += 1
    print(f"\n{len(list(_scripts()))} figure scripts checked, {bad} missing input path(s)")
    sys.exit(1 if bad else 0)
