# manuscript/

The paper and everything that produced it. Named `paper-salvage/` until 2026-09-09, when it
was renamed because "salvage" described a past rescue effort rather than the directory's
contents.

## Current thesis

> High-gain gated-FFN rows recur across text and genomic foundation models, but structural
> prominence, functional criticality, and finite-intervention causal response complexity are
> distinct quantities.

```
STRUCTURAL GEOMETRY  !=  FUNCTIONAL CRITICALITY  !=  CAUSAL RESPONSE COMPLEXITY
```

Two earlier theses are retired: the v1 U_k-led framing (`docs/DECISIONS.md` D-016) and the
"structural amplifier → architectural routing → causal steering" chain that an earlier version
of this README stated as current. Do not reinstate either.

## Layout

| path | contents |
|---|---|
| `actual_manuscript.md` / `.tex` | The paper. Kept content-identical; edit both or neither. Converted from a Word original, so notation is text-extracted — check equations against the source before editing them. |
| `media/` | Figure artwork as referenced by the manuscript (`image1`–`image8`). Both formats point at this one copy. |
| `figures/` | The figure pipeline: render scripts, output, `source_data/` (the exact plotted values), and `FIGURE_PROVENANCE.md` (panel-by-panel manifest) |
| `experiments/E1`–`E13` | **Frozen** experiment harnesses — the code that produced the census and the case studies. *Frozen* means not edited to make later results come out differently. `run_bos_mediation.py` and `e12_lib.py` in particular are reused unmodified by later analyses. |
| `docs/` | Preregistrations (content-locked), `DECISIONS.md`, `CLAIMS_LEDGER.md`, outlines, audit correspondence |
| `docs/prereg/` | 12 locked preregistrations. Verify with `python src/prereg_lock.py verify --all` |
| `src/` | `prereg_lock.py` and manuscript-side helpers |
| `results/keep/` | Curated, provenance-locked artifacts for E1 and E2. `sw_broadcast_impulse.json` is duplicated at `results/` in the repo root: the root copy is the working path several scripts read, this one is the locked archival copy. |
| `CLAUDE.md` | Historical operating rules. Its hard constraints on claims are still worth reading; its stated thesis is not current. |

## Numbering note

The manuscript figures are `image6`→Fig 1, `image1`→Fig 2, `image3`→Fig 3, `image7`→Fig 4,
`image8`→Fig 5, `image4`→Fig S1, `image2`→Fig S2. The names come from the Word export and are
not sequential; `image5` is unused. The mapping to the real artwork in `figures/` is recorded
here because it is not inferable from the filenames.
