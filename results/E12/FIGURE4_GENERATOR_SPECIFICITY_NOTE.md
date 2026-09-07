# Figure 4 GENERator specificity: validation and provenance

No model inference or new experiment was run. The figure is reconstructed from stored final artifacts.

## Numerical anchors

- Exact q1, L4/r2371: `0.968869125237583`.
- Baseline generated GC: `0.420401114004630`.
- L4/r2371 ablation: native NLL `8.754319605827332`; GC `0.306495949074074`.
- Selected random-direction scale: `c=0.0125` (closest tested nonzero condition).
- Random-direction endpoint: native NLL `8.753588829040527`; GC `0.306848596643519`.
- The uncorrected `damage_matching.csv` GC field is `0.306672272858796` and is not used in the plot.
- Random minus ablation: ΔNLL `-0.000730776786805`; ΔGC `+0.000352647569444`.
- Maximum native NLL reached at any tested scale by any of the five non-2371 rows: `6.386865547895431`.
- GC estimates use `96` aligned prompts and generation seeds 42, 43, 44. Error bars use `5000` prompt-bootstrap draws with seed `42`.

The random-direction matching table marks the endpoint as a fallback because the target was not bracketed by distinct nonzero directions. `c=0` is the zero vector and therefore coincides with row ablation; the selected `c=0.0125` value is the closest tested nonzero replacement.

### Documented aggregation bug

`run_e12_full.py::write_damage_matching_csv` filters generation records by `label` but not by the selected `value`. After the direction-fix rerun left both `c=0` and `c=0.0125` records under `matched_random_direction`, the CSV pooled all six condition records. Its value `0.3066722728587963` is exactly the mean of the scale-specific `c=0` GC (`0.3064959490740741`) and `c=0.0125` GC (`0.3068485966435185`). Figure 4 and its plotting CSV use only the three stored `c=0.0125` records (seeds 42, 43, and 44). The original E12 artifacts are not altered.

## Exact sources

- `results/e7_legacy_reanalysis.json` — exact q1.
- `results/mechanism/attention_sink_implicit_bias.json` — BOS attention and activation co-occurrence.
- `results/E12/damage_matching.csv` — selected scales, native NLL, final GC means, reachability, and full tested grids.
- `results/E12/raw/damage_evals.jsonl` — native-NLL measurements.
- `results/E12/raw/gen_records.jsonl` — per-seed, per-prompt generated GC measurements.
- `results/E12/E12_summary.md` — final interpretation and paired-difference bootstrap results.
- `paper-salvage/experiments/E12_generator_degradation_control/run_e12_full.py` and `e12_lib.py` — intervention, damage, generation, and bootstrap implementation.

## Caption

**Figure 4. Functional consequences of disrupting a concentrated high-gain location in GENERator EUK.** (A) Exact spectral concentration of the primary high-gain row L4/r2371. (B) Reproduced BOS-centered attention/activation phenotype. Mean incoming attention mass at position 0 is 37.9% (33.0× uniform), 78.8% of layer-head observations have their maximum there, and the high-gain activation maximum is also at position 0. These measurements establish co-occurrence only, not that L4/r2371 causes the attention sink. (C) Generated GC fraction versus native NLL for intact baseline, L4/r2371 ablation, the five highest-damage non-2371 endpoints from the preregistered sweep, and a fixed random-direction replacement at L4/r2371. The nonzero random-direction endpoint nearly overlaps ablation in damage and GC, so the low-GC effect does not require the learned row-2371 direction. The other rows never reached comparable native damage and therefore do not establish uniqueness to this location. Error bars are 95% percentile intervals from 5,000 bootstrap resamples of 96 prompt-level GC values after averaging the three generation seeds per prompt.

## Outputs

- `paper-salvage/figures/main/fig4_generator_specificity.png`
- `paper-salvage/figures/main/fig4_generator_specificity.pdf`
- `results/E12/figure4_generator_specificity_panel_c.csv`
- `paper-salvage/figures/fig4_generator_specificity.py`
