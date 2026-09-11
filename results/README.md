# results/

Artifacts. `results/` is in `.gitignore`; files that back a reported number are force-added, so
they **are** committed and do ship in a clone. Everything else here is local working output and
is invisible to a checkout.

| directory | contents |
|---|---|
| `experiments/` | Output of the frozen harnesses in `experiments/frozen/`, one subdirectory per line: `E7` (exact spectral metrics), `E10` (NLP architecture causal), `E11` (scale ladder), `E12` (GENERator degradation), `E13` (the 22-model census) with its `E13_candidate_stability` and `E13_dnabert2_reproducibility` companions, `E_BOS_MEDIATION`, and `gue` (downstream evaluation) |
| `analyses/` | Analyses derived from those outputs: `detector_provenance` (which coordinate each selection rule picks, and the paired causal comparison), `within_layer_sweep` (the graded sweeps and Figure 5), `mechanism_generator` (BOS mediation, attention sink, random directions, DNABERT-2 pair), `census_analysis` (cohort correlations and sensitivity) |
| `negative_results/` | Outputs of the experiments whose result is a null — see `scripts/negative_results/README.md` |
| `supplementary/` | The eight supplementary tables, and the two verbatim inputs under `sources/` |

The subdirectory names under `experiments/` match the harness names under
`experiments/frozen/`, so an artifact can be traced to the code that produced it by name
alone. For the authoritative mapping in the other direction — claim to code to artifact — see
[`docs/EXPERIMENT_MAP.md`](../docs/EXPERIMENT_MAP.md).
