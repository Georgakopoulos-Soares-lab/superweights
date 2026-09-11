# stubs/

**Import shims for optional or unavailable dependencies — not PEP 484 type stubs.**

The name is a historical accident and reads the wrong way to most Python developers. Nothing
here is a `.pyi` file and nothing here is consumed by a type checker.

| shim | stands in for | needed by |
|---|---|---|
| `mamba_ssm/` | the `mamba_ssm` package (selective-scan CUDA kernels) | the Caduceus wrapper, so it can be imported without a CUDA build of `mamba_ssm` present |
| `hf/configuration_hybridna.py` | HybriDNA's remote configuration module | the HybriDNA wrapper |

`scripts/evaluation/run_gue_ablation.py` puts this directory on `sys.path` at runtime. That is
the only consumer, and it is why the directory must keep its current name and location: the
insert is a literal path, so renaming would break it silently at run time rather than at import
time. See `tests/test_entrypoints.py` for the general hazard.

No result reported in the paper depends on these shims. They exist so that wrappers for models
outside the paper's panel can be imported without installing heavy optional dependencies.
