"""
stubs/hf/configuration_hybridna.py

HybriDNA's modeling_hybridna.py has a broken absolute import:
    from hf.configuration_hybridna import HybriDNAConfig
The 'hf/' subdirectory doesn't exist in the public HF repo — it was a local
path in the authors' development environment.

This stub satisfies:
  1. transformers' check_imports scan (importlib.import_module('hf') succeeds)
  2. Runtime: provides the real HybriDNAConfig class by finding it in
     sys.modules (already loaded by AutoConfig.from_pretrained before the
     modeling file is imported).
"""
import sys


def _find_hybridna_config():
    # First pass: check sys.modules — AutoConfig loads configuration_hybridna.py
    # before modeling_hybridna.py, so the class is already registered.
    for _mod_name, _mod in list(sys.modules.items()):
        if (_mod_name.endswith(".configuration_hybridna")
                and hasattr(_mod, "HybriDNAConfig")):
            return _mod.HybriDNAConfig

    # Fallback: locate and load configuration_hybridna.py from the HF modules
    # cache directly (used when this stub is imported standalone, e.g. by
    # check_imports, before AutoConfig has run).
    import os
    import glob
    import importlib.util

    hf_home = os.environ.get(
        "HF_HOME",
        os.environ.get("TRANSFORMERS_CACHE",
                       os.path.expanduser("~/.cache/huggingface"))
    )
    pattern = os.path.join(
        hf_home, "modules", "transformers_modules",
        "Mishamq", "*", "configuration_hybridna.py"
    )
    files = sorted(glob.glob(pattern))
    if files:
        spec = importlib.util.spec_from_file_location(
            "_hybridna_cfg_real", files[-1]
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.HybriDNAConfig

    # Last resort: return a minimal placeholder that won't break isinstance
    # checks (there are none in the current modeling_hybridna.py).
    from transformers import PretrainedConfig
    return PretrainedConfig


HybriDNAConfig = _find_hybridna_config()
