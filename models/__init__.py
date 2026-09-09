from .generator_wrapper import GeneratorWrapper
from .evo2_wrapper import Evo2Wrapper
from .ntv3_wrapper import NTv3Wrapper
from .dnabert2_wrapper import DNABERT2Wrapper

WRAPPER_MAP = {
    "generator":             GeneratorWrapper,
    "generator_prokaryote":  GeneratorWrapper,
    "evo2":                  Evo2Wrapper,
    "ntv3":                  NTv3Wrapper,
    "dnabert2":              DNABERT2Wrapper,
}

# Evo1 wrapper depends on the `evo` package which is only installed in the
# `evo` conda env. Import lazily so other envs (generator/biojepa) can still
# import models.__init__ without crashing.
try:
    from .evo1_wrapper import Evo1Wrapper
    WRAPPER_MAP["evo1"] = Evo1Wrapper
except ImportError:
    pass

# Caduceus wrapper depends on `mamba_ssm` (compiled CUDA extension), only
# installed in the dedicated `caduceus` conda env. Import lazily.
try:
    from .caduceus_wrapper import CaduceusWrapper
    WRAPPER_MAP["caduceus"] = CaduceusWrapper
except ImportError:
    pass
