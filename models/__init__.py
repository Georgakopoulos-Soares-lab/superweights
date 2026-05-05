from .generator_wrapper import GeneratorWrapper
from .evo2_wrapper import Evo2Wrapper
from .ntv3_wrapper import NTv3Wrapper
from .dnabert2_wrapper import DNABERT2Wrapper

WRAPPER_MAP = {
    "generator": GeneratorWrapper,
    "evo2":      Evo2Wrapper,
    "ntv3":      NTv3Wrapper,
    "dnabert2":  DNABERT2Wrapper,
}
