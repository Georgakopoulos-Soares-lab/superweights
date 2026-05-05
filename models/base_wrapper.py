# models/base_wrapper.py
from abc import ABC, abstractmethod
import torch


class BaseGenomicWrapper(ABC):
    """
    Abstract base for all model wrappers.
    Every subclass must implement: load(), forward(), get_target_module(), num_layers.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load(self): ...

    @abstractmethod
    def forward(self, sequence: str): ...

    @abstractmethod
    def get_target_module(self, layer_idx: int): ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...

    def zero_out_weight(self, layer_idx: int, row: int, col: int):
        """Zero out a single scalar weight in-place (used during iterative detection)."""
        module = self.get_target_module(layer_idx)
        weight = module.weight
        with torch.no_grad():
            weight.data[row, col] = 0.0
        # Verify the write actually stuck
        actual = weight.data[row, col].item()
        if actual != 0.0:
            raise RuntimeError(
                f"zero_out_weight failed at layer={layer_idx}, row={row}, col={col}. "
                f"Value is still {actual}. Check device placement and tensor sharing."
            )
        print(f"  → Zeroed layer={layer_idx}[{row},{col}] confirmed.")

    def zero_out_row(self, layer_idx: int, row: int):
        """Zero out an entire output row in-place (super row detection)."""
        module = self.get_target_module(layer_idx)
        with torch.no_grad():
            module.weight.data[row, :] = 0.0
        actual_max = module.weight.data[row, :].abs().max().item()
        if actual_max != 0.0:
            raise RuntimeError(
                f"zero_out_row failed at layer={layer_idx}, row={row}. "
                f"Max abs value still {actual_max}."
            )
        print(f"  → Zeroed row layer={layer_idx}[{row},:] ({module.weight.data.shape[1]} weights) confirmed.")

    def _resolve_module(self, pattern: str, layer_idx: int):
        """Navigate model attribute tree using dot-separated pattern with {i} placeholder."""
        path = pattern.replace("{i}", str(layer_idx))
        module = self.model
        for attr in path.split("."):
            module = getattr(module, attr)
        return module
