# models/megadna_wrapper.py
"""
Wrapper for megaDNA (MEGABYTE-based hierarchical DNA model).

megaDNA is loaded from a pickled .pt file (not a HuggingFace checkpoint).
It requires the megaDNA + MEGABYTE_pytorch packages, which live in a separate
Python 3.9 venv at /work/11034/atzanakak/ls6/venvs/megadna.

Architecture (confirmed from inspection of megaDNA_phage_145M.pt):
  3 stages × 8 layers = 24 total FF modules
  model.transformers        — nn.ModuleList of 3 stage Transformers
  transformer.layers        — nn.ModuleList of 8 [attn, ff] pairs
  ff                        — nn.Sequential:
                                [0] RMSNorm
                                [1] Linear(512 → 2048)    ← up-proj
                                [2] GELU
                                [3] Dropout(p=0.0)
                                [4] Linear(2048 → 512)    ← down-proj (target)

layer_idx in get_target_module is FLATTENED across all stages:
  stage = layer_idx // layers_per_stage
  local = layer_idx % layers_per_stage
  e.g. layer_idx=0..7 → stage 0, 8..15 → stage 1, 16..23 → stage 2

Vocabulary: ['**'(pad=0), 'A'=1, 'T'=2, 'C'=3, 'G'=4, '#'=5]
forward(ids, return_loss=True) returns scalar cross-entropy loss.
"""
import sys
import torch

from .base_wrapper import BaseGenomicWrapper

BASE_TO_TOKEN = {"A": 1, "T": 2, "C": 3, "G": 4}


def _encode(sequence: str) -> torch.Tensor:
    """Encode ACGT string to token id tensor, skipping non-ACGT characters."""
    ids = [BASE_TO_TOKEN[b] for b in sequence.upper() if b in BASE_TO_TOKEN]
    return torch.tensor(ids, dtype=torch.long)


class MegaDNAWrapper(BaseGenomicWrapper):

    LAYERS_PER_STAGE = 8  # confirmed from inspection

    def load(self):
        megadna_src = self.config.get("megadna_src", "/work/11034/atzanakak/ls6/megaDNA")
        if megadna_src not in sys.path:
            sys.path.insert(0, megadna_src)

        # Must import before torch.load so pickle can resolve the class
        from megaDNA.megadna import MEGADNA  # noqa: F401

        model_path = self.config["model_path"]
        device_str = self.config.get("device", "cuda")
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")

        obj = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = obj.to(device)
        self.model.eval()

        # Count actual layers at runtime and patch config
        self._layers_per_stage = self.LAYERS_PER_STAGE
        self._num_stages = len(self.model.transformers)
        self._total_layers = self._num_stages * self._layers_per_stage
        # Respect target_stage: if set, only expose that stage's layers for sweeping
        self._target_stage = self.config.get("target_stage", None)
        if self._target_stage is not None:
            self.config["num_layers"] = self._layers_per_stage
        else:
            self.config["num_layers"] = self._total_layers

        # megaDNA has no HF tokenizer; byte-level encoding is done in _prepare_ids
        self.tokenizer = None

    # ------------------------------------------------------------------
    # BaseGenomicWrapper interface
    # ------------------------------------------------------------------

    def forward(self, sequence: str):
        ids = self._prepare_ids(sequence)
        with torch.no_grad():
            self.model(ids, return_value='logits')

    def get_target_module(self, layer_idx: int):
        """Return the down-proj Linear[4] for the given layer_idx.

        If target_stage is set (e.g. 0), layer_idx is local to that stage.
        Otherwise layer_idx is flattened across all stages.
        """
        L = self._layers_per_stage
        if self._target_stage is not None:
            stage = self._target_stage
            local = layer_idx
        else:
            stage = layer_idx // L
            local = layer_idx % L
        return self.model.transformers[stage].layers[local][1][4]

    @property
    def num_layers(self) -> int:
        if self._target_stage is not None:
            return self._layers_per_stage
        return self._total_layers

    # ------------------------------------------------------------------
    # Perplexity — use model's own cross-entropy loss
    # ------------------------------------------------------------------

    def compute_perplexity(self, sequence: str) -> float:
        ids = self._prepare_ids(sequence)
        with torch.no_grad():
            loss = self.model(ids, return_value='loss')
        return torch.exp(loss).item()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_ids(self, sequence: str) -> torch.Tensor:
        ids = _encode(sequence)
        device = next(self.model.parameters()).device
        return ids.unsqueeze(0).to(device)  # [1, L]
