# models/hybridna_wrapper.py
import torch
from .base_wrapper import BaseGenomicWrapper


class HybriDNAWrapper(BaseGenomicWrapper):
    """
    Wrapper for Mishamq/HybriDNA-7B.

    HybriDNA-7B is a 32-layer causal Mamba2/SSM hybrid model with a byte-level
    DNA tokenizer (vocab=12: A,C,G,T + specials).  Attention layers appear at
    indices 4,12,20,28; all other layers use Mamba2 SSM blocks.  Both layer types
    share an identical MLP structure (HybriDNAMLP) accessible via:

        model.layers.{i}.feed_forward.down_proj

    Mamba custom CUDA kernels are disabled at load time (use_mamba_kernels=False)
    to avoid requiring triton/mamba-ssm; the eager fallback is used instead.
    """

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_id"],
            trust_remote_code=self.config.get("hf_trust_remote_code", True),
            padding_side="left",
        )

        dtype_map = {
            "float16":  torch.float16,
            "float32":  torch.float32,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(self.config.get("dtype", "bfloat16"), torch.bfloat16)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_id"],
            dtype=dtype,
            trust_remote_code=self.config.get("hf_trust_remote_code", True),
            device_map="auto",
            use_mamba_kernels=False,   # disable triton kernels; use eager SSM fallback
        )
        self.model.eval()

    def _device(self):
        return next(self.model.parameters()).device

    def _to_device(self, inputs: dict) -> dict:
        """Move a plain dict of tensors to the model device."""
        dev = self._device()
        return {k: v.to(dev) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()}

    def forward(self, sequence: str):
        inputs = self._to_device(
            self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        )
        with torch.no_grad():
            self.model(**inputs)

    def compute_perplexity(self, sequence: str) -> float:
        """Next-token cross-entropy perplexity on a single sequence."""
        inputs = self._to_device(
            self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        )
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            outputs = self.model(**inputs, labels=input_ids)
        return torch.exp(outputs.loss).item()

    def get_target_module(self, layer_idx: int):
        return self._resolve_module(self.config["down_proj_pattern"], layer_idx)

    @property
    def num_layers(self) -> int:
        return self.config["num_layers"]
