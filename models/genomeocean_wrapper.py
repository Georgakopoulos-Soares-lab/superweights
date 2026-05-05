# models/genomeocean_wrapper.py
import torch
from .base_wrapper import BaseGenomicWrapper


class GenomeOceanWrapper(BaseGenomicWrapper):
    """
    Wrapper for pGenomeOcean/GenomeOcean-4B (and smaller variants).

    GenomeOcean is a causal decoder model trained on metagenomic assemblies.
    The tokenizer is a character/BPE tokenizer with left-padding — no fixed-length
    alignment required.  Loaded with AutoModelForCausalLM; perplexity is computed
    via the model's built-in cross-entropy loss (labels trick).
    """

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_id"],
            trust_remote_code=self.config.get("hf_trust_remote_code", True),
            padding_side="left",
            use_fast=True,
        )

        dtype_map = {
            "float16":  torch.float16,
            "float32":  torch.float32,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(self.config.get("dtype", "bfloat16"), torch.bfloat16)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_id"],
            torch_dtype=dtype,
            trust_remote_code=self.config.get("hf_trust_remote_code", True),
            attn_implementation="sdpa",
            device_map="auto",
        )
        self.model.eval()

    def forward(self, sequence: str):
        inputs = self.tokenizer(
            sequence, return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)
        with torch.no_grad():
            self.model(**inputs)

    def compute_perplexity(self, sequence: str) -> float:
        """Next-token cross-entropy perplexity on a single sequence."""
        inputs = self.tokenizer(
            sequence, return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            outputs = self.model(**inputs, labels=input_ids)
        return torch.exp(outputs.loss).item()

    def get_target_module(self, layer_idx: int):
        return self._resolve_module(self.config["down_proj_pattern"], layer_idx)

    @property
    def num_layers(self) -> int:
        return self.config["num_layers"]
