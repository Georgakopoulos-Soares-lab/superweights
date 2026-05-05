# models/generator_wrapper.py
import torch
from .base_wrapper import BaseGenomicWrapper


class GeneratorWrapper(BaseGenomicWrapper):

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_id"],
            trust_remote_code=self.config.get("hf_trust_remote_code", True),
        )
        dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
        dtype = dtype_map.get(self.config.get("dtype", "float32"), torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_id"],
            torch_dtype=dtype,
            trust_remote_code=self.config.get("hf_trust_remote_code", True),
            device_map="auto",
        )
        self.model.eval()

    def _prepare_sequence(self, sequence: str) -> str:
        # GENERator uses a 6-mer tokenizer — length must be divisible by 6
        remainder = len(sequence) % 6
        if remainder:
            sequence = sequence[remainder:]   # left-truncate to nearest multiple of 6
        bos = self.tokenizer.bos_token or ""
        return bos + sequence

    def forward(self, sequence: str):
        seq = self._prepare_sequence(sequence)
        inputs = self.tokenizer(
            seq, return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)
        with torch.no_grad():
            self.model(**inputs)

    def get_target_module(self, layer_idx: int):
        return self._resolve_module(self.config["down_proj_pattern"], layer_idx)

    @property
    def num_layers(self) -> int:
        return self.config["num_layers"]
