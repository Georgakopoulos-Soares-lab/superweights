# models/evo2_wrapper.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_wrapper import BaseGenomicWrapper


class Evo2Wrapper(BaseGenomicWrapper):

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_id"],
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_id"],
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        self.model.eval()

    def forward(self, sequence: str):
        inputs = self.tokenizer(sequence, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            self.model(**inputs)

    def get_target_module(self, layer_idx: int):
        return self._resolve_module(self.config["down_proj_pattern"], layer_idx)

    @property
    def num_layers(self) -> int:
        return self.config["num_layers"]
