# models/dnabert2_wrapper.py
import torch
from transformers import AutoModel, AutoTokenizer
from .base_wrapper import BaseGenomicWrapper


class DNABERT2Wrapper(BaseGenomicWrapper):

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_id"],
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.config["model_id"],
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        self.model.eval()
        if self.config.get("device") == "cuda":
            self.model = self.model.cuda()

    def forward(self, sequence: str):
        inputs = self.tokenizer(
            sequence, return_tensors="pt", truncation=True, max_length=512
        )
        if self.config.get("device") == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            self.model(**inputs)

    def get_target_module(self, layer_idx: int):
        return self._resolve_module(self.config["down_proj_pattern"], layer_idx)

    @property
    def num_layers(self) -> int:
        return self.config["num_layers"]
