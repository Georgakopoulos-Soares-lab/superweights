# models/ntv3_wrapper.py
import torch
from .base_wrapper import BaseGenomicWrapper

# Pinned remote-code revision (the Hub's default/latest NTv3 code has since
# drifted and breaks the forward pass with internal shape mismatches).
# Matches the pin used in scripts/evaluation/run_gue_multiseed.py and
# scripts/analysis/run_ntv3_uk_per_layer.py.
_NTV3_CODE_REVISION = "0ecff3637f0d3ba5b686d1095083218157c2ca34"


class NTv3Wrapper(BaseGenomicWrapper):

    def load(self):
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        trust_remote_code = self.config.get("hf_trust_remote_code", True)
        code_revision = self.config.get("code_revision", _NTV3_CODE_REVISION)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_id"], trust_remote_code=trust_remote_code,
            code_revision=code_revision)
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.config["model_id"],
            torch_dtype=torch.float32,
            trust_remote_code=trust_remote_code,
            code_revision=code_revision,
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
