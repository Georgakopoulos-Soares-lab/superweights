"""
models/caduceus_wrapper.py
----------------------------
Wrapper for Caduceus (bidirectional, reverse-complement-equivariant Mamba
masked-LM), the Tier 2 non-attention-mixer generality test.

Caduceus has NO gated FFN anywhere -- each block is a pure Mamba (SSM) mixer.
Blocks return a (hidden_states, residual) tuple (mamba_ssm's fused-add-norm
convention); the true residual-stream value at a block boundary is their sum.
get_target_module() is not meaningful here (no down_proj exists) -- ablation
is done at the block level directly by hooks/ablation_trace.py, which special-
cases model_key == "caduceus".

Requires the `caduceus` conda env (mamba_ssm compiled from source; see
results/mechanism/caduceus_env_notes.md for what was required).
"""
import torch
from .base_wrapper import BaseGenomicWrapper


class CaduceusWrapper(BaseGenomicWrapper):

    def load(self):
        # mamba_ssm 2.2.2's utils/generation.py imports names that newer
        # transformers versions removed; transformers==4.44.2 (pinned in the
        # dedicated `caduceus` conda env) still has them, so no monkeypatch
        # is needed there. Import mamba_ssm first so its custom CUDA ops
        # register before Caduceus's remote code (which imports from it).
        import mamba_ssm  # noqa: F401
        from transformers import AutoModel, AutoTokenizer

        dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
        dtype = dtype_map.get(self.config.get("dtype", "float32"), torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_id"], trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.config["model_id"], trust_remote_code=True, torch_dtype=dtype)
        self.model.eval()
        if self.config.get("device") == "cuda":
            self.model = self.model.cuda()

    def forward(self, sequence: str):
        inputs = self.tokenizer(sequence, return_tensors="pt")
        if self.config.get("device") == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            self.model(**inputs)

    def get_target_module(self, layer_idx: int):
        raise NotImplementedError(
            "Caduceus has no down_proj; ablation hooks the block itself "
            "(see hooks/ablation_trace.py's caduceus special case).")

    @property
    def num_layers(self) -> int:
        return self.config["num_layers"]
