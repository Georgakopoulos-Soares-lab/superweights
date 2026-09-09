# models/dnabert2_wrapper.py
import torch
from .base_wrapper import BaseGenomicWrapper

# Pinned revision — matches the tokenizer revision used in run_gue_ablation.py
_DNABERT2_REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"


class DNABERT2Wrapper(BaseGenomicWrapper):

    def load(self):
        from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

        rev = self.config.get("revision", _DNABERT2_REVISION)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_id"],
            trust_remote_code=True,
            revision=rev,
        )

        # transformers ≥ 5.x removed BertConfig defaults for is_decoder and
        # pad_token_id, which the DNABERT-2 custom bert_layers.py still accesses
        # directly.  Load config separately and inject the missing defaults.
        cfg = AutoConfig.from_pretrained(
            self.config["model_id"],
            trust_remote_code=True,
            revision=rev,
        )
        if not hasattr(cfg, "is_decoder") or cfg.is_decoder is None:
            cfg.is_decoder = False
        if not hasattr(cfg, "pad_token_id") or cfg.pad_token_id is None:
            cfg.pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else 0
            )

        self.model = AutoModelForMaskedLM.from_pretrained(
            self.config["model_id"],
            config=cfg,
            trust_remote_code=True,
            revision=rev,
            # device_map="cpu" avoids the meta-device / alibi tensor conflict
            # that arises with transformers ≥5 + the DNABERT-2 custom __init__.
            # The model is then moved to CUDA below.
            device_map={"": "cpu"},
        )
        self.model.eval()
        if self.config.get("device") == "cuda":
            self.model = self.model.cuda()

        # DNABERT-2 ships a 2022-era Triton flash-attention kernel that uses the
        # removed tl.dot(trans_b=...) API and breaks on modern Triton (>=3.x).
        # Force the mathematically-equivalent PyTorch attention fallback
        # (bert_layers.py uses it whenever the Triton func is None), matching
        # the workaround already applied in scripts/evaluation/run_gue_multiseed.py.
        import sys as _sys
        for _name, _mod in list(_sys.modules.items()):
            if _name.endswith("bert_layers") and "DNABERT" in _name:
                _mod.flash_attn_qkvpacked_func = None

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
