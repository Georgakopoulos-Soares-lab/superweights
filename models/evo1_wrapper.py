"""
models/evo1_wrapper.py
----------------------
Thin wrapper around the togethercomputer Evo-1 (StripedHyena) model so that
detection / ablation / mechanistic scripts can target it via the same
interface used for Evo2.

Requires the `evo` Python package (i.e. the `evo` conda env on this machine).
"""
import torch
import torch.nn.functional as F

from .base_wrapper import BaseGenomicWrapper


class Evo1Wrapper(BaseGenomicWrapper):
    def load(self):
        from evo import Evo
        ckpt = self.config.get("model_id", "evo-1-8k-base")
        # Evo() accepts the short name "evo-1-8k-base" or "evo-1-131k-base"
        short = ckpt.split("/")[-1] if "/" in ckpt else ckpt

        # Pass device into Evo() so the loader's to_bfloat16_except_poles_residues()
        # pipeline is preserved (poles/residues must remain fp32; a blanket
        # .to(dtype=fp16) after-the-fact corrupts the StripedHyena conv kernels
        # and yields NaN activations).
        device = self.config.get("device", "cuda")
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.evo = Evo(short, device=device)
        self.model = self.evo.model
        self.tokenizer = self.evo.tokenizer
        self.model.eval()
        # unembed
        self.unembed = None
        for attr in ("unembed", "lm_head", "head"):
            cand = getattr(self.evo, attr, None) or getattr(self.model, attr, None)
            if cand is not None:
                self.unembed = cand
                break

    # --------------------------------------------------------------- helpers
    def _device(self):
        return next(self.model.parameters()).device

    def _tokenize(self, sequence: str) -> torch.Tensor:
        device = self._device()
        if self.tokenizer is not None:
            encode = getattr(self.tokenizer, "encode", None) or \
                     getattr(self.tokenizer, "tokenize", None)
            tokens = encode(sequence)
            if isinstance(tokens, torch.Tensor):
                return tokens.view(1, -1).long().to(device)
            return torch.tensor(list(tokens), dtype=torch.long,
                                device=device).unsqueeze(0)
        # byte-level fallback
        return torch.tensor([[ord(c) for c in sequence]],
                            dtype=torch.long, device=device)

    def _logits(self, input_ids):
        out = self.model(input_ids)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        if logits.dim() == 3:
            logits = logits[0]
        return logits

    # ----------------------------------------------------------- public API
    def forward(self, sequence: str):
        with torch.no_grad():
            self.model(self._tokenize(sequence))

    def compute_perplexity(self, sequence: str) -> float:
        ids = self._tokenize(sequence)
        with torch.no_grad():
            logits = self._logits(ids)
        shift_logits = logits[:-1].float()
        shift_labels = ids[0, 1:]
        return torch.exp(F.cross_entropy(shift_logits, shift_labels)).item()

    def compute_perplexity_ablated(self, sequence: str, sw_list: list,
                                   mode: str = "superrow") -> float:
        ids = self._tokenize(sequence)

        def _unwrap(out):
            return (out[0], out[1:]) if isinstance(out, tuple) else (out, None)

        def _rewrap(o, rest):
            return (o,) + rest if rest is not None else o

        handles = []
        for entry in sw_list:
            layer = entry["layer"]; row = entry["row"]
            module = self.get_target_module(layer)
            if mode == "superrow":
                def make_hook(r):
                    def _hook(mod, inp, output):
                        o, rest = _unwrap(output)
                        o[..., r] = 0.0
                        return _rewrap(o, rest)
                    return _hook
                handles.append(module.register_forward_hook(make_hook(row)))
            else:  # superweight
                col = entry["col"]
                w_rc = module.weight.data[row, col].item()

                def make_hook(r, c, w):
                    def _hook(mod, inp, output):
                        o, rest = _unwrap(output)
                        in_t = inp[0] if isinstance(inp, tuple) else inp
                        if in_t is not None and in_t.shape[-1] > c:
                            o[..., r] = o[..., r] - w * in_t[..., c].float()
                        return _rewrap(o, rest)
                    return _hook
                handles.append(module.register_forward_hook(make_hook(row, col, w_rc)))

        try:
            with torch.no_grad():
                logits = self._logits(ids)
        finally:
            for h in handles:
                h.remove()
        shift_logits = logits[:-1].float()
        shift_labels = ids[0, 1:]
        return torch.exp(F.cross_entropy(shift_logits, shift_labels)).item()

    def get_target_module(self, layer_idx: int):
        return self._resolve_module(self.config["down_proj_pattern"], layer_idx)

    @property
    def num_layers(self) -> int:
        return self.config["num_layers"]
