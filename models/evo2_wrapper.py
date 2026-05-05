# models/evo2_wrapper.py
import torch
from .base_wrapper import BaseGenomicWrapper


class Evo2Wrapper(BaseGenomicWrapper):

    def load(self):
        # A100 is compute capability 8.0; FP8 requires 8.9+.
        # Monkeypatch transformer_engine's fp8_autocast to a no-op before loading.
        try:
            from contextlib import contextmanager
            import transformer_engine.pytorch as te

            @contextmanager
            def _noop_fp8(*args, **kwargs):
                yield

            te.fp8_autocast = _noop_fp8
        except ImportError:
            pass

        from evo2 import Evo2
        self.evo2 = Evo2(self.config["model_id"])
        # StripedHyena inner model — navigation root for get_target_module
        self.model = self.evo2.model
        self.model.eval()

        # Locate the tokenizer (attribute name varies across evo2 versions)
        self.tokenizer = None
        for attr in ("tokenizer", "tok", "token_encoder"):
            if hasattr(self.evo2, attr):
                self.tokenizer = getattr(self.evo2, attr)
                break

        # Locate the unembed (lm_head) layer.
        # unembed.weight is NOT inside self.model (it shows as 'extra key' when
        # loading the checkpoint) — search self.evo2 first, then self.model.
        self.unembed = None
        for attr in ("unembed", "lm_head", "head", "output"):
            candidate = getattr(self.evo2, attr, None) or getattr(self.model, attr, None)
            if candidate is not None and callable(candidate):
                self.unembed = candidate
                print(f"[Evo2Wrapper] unembed found at: {attr}")
                break
        if self.unembed is None:
            # Fallback: scan all nn.Module children of self.evo2 for weight shape matching
            # [vocab_size, hidden_dim] — typically [512, 4096] for evo2_7b.
            import torch.nn as nn
            hidden_dim = self.config.get("hidden_dim", 4096)
            for name, mod in self.evo2.__dict__.items():
                if isinstance(mod, nn.Module):
                    for pname, p in mod.named_parameters():
                        if p.dim() == 2 and p.shape[1] == hidden_dim:
                            self.unembed = mod
                            print(f"[Evo2Wrapper] unembed found by shape scan: evo2.{name}")
                            break
                if self.unembed is not None:
                    break
        if self.unembed is None:
            print("[Evo2Wrapper] WARNING: unembed not found — perplexity will be unavailable")

    def forward(self, sequence: str):
        device = next(self.model.parameters()).device
        with torch.no_grad():
            if self.tokenizer is not None:
                encode = getattr(self.tokenizer, "encode", None) or getattr(self.tokenizer, "tokenize", None)
                tokens = encode(sequence)
                if isinstance(tokens, torch.Tensor):
                    input_ids = tokens.view(1, -1).long().to(device)
                else:
                    # list / numpy array / any iterable of token ids
                    input_ids = torch.tensor(list(tokens), dtype=torch.long, device=device).unsqueeze(0)
            else:
                input_ids = torch.tensor(
                    [[ord(c) for c in sequence]], dtype=torch.long, device=device
                )
            self.model(input_ids)

    def _tokenize(self, sequence: str):
        device = next(self.model.parameters()).device
        if self.tokenizer is not None:
            encode = (getattr(self.tokenizer, "encode", None)
                      or getattr(self.tokenizer, "tokenize", None))
            tokens = encode(sequence)
            if isinstance(tokens, torch.Tensor):
                return tokens.view(1, -1).long().to(device)
            return torch.tensor(list(tokens), dtype=torch.long, device=device).unsqueeze(0)
        return torch.tensor([[ord(c) for c in sequence]], dtype=torch.long, device=device)

    def _logits_from_forward(self, input_ids) -> torch.Tensor:
        """Single forward; return [T, vocab] logits. No hooks."""
        out = self.model(input_ids)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        if logits.dim() == 3:
            logits = logits[0]
        return logits

    def compute_perplexity(self, sequence: str) -> float:
        """Baseline causal perplexity (no ablation)."""
        import torch.nn.functional as F
        input_ids = self._tokenize(sequence)
        with torch.no_grad():
            logits = self._logits_from_forward(input_ids)
        shift_logits = logits[:-1].float()
        shift_labels = input_ids[0, 1:]
        return torch.exp(F.cross_entropy(shift_logits, shift_labels)).item()

    def compute_perplexity_ablated(self, sequence: str, sw_list: list,
                                   mode: str = "superrow") -> float:
        """
        Perplexity with output-hook ablation.

        Transformer-engine layers cache quantized weights and ignore in-place
        writes to .weight.data, so zeroing the weight tensor has no effect on
        the forward pass.  This method intercepts the *output* of each target
        layer using a forward hook and zeros the affected channel(s) directly.

        superrow mode:    output[..., row] = 0
        superweight mode: output[..., row] -= W_orig[row, col] * input[..., col]
        """
        import torch.nn.functional as F
        input_ids = self._tokenize(sequence)

        def _unwrap(out):
            return (out[0], out[1:]) if isinstance(out, tuple) else (out, None)

        def _rewrap(o, rest):
            return (o,) + rest if rest is not None else o

        handles = []
        for entry in sw_list:
            layer  = entry["layer"]
            row    = entry["row"]
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
                # Read the weight value that the *actual* forward will use.
                # TE might use a quantized copy but the sign/scale should still
                # match for the order-of-magnitude test we care about.
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
                logits = self._logits_from_forward(input_ids)
        finally:
            for h in handles:
                h.remove()

        shift_logits = logits[:-1].float()
        shift_labels = input_ids[0, 1:]
        return torch.exp(F.cross_entropy(shift_logits, shift_labels)).item()

    def get_target_module(self, layer_idx: int):
        return self._resolve_module(self.config["down_proj_pattern"], layer_idx)

    @property
    def num_layers(self) -> int:
        return self.config["num_layers"]
