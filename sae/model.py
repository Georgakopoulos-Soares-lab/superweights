# sae/model.py
"""
Batch-TopK Sparse Autoencoder (SAE).

Architecture (following Anthropic / Gao et al. 2024 "Scaling and evaluating
sparse autoencoders"):

  pre_bias  : learnable centre subtracted from input before encoding
  encoder   : Linear(d_in → n_features) + ReLU + TopK sparsification
  decoder   : Linear(n_features → d_in) with unit-norm columns + b_dec offset

Loss
----
  recon_loss = MSE(x, x_hat)
  aux_loss   = MSE(x − x_hat, x_hat_aux)   where x_hat_aux uses only dead
               features (AuxK method) — prevents dead-feature collapse
  total_loss = recon_loss + lambda_aux * aux_loss

Batch-TopK
----------
Per token in the batch, exactly k pre-activations are kept; the rest are
zeroed after ReLU.  This hard constraint on L0 = k makes sparsity stable and
removes the need to tune an L1 coefficient.

Dead-feature tracking
---------------------
steps_since_active counts gradient steps since each feature last fired.
Features inactive for ≥ dead_threshold steps are considered dead and receive
the auxiliary AuxK penalty to resurrect them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchTopKSAE(nn.Module):
    """
    Batch-TopK Sparse Autoencoder.

    Parameters
    ----------
    d_in      : model hidden size (input / output dimension)
    n_features: dictionary size (number of SAE latent features)
    k         : TopK budget per token (== target L0)
    k_aux     : AuxK budget for dead feature penalty (Gao et al. 2024)
    """

    def __init__(
        self,
        d_in: int,
        n_features: int,
        k: int,
        k_aux: int = 512,
    ):
        super().__init__()
        self.d_in       = d_in
        self.n_features = n_features
        self.k          = k
        self.k_aux      = min(k_aux, n_features)

        # Pre-encoder bias: centred on data mean (updated via EMA outside model)
        self.pre_bias = nn.Parameter(torch.zeros(d_in))

        # Encoder: maps centred input to feature pre-activations
        self.W_enc  = nn.Linear(d_in, n_features, bias=True)

        # Decoder: columns must stay at unit norm (call normalise_decoder() each step)
        # Stored as [d_in, n_features] so decode = acts @ W_dec.T + b_dec
        self.W_dec  = nn.Parameter(F.normalize(torch.randn(d_in, n_features), dim=0))
        self.b_dec  = nn.Parameter(torch.zeros(d_in))

        # Dead-feature counter — not a gradient-bearing parameter
        self.register_buffer(
            "steps_since_active", torch.zeros(n_features, dtype=torch.long)
        )

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def dead_threshold(self) -> int:
        """Steps before a feature is considered dead (10× n_features, min 1000)."""
        return max(10 * self.n_features, 1_000)

    @property
    def dead_mask(self) -> torch.Tensor:
        """Boolean [n_features] — True for features inactive ≥ dead_threshold steps."""
        return self.steps_since_active >= self.dead_threshold

    # ── Internal helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _topk_sparse(acts: torch.Tensor, k: int) -> torch.Tensor:
        """
        Zero all but the top-k values (per token) in `acts`.
        acts : [B, n_features]  (already ReLU'd, non-negative)
        """
        if k <= 0 or acts.shape[-1] == 0:
            return torch.zeros_like(acts)
        k = min(k, acts.shape[-1])
        topk_vals, _ = torch.topk(acts, k, dim=-1)          # [B, k]
        threshold    = topk_vals[..., -1:]                   # [B, 1]
        return acts * (acts >= threshold).float()

    # ── Public API ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def normalise_decoder(self):
        """Renormalise all decoder columns to unit L2 norm.  Call after each step."""
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    @torch.no_grad()
    def update_dead_mask(self, acts: torch.Tensor):
        """
        Update per-feature inactivity counters from a batch of activations.
        acts : [B, n_features]
        """
        fired = (acts > 0).any(dim=0)           # [n_features]
        self.steps_since_active[fired]  = 0
        self.steps_since_active[~fired] += 1

    def encode(self, x: torch.Tensor) -> dict:
        """
        Encode input to sparse feature activations (no loss, no decode).
        Returns {'pre_acts': ..., 'acts': ...}
        """
        x_c      = x - self.pre_bias
        pre_acts = self.W_enc(x_c)
        acts     = self._topk_sparse(F.relu(pre_acts), self.k)
        return {"pre_acts": pre_acts, "acts": acts}

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """Decode sparse feature activations back to input space."""
        return acts @ self.W_dec.T + self.b_dec

    def forward(
        self,
        x: torch.Tensor,
        use_aux_loss: bool = True,
        lambda_aux: float = 1.0 / 32,
    ) -> dict:
        """
        Full SAE forward pass with loss computation.

        Parameters
        ----------
        x            : [B, d_in]  residual-stream activations
        use_aux_loss : whether to compute the AuxK dead-feature penalty
        lambda_aux   : weight of the auxiliary loss (default: 1/32)

        Returns
        -------
        dict with keys:
          x_hat      – [B, d_in]  reconstruction
          acts       – [B, n_features]  sparse feature activations
          pre_acts   – [B, n_features]  pre-TopK activations
          loss       – scalar total loss
          recon_loss – scalar MSE reconstruction loss
          aux_loss   – scalar auxiliary loss (0 when not used)
          l0         – mean number of active features per token
          frac_dead  – fraction of features currently dead
        """
        # ── Encode ────────────────────────────────────────────────────────────
        x_c      = x - self.pre_bias                         # [B, d_in]
        pre_acts = self.W_enc(x_c)                           # [B, n_features]
        acts     = self._topk_sparse(F.relu(pre_acts), self.k)

        # ── Decode ────────────────────────────────────────────────────────────
        x_hat = acts @ self.W_dec.T + self.b_dec             # [B, d_in]

        # ── Reconstruction loss ───────────────────────────────────────────────
        recon_loss = F.mse_loss(x_hat, x.detach())

        # ── Auxiliary dead-feature loss (AuxK) ───────────────────────────────
        aux_loss = x.new_tensor(0.0)
        if use_aux_loss:
            dead    = self.dead_mask                          # [n_features]
            n_dead  = int(dead.sum().item())
            if n_dead > 0:
                k_aux     = min(self.k_aux, n_dead)
                dead_pre  = pre_acts.clone()
                dead_pre[:, ~dead] = float("-inf")           # mask live features
                dead_acts = self._topk_sparse(F.relu(dead_pre), k_aux)
                x_hat_aux = dead_acts @ self.W_dec.T + self.b_dec
                residual  = (x - x_hat).detach()
                aux_loss  = F.mse_loss(x_hat_aux, residual)

        # ── Total loss ────────────────────────────────────────────────────────
        loss = recon_loss + lambda_aux * aux_loss

        l0       = (acts > 0).float().sum(dim=-1).mean()
        frac_dead = self.dead_mask.float().mean()

        return {
            "x_hat":      x_hat,
            "acts":       acts,
            "pre_acts":   pre_acts,
            "loss":       loss,
            "recon_loss": recon_loss,
            "aux_loss":   aux_loss,
            "l0":         l0,
            "frac_dead":  frac_dead,
        }

    # ── Save / load ─────────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save model state + hyperparameters."""
        torch.save(
            {
                "state_dict":  self.state_dict(),
                "d_in":        self.d_in,
                "n_features":  self.n_features,
                "k":           self.k,
                "k_aux":       self.k_aux,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "BatchTopKSAE":
        """Load a saved SAE."""
        ckpt = torch.load(path, map_location=device)
        sae  = cls(
            d_in       = ckpt["d_in"],
            n_features = ckpt["n_features"],
            k          = ckpt["k"],
            k_aux      = ckpt["k_aux"],
        )
        sae.load_state_dict(ckpt["state_dict"])
        return sae.to(device)
