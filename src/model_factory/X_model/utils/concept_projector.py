"""Concept projector utilities for TSPN-CL.

Maps selected physical features h_sel -> concept space c with a row-stochastic
weight matrix W obtained via softmax over learnable parameters A.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptProjector(nn.Module):
    """Project physical features into a low-dimensional concept space."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        init_W: Optional[torch.Tensor] = None,
        trainable: bool = True,
        mode: str = "softmax",
        linear_bias: bool = False,
    ) -> None:
        """
        mode:
            - "softmax": original row-stochastic projector
            - "linear": learnable linear map without softmax (metric-head friendly)
            - "identity": bypass (requires in_dim == out_dim)
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.trainable = trainable
        self.mode = mode

        if mode == "softmax":
            self.A = nn.Parameter(torch.randn(out_dim, in_dim) * 0.01, requires_grad=trainable)
            if init_W is not None:
                with torch.no_grad():
                    if init_W.shape != (out_dim, in_dim):
                        raise ValueError(f"init_W shape {init_W.shape} != ({out_dim}, {in_dim})")
                    self.A.copy_(init_W)
        elif mode == "linear":
            self.linear = nn.Linear(in_dim, out_dim, bias=linear_bias)
            if init_W is not None:
                with torch.no_grad():
                    if init_W.shape != (out_dim, in_dim):
                        raise ValueError(f"init_W shape {init_W.shape} != ({out_dim}, {in_dim})")
                    self.linear.weight.copy_(init_W)
        elif mode == "identity":
            if in_dim != out_dim:
                raise ValueError(f"Identity projector requires in_dim == out_dim, got {in_dim} vs {out_dim}")
        else:
            raise ValueError(f"Unsupported projector mode: {mode}")

    def forward(self, h_sel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute concept vectors and their weights.

        Args:
            h_sel: Tensor of shape [B, K] containing selected physical features.

        Returns:
            c: Tensor of shape [B, d] concept representations.
            W: Tensor of shape [d, K] row-stochastic weights (non-negative, row-sum=1).
        """
        if self.mode == "softmax":
            # Row-wise softmax to ensure non-negativity and row sums to 1
            W = F.softmax(self.A, dim=-1)
            c = torch.matmul(h_sel, W.t())
            return c, W

        if self.mode == "linear":
            c = self.linear(h_sel)
            return c, self.linear.weight

        # identity mode
        identity_w = torch.eye(self.in_dim, device=h_sel.device, dtype=h_sel.dtype)
        return h_sel, identity_w

    def sparsity_loss(self, mode: str = "l1", strength: float = 1.0) -> torch.Tensor:
        """Optional sparsity/entropy regularizer on W.

        Args:
            mode: 'l1' or 'entropy'
            strength: scaling factor

        Returns:
            Scalar tensor loss (may be zero if strength==0).
        """
        if strength <= 0 or self.mode == "identity":
            return torch.tensor(0.0, device=self._device(), dtype=self._dtype())

        if self.mode == "softmax":
            W = F.softmax(self.A, dim=-1)
        elif self.mode == "linear":
            W = self.linear.weight
        else:
            raise ValueError(f"Unsupported projector mode for sparsity: {self.mode}")

        if mode == "l1":
            return strength * W.abs().mean()
        if mode == "entropy":
            entropy = -(W * (W + 1e-8).log()).sum(dim=-1).mean()
            return strength * entropy
        raise ValueError(f"Unsupported sparsity mode: {mode}")

    def _device(self) -> torch.device:
        if self.mode == "softmax":
            return self.A.device
        if self.mode == "linear":
            return self.linear.weight.device
        return torch.device("cpu")

    def _dtype(self) -> torch.dtype:
        if self.mode == "softmax":
            return self.A.dtype
        if self.mode == "linear":
            return self.linear.weight.dtype
        return torch.float32
