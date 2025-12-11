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
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.trainable = trainable

        self.A = nn.Parameter(torch.randn(out_dim, in_dim) * 0.01, requires_grad=trainable)
        if init_W is not None:
            with torch.no_grad():
                if init_W.shape != (out_dim, in_dim):
                    raise ValueError(f"init_W shape {init_W.shape} != ({out_dim}, {in_dim})")
                self.A.copy_(init_W)

    def forward(self, h_sel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute concept vectors and their weights.

        Args:
            h_sel: Tensor of shape [B, K] containing selected physical features.

        Returns:
            c: Tensor of shape [B, d] concept representations.
            W: Tensor of shape [d, K] row-stochastic weights (non-negative, row-sum=1).
        """
        # Row-wise softmax to ensure non-negativity and row sums to 1
        W = F.softmax(self.A, dim=-1)
        c = torch.matmul(h_sel, W.t())
        return c, W

    def sparsity_loss(self, mode: str = "l1", strength: float = 1.0) -> torch.Tensor:
        """Optional sparsity/entropy regularizer on W.

        Args:
            mode: 'l1' or 'entropy'
            strength: scaling factor

        Returns:
            Scalar tensor loss (may be zero if strength==0).
        """
        if strength <= 0:
            return torch.tensor(0.0, device=self.A.device, dtype=self.A.dtype)
        W = F.softmax(self.A, dim=-1)
        if mode == "l1":
            return strength * W.abs().mean()
        if mode == "entropy":
            entropy = -(W * (W + 1e-8).log()).sum(dim=-1).mean()
            return strength * entropy
        raise ValueError(f"Unsupported sparsity mode: {mode}")
