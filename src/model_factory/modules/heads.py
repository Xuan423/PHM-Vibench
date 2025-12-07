"""Lightweight heads operating on physics subspace features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveHead(nn.Module):
    """Linear → LayerNorm head followed by L2 normalisation."""

    def __init__(self, in_dim: int, proj_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, z_phys: torch.Tensor) -> torch.Tensor:
        proj = self.linear(z_phys)
        proj = self.norm(proj)
        proj = F.normalize(proj, p=2, dim=-1)
        return proj


class ClassifierHead(nn.Module):
    """Single linear layer mapping physics subspace features to logits."""

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, z_phys: torch.Tensor) -> torch.Tensor:
        return self.linear(z_phys)
