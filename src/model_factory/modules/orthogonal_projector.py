"""Orthogonal projector that splits physics-stat features into subspaces."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OrthogonalProjectorConfig:
    in_dim: int
    proj_dim: int
    reorth_every: int = 1


class OrthogonalProjector(nn.Module):
    """Learn a Stiefel matrix ``W`` whose columns span the physics subspace."""

    def __init__(self, config: OrthogonalProjectorConfig) -> None:
        super().__init__()
        if config.proj_dim <= 0 or config.proj_dim > config.in_dim:
            raise ValueError("proj_dim must be in (0, in_dim]")
        self.config = config
        self.W = nn.Parameter(torch.empty(config.in_dim, config.proj_dim))
        nn.init.orthogonal_(self.W)
        self.reorth_every = max(1, int(getattr(config, "reorth_every", 1)))

    @property
    def proj(self) -> torch.Tensor:
        return self.W @ self.W.transpose(0, 1)

    def forward(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if v.dim() != 2 or v.size(1) != self.config.in_dim:
            raise ValueError(f"Expected tensor (B, {self.config.in_dim}); got {v.shape}")
        P = self.proj
        z_phys = v @ P
        z_spur = v - z_phys
        return z_phys, z_spur

    def ortho_loss(self) -> torch.Tensor:
        wtw = self.W.transpose(0, 1) @ self.W
        identity = torch.eye(
            self.config.proj_dim,
            device=self.W.device,
            dtype=self.W.dtype,
        )
        return F.mse_loss(wtw, identity)

    @torch.no_grad()
    def reorthogonalize(self) -> None:
        """Project ``W`` back to the Stiefel manifold via QR."""
        q, _ = torch.linalg.qr(self.W)
        self.W.copy_(q)

    def maybe_reorthogonalize(self, step: int) -> None:
        if self.reorth_every <= 0:
            return
        if (step + 1) % self.reorth_every == 0:
            self.reorthogonalize()
