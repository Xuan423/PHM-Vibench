"""Physics-statistics SPD layer that augments transparent features with log-Euclidean covariance vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class PhysicsStatSPDConfig:
    feat_dim: int
    eps: float = 1e-3
    momentum: float = 0.1
    block_size: int = 0


class PhysicsStatSPDLayer(nn.Module):
    """Build SPD covariance on physics statistics and map to tangent space."""

    def __init__(self, config: PhysicsStatSPDConfig) -> None:
        super().__init__()
        self.config = config
        feat_dim = config.feat_dim
        self.register_buffer("running_mean", torch.zeros(1, feat_dim))
        self.block_size = int(getattr(config, "block_size", 0))

    @property
    def feat_dim(self) -> int:
        return int(self.config.feat_dim)

    def _spd_block(self, block: torch.Tensor, eps: float) -> torch.Tensor:
        d = block.size(1)
        eye = torch.eye(d, device=block.device, dtype=block.dtype)
        cov = torch.einsum("bi,bj->bij", block, block)
        cov = cov + eps * eye.unsqueeze(0)
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        try:
            eigvals, eigvecs = torch.linalg.eigh(cov)
        except RuntimeError:
            jitter = eps
            mat = cov
            success = False
            for _ in range(4):
                try:
                    eigvals, eigvecs = torch.linalg.eigh(mat)
                    success = True
                    break
                except RuntimeError:
                    mat = mat + jitter * eye.unsqueeze(0)
                    jitter *= 10.0
            if not success:
                mat64 = mat.to(torch.double)
                u, s, _ = torch.linalg.svd(mat64)
                eigvals = s.to(block.dtype)
                eigvecs = u.to(block.dtype)
        eigvals = torch.clamp(eigvals, min=eps)
        log_cov = eigvecs @ torch.diag_embed(torch.log(eigvals)) @ eigvecs.transpose(-1, -2)
        log_cov = torch.nan_to_num(log_cov, nan=0.0, posinf=0.0, neginf=0.0)
        indices = torch.triu_indices(d, d, device=block.device)
        spd_vec = log_cov[..., indices[0], indices[1]]
        spd_vec = torch.nan_to_num(spd_vec, nan=0.0, posinf=0.0, neginf=0.0)
        return spd_vec

    @property
    def output_dim(self) -> int:
        spd_dim = 0
        block_size = self.block_size if self.block_size > 0 else self.feat_dim
        for block in range(0, self.feat_dim, block_size):
            d = min(block_size, self.feat_dim - block)
            spd_dim += d * (d + 1) // 2
        return self.feat_dim + spd_dim

    def reset_running_stats(self) -> None:
        self.running_mean.zero_()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() != 2:
            raise ValueError(f"Expected 2D tensor (B, D_feat); got {h.shape}")
        if h.size(1) != self.feat_dim:
            raise ValueError(f"Expected feat_dim={self.feat_dim}, got {h.size(1)}")

        if self.training:
            batch_mean = h.mean(dim=0, keepdim=True)
            momentum = self.config.momentum
            self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean.detach()
            mean = batch_mean
        else:
            mean = self.running_mean

        centred = h - mean
        eps = self.config.eps
        block_size = self.block_size if self.block_size > 0 else self.feat_dim
        spd_vecs = []
        for start in range(0, self.feat_dim, block_size):
            end = min(start + block_size, self.feat_dim)
            block = centred[:, start:end]
            spd_vecs.append(self._spd_block(block, eps))
        spd_vec = torch.cat(spd_vecs, dim=-1)
        spd_vec = torch.nan_to_num(spd_vec, nan=0.0, posinf=0.0, neginf=0.0)
        v = torch.cat([h, spd_vec], dim=-1)
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return v
