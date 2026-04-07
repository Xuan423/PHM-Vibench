from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalChannelFusion(nn.Module):
    def __init__(self, in_features: int, num_channels: int, norm: str = "layernorm") -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.num_channels = int(num_channels)
        self.norm = str(norm)
        self.score = nn.Linear(self.in_features, 1, bias=False)
        self.channel_bias = nn.Parameter(torch.zeros(self.num_channels))
        self.layer_norm = nn.LayerNorm(self.in_features)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm == "layernorm":
            return self.layer_norm(x)
        if self.norm == "l2":
            return F.normalize(x, dim=-1)
        if self.norm == "none":
            return x
        raise ValueError(f"Unsupported channel fusion norm: {self.norm}")

    def forward(self, e_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if e_flat.ndim != 4:
            raise ValueError(f"Expected [B, C, R, M], got {tuple(e_flat.shape)}")
        if e_flat.shape[1] != self.num_channels or e_flat.shape[-1] != self.in_features:
            raise ValueError(
                f"LocalChannelFusion shape mismatch: expected channels={self.num_channels}, "
                f"features={self.in_features}, got {tuple(e_flat.shape)}."
            )

        normalized = self._normalize(e_flat)
        scores = self.score(normalized).squeeze(-1)
        scores = scores + self.channel_bias.view(1, self.num_channels, 1)
        alpha_bc = torch.softmax(scores, dim=1)
        fused = torch.einsum("bcrm,bcr->brm", e_flat, alpha_bc)
        alpha = alpha_bc.permute(0, 2, 1).contiguous()
        return fused, alpha
