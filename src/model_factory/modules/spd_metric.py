"""Low-rank SPD coupling metric for physics-aware embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SPDCouplingState:
    """Snapshot of SPD factors for explainability/logging."""

    diag: torch.Tensor
    low_rank: torch.Tensor
    temperature: torch.Tensor


class SPDCouplingMetric(nn.Module):
    """Applies ``M = D + L Lᵀ`` with per-dimension temperature scaling."""

    def __init__(
        self,
        axis_dim: int,
        *,
        rank: int = 4,
        tau_init: float = 1.0,
        tau_min: float = 1e-4,
    ) -> None:
        super().__init__()
        if axis_dim <= 0:
            raise ValueError("SPDCouplingMetric requires axis_dim > 0")
        if rank <= 0:
            raise ValueError("SPDCouplingMetric rank must be positive")

        self.axis_dim = axis_dim
        self.rank = rank
        self.tau_min = tau_min

        self.diag_raw = nn.Parameter(torch.randn(axis_dim) * 0.02)
        self.low_rank = nn.Parameter(torch.randn(axis_dim, rank) * 0.02)
        self.tau_raw = nn.Parameter(torch.full((axis_dim,), float(tau_init)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self._apply_metric(h)
        z = self._apply_temperature(z)
        return F.normalize(z, p=2, dim=1)

    def _apply_metric(self, h: torch.Tensor) -> torch.Tensor:
        diag = self.diag()
        scaled = h * diag
        coupled = (h @ self.low_rank) @ self.low_rank.transpose(0, 1)
        return scaled + coupled

    def _apply_temperature(self, z: torch.Tensor) -> torch.Tensor:
        tau = self.temperature()
        return z * tau

    def diag(self) -> torch.Tensor:
        return F.softplus(self.diag_raw) + 1e-4

    def temperature(self) -> torch.Tensor:
        return F.softplus(self.tau_raw) + self.tau_min

    def regularization(self) -> torch.Tensor:
        return torch.sum(self.low_rank.pow(2))

    def explain(self) -> SPDCouplingState:
        return SPDCouplingState(
            diag=self.diag().detach().cpu(),
            low_rank=self.low_rank.detach().cpu(),
            temperature=self.temperature().detach().cpu(),
        )

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute ``s_M(a_i, b_j) = a_iᵀ M b_j`` for two sets of embeddings."""
        diag = self.diag()
        low_rank = self.low_rank
        term_diag = (a * diag) @ b.transpose(0, 1)
        a_lr = a @ low_rank
        b_lr = b @ low_rank
        term_lr = a_lr @ b_lr.transpose(0, 1)
        return term_diag + term_lr

    def metric_matrix(self) -> torch.Tensor:
        diag = torch.diag(self.diag())
        coupling = self.low_rank @ self.low_rank.transpose(0, 1)
        return diag + coupling
