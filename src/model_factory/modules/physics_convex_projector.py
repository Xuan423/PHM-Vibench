"""Convex-combination projector that mixes physical operators into interpretable axes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SimplexStats:
    """Diagnostic values for simplex-constrained weights."""

    max_deviation: torch.Tensor
    min_value: torch.Tensor


class PhysicsConvexProjector(nn.Module):
    """Project backbone features via column-wise convex combinations.

    Given backbone output ``S ∈ ℝ^{B×d}``, the projector learns a matrix ``A`` with
    non-negative columns that each sum to 1, producing ``H = S A``.
    """

    def __init__(
        self,
        input_dim: int,
        axis_dim: int,
        *,
        simplex_eps: float = 1e-6,
        prior_mask: Optional[torch.Tensor] = None,
        init: str = "uniform",
    ) -> None:
        super().__init__()
        if axis_dim <= 0:
            raise ValueError("PhysicsConvexProjector requires axis_dim > 0")
        if input_dim <= 0:
            raise ValueError("PhysicsConvexProjector requires input_dim > 0")

        self.input_dim = input_dim
        self.axis_dim = axis_dim
        self.simplex_eps = simplex_eps
        self.register_buffer("prior_mask", None if prior_mask is None else prior_mask.float())

        weight = torch.empty(input_dim, axis_dim)
        if init == "kaiming":
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        elif init == "uniform":
            nn.init.uniform_(weight, -0.02, 0.02)
        else:
            nn.init.xavier_uniform_(weight)
        self.logits = nn.Parameter(weight)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        axes = self._current_axes()
        h = features @ axes
        stats = self._compute_stats(axes)
        return h, {
            "simplex_max_deviation": stats.max_deviation,
            "simplex_min_value": stats.min_value,
        }

    # ------------------------------------------------------------------
    # Diagnostics / helpers
    # ------------------------------------------------------------------
    def _current_axes(self) -> torch.Tensor:
        logits = self.logits
        if self.prior_mask is not None:
            large_neg = torch.finfo(logits.dtype).min
            logits = torch.where(self.prior_mask > 0.0, logits, torch.full_like(logits, large_neg))
        axes = torch.softmax(logits, dim=0)
        return axes

    def _compute_stats(self, axes: torch.Tensor) -> SimplexStats:
        col_sums = axes.sum(dim=0)
        deviation = torch.max(torch.abs(col_sums - 1.0))
        min_value = torch.min(axes)
        return SimplexStats(max_deviation=deviation, min_value=min_value)

    def explain_axes(self) -> torch.Tensor:
        return self._current_axes().detach().cpu()
