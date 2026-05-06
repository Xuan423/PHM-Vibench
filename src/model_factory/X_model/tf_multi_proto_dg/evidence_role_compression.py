from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceRoleCompression(nn.Module):
    def __init__(
        self,
        in_features: int,
        role_dim: int,
        nonneg: str = "none",
        input_norm: str = "layernorm",
        output_norm: str = "none",
        init_mode: str = "random_normal",
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.role_dim = int(role_dim)
        self.nonneg = str(nonneg)
        self.input_norm = str(input_norm)
        self.output_norm = str(output_norm)
        self.init_mode = str(init_mode)
        if self.init_mode == "random_normal":
            basis = torch.randn(self.in_features, self.role_dim) * 0.02
        elif self.init_mode == "dct":
            basis = self._build_dct_basis(self.in_features, self.role_dim) * 0.02
        else:
            raise ValueError(
                f"Unsupported role_init_mode={init_mode!r}. "
                "Expected one of {'random_normal', 'dct'}."
            )
        self.raw_basis = nn.Parameter(basis)
        # Compatibility-only parameter: kept to preserve historical initialization
        # order and checkpoint key layout (previously part of the removed residual path).
        # It is intentionally not used in forward computation.
        self.residual_proj = nn.Linear(self.in_features, self.role_dim, bias=False)

    @staticmethod
    def _build_dct_basis(in_features: int, role_dim: int) -> torch.Tensor:
        """Seed-independent smooth basis over fixed operator-indicator coordinates."""
        positions = torch.arange(int(in_features), dtype=torch.float32).add(0.5)
        frequencies = torch.arange(int(role_dim), dtype=torch.float32)
        basis = torch.cos(torch.pi * positions[:, None] * frequencies[None, :] / float(max(in_features, 1)))
        if role_dim > 0:
            basis[:, 0] = 1.0
        basis = basis - basis.mean(dim=0, keepdim=True)
        basis[:, 0] = 1.0
        basis = basis / basis.norm(dim=0, keepdim=True).clamp_min(1e-6)
        return basis

    def _build_basis(self) -> torch.Tensor:
        if self.nonneg == "softplus":
            basis = F.softplus(self.raw_basis)
            return basis / basis.sum(dim=0, keepdim=True).clamp_min(1e-6)
        elif self.nonneg == "relu":
            basis = torch.relu(self.raw_basis)
            return basis / basis.sum(dim=0, keepdim=True).clamp_min(1e-6)
        elif self.nonneg == "abs":
            basis = self.raw_basis.abs()
            return basis / basis.sum(dim=0, keepdim=True).clamp_min(1e-6)
        elif self.nonneg == "none":
            basis = self.raw_basis
            return basis / basis.norm(dim=0, keepdim=True).clamp_min(1e-6)
        else:
            raise ValueError(f"Unsupported role_nonneg mode: {self.nonneg}")

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_norm == "layernorm":
            return F.layer_norm(x, (int(x.shape[-1]),))
        if self.input_norm == "l2":
            return F.normalize(x, dim=-1)
        if self.input_norm == "none":
            return x
        raise ValueError(f"Unsupported role_input_norm mode: {self.input_norm}")

    def _normalize_output(self, z: torch.Tensor) -> torch.Tensor:
        if self.output_norm == "layernorm":
            return F.layer_norm(z, (int(z.shape[-1]),))
        if self.output_norm == "l2":
            return F.normalize(z, dim=-1)
        if self.output_norm == "none":
            return z
        raise ValueError(f"Unsupported role_output_norm mode: {self.output_norm}")

    def forward(self, e_bar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if e_bar.ndim != 3:
            raise ValueError(f"Expected [B, R, M], got {tuple(e_bar.shape)}")
        if e_bar.shape[-1] != self.in_features:
            raise ValueError(
                f"EvidenceRoleCompression expected in_features={self.in_features}, got {tuple(e_bar.shape)}."
            )

        basis = self._build_basis()
        e_norm = self._normalize_input(e_bar)
        z = torch.matmul(e_norm, basis)
        z = self._normalize_output(z)
        return z, basis
