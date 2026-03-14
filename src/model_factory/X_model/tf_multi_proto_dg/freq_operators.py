from __future__ import annotations

from typing import Any, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tensor_ops import normalize_freq, safe_log1p


class FreqOperatorBase(nn.Module):
    name = "FreqOperator"

    def forward(self, x_bcf: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class IdentitySpectrum(FreqOperatorBase):
    name = "IdentitySpectrum"

    def forward(self, x_bcf: torch.Tensor) -> torch.Tensor:
        return x_bcf


class LogSpectrum(FreqOperatorBase):
    name = "LogSpectrum"

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x_bcf: torch.Tensor) -> torch.Tensor:
        return safe_log1p(x_bcf, alpha=self.alpha)


class SpectralWhitening(FreqOperatorBase):
    name = "SpectralWhitening"

    def __init__(self, kernel_size: int = 5, eps: float = 1e-6) -> None:
        super().__init__()
        self.kernel_size = max(3, int(kernel_size) | 1)
        self.eps = eps

    def forward(self, x_bcf: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        smooth = F.avg_pool1d(
            F.pad(x_bcf, (pad, pad), mode="replicate"),
            kernel_size=self.kernel_size,
            stride=1,
        )
        return x_bcf / smooth.clamp_min(self.eps)


class GaussianBandMask(FreqOperatorBase):
    name = "GaussianBandMask"

    def __init__(self, center_ratio: float = 0.5, sigma_ratio: float = 0.15) -> None:
        super().__init__()
        self.center_ratio = float(center_ratio)
        self.sigma_ratio = float(sigma_ratio)

    def forward(self, x_bcf: torch.Tensor) -> torch.Tensor:
        freq_bins = x_bcf.shape[-1]
        grid = torch.linspace(0.0, 1.0, freq_bins, device=x_bcf.device, dtype=x_bcf.dtype)
        sigma = max(self.sigma_ratio, 1e-4)
        mask = torch.exp(-0.5 * ((grid - self.center_ratio) / sigma) ** 2)
        mask = mask / mask.norm(p=2).clamp_min(1e-6)
        return x_bcf * mask.view(1, 1, -1)


FREQ_OPERATOR_REGISTRY = {
    "IdentitySpectrum": IdentitySpectrum,
    "LogSpectrum": LogSpectrum,
    "SpectralWhitening": SpectralWhitening,
    "GaussianBandMask": GaussianBandMask,
}


class FreqOperatorBank(nn.Module):
    def __init__(self, operators: Iterable[FreqOperatorBase]) -> None:
        super().__init__()
        self.operators = nn.ModuleList(list(operators))
        self.operator_names = [op.name for op in self.operators]

    def forward(self, x_bcf: torch.Tensor) -> torch.Tensor:
        outputs = [normalize_freq(operator(x_bcf)) for operator in self.operators]
        return torch.stack(outputs, dim=2)


def build_freq_operator_bank(config: Any) -> FreqOperatorBank:
    operators: List[FreqOperatorBase] = []
    for name in config.freq_operators:
        operator_cls = FREQ_OPERATOR_REGISTRY.get(name)
        if operator_cls is None:
            raise ValueError(f"Unknown frequency operator: {name}")
        operators.append(operator_cls())
    return FreqOperatorBank(operators)
