from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tensor_ops import normalize_time


class TimeOperatorBase(nn.Module):
    name = "TimeOperator"

    def forward(self, x_bcl: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Identity(TimeOperatorBase):
    name = "Identity"

    def forward(self, x_bcl: torch.Tensor) -> torch.Tensor:
        return x_bcl


class FirstDifference(TimeOperatorBase):
    name = "FirstDifference"

    def forward(self, x_bcl: torch.Tensor) -> torch.Tensor:
        diff = x_bcl[..., 1:] - x_bcl[..., :-1]
        return F.pad(diff, (1, 0))


class DetrendMovingAverage(TimeOperatorBase):
    name = "DetrendMovingAverage"

    def __init__(self, kernel_size: int = 9) -> None:
        super().__init__()
        self.kernel_size = max(3, int(kernel_size) | 1)

    def forward(self, x_bcl: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        pooled = F.avg_pool1d(
            F.pad(x_bcl, (pad, pad), mode="replicate"),
            kernel_size=self.kernel_size,
            stride=1,
        )
        return x_bcl - pooled


class HilbertEnvelope(TimeOperatorBase):
    name = "HilbertEnvelope"

    def forward(self, x_bcl: torch.Tensor) -> torch.Tensor:
        length = x_bcl.shape[-1]
        spectrum = torch.fft.fft(x_bcl, dim=-1)
        h = torch.zeros(length, device=x_bcl.device, dtype=spectrum.dtype)
        if length % 2 == 0:
            h[0] = 1
            h[length // 2] = 1
            h[1:length // 2] = 2
        else:
            h[0] = 1
            h[1:(length + 1) // 2] = 2
        analytic = torch.fft.ifft(spectrum * h, dim=-1)
        return analytic.abs().real


class TKEO(TimeOperatorBase):
    name = "TKEO"

    def forward(self, x_bcl: torch.Tensor) -> torch.Tensor:
        center = x_bcl[..., 1:-1]
        left = x_bcl[..., :-2]
        right = x_bcl[..., 2:]
        energy = center.square() - left * right
        return F.pad(energy, (1, 1))


TIME_OPERATOR_REGISTRY = {
    "Identity": Identity,
    "FirstDifference": FirstDifference,
    "DetrendMovingAverage": DetrendMovingAverage,
    "HilbertEnvelope": HilbertEnvelope,
    "TKEO": TKEO,
}


class TimeOperatorBank(nn.Module):
    def __init__(self, operators: Iterable[TimeOperatorBase]) -> None:
        super().__init__()
        self.operators = nn.ModuleList(list(operators))
        self.operator_names = [op.name for op in self.operators]

    def forward(self, x_bcl: torch.Tensor) -> torch.Tensor:
        outputs = [normalize_time(operator(x_bcl)) for operator in self.operators]
        return torch.stack(outputs, dim=2)


def build_time_operator_bank(config: Any) -> TimeOperatorBank:
    operators: List[TimeOperatorBase] = []
    for name in config.time_operators:
        operator_cls = TIME_OPERATOR_REGISTRY.get(name)
        if operator_cls is None:
            raise ValueError(f"Unknown time operator: {name}")
        operators.append(operator_cls())
    return TimeOperatorBank(operators)
