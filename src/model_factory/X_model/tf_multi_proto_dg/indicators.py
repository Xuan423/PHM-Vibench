from __future__ import annotations

from typing import Any, Iterable, List

import torch

from .tensor_ops import entropy_from_abs


TIME_INDICATORS = ("Mean", "Std", "RMS", "Kurtosis", "CrestFactor", "Entropy")
FREQ_INDICATORS = ("BandEnergy", "BandRMS", "SpectralEntropy", "PeakRatio")


def _compute_time_indicator(name: str, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if name == "Mean":
        return x.mean(dim=-1)
    if name == "Std":
        return x.std(dim=-1, unbiased=False)
    if name == "RMS":
        return torch.sqrt(x.square().mean(dim=-1) + eps)
    if name == "Kurtosis":
        centered = x - x.mean(dim=-1, keepdim=True)
        std = centered.std(dim=-1, unbiased=False, keepdim=True).clamp_min(eps)
        return (centered / std).pow(4).mean(dim=-1)
    if name == "CrestFactor":
        peak = x.abs().amax(dim=-1)
        rms = torch.sqrt(x.square().mean(dim=-1) + eps)
        return peak / rms.clamp_min(eps)
    if name == "Entropy":
        return entropy_from_abs(x, eps=eps)
    raise ValueError(f"Unknown time indicator: {name}")


def _compute_freq_indicator(name: str, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if name == "BandEnergy":
        return x.square().sum(dim=-1)
    if name == "BandRMS":
        return torch.sqrt(x.square().mean(dim=-1) + eps)
    if name == "SpectralEntropy":
        return entropy_from_abs(x, eps=eps)
    if name == "PeakRatio":
        return x.amax(dim=-1) / x.mean(dim=-1).clamp_min(eps)
    raise ValueError(f"Unknown frequency indicator: {name}")


def compute_time_indicators(time_patches: torch.Tensor, indicator_names: Iterable[str]) -> torch.Tensor:
    outputs = [_compute_time_indicator(name, time_patches) for name in indicator_names]
    return torch.stack(outputs, dim=-1)


def compute_freq_indicators(freq_bands: torch.Tensor, indicator_names: Iterable[str]) -> torch.Tensor:
    outputs = [_compute_freq_indicator(name, freq_bands) for name in indicator_names]
    return torch.stack(outputs, dim=-1)


def list_indicator_names(config: Any) -> dict[str, List[str]]:
    return {
        "time": list(config.time_indicators),
        "freq": list(config.freq_indicators),
    }
