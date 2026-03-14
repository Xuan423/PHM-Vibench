from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def to_bcl(x: torch.Tensor) -> torch.Tensor:
    """Convert inputs to [B, C, L] with a conservative time-major heuristic."""
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    if x.ndim == 2:
        if x.shape[0] > x.shape[1]:
            x = x.transpose(0, 1)
        x = x.unsqueeze(0)
    elif x.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D tensor, got shape {tuple(x.shape)}.")

    if x.ndim == 3 and x.shape[1] > x.shape[2]:
        x = x.transpose(1, 2)
    return x.contiguous().float()


def maybe_zero_pad_last_dim(x: torch.Tensor, target_length: int) -> torch.Tensor:
    current = int(x.shape[-1])
    if current > target_length:
        raise ValueError(f"Cannot pad to a smaller target length: {current} > {target_length}.")
    if current == target_length:
        return x
    return F.pad(x, (0, target_length - current))


def compute_partition_width(length: int, part_count: int, padding_mode: str) -> int:
    if part_count <= 0:
        raise ValueError("part_count must be positive.")
    if padding_mode == "error":
        if length % part_count != 0:
            raise ValueError(f"Length {length} is not divisible by {part_count}.")
        return length // part_count
    if padding_mode == "zero_right":
        return int(math.ceil(length / part_count))
    raise ValueError(f"Unsupported padding_mode: {padding_mode}")


def safe_rfft_amplitude(x_bcl: torch.Tensor) -> torch.Tensor:
    return torch.fft.rfft(x_bcl, dim=-1).abs()


def safe_log1p(x: torch.Tensor, alpha: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    return torch.log1p(torch.clamp(alpha * x, min=eps))


def normalize_time(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, unbiased=False, keepdim=True)
    return (x - mean) / (std + eps)


def normalize_freq(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(x.square(), dim=-1, keepdim=True) + eps)
    return x / rms


def entropy_from_abs(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = x.abs()
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    return -(probs * probs.clamp_min(eps).log()).sum(dim=-1)
