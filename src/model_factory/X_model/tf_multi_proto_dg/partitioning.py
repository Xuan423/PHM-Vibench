from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch

from .tensor_ops import compute_partition_width, maybe_zero_pad_last_dim


@dataclass(frozen=True)
class RegionMeta:
    domain: str
    operator_name: str
    channel_index: int
    region_index: int
    region_start: int
    region_end: int


def make_time_patches(x_bcol: torch.Tensor, patch_count: int, padding_mode: str) -> torch.Tensor:
    if x_bcol.ndim != 4:
        raise ValueError(f"Expected [B, C, O, L], got {tuple(x_bcol.shape)}")
    length = int(x_bcol.shape[-1])
    patch_width = compute_partition_width(length, patch_count, padding_mode)
    padded_length = patch_width * patch_count
    x_bcol = maybe_zero_pad_last_dim(x_bcol, padded_length)
    return x_bcol.view(*x_bcol.shape[:-1], patch_count, patch_width)


def make_freq_bands(x_bcof: torch.Tensor, band_count: int, padding_mode: str) -> torch.Tensor:
    if x_bcof.ndim != 4:
        raise ValueError(f"Expected [B, C, O, F], got {tuple(x_bcof.shape)}")
    freq_bins = int(x_bcof.shape[-1])
    band_width = compute_partition_width(freq_bins, band_count, padding_mode)
    padded_bins = band_width * band_count
    x_bcof = maybe_zero_pad_last_dim(x_bcof, padded_bins)
    return x_bcof.view(*x_bcof.shape[:-1], band_count, band_width)


def build_region_index(
    domain: str,
    operator_names: Iterable[str],
    num_channels: int,
    region_count: int,
    region_width: int,
) -> List[RegionMeta]:
    regions: List[RegionMeta] = []
    for channel_index in range(num_channels):
        for operator_name in operator_names:
            for region_index in range(region_count):
                start = region_index * region_width
                regions.append(
                    RegionMeta(
                        domain=domain,
                        operator_name=str(operator_name),
                        channel_index=channel_index,
                        region_index=region_index,
                        region_start=start,
                        region_end=start + region_width,
                    )
                )
    return regions
