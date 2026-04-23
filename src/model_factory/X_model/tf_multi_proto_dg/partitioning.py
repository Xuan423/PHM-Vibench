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


def resolve_time_patch_width(
    length: int,
    patch_count: int,
    padding_mode: str,
    patch_width: int | None = None,
) -> int:
    if patch_width is not None:
        resolved = int(patch_width)
        if resolved <= 0:
            raise ValueError(f"patch_width must be positive, got {resolved}.")
        return resolved
    return compute_partition_width(length, patch_count, padding_mode)


def _sample_random_regions(
    x_bcod: torch.Tensor,
    region_count: int,
    region_width: int,
    sampling_mode: str = "random",
    sample_keys: torch.Tensor | None = None,
    sampling_epoch: int = 0,
    sampling_seed_offset: int = 0,
    return_start_indices: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_channels, num_operators, length = x_bcod.shape
    padded_length = max(length, region_width)
    x_bcod = maybe_zero_pad_last_dim(x_bcod, padded_length)

    max_start = padded_length - region_width

    deterministic_mode = sample_keys is not None
    hashed_uniform: torch.Tensor | None = None
    if deterministic_mode:
        sample_keys = sample_keys.to(device=x_bcod.device, dtype=torch.long).view(batch_size, 1)
        region_ids = torch.arange(region_count, device=x_bcod.device, dtype=torch.long).view(1, region_count)
        epoch_term = int(sampling_epoch) * 1000003 + int(sampling_seed_offset)
        h = sample_keys * 6364136223846793005
        h = h + region_ids * 2862933555777941757
        h = h + int(epoch_term) + 1442695040888963407
        h = h ^ (h >> 33)
        h = h * 3202034522624059733
        h = h ^ (h >> 29)
        h = h * 3935559000370003845
        h = h ^ (h >> 32)
        hashed_uniform = torch.abs(h)

    if max_start <= 0:
        start_indices = torch.zeros(
            batch_size,
            region_count,
            device=x_bcod.device,
            dtype=torch.long,
        )
    else:
        if str(sampling_mode) == "random_stratified":
            edges = torch.linspace(
                0,
                max_start + 1,
                steps=region_count + 1,
                device=x_bcod.device,
            )
            lower = torch.floor(edges[:-1]).long()
            upper_exclusive = torch.ceil(edges[1:]).long()
            widths = (upper_exclusive - lower).clamp_min(1)
            if deterministic_mode:
                random_offsets = torch.remainder(hashed_uniform, widths.unsqueeze(0))
            else:
                random_offsets = torch.floor(
                    torch.rand(batch_size, region_count, device=x_bcod.device) * widths.unsqueeze(0)
                ).long()
            start_indices = lower.unsqueeze(0) + random_offsets
            start_indices = start_indices.clamp_max(max_start)
        elif str(sampling_mode) == "random":
            if deterministic_mode:
                start_indices = torch.remainder(hashed_uniform, max_start + 1)
            else:
                start_indices = torch.randint(
                    0,
                    max_start + 1,
                    (batch_size, region_count),
                    device=x_bcod.device,
                )
        else:
            raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")

    offsets = torch.arange(region_width, device=x_bcod.device, dtype=torch.long)
    gather_index = start_indices[:, None, None, :, None] + offsets[None, None, None, None, :]
    gather_index = gather_index.expand(
        batch_size,
        num_channels,
        num_operators,
        region_count,
        region_width,
    )
    expanded = x_bcod.unsqueeze(-2).expand(
        batch_size,
        num_channels,
        num_operators,
        region_count,
        padded_length,
    )
    regions = expanded.gather(-1, gather_index)
    if return_start_indices:
        return regions, start_indices
    return regions


def make_time_patches(
    x_bcol: torch.Tensor,
    patch_count: int,
    padding_mode: str,
    patch_mode: str = "uniform",
    patch_width: int | None = None,
    sample_keys: torch.Tensor | None = None,
    sampling_epoch: int = 0,
    sampling_seed_offset: int = 0,
    return_start_indices: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if x_bcol.ndim != 4:
        raise ValueError(f"Expected [B, C, O, L], got {tuple(x_bcol.shape)}")
    length = int(x_bcol.shape[-1])
    patch_width = resolve_time_patch_width(
        length=length,
        patch_count=patch_count,
        padding_mode=padding_mode,
        patch_width=patch_width,
    )
    if str(patch_mode) == "random":
        return _sample_random_regions(
            x_bcol,
            region_count=patch_count,
            region_width=patch_width,
            sampling_mode="random",
            sample_keys=sample_keys,
            sampling_epoch=sampling_epoch,
            sampling_seed_offset=sampling_seed_offset,
            return_start_indices=return_start_indices,
        )
    if str(patch_mode) == "random_stratified":
        return _sample_random_regions(
            x_bcol,
            region_count=patch_count,
            region_width=patch_width,
            sampling_mode="random_stratified",
            sample_keys=sample_keys,
            sampling_epoch=sampling_epoch,
            sampling_seed_offset=sampling_seed_offset,
            return_start_indices=return_start_indices,
        )
    if str(patch_mode) != "uniform":
        raise ValueError(f"Unsupported patch_mode: {patch_mode}")
    padded_length = patch_width * patch_count
    x_bcol = maybe_zero_pad_last_dim(x_bcol, padded_length)
    patches = x_bcol.view(*x_bcol.shape[:-1], patch_count, patch_width)
    if return_start_indices:
        starts = torch.arange(
            0,
            patch_count * patch_width,
            patch_width,
            device=x_bcol.device,
            dtype=torch.long,
        ).view(1, patch_count).expand(x_bcol.shape[0], patch_count)
        return patches, starts
    return patches


def resolve_freq_band_width(
    freq_bins: int,
    band_count: int,
    padding_mode: str,
    band_width: int | None = None,
) -> int:
    if band_width is not None:
        resolved = int(band_width)
        if resolved <= 0:
            raise ValueError(f"band_width must be positive, got {resolved}.")
        return resolved
    return compute_partition_width(freq_bins, band_count, padding_mode)


def make_freq_bands(
    x_bcof: torch.Tensor,
    band_count: int,
    padding_mode: str,
    band_mode: str = "uniform",
    band_width: int | None = None,
    sample_keys: torch.Tensor | None = None,
    sampling_epoch: int = 0,
    sampling_seed_offset: int = 0,
    return_start_indices: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if x_bcof.ndim != 4:
        raise ValueError(f"Expected [B, C, O, F], got {tuple(x_bcof.shape)}")
    freq_bins = int(x_bcof.shape[-1])
    band_width = resolve_freq_band_width(
        freq_bins=freq_bins,
        band_count=band_count,
        padding_mode=padding_mode,
        band_width=band_width,
    )
    if str(band_mode) == "random":
        return _sample_random_regions(
            x_bcof,
            region_count=band_count,
            region_width=band_width,
            sampling_mode="random",
            sample_keys=sample_keys,
            sampling_epoch=sampling_epoch,
            sampling_seed_offset=sampling_seed_offset,
            return_start_indices=return_start_indices,
        )
    if str(band_mode) == "random_stratified":
        return _sample_random_regions(
            x_bcof,
            region_count=band_count,
            region_width=band_width,
            sampling_mode="random_stratified",
            sample_keys=sample_keys,
            sampling_epoch=sampling_epoch,
            sampling_seed_offset=sampling_seed_offset,
            return_start_indices=return_start_indices,
        )
    if str(band_mode) != "uniform":
        raise ValueError(f"Unsupported band_mode: {band_mode}")
    padded_bins = band_width * band_count
    x_bcof = maybe_zero_pad_last_dim(x_bcof, padded_bins)
    bands = x_bcof.view(*x_bcof.shape[:-1], band_count, band_width)
    if return_start_indices:
        starts = torch.arange(
            0,
            band_count * band_width,
            band_width,
            device=x_bcof.device,
            dtype=torch.long,
        ).view(1, band_count).expand(x_bcof.shape[0], band_count)
        return bands, starts
    return bands


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
