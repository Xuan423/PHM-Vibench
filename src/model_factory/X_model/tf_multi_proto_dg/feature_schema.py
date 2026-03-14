from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, List

import torch


@dataclass(frozen=True)
class FeatureMeta:
    feature_index: int
    domain: str
    operator_name: str
    indicator_name: str
    channel_index: int
    region_index: int
    region_start: int
    region_end: int
    display_name: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_feature_metadata(
    num_channels: int,
    time_operator_names: Iterable[str],
    freq_operator_names: Iterable[str],
    time_patch_count: int,
    freq_band_count: int,
    time_patch_width: int,
    freq_band_width: int,
    time_indicator_names: Iterable[str],
    freq_indicator_names: Iterable[str],
) -> List[FeatureMeta]:
    feature_meta: List[FeatureMeta] = []
    feature_index = 0
    for channel_index in range(num_channels):
        for operator_name in time_operator_names:
            for region_index in range(time_patch_count):
                start = region_index * time_patch_width
                end = start + time_patch_width
                for indicator_name in time_indicator_names:
                    display_name = (
                        f"time.ch{channel_index}.{operator_name}.patch{region_index}.{indicator_name}"
                    )
                    feature_meta.append(
                        FeatureMeta(
                            feature_index=feature_index,
                            domain="time",
                            operator_name=str(operator_name),
                            indicator_name=str(indicator_name),
                            channel_index=channel_index,
                            region_index=region_index,
                            region_start=start,
                            region_end=end,
                            display_name=display_name,
                        )
                    )
                    feature_index += 1
        for operator_name in freq_operator_names:
            for region_index in range(freq_band_count):
                start = region_index * freq_band_width
                end = start + freq_band_width
                for indicator_name in freq_indicator_names:
                    display_name = (
                        f"freq.ch{channel_index}.{operator_name}.band{region_index}.{indicator_name}"
                    )
                    feature_meta.append(
                        FeatureMeta(
                            feature_index=feature_index,
                            domain="freq",
                            operator_name=str(operator_name),
                            indicator_name=str(indicator_name),
                            channel_index=channel_index,
                            region_index=region_index,
                            region_start=start,
                            region_end=end,
                            display_name=display_name,
                        )
                    )
                    feature_index += 1
    return feature_meta


def flatten_feature_tensors(time_features: torch.Tensor, freq_features: torch.Tensor) -> torch.Tensor:
    time_flat = time_features.reshape(time_features.shape[0], -1)
    freq_flat = freq_features.reshape(freq_features.shape[0], -1)
    return torch.cat([time_flat, freq_flat], dim=1)


def decode_feature_index(feature_meta: List[FeatureMeta], feature_index: int) -> FeatureMeta:
    return feature_meta[int(feature_index)]
