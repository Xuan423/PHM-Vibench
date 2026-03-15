from __future__ import annotations

from typing import Dict, List

import torch

from .config_schema import AblationConfig
from .feature_schema import FeatureMeta


def build_branch_mask(feature_meta: List[FeatureMeta], ablation_cfg: AblationConfig) -> torch.Tensor:
    values = []
    for item in feature_meta:
        if item.domain == "time":
            values.append(float(ablation_cfg.branch_masks.time))
        elif item.domain == "freq":
            values.append(float(ablation_cfg.branch_masks.freq))
        else:
            values.append(1.0)
    return torch.as_tensor(values, dtype=torch.float32)


def build_operator_mask(feature_meta: List[FeatureMeta], ablation_cfg: AblationConfig) -> torch.Tensor:
    values = []
    for item in feature_meta:
        if item.domain == "time":
            values.append(float(ablation_cfg.operator_masks_time.get(item.operator_name, 1.0)))
        elif item.domain == "freq":
            values.append(float(ablation_cfg.operator_masks_freq.get(item.operator_name, 1.0)))
        else:
            values.append(1.0)
    return torch.as_tensor(values, dtype=torch.float32)


def build_indicator_mask(feature_meta: List[FeatureMeta], ablation_cfg: AblationConfig) -> torch.Tensor:
    values = []
    for item in feature_meta:
        if item.domain == "time":
            values.append(float(ablation_cfg.indicator_masks_time.get(item.indicator_name, 1.0)))
        elif item.domain == "freq":
            values.append(float(ablation_cfg.indicator_masks_freq.get(item.indicator_name, 1.0)))
        else:
            values.append(1.0)
    return torch.as_tensor(values, dtype=torch.float32)


def build_feature_mask(feature_meta: List[FeatureMeta], ablation_cfg: AblationConfig) -> torch.Tensor:
    branch_mask = build_branch_mask(feature_meta, ablation_cfg)
    operator_mask = build_operator_mask(feature_meta, ablation_cfg)
    indicator_mask = build_indicator_mask(feature_meta, ablation_cfg)
    return branch_mask * operator_mask * indicator_mask


def describe_active_components(feature_meta: List[FeatureMeta], feature_mask: torch.Tensor) -> Dict[str, object]:
    active_time_ops = set()
    active_freq_ops = set()
    active_time_inds = set()
    active_freq_inds = set()
    active_branches = set()
    active_feature_indices: List[int] = []

    mask_cpu = feature_mask.detach().cpu().view(-1)
    for meta, mask_value in zip(feature_meta, mask_cpu.tolist()):
        if float(mask_value) <= 0.0:
            continue
        active_feature_indices.append(int(meta.feature_index))
        active_branches.add(str(meta.domain))
        if meta.domain == "time":
            active_time_ops.add(str(meta.operator_name))
            active_time_inds.add(str(meta.indicator_name))
        elif meta.domain == "freq":
            active_freq_ops.add(str(meta.operator_name))
            active_freq_inds.add(str(meta.indicator_name))

    return {
        "active_branches": sorted(active_branches),
        "active_operators": {
            "time": sorted(active_time_ops),
            "freq": sorted(active_freq_ops),
        },
        "active_indicators": {
            "time": sorted(active_time_inds),
            "freq": sorted(active_freq_inds),
        },
        "active_feature_indices": active_feature_indices,
    }
