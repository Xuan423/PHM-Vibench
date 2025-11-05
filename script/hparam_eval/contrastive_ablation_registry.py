"""Dataset registry for unified TSPN contrastive ablation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DomainTask:
    """Container for a single source→target domain assignment."""

    source_domain_id: Sequence[int]
    target_domain_id: Sequence[int]
    label: str


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset-specific ablation configuration."""

    name: str
    config_root: Path
    base_config: str
    domain_tasks: Sequence[DomainTask]
    target_system_id: Sequence[int]
    few_shot_overrides: Mapping[str, object]
    contrastive_weights: Sequence[float]
    notes: str = ""

    @property
    def config_root_path(self) -> Path:
        return self.config_root

    def resolve_base_config(self) -> Path:
        return self.config_root / self.base_config


def _build_hust_spec() -> DatasetSpec:
    domain_tasks = [
        DomainTask(source_domain_id=[0, 1, 2, 3, 4], target_domain_id=[10], label="src_full"),
        DomainTask(source_domain_id=[0, 2, 4, 6, 8], target_domain_id=[10], label="src_even"),
        DomainTask(source_domain_id=[1, 3, 5, 7, 9], target_domain_id=[10], label="src_odd"),
    ]
    few_shot_overrides: Dict[str, object] = {}
    return DatasetSpec(
        name="hust",
        config_root=PROJECT_ROOT / "configs" / "experiments" / "tspn_contrastive_ablation" / "hust",
        base_config="base.yaml",
        domain_tasks=domain_tasks,
        target_system_id=[19],
        few_shot_overrides=few_shot_overrides,
        contrastive_weights=(0.2, 0.3),
        notes="HUST contrastive ablation suite derived from demo few-shot config.",
    )


def _build_sdust_spec() -> DatasetSpec:
    domain_tasks = [
        DomainTask(source_domain_id=[0, 4, 9, 13, 17, 21], target_domain_id=[37], label="src_group_a"),
        DomainTask(source_domain_id=[1, 5, 10, 14, 18, 22], target_domain_id=[38], label="src_group_b"),
        DomainTask(source_domain_id=[2, 6, 11, 15, 19, 23], target_domain_id=[39], label="src_group_c"),
        DomainTask(source_domain_id=[3, 7, 12, 16, 20, 24], target_domain_id=[40], label="src_group_d"),
    ]
    few_shot_overrides = {
        "enabled": True,
        "sampler": "hierarchical",
        "systems_per_episode": 1,
        "domains_per_episode": 4,
        "classes_per_domain": 8,
        "support_per_class": 3,
        "query_per_class": 10,
        "preserve_labels": True,
        "episodes_per_epoch": 10,
        "warn_on_shortfall": True,
        "format": "episode",
    }
    return DatasetSpec(
        name="sdust",
        config_root=PROJECT_ROOT / "configs" / "experiments" / "tspn_contrastive_ablation" / "sdust",
        base_config="base.yaml",
        domain_tasks=domain_tasks,
        target_system_id=[21],
        few_shot_overrides=few_shot_overrides,
        contrastive_weights=(0.2, 0.3),
        notes="SDUST contrastive ablation suite with hierarchical few-shot episodes.",
    )


_DATASET_BUILDERS = {
    "hust": _build_hust_spec,
    "sdust": _build_sdust_spec,
}


def get_dataset_spec(name: str) -> DatasetSpec:
    key = name.lower()
    if key not in _DATASET_BUILDERS:
        raise ValueError(f"Unsupported dataset '{name}'. Available: {', '.join(sorted(_DATASET_BUILDERS))}")
    return _DATASET_BUILDERS[key]()


def list_dataset_names() -> List[str]:
    return sorted(_DATASET_BUILDERS.keys())
