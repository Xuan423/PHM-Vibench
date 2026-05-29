from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import yaml


ALLOWED_HPARAM_KEYS = (
    "model.proto_contrastive_weight",
    "model.proto_contrastive_temperature",
    "task.label_smoothing",
)

_TASK_REQUIRED_KEYS = ("task_id", "target_system_id", "source_domain_id", "target_domain_id")
_TASK_ALLOWED_KEYS = set(_TASK_REQUIRED_KEYS) | {"description", "overrides"}
_STUDY_ALLOWED_TYPES = {"ablation", "hparam"}

_HPARAM_ID_ALIASES = {
    "model.proto_contrastive_weight": "pcw",
    "model.proto_contrastive_temperature": "pctau",
    "task.label_smoothing": "ls",
}

_LEGACY_HPARAM_KEY_ALIASES = {
    "model.prototype_temperature": "model.proto_contrastive_temperature",
}


@dataclass(frozen=True)
class SystemFamilyPreset:
    target_system_id: int
    candidate_domains: tuple[int, ...]
    base_config: str
    default_label_smoothing: float
    default_lr: float
    default_weight_decay: float
    default_num_epochs: int
    default_patience: int
    default_proto_contrastive_weight: float
    default_num_prototypes_per_class: int


SUPPORTED_SYSTEM_PRESETS: Dict[int, SystemFamilyPreset] = {
    27: SystemFamilyPreset(
        target_system_id=27,
        candidate_domains=(0, 1, 2),
        base_config="configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg.yaml",
        default_label_smoothing=0.08,
        default_lr=0.001,
        default_weight_decay=0.001,
        default_num_epochs=100,
        default_patience=50,
        default_proto_contrastive_weight=0.05,
        default_num_prototypes_per_class=3,
    ),
    16: SystemFamilyPreset(
        target_system_id=16,
        candidate_domains=(10, 11, 14, 15),
        base_config="configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg.yaml",
        default_label_smoothing=0.02,
        default_lr=0.001,
        default_weight_decay=0.0001,
        default_num_epochs=100,
        default_patience=50,
        default_proto_contrastive_weight=0.1,
        default_num_prototypes_per_class=4,
    ),
    21: SystemFamilyPreset(
        target_system_id=21,
        candidate_domains=(14, 16, 22, 24),
        base_config="configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg.yaml",
        default_label_smoothing=0.05,
        default_lr=0.001,
        default_weight_decay=0.001,
        default_num_epochs=100,
        default_patience=50,
        default_proto_contrastive_weight=0.08,
        default_num_prototypes_per_class=4,
    ),
    13: SystemFamilyPreset(
        target_system_id=13,
        candidate_domains=(0, 1, 2, 3),
        base_config="configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg_full_baseline_k2_w002_adapteffk.yaml",
        default_label_smoothing=0.08,
        default_lr=0.01,
        default_weight_decay=0.0001,
        default_num_epochs=100,
        default_patience=50,
        default_proto_contrastive_weight=0.1,
        default_num_prototypes_per_class=4,
    ),
}


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    target_system_id: tuple[int, ...]
    source_domain_id: tuple[int, ...]
    target_domain_id: tuple[int, ...]
    description: str = ""
    overrides: Dict[str, Any] = field(default_factory=dict)

    def to_overrides(self) -> Dict[str, Any]:
        overrides = dict(self.overrides)
        overrides["task.target_system_id"] = list(self.target_system_id)
        overrides["task.source_domain_id"] = list(self.source_domain_id)
        overrides["task.target_domain_id"] = list(self.target_domain_id)
        return overrides

    def to_manifest_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "target_system_id": list(self.target_system_id),
            "source_domain_id": list(self.source_domain_id),
            "target_domain_id": list(self.target_domain_id),
            "overrides": dict(self.overrides),
        }


@dataclass(frozen=True)
class TasksetConfig:
    name: str
    description: str
    tasks: tuple[TaskSpec, ...]
    path: str


@dataclass(frozen=True)
class StudyExecutionConfig:
    iterations: int
    default_num_epochs: int
    smoke_iterations: int
    smoke_num_epochs: int
    smoke_num_workers: int | None
    continue_on_error: bool = True
    diagnostics_extract: bool = True


@dataclass(frozen=True)
class StudyItemSpec:
    item_id: str
    group: str
    description: str
    overrides: Dict[str, Any]
    diagnostics_expected: bool = False

    def to_manifest_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "group": self.group,
            "description": self.description,
            "diagnostics_expected": self.diagnostics_expected,
            "overrides": dict(self.overrides),
        }


@dataclass(frozen=True)
class StudyConfig:
    name: str
    study_type: str
    base_config: str
    output_dir: str
    summary: Dict[str, Any]
    execution: StudyExecutionConfig
    items: tuple[StudyItemSpec, ...]
    path: str


def get_system_family_preset(target_system_id: int) -> SystemFamilyPreset | None:
    return SUPPORTED_SYSTEM_PRESETS.get(int(target_system_id))


def _load_yaml_mapping(path: str | Path) -> Dict[str, Any]:
    file_path = Path(path)
    data = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML at {file_path}.")
    return data


def _ensure_int_list(field_name: str, value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list of integers.")
    values: List[int] = []
    for item in value:
        try:
            values.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must contain only integers.") from exc
    return tuple(values)


def _sanitize_component(text: Any) -> str:
    rendered = str(text)
    rendered = rendered.replace("-", "m").replace(".", "p")
    rendered = rendered.replace("[", "").replace("]", "").replace(",", "_")
    return rendered


def _validate_task_against_supported_systems(
    *,
    task_id: str,
    target_system_id: tuple[int, ...],
    source_domain_id: tuple[int, ...],
    target_domain_id: tuple[int, ...],
    overrides: Mapping[str, Any],
) -> None:
    if len(target_system_id) != 1:
        return
    preset = get_system_family_preset(target_system_id[0])
    if preset is None:
        return

    candidate_set = set(preset.candidate_domains)
    source_set = set(source_domain_id)
    target_set = set(target_domain_id)

    if len(target_domain_id) != 1:
        raise ValueError(
            f"Task {task_id!r} for supported system {preset.target_system_id} must target exactly one domain."
        )
    if not target_set.issubset(candidate_set):
        raise ValueError(
            f"Task {task_id!r} target domains {sorted(target_set)} are outside supported system "
            f"{preset.target_system_id} candidate domains {list(preset.candidate_domains)}."
        )
    if not source_set.issubset(candidate_set):
        raise ValueError(
            f"Task {task_id!r} source domains {sorted(source_set)} are outside supported system "
            f"{preset.target_system_id} candidate domains {list(preset.candidate_domains)}."
        )
    if source_set & target_set:
        raise ValueError(f"Task {task_id!r} source and target domains must be disjoint.")
    if source_set | target_set != candidate_set:
        raise ValueError(
            f"Task {task_id!r} must form a leave-one-out split over {list(preset.candidate_domains)}."
        )

    expected_override_values = {
        "task.label_smoothing": preset.default_label_smoothing,
        "task.lr": preset.default_lr,
        "task.weight_decay": preset.default_weight_decay,
        "trainer.num_epochs": preset.default_num_epochs,
        "trainer.patience": preset.default_patience,
        "model.proto_contrastive_weight": preset.default_proto_contrastive_weight,
    }
    for key, expected in expected_override_values.items():
        if key not in overrides:
            continue
        actual = overrides[key]
        if float(actual) != float(expected):
            raise ValueError(
                f"Task {task_id!r} override {key}={actual!r} does not match supported system "
                f"{preset.target_system_id} default {expected!r}."
            )

    if "model.num_prototypes_per_class" in overrides:
        actual_proto_count = int(overrides["model.num_prototypes_per_class"])
        if actual_proto_count != preset.default_num_prototypes_per_class:
            raise ValueError(
                f"Task {task_id!r} override model.num_prototypes_per_class={actual_proto_count} "
                f"does not match supported system {preset.target_system_id} expected value "
                f"{preset.default_num_prototypes_per_class}."
            )


def _parse_task_spec(raw_task: Mapping[str, Any], index: int) -> TaskSpec:
    unknown_keys = sorted(set(raw_task.keys()) - _TASK_ALLOWED_KEYS)
    if unknown_keys:
        raise ValueError(f"Task {index} contains unsupported keys: {unknown_keys}.")
    missing = [key for key in _TASK_REQUIRED_KEYS if key not in raw_task]
    if missing:
        raise ValueError(f"Task {index} is missing required keys: {missing}.")
    overrides = raw_task.get("overrides", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise ValueError(f"Task {index} overrides must be a mapping.")
    task_spec = TaskSpec(
        task_id=str(raw_task["task_id"]),
        description=str(raw_task.get("description", "")),
        target_system_id=_ensure_int_list("target_system_id", raw_task["target_system_id"]),
        source_domain_id=_ensure_int_list("source_domain_id", raw_task["source_domain_id"]),
        target_domain_id=_ensure_int_list("target_domain_id", raw_task["target_domain_id"]),
        overrides=dict(overrides),
    )
    _validate_task_against_supported_systems(
        task_id=task_spec.task_id,
        target_system_id=task_spec.target_system_id,
        source_domain_id=task_spec.source_domain_id,
        target_domain_id=task_spec.target_domain_id,
        overrides=task_spec.overrides,
    )
    return task_spec


def load_taskset(taskset_path: str | Path) -> TasksetConfig:
    data = _load_yaml_mapping(taskset_path)
    payload = data.get("taskset")
    if not isinstance(payload, Mapping):
        raise ValueError("Taskset YAML must contain a 'taskset' mapping.")
    tasks_payload = payload.get("tasks")
    if not isinstance(tasks_payload, list) or not tasks_payload:
        raise ValueError("Taskset must contain a non-empty 'tasks' list.")
    tasks = tuple(_parse_task_spec(task, index) for index, task in enumerate(tasks_payload))
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"Duplicate task_id detected in taskset: {task_ids}.")
    return TasksetConfig(
        name=str(payload.get("name", Path(taskset_path).stem)),
        description=str(payload.get("description", "")),
        tasks=tasks,
        path=str(Path(taskset_path)),
    )


def expand_tasks(taskset_cfg: TasksetConfig) -> List[TaskSpec]:
    return list(taskset_cfg.tasks)


def _parse_execution(payload: Mapping[str, Any]) -> StudyExecutionConfig:
    required = ("iterations", "default_num_epochs", "smoke_iterations", "smoke_num_epochs")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Study execution config missing required keys: {missing}.")
    smoke_num_workers = payload.get("smoke_num_workers")
    return StudyExecutionConfig(
        iterations=int(payload["iterations"]),
        default_num_epochs=int(payload["default_num_epochs"]),
        smoke_iterations=int(payload["smoke_iterations"]),
        smoke_num_epochs=int(payload["smoke_num_epochs"]),
        smoke_num_workers=int(smoke_num_workers) if smoke_num_workers is not None else None,
        continue_on_error=bool(payload.get("continue_on_error", True)),
        diagnostics_extract=bool(payload.get("diagnostics_extract", True)),
    )


def _parse_study_header(data: Mapping[str, Any]) -> tuple[Dict[str, Any], StudyExecutionConfig]:
    study_payload = data.get("study")
    execution_payload = data.get("execution")
    if not isinstance(study_payload, Mapping) or not isinstance(execution_payload, Mapping):
        raise ValueError("Study YAML must contain 'study' and 'execution' mappings.")
    study_type = str(study_payload.get("type", "")).strip()
    if study_type not in _STUDY_ALLOWED_TYPES:
        raise ValueError(f"study.type must be one of {_STUDY_ALLOWED_TYPES}, got {study_type!r}.")
    for field_name in ("name", "base_config", "output_dir"):
        if field_name not in study_payload:
            raise ValueError(f"study.{field_name} is required.")
    return (
        {
            "name": str(study_payload["name"]),
            "study_type": study_type,
            "base_config": str(study_payload["base_config"]),
            "output_dir": str(study_payload["output_dir"]),
            "summary": dict(study_payload.get("summary", {})),
        },
        _parse_execution(execution_payload),
    )


def _parse_ablation_items(variants: Sequence[Mapping[str, Any]]) -> tuple[StudyItemSpec, ...]:
    items: List[StudyItemSpec] = []
    item_ids: List[str] = []
    for index, variant in enumerate(variants):
        if not isinstance(variant, Mapping):
            raise ValueError(f"Variant {index} must be a mapping.")
        variant_id = variant.get("variant_id") or variant.get("item_id")
        if variant_id is None:
            raise ValueError(f"Variant {index} must define variant_id.")
        overrides = variant.get("overrides", {})
        if not isinstance(overrides, Mapping):
            raise ValueError(f"Variant {variant_id} overrides must be a mapping.")
        item_id = str(variant_id)
        item_ids.append(item_id)
        items.append(
            StudyItemSpec(
                item_id=item_id,
                group=str(variant.get("group", "")),
                description=str(variant.get("description", "")),
                diagnostics_expected=bool(variant.get("diagnostics_expected", False)),
                overrides=dict(overrides),
            )
        )
    if len(item_ids) != len(set(item_ids)):
        raise ValueError(f"Duplicate ablation item ids detected: {item_ids}.")
    if "branch_full_ce_only" in item_ids:
        raise ValueError("Redundant branch_full_ce_only must not appear in the new batch ablation study.")
    return tuple(items)


def _format_hparam_item_id(keys: Sequence[str], values: Sequence[Any]) -> str:
    parts = []
    for key, value in zip(keys, values):
        alias = _HPARAM_ID_ALIASES[key]
        parts.append(f"{alias}_{_sanitize_component(value)}")
    return "__".join(parts)


def expand_hparam_grid(grid: Mapping[str, Sequence[Any]]) -> List[StudyItemSpec]:
    keys = list(grid.keys())
    items: List[StudyItemSpec] = []
    for values in product(*(grid[key] for key in keys)):
        overrides = {key: value for key, value in zip(keys, values)}
        item_id = _format_hparam_item_id(keys, values)
        description = ", ".join(f"{key}={value}" for key, value in overrides.items())
        items.append(
            StudyItemSpec(
                item_id=item_id,
                group="hparam",
                description=description,
                diagnostics_expected=False,
                overrides=overrides,
            )
        )
    return items


def _parse_hparam_items(grid: Mapping[str, Any]) -> tuple[StudyItemSpec, ...]:
    normalized_grid: Dict[str, Any] = {}
    for key, value in grid.items():
        canonical_key = _LEGACY_HPARAM_KEY_ALIASES.get(key, key)
        if canonical_key in normalized_grid:
            raise ValueError(
                f"Duplicate hparam key after alias normalization: {canonical_key!r}."
            )
        normalized_grid[canonical_key] = value

    unknown_keys = sorted(set(normalized_grid.keys()) - set(ALLOWED_HPARAM_KEYS))
    if unknown_keys:
        raise ValueError(f"Unsupported hparam keys: {unknown_keys}.")
    missing_keys = [key for key in ALLOWED_HPARAM_KEYS if key not in normalized_grid]
    if missing_keys:
        raise ValueError(f"Hparam study must define all prioritized keys: {missing_keys}.")
    for key, values in normalized_grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Hparam grid '{key}' must be a non-empty list.")
    items = expand_hparam_grid(normalized_grid)
    return tuple(items)


def load_study_config(study_path: str | Path) -> StudyConfig:
    data = _load_yaml_mapping(study_path)
    header, execution = _parse_study_header(data)
    study_type = header["study_type"]
    if study_type == "ablation":
        if "variants" not in data or "grid" in data:
            raise ValueError("Ablation study must contain 'variants' and must not contain 'grid'.")
        variants = data["variants"]
        if not isinstance(variants, list) or not variants:
            raise ValueError("Ablation study variants must be a non-empty list.")
        items = _parse_ablation_items(variants)
    else:
        if "grid" not in data or "variants" in data:
            raise ValueError("Hparam study must contain 'grid' and must not contain 'variants'.")
        grid = data["grid"]
        if not isinstance(grid, Mapping):
            raise ValueError("Hparam study grid must be a mapping.")
        items = _parse_hparam_items(grid)
    return StudyConfig(
        name=header["name"],
        study_type=study_type,
        base_config=header["base_config"],
        output_dir=header["output_dir"],
        summary=header["summary"],
        execution=execution,
        items=items,
        path=str(Path(study_path)),
    )


def load_ablation_study(study_path: str | Path) -> StudyConfig:
    study = load_study_config(study_path)
    if study.study_type != "ablation":
        raise ValueError(f"Expected ablation study, got {study.study_type!r}.")
    return study


def load_hparam_study(study_path: str | Path) -> StudyConfig:
    study = load_study_config(study_path)
    if study.study_type != "hparam":
        raise ValueError(f"Expected hparam study, got {study.study_type!r}.")
    return study


def select_study_items(
    items: Sequence[StudyItemSpec],
    limit_items: Sequence[str] | None = None,
) -> List[StudyItemSpec]:
    if not limit_items:
        return list(items)
    allowed = {str(item) for item in limit_items}
    selected = [item for item in items if item.item_id in allowed]
    if len(selected) != len(allowed):
        known = sorted(item.item_id for item in items)
        missing = sorted(allowed - {item.item_id for item in selected})
        raise ValueError(f"Unknown item ids: {missing}. Known items: {known}.")
    return selected


def manifest_from_tasks(tasks: Iterable[TaskSpec]) -> List[Dict[str, Any]]:
    return [task.to_manifest_dict() for task in tasks]


def manifest_from_items(items: Iterable[StudyItemSpec]) -> List[Dict[str, Any]]:
    return [item.to_manifest_dict() for item in items]
