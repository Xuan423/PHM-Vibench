"""Stage configuration helpers for the two-stage TSPN pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ...configs.config_utils import ConfigWrapper, dict_to_namespace, load_config


def _ensure_namespace(parent: ConfigWrapper, attr: str) -> ConfigWrapper:
    value = getattr(parent, attr, None)
    if isinstance(value, ConfigWrapper):
        return value
    if value is None:
        value = ConfigWrapper()
        setattr(parent, attr, value)
        return value
    if not isinstance(value, ConfigWrapper):
        value = ConfigWrapper(**getattr(value, "__dict__", {}))
        setattr(parent, attr, value)
    return value


def _set_nested_attr(root: ConfigWrapper, path: str, value: Any) -> None:
    parts = path.split(".")
    current = root
    for part in parts[:-1]:
        current = _ensure_namespace(current, part)
    setattr(current, parts[-1], value)


@dataclass
class StageConfigComposer:
    """Creates stage-specific configs derived from a common base."""

    base_config_path: str
    overrides: Optional[Dict[str, Any]] = None
    stage2_template_path: Optional[str] = None
    output_root: Optional[str] = None

    def __post_init__(self) -> None:
        self._base = load_config(self.base_config_path, self.overrides)
        env = getattr(self._base, "environment", ConfigWrapper())
        self._base_project = getattr(env, "project", "tspn_two_stage")
        default_output = getattr(env, "output_dir", None)
        base_output = Path(self.output_root) if self.output_root else (
            Path(default_output) if default_output else Path("save") / self._base_project
        )
        self._output_root = base_output
        pipeline_cfg = getattr(self._base, "pipeline", None)
        self._stage2_inline: Optional[Dict[str, Any]] = None
        if pipeline_cfg is not None:
            stage2_inline = getattr(pipeline_cfg, "stage2_overrides", None)
            if stage2_inline is not None:
                self._stage2_inline = self._namespace_to_dict(stage2_inline)
        self._stage2_template = dict_to_namespace(self._load_stage2_template())

    # ------------------------------------------------------------------
    def build_stage1(self) -> ConfigWrapper:
        cfg = self._base.copy()
        self._apply_common_env(cfg, "stage1")
        _set_nested_attr(cfg, "task.stage_mode", "stage1")
        _set_nested_attr(cfg, "task.few_shot.enabled", False)
        _set_nested_attr(cfg, "task.contrastive.loss_weight", 0.0)
        _set_nested_attr(cfg, "model.contrastive.loss_weight", 0.0)
        _set_nested_attr(cfg, "task.contrastive.prototype_matching.enabled", False)
        _set_nested_attr(cfg, "model.contrastive.prototype_matching.enabled", False)
        return cfg

    def build_stage2(self) -> ConfigWrapper:
        cfg = self._base.copy()
        cfg.update(self._stage2_template)
        self._apply_common_env(cfg, "stage2")
        _set_nested_attr(cfg, "task.stage_mode", "stage2")
        _set_nested_attr(cfg, "task.few_shot.enabled", True)
        few_shot_ns = getattr(cfg.task, "few_shot", None)
        if few_shot_ns and hasattr(few_shot_ns, "warmup_flat_epochs"):
            raise ValueError(
                "Stage2 no longer supports `warmup_flat_epochs`. Remove the key and rely on task.warmup."
            )
        return cfg

    # ------------------------------------------------------------------
    def _apply_common_env(self, cfg: ConfigWrapper, stage_suffix: str) -> None:
        env = _ensure_namespace(cfg, "environment")
        project = f"{self._base_project}_{stage_suffix}"
        env.project = project
        env.iterations = 1
        env.output_dir = str((self._output_root / stage_suffix).resolve())

    @property
    def stage_output_root(self) -> Path:
        return self._output_root

    def _load_stage2_template(self) -> Dict[str, Any]:
        if self._stage2_inline:
            return self._stage2_inline
        candidates: List[Path] = []
        if self.stage2_template_path:
            candidates.append(Path(self.stage2_template_path))
        else:
            candidates.extend(
                [
                    Path("configs/demo/X_Single_DG/TSPN_FewShot/tspn_two_stage_stage2.yaml"),
                    Path("configs/pipelines/tspn_two_stage_stage2.yaml"),
                ]
            )
        for path in candidates:
            if path.exists():
                return self._load_partial_yaml(path)
        raise FileNotFoundError(
            "No stage2 overrides found. Either embed `pipeline.stage2_overrides` in the config or "
            "pass --stage2_template pointing to a YAML file."
        )

    @staticmethod
    def _load_partial_yaml(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    @staticmethod
    def _namespace_to_dict(obj: Any) -> Any:
        if isinstance(obj, ConfigWrapper):
            return {
                key: StageConfigComposer._namespace_to_dict(value)
                for key, value in obj.__dict__.items()
            }
        if isinstance(obj, dict):
            return {key: StageConfigComposer._namespace_to_dict(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [StageConfigComposer._namespace_to_dict(value) for value in obj]
        return obj
