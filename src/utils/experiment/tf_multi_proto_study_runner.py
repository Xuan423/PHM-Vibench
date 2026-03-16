from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd

from src.Pipeline_01_default import pipeline
from src.configs.config_utils import merge_with_local_override
from src.utils.experiment.tf_multi_proto_ablation_summary import (
    aggregate_study_results,
    collect_study_results,
    render_study_markdown_summary,
    write_study_summary_bundle,
)
from src.utils.experiment.tf_multi_proto_study_config import (
    StudyConfig,
    StudyItemSpec,
    TaskSpec,
    expand_tasks,
    load_study_config,
    load_taskset,
    manifest_from_items,
    manifest_from_tasks,
    select_study_items,
)


def build_override_list(overrides: Mapping[str, Any]) -> List[str]:
    return [f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in overrides.items()]


def snapshot_iteration_dirs(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {str(path.resolve()) for path in root.glob("**/iter_*") if path.is_dir()}


def find_new_run_dirs(root: Path, before: set[str]) -> List[str]:
    after = snapshot_iteration_dirs(root)
    new_dirs = [Path(path) for path in sorted(after - before, key=lambda item: Path(item).stat().st_mtime)]
    return [str(path) for path in new_dirs]


def execute_study_iteration(
    base_config_path: str,
    local_config: str | None,
    overrides: Mapping[str, Any],
    output_root: Path,
) -> tuple[Dict[str, Any], List[str]]:
    before = snapshot_iteration_dirs(output_root)
    args = SimpleNamespace(
        config_path=base_config_path,
        local_config=local_config,
        override=build_override_list(overrides),
    )
    results = pipeline(args)
    run_dirs = find_new_run_dirs(output_root, before)
    if not results:
        raise RuntimeError("Pipeline returned no results.")
    return dict(results[0]), run_dirs


def _merge_study_overrides(
    task_spec: TaskSpec,
    item_spec: StudyItemSpec,
    base_seed: int,
    iteration: int,
    run_output_root: Path,
    num_epochs: int,
    smoke_num_workers: int | None,
    export_diagnostics_override: bool | None,
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    overrides.update(task_spec.to_overrides())
    overrides.update(item_spec.overrides)
    overrides["environment.output_dir"] = str(run_output_root)
    overrides["environment.iterations"] = 1
    overrides["environment.seed"] = int(base_seed + iteration)
    overrides["trainer.num_epochs"] = int(num_epochs)
    if smoke_num_workers is not None:
        overrides["data.num_workers"] = int(smoke_num_workers)

    diagnostics_enabled = bool(item_spec.overrides.get("model.export_diagnostics", False))
    if export_diagnostics_override is not None:
        diagnostics_enabled = bool(export_diagnostics_override)
    overrides["model.export_diagnostics"] = diagnostics_enabled

    notes = [
        task_spec.description,
        item_spec.description,
        f"task_id={task_spec.task_id}",
        f"item_id={item_spec.item_id}",
        f"iteration={iteration}",
    ]
    overrides["environment.notes"] = " | ".join(str(note) for note in notes if note)
    return overrides


def _study_output_root(study_output_root: Path, task_spec: TaskSpec, item_spec: StudyItemSpec) -> Path:
    return study_output_root / task_spec.task_id / item_spec.item_id


def run_study(
    taskset_path: str | Path,
    study_config_path: str | Path,
    local_config: str | None = None,
    limit_items: Sequence[str] | None = None,
    smoke: bool = False,
    iterations_override: int | None = None,
    num_epochs_override: int | None = None,
    output_dir_override: str | None = None,
    export_diagnostics_override: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    taskset_cfg = load_taskset(taskset_path)
    study_cfg = load_study_config(study_config_path)
    selected_items = select_study_items(study_cfg.items, limit_items)
    tasks = expand_tasks(taskset_cfg)

    base_config = merge_with_local_override(study_cfg.base_config, local_config)
    base_seed = int(getattr(base_config.environment, "seed", 0))

    iterations = int(
        iterations_override if iterations_override is not None else (
            study_cfg.execution.smoke_iterations if smoke else study_cfg.execution.iterations
        )
    )
    num_epochs = int(
        num_epochs_override if num_epochs_override is not None else (
            study_cfg.execution.smoke_num_epochs if smoke else study_cfg.execution.default_num_epochs
        )
    )
    smoke_num_workers = study_cfg.execution.smoke_num_workers if smoke else None
    study_output_root = Path(output_dir_override or study_cfg.output_dir).resolve()
    study_output_root.mkdir(parents=True, exist_ok=True)

    all_runs: List[pd.DataFrame] = []
    for task_spec in tasks:
        for item_spec in selected_items:
            item_output_root = _study_output_root(study_output_root, task_spec, item_spec)
            iteration_records: List[Dict[str, Any]] = []
            run_dirs: List[str] = []
            for iteration in range(iterations):
                iteration_overrides = _merge_study_overrides(
                    task_spec=task_spec,
                    item_spec=item_spec,
                    base_seed=base_seed,
                    iteration=iteration,
                    run_output_root=item_output_root,
                    num_epochs=num_epochs,
                    smoke_num_workers=smoke_num_workers,
                    export_diagnostics_override=export_diagnostics_override,
                )
                try:
                    result, new_run_dirs = execute_study_iteration(
                        base_config_path=study_cfg.base_config,
                        local_config=local_config,
                        overrides=iteration_overrides,
                        output_root=item_output_root,
                    )
                    iteration_records.append({"success": True, **result})
                    run_dirs.append(new_run_dirs[-1] if new_run_dirs else "")
                except Exception as exc:
                    iteration_records.append({"success": False, "error_message": str(exc)})
                    run_dirs.append("")
                    if not study_cfg.execution.continue_on_error:
                        raise
            all_runs.append(
                collect_study_results(
                    item_spec=item_spec.to_manifest_dict(),
                    task_spec=task_spec.to_manifest_dict(),
                    iteration_records=iteration_records,
                    run_dirs=run_dirs,
                    smoke=smoke,
                    num_epochs=num_epochs,
                    study_type=study_cfg.study_type,
                )
            )

    runs_df = pd.concat(all_runs, ignore_index=True) if all_runs else pd.DataFrame()
    summary_df = aggregate_study_results(runs_df)
    markdown = render_study_markdown_summary(
        runs_df=runs_df,
        summary_df=summary_df,
        item_manifest=manifest_from_items(selected_items),
        task_manifest=manifest_from_tasks(tasks),
        study_name=study_cfg.name,
        study_type=study_cfg.study_type,
    )
    write_study_summary_bundle(
        output_dir=study_output_root,
        runs_df=runs_df,
        summary_df=summary_df,
        markdown=markdown,
        item_manifest=manifest_from_items(selected_items),
        task_manifest=manifest_from_tasks(tasks),
        artifact_prefix=study_cfg.study_type,
    )
    return runs_df, summary_df


def load_taskset_and_study(
    taskset_path: str | Path,
    study_path: str | Path,
) -> tuple[List[TaskSpec], StudyConfig]:
    taskset_cfg = load_taskset(taskset_path)
    study_cfg = load_study_config(study_path)
    return expand_tasks(taskset_cfg), study_cfg
