from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.Pipeline_01_default import pipeline
from src.configs.config_utils import merge_with_local_override
from src.utils.experiment import (
    aggregate_variant_results,
    collect_variant_results,
    render_markdown_summary,
    write_summary_bundle,
)


def load_study_config(study_config_path: str | Path) -> Dict[str, Any]:
    path = Path(study_config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "study" not in data or "execution" not in data or "variants" not in data:
        raise ValueError("Study config must contain 'study', 'execution', and 'variants' sections.")
    return data


def select_variants(
    variants: Sequence[Mapping[str, Any]],
    limit_variants: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    if not limit_variants:
        return [dict(item) for item in variants]
    allowed = {str(item) for item in limit_variants}
    selected = [dict(item) for item in variants if str(item.get("variant_id")) in allowed]
    if len(selected) != len(allowed):
        known = sorted(str(item.get("variant_id")) for item in variants)
        missing = sorted(allowed - {str(item.get("variant_id")) for item in selected})
        raise ValueError(f"Unknown variant ids: {missing}. Known variants: {known}.")
    return selected


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


def execute_variant_iteration(
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


def prepare_iteration_overrides(
    variant_spec: Mapping[str, Any],
    base_seed: int,
    iteration: int,
    variant_output_root: Path,
    num_epochs: int,
    smoke_num_workers: int | None = None,
) -> Dict[str, Any]:
    overrides = dict(variant_spec.get("overrides", {}))
    overrides["environment.output_dir"] = str(variant_output_root)
    overrides["environment.iterations"] = 1
    overrides["environment.seed"] = int(base_seed + iteration)
    overrides["trainer.num_epochs"] = int(num_epochs)
    if smoke_num_workers is not None:
        overrides["data.num_workers"] = int(smoke_num_workers)
    overrides["environment.notes"] = (
        f"{variant_spec.get('description', '')} | variant={variant_spec.get('variant_id')} | iteration={iteration}"
    )
    return overrides


def run_ablation_study(
    study_config_path: str | Path,
    local_config: str | None = None,
    limit_variants: Sequence[str] | None = None,
    smoke: bool = False,
    iterations_override: int | None = None,
    num_epochs_override: int | None = None,
    output_dir_override: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    study_cfg = load_study_config(study_config_path)
    selected_variants = select_variants(study_cfg["variants"], limit_variants)
    base_config_path = str(study_cfg["study"]["base_config"])
    base_config = merge_with_local_override(base_config_path, local_config)
    base_seed = int(getattr(base_config.environment, "seed", 0))

    execution = study_cfg["execution"]
    iterations = int(
        iterations_override
        if iterations_override is not None
        else execution["smoke_iterations"] if smoke else execution["iterations"]
    )
    num_epochs = int(
        num_epochs_override
        if num_epochs_override is not None
        else execution["smoke_num_epochs"] if smoke else execution["default_num_epochs"]
    )
    smoke_num_workers = int(execution["smoke_num_workers"]) if smoke and "smoke_num_workers" in execution else None
    study_output_root = Path(output_dir_override or study_cfg["study"]["output_dir"]).resolve()
    study_output_root.mkdir(parents=True, exist_ok=True)

    all_runs = []
    for variant_spec in selected_variants:
        variant_output_root = study_output_root / str(variant_spec["variant_id"])
        iteration_records: List[Dict[str, Any]] = []
        run_dirs: List[str] = []
        for iteration in range(iterations):
            iteration_overrides = prepare_iteration_overrides(
                variant_spec=variant_spec,
                base_seed=base_seed,
                iteration=iteration,
                variant_output_root=variant_output_root,
                num_epochs=num_epochs,
                smoke_num_workers=smoke_num_workers,
            )
            try:
                result, new_run_dirs = execute_variant_iteration(
                    base_config_path=base_config_path,
                    local_config=local_config,
                    overrides=iteration_overrides,
                    output_root=variant_output_root,
                )
                iteration_records.append({"success": True, **result})
                run_dirs.append(new_run_dirs[-1] if new_run_dirs else "")
            except Exception as exc:
                iteration_records.append({"success": False, "error_message": str(exc)})
                run_dirs.append("")
                if not execution.get("continue_on_error", True):
                    raise
        all_runs.append(
            collect_variant_results(
                variant_spec=variant_spec,
                iteration_records=iteration_records,
                run_dirs=run_dirs,
                smoke=smoke,
                num_epochs=num_epochs,
            )
        )

    runs_df = pd.concat(all_runs, ignore_index=True) if all_runs else pd.DataFrame()
    summary_df = aggregate_variant_results(runs_df)
    markdown = render_markdown_summary(
        runs_df=runs_df,
        summary_df=summary_df,
        variant_manifest=selected_variants,
        study_name=str(study_cfg["study"]["name"]),
    )
    write_summary_bundle(study_output_root, runs_df, summary_df, markdown, selected_variants)
    return runs_df, summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TF_MultiProtoDG ablation study.")
    parser.add_argument(
        "--study-config",
        type=str,
        default="configs/experiments/01_cross_domain/X_DG/tf_multi_proto_dg_ablation/study.yaml",
        help="Path to the study-level ablation YAML.",
    )
    parser.add_argument("--local-config", type=str, default=None, help="Optional local override YAML.")
    parser.add_argument(
        "--limit-variants",
        nargs="*",
        default=None,
        help="Optional subset of variant ids to execute.",
    )
    parser.add_argument("--smoke", action="store_true", help="Use smoke iterations and smoke epochs.")
    parser.add_argument("--iterations", type=int, default=None, help="Override study iteration count.")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override trainer.num_epochs.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override study output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ablation_study(
        study_config_path=args.study_config,
        local_config=args.local_config,
        limit_variants=args.limit_variants,
        smoke=args.smoke,
        iterations_override=args.iterations,
        num_epochs_override=args.num_epochs,
        output_dir_override=args.output_dir,
    )


if __name__ == "__main__":
    main()
