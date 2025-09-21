"""CLI entrypoint for orchestrating TSPN hyperparameter sweeps."""

from __future__ import annotations

import argparse
import itertools
import math
import shutil
import textwrap
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
import yaml

from .utils import (
    LaunchResult,
    build_override,
    export_json,
    export_yaml,
    launch_experiment,
    load_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT_DEFAULT = PROJECT_ROOT / "configs" / "experiments" / "tspn_hparam_eval"
OUTPUT_ROOT_DEFAULT = PROJECT_ROOT / "save" / "hparam_eval"


@dataclass
class ParameterOption:
    """Concrete value for a sweep dimension."""

    key: str
    name: str
    overrides: Dict[str, Any]
    value: Any


@dataclass
class RunSpec:
    """Fully materialised run specification."""

    name: str
    group: str
    base_config: Path
    overrides: Dict[str, Any]
    hyperparams: Dict[str, Any]
    output_dir: Path
    pipeline: Optional[str] = None
    config_path: Optional[Path] = None
    log_dir: Optional[Path] = None


@dataclass
class RunResult:
    """Captures execution outcome for a run specification."""

    spec: RunSpec
    launch: LaunchResult
    status: str
    metrics: Dict[str, Any]
    log_dir: Optional[Path]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TSPN hyperparameter sweep orchestrator")
    parser.add_argument(
        "--sweep",
        choices=["contrastive", "fewshot", "optimizer", "baseline", "all"],
        default="all",
        help="Sweep group to execute.",
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=CONFIG_ROOT_DEFAULT,
        help="Directory containing sweep configuration YAMLs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT_DEFAULT,
        help="Root directory for aggregated outputs.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum concurrent experiment processes.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Optional timeout (seconds) for each run; 0 disables the limit.",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="Retry failed runs once after the initial sweep completes.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma separated CUDA device list forwarded as CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="Pipeline_01_default",
        help="Pipeline module to pass into main.py.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on the number of generated configs per sweep.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing experiments.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Additional notes appended to environment.notes for each run.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    config_root: Path = args.config_root
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    groups = [args.sweep] if args.sweep != "all" else ["contrastive", "fewshot", "optimizer", "baseline"]

    run_specs: List[RunSpec] = []
    for group in groups:
        run_specs.extend(
            build_run_specs_for_group(
                group=group,
                config_root=config_root,
                output_root=output_root,
                pipeline=args.pipeline,
                global_notes=args.notes,
                limit=args.limit,
            )
        )

    if not run_specs:
        print("[WARN] No run specifications generated; aborting.")
        return 0

    print(f"[INFO] Generated {len(run_specs)} run specifications across groups: {', '.join(groups)}")

    results = execute_runs(
        run_specs,
        max_parallel=max(1, args.max_parallel),
        timeout=args.timeout if args.timeout > 0 else None,
        devices=args.devices,
        pipeline=args.pipeline,
        dry_run=args.dry_run,
    )

    if args.rerun_failed and not args.dry_run:
        failed_specs = [res.spec for res in results if res.status == "failed"]
        if failed_specs:
            print(f"[INFO] Retrying {len(failed_specs)} failed runs once more...")
            retry_results = execute_runs(
                failed_specs,
                max_parallel=max(1, args.max_parallel),
                timeout=args.timeout if args.timeout > 0 else None,
                devices=args.devices,
                pipeline=args.pipeline,
                dry_run=args.dry_run,
            )
            # Replace results with retries for the failed specs
            retry_lookup = {res.spec.name: res for res in retry_results}
            results = [retry_lookup.get(res.spec.name, res) for res in results]

    summarise_results(results, output_root)
    return 0


def build_run_specs_for_group(
    *,
    group: str,
    config_root: Path,
    output_root: Path,
    pipeline: Optional[str],
    global_notes: str,
    limit: int,
) -> List[RunSpec]:
    if group == "baseline":
        baseline_path = config_root / "baseline_original.yaml"
        if not baseline_path.exists():
            raise FileNotFoundError(f"Missing baseline config: {baseline_path}")
        run_name = "baseline_original"
        output_dir = output_root / "baseline" / run_name
        overrides = {
            "environment.project": f"baseline_{run_name}",
            "environment.output_dir": str(output_dir),
            "environment.notes": collapse_notes("Baseline reference run.", global_notes),
        }
        spec = RunSpec(
            name=run_name,
            group="baseline",
            base_config=baseline_path,
            overrides=overrides,
            hyperparams={},
            output_dir=output_dir,
            pipeline=pipeline,
        )
        return [spec]

    grid_path = config_root / f"{group}_grid.yaml"
    if not grid_path.exists():
        raise FileNotFoundError(f"Missing sweep configuration: {grid_path}")

    with grid_path.open("r", encoding="utf-8") as fp:
        grid_def = yaml.safe_load(fp)

    base_config = (grid_path.parent / grid_def.get("base_config", "")).resolve()
    if not base_config.exists():
        raise FileNotFoundError(f"Base config referenced by {grid_path} not found: {base_config}")

    fixed = grid_def.get("fixed", {})
    parameters = grid_def.get("parameters", {})
    extra_runs = grid_def.get("extra_runs", [])

    parameter_order = list(parameters.keys())
    option_sets: List[List[ParameterOption]] = []
    for param_name in parameter_order:
        option_sets.append(list(expand_parameter_options(param_name, parameters[param_name])))

    combos: Iterable[Tuple[ParameterOption, ...]]
    if option_sets:
        combos = itertools.product(*option_sets)
    else:
        combos = [tuple()]

    run_specs: List[RunSpec] = []
    seen_names: set[str] = set()
    for combo_idx, combo in enumerate(combos):
        hyperparams: Dict[str, Any] = {}
        overrides: Dict[str, Any] = dict(fixed)
        name_parts = [group]
        for option in combo:
            hyperparams[option.key] = option.value
            overrides.update(option.overrides)
            name_parts.append(f"{slug(option.key)}_{slug(option.value)}")
        run_name = "__".join(name_parts)
        if run_name in seen_names:
            run_name = f"{run_name}__{combo_idx}"
        seen_names.add(run_name)

        output_dir = output_root / group / run_name
        overrides.update(
            {
                "environment.project": f"{group}_{run_name}",
                "environment.output_dir": str(output_dir),
                "environment.notes": collapse_notes(
                    describe_hyperparams(group, hyperparams),
                    global_notes,
                ),
            }
        )
        spec = RunSpec(
            name=run_name,
            group=group,
            base_config=base_config,
            overrides=overrides,
            hyperparams=hyperparams,
            output_dir=output_dir,
            pipeline=pipeline,
        )
        run_specs.append(spec)
        if limit and len(run_specs) >= limit:
            break

    for extra in extra_runs:
        name = extra.get("name")
        if not name:
            raise ValueError(f"extra_runs entry in {grid_path} missing name")
        overrides = dict(fixed)
        overrides.update(extra.get("overrides", {}))
        hyperparams = extra.get("hyperparams", {})
        output_dir = output_root / group / name
        overrides.update(
            {
                "environment.project": f"{group}_{name}",
                "environment.output_dir": str(output_dir),
                "environment.notes": collapse_notes(
                    extra.get("description", "Extra run"),
                    global_notes,
                ),
            }
        )
        spec = RunSpec(
            name=name,
            group=group,
            base_config=base_config,
            overrides=overrides,
            hyperparams=hyperparams,
            output_dir=output_dir,
            pipeline=pipeline,
        )
        if limit and len(run_specs) >= limit:
            break
        run_specs.append(spec)

    return run_specs


def expand_parameter_options(parameter: str, definition: Mapping[str, Any]) -> Iterable[ParameterOption]:
    if "values" in definition:
        values = definition.get("values", [])
        apply_paths = definition.get("apply") or []
        for value in values:
            overrides: Dict[str, Any]
            if apply_paths:
                overrides = {path: value for path in apply_paths}
            else:
                overrides = {parameter: value}
            yield ParameterOption(parameter, slug_value(value), overrides, value)
    elif "options" in definition:
        for option in definition["options"]:
            name = option.get("name")
            if not name:
                raise ValueError(f"Parameter {parameter} options entry missing name")
            overrides = option.get("overrides", {})
            value = option.get("value", name)
            yield ParameterOption(parameter, slug(name), overrides, value)
    else:
        raise ValueError(f"Unsupported parameter definition for {parameter}: {definition}")


def slug(value: Any) -> str:
    text = str(value).strip().replace("/", "-").replace(" ", "-")
    return text.replace("__", "_")


def slug_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isclose(value, 0.0):
            return "0"
        magnitude = f"{value:.4g}"  # compact formatting
        return magnitude.replace(".", "p").replace("-", "m")
    return slug(value)


def collapse_notes(*notes: str) -> str:
    parts = [note for note in notes if note]
    return " | ".join(parts)


def describe_hyperparams(group: str, params: Mapping[str, Any]) -> str:
    if not params:
        return f"{group} sweep default configuration"
    items = [f"{key}={value}" for key, value in params.items()]
    return f"{group}: " + ", ".join(items)


def execute_runs(
    run_specs: Sequence[RunSpec],
    *,
    max_parallel: int,
    timeout: Optional[int],
    devices: Optional[str],
    pipeline: Optional[str],
    dry_run: bool,
) -> List[RunResult]:
    results: List[RunResult] = []

    def run_single(spec: RunSpec) -> RunResult:
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        cache_path = build_override(spec.base_config, spec.overrides, spec.name)
        spec.config_path = cache_path
        log_path = spec.output_dir / "train.log"
        extra_env = {"CUDA_VISIBLE_DEVICES": devices} if devices else None
        launch = launch_experiment(
            cache_path,
            log_path=log_path,
            timeout=timeout,
            extra_env=extra_env,
            pipeline=pipeline,
            dry_run=dry_run,
        )
        status = "success"
        if launch.timed_out:
            status = "timeout"
        elif launch.returncode != 0:
            status = "failed"

        log_dir = None
        metrics_info: Dict[str, Any] = {"summary": {}}
        if not dry_run:
            project_name = spec.overrides.get("environment.project")
            log_dir = locate_lightning_log(project_name)
            if log_dir:
                spec.log_dir = log_dir
                metrics_info = load_metrics(log_dir)
                copy_artifacts(log_dir, spec.output_dir)
        export_json(
            {
                "run_name": spec.name,
                "group": spec.group,
                "hyperparams": spec.hyperparams,
                "overrides": spec.overrides,
                "config_path": str(cache_path),
                "log_dir": str(log_dir) if log_dir else None,
                "status": status,
                "returncode": launch.returncode,
                "timed_out": launch.timed_out,
                "runtime_sec": launch.runtime,
                "metrics": metrics_info.get("summary", {}),
            },
            spec.output_dir / "run_summary.json",
        )
        resolved_config_path = spec.output_dir / "resolved_config.yaml"
        if not dry_run:
            shutil.copy(cache_path, resolved_config_path)
        else:
            resolved_config_path.write_text("dry-run: config not generated", encoding="utf-8")
        return RunResult(spec=spec, launch=launch, status=status, metrics=metrics_info, log_dir=log_dir)

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_spec = {executor.submit(run_single, spec): spec for spec in run_specs}
        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - guard against runtime failures
                print(f"[ERROR] Run {spec.name} crashed: {exc}")
                result = RunResult(
                    spec=spec,
                    launch=LaunchResult(returncode=-1, runtime=0.0, error=str(exc)),
                    status="failed",
                    metrics={"summary": {}},
                    log_dir=None,
                )
            results.append(result)
            print(f"[INFO] Completed run {spec.name} with status {result.status}")
    return results


def locate_lightning_log(project_name: Optional[str], retries: int = 5, delay: int = 3) -> Optional[Path]:
    if not project_name:
        return None
    logs_root = PROJECT_ROOT / "save"
    for attempt in range(retries):
        matches: List[Path] = []
        for hparams_path in logs_root.rglob("hparams.yaml"):
            try:
                with hparams_path.open("r", encoding="utf-8") as fp:
                    data = yaml.safe_load(fp)
            except Exception:
                continue
            environment = data.get("environment") if isinstance(data, Mapping) else None
            if isinstance(environment, Mapping) and environment.get("project") == project_name:
                matches.append(hparams_path.parent)
        if matches:
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return matches[0]
        time.sleep(delay)
    return None


def copy_artifacts(log_dir: Path, output_dir: Path) -> None:
    target_dir = output_dir / "lightning_logs"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(log_dir.parent, target_dir)


def summarise_results(results: Sequence[RunResult], output_root: Path) -> None:
    if not results:
        return

    rows: List[Dict[str, Any]] = []
    baseline_summary: Dict[str, float] = {}
    for result in results:
        summary = result.metrics.get("summary", {}) if result.metrics else {}
        if result.spec.group == "baseline" and result.status == "success":
            baseline_summary = summary
        for metric, value in summary.items():
            rows.append(
                {
                    "group": result.spec.group,
                    "run_name": result.spec.name,
                    "metric": metric,
                    "value": value,
                    "status": result.status,
                }
            )
        if not summary:
            rows.append(
                {
                    "group": result.spec.group,
                    "run_name": result.spec.name,
                    "metric": "n/a",
                    "value": float("nan"),
                    "status": result.status,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty and baseline_summary:
        df["baseline_delta"] = df.apply(
            lambda row: row["value"] - baseline_summary.get(row["metric"], math.nan)
            if isinstance(row["value"], (int, float)) and not math.isnan(row["value"])
            else math.nan,
            axis=1,
        )
    summary_csv = output_root / "summary_metrics.csv"
    df.to_csv(summary_csv, index=False)

    summary_md = output_root / "summary_report.md"
    with summary_md.open("w", encoding="utf-8") as fp:
        fp.write("# TSPN Hyperparameter Evaluation Summary\n\n")
        if baseline_summary:
            fp.write("## Baseline Reference\n\n")
            for metric, value in baseline_summary.items():
                fp.write(f"- **{metric}**: {value}\n")
            fp.write("\n")
        for group, group_df in df.groupby("group"):
            fp.write(f"## {group.capitalize()} sweep\n\n")
            try:
                table_repr = group_df.to_markdown(index=False)
            except ImportError:
                table_repr = group_df.to_string(index=False)
            fp.write(table_repr)
            fp.write("\n\n")


if __name__ == "__main__":
    raise SystemExit(main())
