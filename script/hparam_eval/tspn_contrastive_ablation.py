"""CLI entrypoint for unified TSPN contrastive ablation experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import yaml

from .contrastive_ablation_registry import DatasetSpec, DomainTask, get_dataset_spec, list_dataset_names
from .utils import LaunchResult, build_override, export_json, launch_experiment, load_metrics
from .tspn_hparam_eval import parse_device_pool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "save" / "contrastive_ablation"


@dataclass(frozen=True)
class Variant:
    name: str
    config: str
    description: str
    requires_weight_sweep: bool = True


@dataclass
class RunSpec:
    dataset: DatasetSpec
    domain_task: DomainTask
    variant: Variant
    weight: Optional[float]
    base_config: Path
    overrides: Dict[str, Any]
    output_dir: Path
    metadata: Dict[str, Any]
    pipeline: Optional[str] = None
    config_path: Optional[Path] = None
    log_dir: Optional[Path] = None


@dataclass
class RunResult:
    spec: RunSpec
    launch: LaunchResult
    status: str
    metrics: Dict[str, Any]
    log_dir: Optional[Path]


VARIANT_REGISTRY: List[Variant] = [
    Variant("contrastive_full", "base.yaml", "All contrastive branches active with episodic sampling.", True),
    Variant("contrastive_ssqq", "contrastive_ssqq.yaml", "Disable support-query branch; keep support/query self branches.", True),
    Variant("contrastive_ss_only", "contrastive_ss_only.yaml", "Only support-support branch participates.", True),
    Variant("contrastive_qq_only", "contrastive_qq_only.yaml", "Only query-query branch participates.", True),
    Variant("contrastive_qs_only", "contrastive_qs.yaml", "Keep only the support-query contrastive branch.", True),
    Variant("support_no_align", "support_no_align.yaml", "Disable support prototype alignment and CE support loss.", True),
    Variant("contrastive_disabled", "contrastive_disabled.yaml", "Remove contrastive loss from optimisation.", False),
    Variant("baseline_tspn", "baseline_tspn.yaml", "Baseline TSPN without episodic contrastive sampling.", False),
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified contrastive ablation experiment orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help=f"Datasets to launch (default: all). Choices: {', '.join(list_dataset_names())}",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="*",
        default=None,
        help="Subset of variant names to run (default: all).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for ablation outputs.",
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
        "--devices",
        type=str,
        default=None,
        help="Comma separated CUDA device list forwarded as CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--device-pool",
        type=str,
        default=None,
        help="Optional comma separated list of devices to pin one run per GPU.",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="Pipeline_01_default",
        help="Pipeline module to pass into main.py.",
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
    parser.add_argument(
        "--resume-failed",
        action="store_true",
        help="Run only specifications whose previous run_summary does not report success.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    dataset_names = args.datasets or list_dataset_names()
    dataset_specs = [_resolve_dataset(name) for name in dataset_names]

    variants = _filter_variants(args.variants)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    run_specs = build_run_specs(
        datasets=dataset_specs,
        variants=variants,
        output_root=output_root,
        pipeline=args.pipeline,
        global_notes=args.notes,
    )

    if args.resume_failed:
        run_specs, skipped = _filter_failed_specs(run_specs)
        print(f"[INFO] Resume mode active: skipped {skipped} completed runs.")
        if not run_specs:
            print("[INFO] No failed or pending runs remain; exiting.")
            return 0

    if not run_specs:
        print("[WARN] No run specifications generated; aborting.")
        return 0

    print(f"[INFO] Prepared {len(run_specs)} runs:")
    for spec in run_specs:
        print(f"  - {spec.metadata.get('run_name')}")

    device_pool = parse_device_pool(args.device_pool)

    results = execute_runs(
        run_specs,
        max_parallel=max(1, args.max_parallel),
        timeout=args.timeout if args.timeout > 0 else None,
        devices=args.devices,
        device_pool=device_pool,
        pipeline=args.pipeline,
        dry_run=args.dry_run,
    )

    summarise_results(results, output_root)
    return 0


def _resolve_dataset(name: str) -> DatasetSpec:
    try:
        return get_dataset_spec(name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _filter_variants(requested: Optional[Sequence[str]]) -> List[Variant]:
    registry = {variant.name: variant for variant in VARIANT_REGISTRY}
    if not requested:
        return list(registry.values())
    selected: List[Variant] = []
    missing = []
    for name in requested:
        key = name.strip()
        variant = registry.get(key)
        if variant is None:
            missing.append(key)
        else:
            selected.append(variant)
    if missing:
        raise SystemExit(f"Unknown variants requested: {', '.join(sorted(missing))}")
    return selected


def build_run_specs(
    *,
    datasets: Sequence[DatasetSpec],
    variants: Sequence[Variant],
    output_root: Path,
    pipeline: Optional[str],
    global_notes: str,
) -> List[RunSpec]:
    specs: List[RunSpec] = []
    for dataset in datasets:
        for variant in variants:
            config_path = (dataset.config_root / variant.config).resolve()
            if not config_path.exists():
                raise FileNotFoundError(f"Variant config not found: {config_path}")

            weight_options = dataset.contrastive_weights if variant.requires_weight_sweep else (None,)

            for domain_task in dataset.domain_tasks:
                for weight in weight_options:
                    run_name = _compose_run_name(dataset.name, variant.name, domain_task.label, weight)
                    run_output = output_root / dataset.name / run_name
                    overrides = _build_overrides(
                        dataset=dataset,
                        domain_task=domain_task,
                        variant=variant,
                        weight=weight,
                        run_name=run_name,
                        output_dir=run_output,
                        global_notes=global_notes,
                    )
                    cached_config_path = build_override(config_path, overrides, run_name)
                    metadata = _extract_metadata(cached_config_path)
                    metadata.update(
                        {
                            "dataset": dataset.name,
                            "variant": variant.name,
                            "domain_label": domain_task.label,
                            "source_domain_id": ",".join(map(str, domain_task.source_domain_id)),
                            "target_domain_id": ",".join(map(str, domain_task.target_domain_id)),
                            "run_name": run_name,
                        }
                    )
                    if weight is not None:
                        metadata.setdefault("contrastive_loss_weight", weight)

                    specs.append(
                        RunSpec(
                            dataset=dataset,
                            domain_task=domain_task,
                            variant=variant,
                            weight=weight,
                            base_config=config_path,
                            overrides=overrides,
                            output_dir=run_output,
                            metadata=metadata,
                            pipeline=pipeline,
                            config_path=cached_config_path,
                        )
                    )
    return specs


def _build_overrides(
    *,
    dataset: DatasetSpec,
    domain_task: DomainTask,
    variant: Variant,
    weight: Optional[float],
    run_name: str,
    output_dir: Path,
    global_notes: str,
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {
        "environment.project": f"{dataset.name}_contrastive_ablation_{run_name}",
        "environment.output_dir": str(output_dir),
        "environment.notes": _collapse_notes(dataset.notes, variant.description, global_notes),
        "task.metadata.run_name": run_name,
        "task.metadata.dataset": dataset.name,
        "task.metadata.variant_name": variant.name,
        "task.metadata.domain_label": domain_task.label,
        "task.metadata.source_domain_id": ",".join(map(str, domain_task.source_domain_id)),
        "task.metadata.target_domain_id": ",".join(map(str, domain_task.target_domain_id)),
        "task.source_domain_id": list(domain_task.source_domain_id),
        "task.target_domain_id": list(domain_task.target_domain_id),
    }
    if weight is not None:
        overrides["task.contrastive.loss_weight"] = weight
        overrides["model.contrastive.loss_weight"] = weight
        overrides["task.metadata.contrastive_loss_weight"] = weight
    if dataset.target_system_id:
        overrides["task.target_system_id"] = list(dataset.target_system_id)
        overrides["task.metadata.target_system_id"] = ",".join(map(str, dataset.target_system_id))
    if dataset.few_shot_overrides:
        overrides["task.few_shot"] = dict(dataset.few_shot_overrides)
    return overrides


def execute_runs(
    run_specs: Sequence[RunSpec],
    *,
    max_parallel: int,
    timeout: Optional[int],
    devices: Optional[str],
    device_pool: Optional[Sequence[str]],
    pipeline: Optional[str],
    dry_run: bool,
) -> List[RunResult]:
    results: List[RunResult] = []

    pool: Optional[SimpleQueue]
    if device_pool:
        pool = SimpleQueue()
        for token in device_pool:
            pool.put(token)
    else:
        pool = None

    def run_single(spec: RunSpec) -> RunResult:
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        cache_path = spec.config_path or build_override(spec.base_config, spec.overrides, spec.metadata["run_name"])
        spec.config_path = cache_path
        log_path = spec.output_dir / "train.log"
        device_token: Optional[str] = None
        if pool:
            device_token = pool.get()
        try:
            extra_env = None
            if device_token is not None:
                extra_env = {"CUDA_VISIBLE_DEVICES": device_token}
            elif devices:
                extra_env = {"CUDA_VISIBLE_DEVICES": devices}
            launch = launch_experiment(
                cache_path,
                log_path=log_path,
                timeout=timeout,
                extra_env=extra_env,
                pipeline=pipeline,
                dry_run=dry_run,
            )
        finally:
            if pool and device_token is not None:
                pool.put(device_token)
        status = "success"
        if launch.timed_out:
            status = "timeout"
        elif launch.returncode != 0:
            status = "failed"

        log_dir = None
        metrics_info: Dict[str, Any] = {"summary": {}}
        if not dry_run:
            project_name = spec.overrides.get("environment.project")
            log_dir = _locate_lightning_log(project_name)
            if log_dir:
                spec.log_dir = log_dir
                metrics_info = load_metrics(log_dir)
        export_json(
            {
                "dataset": spec.dataset.name,
                "variant": spec.variant.name,
                "metadata": spec.metadata,
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
        if not dry_run and cache_path.exists():
            resolved_config_path.write_text(Path(cache_path).read_text(encoding="utf-8"), encoding="utf-8")
        else:
            resolved_config_path.write_text("dry-run: config not generated", encoding="utf-8")
        return RunResult(spec=spec, launch=launch, status=status, metrics=metrics_info, log_dir=log_dir)

    if max_parallel <= 1:
        for spec in run_specs:
            results.append(run_single(spec))
            print(f"[INFO] Completed {spec.metadata.get('run_name')} with status {results[-1].status}")
        return results

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_spec = {executor.submit(run_single, spec): spec for spec in run_specs}
        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - safety net
                print(f"[ERROR] Run {spec.metadata.get('run_name')} crashed: {exc}")
                result = RunResult(
                    spec=spec,
                    launch=LaunchResult(returncode=-1, runtime=0.0, error=str(exc)),
                    status="failed",
                    metrics={"summary": {}},
                    log_dir=None,
                )
            results.append(result)
            print(f"[INFO] Completed {spec.metadata.get('run_name')} with status {result.status}")
    return results


def summarise_results(results: Sequence[RunResult], output_root: Path) -> None:
    if not results:
        return

    rows: List[Dict[str, Any]] = []
    for result in results:
        metadata = result.spec.metadata
        summary = result.metrics.get("summary", {}) if result.metrics else {}
        row: Dict[str, Any] = {
            "dataset": result.spec.dataset.name,
            "variant": result.spec.variant.name,
            "status": result.status,
            "runtime_sec": result.launch.runtime,
            "config_path": str(result.spec.config_path) if result.spec.config_path else "",
        }
        for key, value in metadata.items():
            row[f"meta_{key}"] = value

        accuracy_metrics = _extract_accuracy_metrics(summary)
        if not accuracy_metrics:
            print(
                f"[WARN] No accuracy metrics found for run {metadata.get('run_name')}; "
                "summary row will omit accuracy columns."
            )
        for key, value in accuracy_metrics.items():
            row[key] = value

        rows.append(row)

    df = pd.DataFrame(rows)
    dataset_output = output_root
    dataset_output.mkdir(parents=True, exist_ok=True)
    summary_csv = dataset_output / "ablation_summary.csv"
    df.to_csv(summary_csv, index=False)

    summary_md = dataset_output / "ablation_summary.md"
    try:
        markdown = df.to_markdown(index=False)
    except ImportError:
        header = " | ".join(df.columns)
        separator = " | ".join("---" for _ in df.columns)
        body = "\n".join(" | ".join(map(str, row)) for row in df.to_numpy())
        markdown = "\n".join([header, separator, body])
    summary_md.write_text(markdown, encoding="utf-8")


def _locate_lightning_log(project_name: Optional[str]) -> Optional[Path]:
    if not project_name:
        return None
    logs_root = PROJECT_ROOT / "save"
    matches: List[Path] = []
    for hparams_path in logs_root.rglob("hparams.yaml"):
        try:
            with hparams_path.open("r", encoding="utf-8") as fp:
                data = yaml.safe_load(fp) or {}
        except Exception:
            continue
        environment = data.get("environment", {})
        if isinstance(environment, Mapping) and environment.get("project") == project_name:
            matches.append(hparams_path.parent)
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _extract_metadata(resolved_config: Path) -> Dict[str, Any]:
    try:
        with resolved_config.open("r", encoding="utf-8") as fp:
            config_dict = yaml.safe_load(fp) or {}
    except FileNotFoundError:
        return {}
    metadata = _deep_get(config_dict, ["task", "metadata"], default={})
    if not isinstance(metadata, dict):
        metadata = {}
    branches = _deep_get(config_dict, ["task", "contrastive", "branches"], default={})
    if isinstance(branches, dict):
        metadata.setdefault(
            "branch_participation",
            ",".join(
                f"{name}:{'on' if branch_cfg.get('participates', True) else 'skip'}"
                for name, branch_cfg in sorted(branches.items())
                if isinstance(branch_cfg, dict)
            ),
        )
    weighting = _deep_get(config_dict, ["task", "contrastive", "weighting", "strategy"], default=None)
    if weighting:
        metadata.setdefault("contrastive_weighting_strategy", weighting)
    return metadata


def _deep_get(mapping: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cursor: Any = mapping
    for key in keys:
        if isinstance(cursor, Mapping) and key in cursor:
            cursor = cursor[key]
        else:
            return default
    return cursor


def _compose_run_name(dataset: str, variant: str, domain_label: str, weight: Optional[float]) -> str:
    parts = [dataset, variant, domain_label]
    if weight is not None:
        weight_str = str(weight).replace(".", "p")
        parts.append(f"lw{weight_str}")
    return "__".join(parts)


def _collapse_notes(*notes: str) -> str:
    return " | ".join(note for note in notes if note)


def _extract_accuracy_metrics(summary: Mapping[str, Any]) -> Dict[str, Any]:
    accuracy_keys = [
        key
        for key in summary.keys()
        if key.lower().startswith("test_acc") or key.lower().endswith("_accuracy") or key.lower() == "test_accuracy"
    ]
    metrics: Dict[str, Any] = {}
    for key in accuracy_keys:
        metrics[key] = summary[key]
    # Fallback to generic accuracy fields
    for fallback in ("test_acc", "val_acc"):
        if fallback in summary and fallback not in metrics:
            metrics[fallback] = summary[fallback]
    return metrics


def _load_run_status(output_dir: Path) -> Optional[str]:
    summary_path = output_dir / "run_summary.json"
    if not summary_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    status = data.get("status")
    if isinstance(status, str):
        return status
    return None


def _filter_failed_specs(run_specs: Sequence[RunSpec]) -> Tuple[List[RunSpec], int]:
    filtered: List[RunSpec] = []
    skipped = 0
    for spec in run_specs:
        status = _load_run_status(spec.output_dir)
        if status and status.lower() == "success":
            skipped += 1
            continue
        filtered.append(spec)
    return filtered, skipped


if __name__ == "__main__":
    raise SystemExit(main())
