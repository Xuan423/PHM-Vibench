"""CLI entrypoint for orchestrating TSPN HUST ablation experiments."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from queue import SimpleQueue

import pandas as pd
import yaml

from .utils import (
    LaunchResult,
    build_override,
    export_json,
    launch_experiment,
    load_metrics,
)
from .tspn_hparam_eval import parse_device_pool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT_DEFAULT = PROJECT_ROOT / "configs" / "experiments" / "tspn_hust_ablation"
OUTPUT_ROOT_DEFAULT = PROJECT_ROOT / "save" / "hust_ablation"


@dataclass
class Variant:
    name: str
    config: str
    description: str


@dataclass
class RunSpec:
    name: str
    variant: Variant
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
    Variant("contrastive_full", "base.yaml", "Full contrastive configuration (all branches, episodic)."),
    Variant("contrastive_ssqq", "contrastive_ssqq.yaml", "Disable support-query branch (keep support & query self branches)."),
    Variant("contrastive_qs_only", "contrastive_qs.yaml", "Keep only support-query branch."),
    Variant("contrastive_disabled", "contrastive_disabled.yaml", "Remove contrastive loss from optimisation."),
    Variant("no_episode_single_branch", "no_episode.yaml", "Disable episodic sampling and use single branch contrastive."),
    Variant("support_no_align", "support_no_align.yaml", "Keep episodes but drop prototype alignment for support."),
    Variant("baseline_tspn", "baseline_tspn.yaml", "Baseline TSPN without contrastive or episodic sampling."),
]

CONTRASTIVE_WEIGHTED_VARIANTS = {
    "contrastive_full",
    "contrastive_ssqq",
    "contrastive_qs_only",
    "no_episode_single_branch",
    "support_no_align",
}

CONTRASTIVE_LOSS_WEIGHTS: Sequence[float] = (0.2, 0.3)

SOURCE_DOMAIN_COMBINATIONS: Dict[str, List[int]] = {
    "even": [0, 2, 4, 6, 8],
    "odd": [1, 3, 5, 7, 9],
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TSPN HUST ablation experiment orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=CONFIG_ROOT_DEFAULT,
        help="Directory containing ablation configuration YAMLs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT_DEFAULT,
        help="Root directory for ablation outputs.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="*",
        default=None,
        help="Subset of variant names to run (default: all).",
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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    config_root = args.config_root
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    requested = set(args.variants) if args.variants else None
    variants = [variant for variant in VARIANT_REGISTRY if requested is None or variant.name in requested]
    if requested:
        missing = requested - {variant.name for variant in variants}
        if missing:
            raise ValueError(f"Unknown variants requested: {', '.join(sorted(missing))}")

    run_specs = build_run_specs(
        variants=variants,
        config_root=config_root,
        output_root=output_root,
        pipeline=args.pipeline,
        global_notes=args.notes,
    )

    if not run_specs:
        print("[WARN] No run specifications generated; aborting.")
        return 0

    print(f"[INFO] Prepared {len(run_specs)} ablation variants: {', '.join(spec.name for spec in run_specs)}")

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


def build_run_specs(
    *,
    variants: Iterable[Variant],
    config_root: Path,
    output_root: Path,
    pipeline: Optional[str],
    global_notes: str,
) -> List[RunSpec]:
    specs: List[RunSpec] = []
    for variant in variants:
        config_path = (config_root / variant.config).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Variant config not found: {config_path}")

        weight_options: Sequence[Optional[float]]
        if variant.name in CONTRASTIVE_WEIGHTED_VARIANTS:
            weight_options = tuple(CONTRASTIVE_LOSS_WEIGHTS)
        else:
            weight_options = (None,)

        for weight in weight_options:
            for domain_tag, domain_ids in SOURCE_DOMAIN_COMBINATIONS.items():
                run_name = _compose_run_name(variant.name, weight, domain_tag)
                run_output = output_root / run_name
                overrides: Dict[str, Any] = {
                    "environment.project": f"hust_ablation_{run_name}",
                    "environment.output_dir": str(run_output),
                    "environment.notes": _collapse_notes(variant.description, global_notes),
                    "task.metadata.run_name": run_name,
                    "task.metadata.base_variant": variant.name,
                    "task.metadata.source_domain_tag": domain_tag,
                    "task.metadata.source_domain_ids": ",".join(str(i) for i in domain_ids),
                    "task.source_domain_id": domain_ids,
                }

                if weight is not None:
                    overrides["task.metadata.contrastive_loss_weight"] = weight
                    overrides["task.contrastive.loss_weight"] = weight
                    overrides["model.contrastive.loss_weight"] = weight

                cached_config_path = build_override(config_path, overrides, run_name)
                metadata = _extract_metadata(cached_config_path)
                metadata.setdefault("variant_label", variant.name)
                metadata.setdefault("base_variant", variant.name)
                if weight is not None:
                    metadata.setdefault("contrastive_loss_weight", weight)
                metadata.setdefault("source_domain_tag", domain_tag)
                metadata.setdefault("source_domain_ids", ",".join(str(i) for i in domain_ids))

                specs.append(
                    RunSpec(
                        name=run_name,
                        variant=variant,
                        base_config=config_path,
                        overrides=overrides,
                        output_dir=run_output,
                        metadata=metadata,
                        pipeline=pipeline,
                        config_path=cached_config_path,
                    )
                )
    return specs


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
        cache_path = spec.config_path or build_override(spec.base_config, spec.overrides, spec.name)
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
            print(f"[INFO] Completed {spec.name} with status {results[-1].status}")
        return results

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_spec = {executor.submit(run_single, spec): spec for spec in run_specs}
        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"[ERROR] Run {spec.name} crashed: {exc}")
                result = RunResult(
                    spec=spec,
                    launch=LaunchResult(returncode=-1, runtime=0.0, error=str(exc)),
                    status="failed",
                    metrics={"summary": {}},
                    log_dir=None,
                )
            results.append(result)
            print(f"[INFO] Completed {spec.name} with status {result.status}")
    return results


def summarise_results(results: Sequence[RunResult], output_root: Path) -> None:
    if not results:
        return

    rows: List[Dict[str, Any]] = []
    for result in results:
        metadata = result.spec.metadata
        summary = result.metrics.get("summary", {}) if result.metrics else {}
        row: Dict[str, Any] = {
            "variant": result.spec.variant.name,
            "status": result.status,
            "runtime_sec": result.launch.runtime,
            "config_path": str(result.spec.config_path) if result.spec.config_path else "",
        }
        for key, value in metadata.items():
            row[f"meta_{key}"] = value

        test_metrics = {k: v for k, v in summary.items() if k.lower().startswith("test")}
        if not test_metrics:
            # Fallback: include highest-level metric if available
            for candidate in ("val_acc", "test_acc"):
                if candidate in summary:
                    test_metrics[candidate] = summary[candidate]
        for key, value in test_metrics.items():
            row[key] = value

        rows.append(row)

    df = pd.DataFrame(rows)
    summary_csv = output_root / "ablation_summary.csv"
    df.to_csv(summary_csv, index=False)

    summary_md = output_root / "ablation_summary.md"
    try:
        markdown = df.to_markdown(index=False)
    except ImportError:
        header = " | ".join(df.columns)
        separator = " | ".join("---" for _ in df.columns)
        body = "\n".join(" | ".join(map(str, row)) for row in df.to_numpy())
        markdown = "\n".join([header, separator, body])
    summary_md.write_text(markdown, encoding="utf-8")


def _compose_run_name(variant_name: str, weight: Optional[float], domain_tag: str) -> str:
    parts = [variant_name]
    if weight is not None:
        weight_str = str(weight).replace(".", "p")
        parts.append(f"lw{weight_str}")
    parts.append(f"src_{domain_tag}")
    return "__".join(parts)


def _collapse_notes(*notes: str) -> str:
    return " | ".join(note for note in notes if note)


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
    if "active_branches" not in metadata:
        branches = _deep_get(config_dict, ["task", "contrastive", "active_branches"], default=[])
        if isinstance(branches, list):
            metadata["active_branches"] = ",".join(str(b) for b in branches)
        elif isinstance(branches, str):
            metadata["active_branches"] = branches
    return metadata


def _deep_get(mapping: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cursor: Any = mapping
    for key in keys:
        if isinstance(cursor, Mapping) and key in cursor:
            cursor = cursor[key]
        else:
            return default
    return cursor


if __name__ == "__main__":
    raise SystemExit(main())
