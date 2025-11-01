"""Entry point for resuming incomplete TSPN hyperparameter evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .tspn_hparam_eval import (
    CONFIG_ROOT_DEFAULT,
    OUTPUT_ROOT_DEFAULT,
    RunSpec,
    build_run_specs,
    execute_runs,
    parse_device_pool,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume incomplete TSPN sweep runs.")
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
        "--grid",
        type=Path,
        default=None,
        help="Optional explicit path to the sweep grid YAML. Defaults to config-root/contrastive_grid.yaml.",
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
        help="Retry failed runs once after the resume sweep completes.",
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
        "--limit",
        type=int,
        default=0,
        help="Optional cap on the number of generated configs per sweep.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned resume actions without executing experiments.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Additional notes appended to environment.notes for each resumed run.",
    )
    return parser.parse_args(argv)


def load_run_record(summary_path: Path) -> Optional[Dict[str, Any]]:
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    config_root: Path = args.config_root
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    grid_path = args.grid if args.grid else (config_root / "contrastive_grid.yaml")
    run_specs = build_run_specs(
        grid_path=grid_path,
        output_root=output_root,
        pipeline=args.pipeline,
        global_notes=args.notes,
        limit=args.limit,
    )

    if not run_specs:
        print(f"[WARN] No run specifications generated from grid {grid_path}; aborting.")
        return 0

    completed = 0
    success_names: list[str] = []
    todo_specs: list[RunSpec] = []

    for spec in run_specs:
        record = load_run_record(spec.output_dir / "run_summary.json")
        if record and record.get("status") == "success":
            completed += 1
            success_names.append(spec.name)
        else:
            todo_specs.append(spec)

    total = len(run_specs)
    print(f"[INFO] Loaded {total} run specifications from {grid_path}")
    print(f"[INFO] Detected {completed} successful runs and {len(todo_specs)} incomplete runs")

    if args.dry_run:
        if todo_specs:
            print("[DRY-RUN] Pending runs:")
            for spec in todo_specs:
                print(f"  - {spec.name}")
        else:
            print("[DRY-RUN] No runs require resuming.")
        return 0

    if not todo_specs:
        print("[INFO] All runs already completed; nothing to resume.")
        print(
            "[INFO] To regenerate consolidated metrics (for example test_acc per system), "
            "run the standalone summariser:\n"
            f"       python script/hparam_eval/tspn_resummarise.py --root {output_root}"
        )
        return 0

    device_pool = parse_device_pool(args.device_pool)

    results = execute_runs(
        todo_specs,
        max_parallel=max(1, args.max_parallel),
        timeout=args.timeout if args.timeout > 0 else None,
        devices=args.devices,
        device_pool=device_pool,
        pipeline=args.pipeline,
        dry_run=args.dry_run,
    )

    if args.rerun_failed:
        failed_specs = [res.spec for res in results if res.status == "failed"]
        if failed_specs:
            print(f"[INFO] Retrying {len(failed_specs)} failed runs once more...")
            retry_results = execute_runs(
                failed_specs,
                max_parallel=max(1, args.max_parallel),
                timeout=args.timeout if args.timeout > 0 else None,
                devices=args.devices,
                device_pool=device_pool,
                pipeline=args.pipeline,
                dry_run=args.dry_run,
            )
            retry_lookup = {res.spec.name: res for res in retry_results}
            results = [retry_lookup.get(res.spec.name, res) for res in results]

    if not args.dry_run:
        print(
            "[INFO] Resume pass complete. Rebuild the aggregate table with:\n"
            f"       python script/hparam_eval/tspn_resummarise.py --root {output_root}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
