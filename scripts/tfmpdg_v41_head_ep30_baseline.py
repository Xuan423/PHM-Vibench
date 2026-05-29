from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


CONFIG = "configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg.yaml"


@dataclass(frozen=True)
class TargetTask:
    name: str
    source_domains: tuple[int, ...]
    target_domains: tuple[int, ...]


TARGETS: dict[str, TargetTask] = {
    "t0": TargetTask("sys27_t0", (1, 2), (0,)),
    "t1": TargetTask("sys27_t1", (0, 2), (1,)),
    "t2": TargetTask("sys27_t2", (0, 1), (2,)),
}


def _override(key: str, value: Any) -> str:
    if isinstance(value, bool):
        encoded = "true" if value else "false"
    elif isinstance(value, str):
        encoded = json.dumps(value, ensure_ascii=False)
    elif isinstance(value, (list, tuple)):
        encoded = "[" + ",".join(str(item) for item in value) + "]"
    else:
        encoded = str(value)
    return f"{key}={encoded}"


def _command(
    python_bin: str,
    target: TargetTask,
    output_dir: Path,
    epochs: int,
    iterations: int,
    seed: int,
) -> list[str]:
    overrides = [
        _override("environment.seed", seed),
        _override("environment.iterations", iterations),
        _override("environment.output_dir", str(output_dir)),
        _override("environment.notes", f"tfmpdg_v41_head_ep{epochs}_{target.name}"),
        _override("task.target_system_id", [27]),
        _override("task.source_domain_id", list(target.source_domains)),
        _override("task.target_domain_id", list(target.target_domains)),
        _override("data.num_workers", 0),
        _override("data.pin_memory", False),
        _override("trainer.num_workers", 0),
        _override("trainer.pin_memory", False),
        _override("trainer.monitor", "val_total_loss"),
        _override("trainer.num_epochs", epochs),
        _override("trainer.patience", epochs),
        _override("trainer.save_last", False),
    ]
    command = [python_bin, "main.py", "--config", CONFIG]
    for override in overrides:
        command.extend(["--override", override])
    return command


def _read_hparams(path: Path) -> dict[str, Any]:
    hparams_paths = sorted(path.glob("**/hparams.yaml"))
    if not hparams_paths:
        return {}
    hparams_path = hparams_paths[0]
    try:
        with hparams_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
            return dict(data)
    except Exception:
        return {}


def _as_float(row: pd.Series, key: str) -> float | None:
    if key not in row:
        return None
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return None


def _first_metric(row: pd.Series, keys: tuple[str, ...], prefix: str | None = None) -> float | None:
    for key in keys:
        value = _as_float(row, key)
        if value is not None:
            return value
    if prefix is not None:
        for key in row.index:
            if str(key).startswith(prefix):
                value = _as_float(row, str(key))
                if value is not None:
                    return value
    return None


def summarize(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(root.glob("sys27_t*/**/test_result_*.csv")):
        parts = result_path.relative_to(root).parts
        if not parts:
            continue
        target_name = parts[0]
        frame = pd.read_csv(result_path)
        if frame.empty:
            continue
        row = frame.iloc[0]
        hparams = _read_hparams(result_path.parent)
        rows.append(
            {
                "target": target_name,
                "seed": hparams.get("current_seed", hparams.get("seed")),
                "logger_name": hparams.get("logger_name"),
                "acc": _first_metric(row, ("acc", "test_acc"), prefix="test_acc_"),
                "anchor_acc": _as_float(row, "test_decision_anchor_acc"),
                "residual_acc": _first_metric(
                    row,
                    ("test_residual_logits_acc", "test_decision_residual_acc"),
                ),
                "proto_nce": _as_float(row, "test_proto_nce"),
                "total_loss": _as_float(row, "test_total_loss"),
                "best_val_total_loss": _as_float(row, "best_val_total_loss"),
                "assignment_ratio": _as_float(row, "test_assignment_ratio"),
                "best_val_assignment_ratio": _as_float(row, "best_val_assignment_ratio"),
                "result_path": str(result_path),
            }
        )
    seed_frame = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    if not seed_frame.empty:
        for target_name, group in seed_frame.groupby("target"):
            summary_rows.append(
                {
                    "target": target_name,
                    "n": int(len(group)),
                    "mean_acc": float(group["acc"].mean()),
                    "min_acc": float(group["acc"].min()),
                    "std_acc": float(group["acc"].std(ddof=0)),
                    "mean_anchor_acc": float(group["anchor_acc"].mean()),
                    "mean_proto_nce": float(group["proto_nce"].mean()),
                    "max_proto_nce": float(group["proto_nce"].max()),
                    "mean_total_loss": float(group["total_loss"].mean()),
                }
            )
    summary_frame = pd.DataFrame(summary_rows)
    seed_frame.to_csv(root / "seed_results.csv", index=False)
    summary_frame.to_csv(root / "summary.csv", index=False)
    return seed_frame, summary_frame


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the pristine tf_multi_proto_dg_v4.1 HEAD config for 30-epoch cross checks."
    )
    parser.add_argument(
        "--repo",
        default="/home/xuanli/work/PHM-Vibench_v41_clean",
        help="Pristine v4.1 worktree path.",
    )
    parser.add_argument(
        "--root",
        default="results/tmp/tfmpdg_v41_head_ep30_baseline",
        help="Output root, relative paths are resolved in the current repository.",
    )
    parser.add_argument("--targets", nargs="+", default=["t0", "t1", "t2"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--python",
        default="/home/xuanli/miniforge/envs/phmbench/bin/python",
        help="Python executable for training.",
    )
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if not (repo / "main.py").exists():
        raise FileNotFoundError(f"Cannot find main.py under v4.1 repo: {repo}")
    root = Path(args.root)
    if not root.is_absolute():
        root = Path.cwd() / root
    root.mkdir(parents=True, exist_ok=True)

    if not args.summary_only:
        for target_key in args.targets:
            if target_key not in TARGETS:
                raise KeyError(f"Unknown target {target_key!r}. Expected one of {sorted(TARGETS)}.")
            target = TARGETS[target_key]
            output_dir = root / target.name
            output_dir.mkdir(parents=True, exist_ok=True)
            command = _command(
                python_bin=args.python,
                target=target,
                output_dir=output_dir,
                epochs=args.epochs,
                iterations=args.iterations,
                seed=args.seed,
            )
            print(" ".join(command), flush=True)
            if args.dry_run:
                continue
            completed = subprocess.run(command, cwd=repo, check=False)
            if completed.returncode != 0:
                return completed.returncode

    _, summary_frame = summarize(root)
    if summary_frame.empty:
        print(f"No results found under {root}.")
    else:
        print(summary_frame.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
