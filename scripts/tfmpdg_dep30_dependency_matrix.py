from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg.yaml"


@dataclass(frozen=True)
class Condition:
    name: str
    local_mode: str
    proto_weight: float
    proto_tau: float
    weight_decay: float


@dataclass(frozen=True)
class TargetTask:
    name: str
    source_domains: tuple[int, ...]
    target_domains: tuple[int, ...]


CONDITIONS: tuple[Condition, ...] = (
    Condition(
        name="off_harsh_wd0003",
        local_mode="off",
        proto_weight=0.1,
        proto_tau=0.2,
        weight_decay=0.0003,
    ),
    Condition(
        name="mlp_harsh_wd0003",
        local_mode="mlp",
        proto_weight=0.1,
        proto_tau=0.2,
        weight_decay=0.0003,
    ),
    Condition(
        name="off_soft_wd001",
        local_mode="off",
        proto_weight=0.05,
        proto_tau=0.3,
        weight_decay=0.001,
    ),
    Condition(
        name="mlp_soft_wd001",
        local_mode="mlp",
        proto_weight=0.05,
        proto_tau=0.3,
        weight_decay=0.001,
    ),
)

TARGETS: tuple[TargetTask, ...] = (
    TargetTask(name="sys27_t0", source_domains=(1, 2), target_domains=(0,)),
    TargetTask(name="sys27_t1", source_domains=(0, 2), target_domains=(1,)),
)


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


def _command(condition: Condition, target: TargetTask, output_dir: Path) -> list[str]:
    overrides = [
        _override("environment.seed", 42),
        _override("environment.iterations", 5),
        _override("environment.output_dir", str(output_dir)),
        _override("environment.notes", f"tfmpdg_dep30_{condition.name}_{target.name}"),
        _override("task.target_system_id", [27]),
        _override("task.source_domain_id", list(target.source_domains)),
        _override("task.target_domain_id", list(target.target_domains)),
        _override("task.weight_decay", condition.weight_decay),
        _override("model.local_anomaly_residual_mode", condition.local_mode),
        _override("model.proto_contrastive_weight", condition.proto_weight),
        _override("model.proto_contrastive_temperature", condition.proto_tau),
        _override("model.prototype_residual_logit_mode", "agreement"),
        _override("model.prototype_concept_input", "h"),
        _override("model.prototype_assignment_input", "concept"),
        _override("data.num_workers", 0),
        _override("data.pin_memory", False),
        _override("trainer.num_workers", 0),
        _override("trainer.pin_memory", False),
        _override("trainer.monitor", "val_total_loss"),
        _override("trainer.num_epochs", 30),
        _override("trainer.patience", 30),
        _override("trainer.save_last", False),
    ]
    command = [sys.executable, "main.py", "--config", CONFIG]
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
        parsed: dict[str, Any] = {}
        for line in hparams_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ":" not in line or line.startswith(" "):
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key in {"seed", "current_seed", "logger_name"}:
                try:
                    parsed[key] = int(value)
                except ValueError:
                    parsed[key] = value
        return parsed


def _as_float(row: pd.Series, key: str) -> float | None:
    if key not in row:
        return None
    value = row[key]
    try:
        return float(value)
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


def summarize(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(root.glob("*/sys27_t*/**/test_result_*.csv")):
        parts = result_path.relative_to(root).parts
        if len(parts) < 2:
            continue
        condition_name, target_name = parts[0], parts[1]
        frame = pd.read_csv(result_path)
        if frame.empty:
            continue
        row = frame.iloc[0]
        hparams = _read_hparams(result_path.parent)
        rows.append(
            {
                "condition": condition_name,
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
            }
        )

    summaries: list[dict[str, Any]] = []
    if rows:
        frame = pd.DataFrame(rows)
        for (condition_name, target_name), group in frame.groupby(["condition", "target"]):
            summaries.append(
                {
                    "condition": condition_name,
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
        frame.to_csv(root / "seed_results.csv", index=False)
        pd.DataFrame(summaries).to_csv(root / "summary.csv", index=False)
    return rows, summaries


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run 30-epoch TF_MultiProtoDG dependency matrix sequentially."
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Output root. Defaults to results/tmp/tfmpdg_dep30_dependency_matrix_<timestamp>.",
    )
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = REPO_ROOT / "results" / "tmp" / f"tfmpdg_dep30_dependency_matrix_{stamp}"
    else:
        root = Path(args.root)
        if not root.is_absolute():
            root = REPO_ROOT / root
    root.mkdir(parents=True, exist_ok=True)

    plan = []
    for condition in CONDITIONS:
        for target in TARGETS:
            output_dir = root / condition.name / target.name
            plan.append(
                {
                    "condition": condition.__dict__,
                    "target": target.__dict__,
                    "output_dir": str(output_dir),
                    "command": _command(condition, target, output_dir),
                }
            )
    (root / "manifest.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.dry_run:
        for item in plan:
            print(" ".join(item["command"]))
        return 0

    if not args.summary_only:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        for index, item in enumerate(plan, start=1):
            condition_name = item["condition"]["name"]
            target_name = item["target"]["name"]
            print(f"[dep30] {index}/{len(plan)} START {condition_name} {target_name}", flush=True)
            result = subprocess.run(
                item["command"],
                cwd=REPO_ROOT,
                env=env,
                check=False,
            )
            print(
                f"[dep30] {index}/{len(plan)} END {condition_name} {target_name} "
                f"returncode={result.returncode}",
                flush=True,
            )
            if result.returncode != 0:
                return int(result.returncode)

    rows, summaries = summarize(root)
    print(f"[dep30] root={root}", flush=True)
    print(f"[dep30] seed_rows={len(rows)} summary_rows={len(summaries)}", flush=True)
    if summaries:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
