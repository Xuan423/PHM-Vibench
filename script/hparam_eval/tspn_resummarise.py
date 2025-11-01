"""Rebuild TSPN hyperparameter sweep summaries from existing artefacts.

The original sweep runner expected Lightning to log plain ``test_acc`` metrics,
but real runs store system-qualified names instead (for example
``test_acc_RM_032_SDUST``). This CLI scans a directory tree of prior runs
(``iter_*`` folders containing ``hparams.yaml`` / ``metrics.csv`` /
``test_result_*.csv`` / ``train.log``) and consolidates the information into a
single table without relaunching training jobs.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_METRIC_PATTERN = re.compile(
    r"(?P<name>test_acc[\w\-/\.]*)\s*(?:=|:)\s*(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?P<percent>%?)"
)


@dataclass
class IterationSummary:
    """Aggregate of hyperparameters and evaluation metrics for one iteration."""

    run_dir: Path
    iteration: str
    hyperparams: Dict[str, Any]
    metrics: Dict[str, float]


class HparamsLoader(yaml.SafeLoader):
    """Collapse ConfigWrapper dumps into simple dictionaries."""


def _config_wrapper_constructor(loader: yaml.Loader, node: yaml.Node) -> MutableMapping[str, Any]:
    mapping: MutableMapping[str, Any] = loader.construct_mapping(node, deep=True)
    state = mapping.get("state")
    if isinstance(state, Mapping):
        return dict(state)
    return dict(mapping)


HparamsLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:src.configs.config_utils.ConfigWrapper",
    _config_wrapper_constructor,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate TSPN sweep metrics from existing logs.")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "save",
        help="Root directory containing run folders (defaults to <repo>/save).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path (defaults to <root>/tspn_resummary.csv).",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Optional Markdown table output path.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Substring filter applied to run directory names.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Limit the number of run directories processed (0 disables).",
    )
    return parser.parse_args(argv)


def extract_test_metrics_from_log(log_path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not log_path.exists():
        return metrics
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                for match in TEST_METRIC_PATTERN.finditer(line):
                    name = match.group("name")
                    value_raw = match.group("value")
                    percent = match.group("percent")
                    try:
                        value = float(value_raw)
                    except ValueError:
                        continue
                    if percent:
                        value /= 100.0
                    metrics[name] = value
    except OSError:
        return {}
    return metrics


def load_hparams(version_dir: Path) -> Dict[str, Any]:
    hparams_path = version_dir / "hparams.yaml"
    if not hparams_path.exists():
        return {}
    try:
        with hparams_path.open("r", encoding="utf-8") as fp:
            data = yaml.load(fp, Loader=HparamsLoader)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def parse_metrics_csv(version_dir: Path) -> Dict[str, float]:
    metrics_path = version_dir / "metrics.csv"
    if not metrics_path.exists():
        return {}
    try:
        df = pd.read_csv(metrics_path)
    except Exception:
        return {}

    metrics: Dict[str, float] = {}
    for column in df.columns:
        if column.lower() in {"step", "epoch"}:
            continue
        series = df[column].dropna()
        if series.empty:
            continue
        value = series.iloc[-1]
        try:
            metrics[column] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def parse_test_results(iter_dir: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for csv_path in sorted(iter_dir.glob("test_result_*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue
        row = df.iloc[0]
        for column, value in row.items():
            try:
                metrics[column] = float(value)
            except (TypeError, ValueError):
                continue
    return metrics


def merge_metrics(sources: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    for source in sources:
        for key, value in source.items():
            merged[key] = value
    return merged


def pick_primary_test_acc(metrics: Mapping[str, float]) -> Tuple[float, Optional[str]]:
    preferred = ("test_acc", "test_accuracy")
    for key in preferred:
        if key in metrics:
            try:
                return float(metrics[key]), key
            except (TypeError, ValueError):
                continue
    for key in sorted(metrics.keys()):
        if key.startswith("test_acc"):
            try:
                return float(metrics[key]), key
            except (TypeError, ValueError):
                continue
    return math.nan, None


def extract_hyperparams(hparams: Mapping[str, Any]) -> Dict[str, Any]:
    few_shot = hparams.get("few_shot") if isinstance(hparams, Mapping) else {}
    contrastive = hparams.get("contrastive") if isinstance(hparams, Mapping) else {}

    return {
        "contrastive_loss_weight": contrastive.get("loss_weight") if isinstance(contrastive, Mapping) else None,
        "domains_per_episode": few_shot.get("domains_per_episode") if isinstance(few_shot, Mapping) else None,
        "classes_per_domain": few_shot.get("classes_per_domain") if isinstance(few_shot, Mapping) else None,
        "support_per_class": few_shot.get("support_per_class") if isinstance(few_shot, Mapping) else None,
        "query_per_class": few_shot.get("query_per_class") if isinstance(few_shot, Mapping) else None,
        "systems_per_episode": few_shot.get("systems_per_episode") if isinstance(few_shot, Mapping) else None,
        "seed": hparams.get("seed"),
    }


def find_version_dir(iter_dir: Path) -> Optional[Path]:
    logs_dir = iter_dir / "logs"
    if not logs_dir.exists():
        return None
    candidates = [path for path in logs_dir.glob("version_*") if path.is_dir()]
    if not candidates:
        return None
    try:
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    except OSError:
        candidates.sort()
    return candidates[0]


def locate_train_log(iter_dir: Path, version_dir: Optional[Path]) -> Optional[Path]:
    candidates = [
        iter_dir / "train.log",
        iter_dir / "logs" / "train.log",
    ]
    if version_dir is not None:
        candidates.append(version_dir / "train.log")
    for path in candidates:
        if path.exists():
            return path
    return None


def summarise_iteration(iter_dir: Path) -> Optional[IterationSummary]:
    if not iter_dir.is_dir():
        return None
    version_dir = find_version_dir(iter_dir)
    hparams = load_hparams(version_dir) if version_dir else {}

    sources: List[Mapping[str, float]] = []
    train_log = locate_train_log(iter_dir, version_dir)
    if train_log:
        sources.append(extract_test_metrics_from_log(train_log))
    if version_dir:
        sources.append(parse_metrics_csv(version_dir))
    sources.append(parse_test_results(iter_dir))

    metrics = merge_metrics([src for src in sources if src])
    if not metrics:
        print(f"[WARN] No metrics captured for {iter_dir}")

    hyperparams = extract_hyperparams(hparams)
    return IterationSummary(run_dir=iter_dir.parent, iteration=iter_dir.name, hyperparams=hyperparams, metrics=metrics)


def discover_run_dirs(root: Path, name_filter: str, limit: int) -> List[Path]:
    iter_dirs = list(root.rglob("iter_*"))
    run_dirs = sorted({path.parent for path in iter_dirs if path.is_dir()})
    if name_filter:
        run_dirs = [path for path in run_dirs if name_filter in str(path)]
    if limit > 0:
        run_dirs = run_dirs[:limit]
    return run_dirs


def build_dataframe(summaries: Sequence[IterationSummary], root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for summary in summaries:
        if summary is None:
            continue
        try:
            relative_run = summary.run_dir.relative_to(root)
        except ValueError:
            relative_run = summary.run_dir
        row: Dict[str, Any] = {
            "run_dir": str(relative_run),
            "iteration": summary.iteration,
        }
        for key, value in summary.hyperparams.items():
            row[key] = value

        test_acc_value, source_key = pick_primary_test_acc(summary.metrics)
        row["test_acc"] = test_acc_value
        if source_key:
            row["test_acc_source"] = source_key

        for key, value in summary.metrics.items():
            if key == source_key:
                continue
            metric_name = key
            if metric_name in row:
                metric_name = f"metric_{metric_name}"
            row[metric_name] = value

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    preferred_columns = [
        "run_dir",
        "iteration",
        "contrastive_loss_weight",
        "domains_per_episode",
        "classes_per_domain",
        "support_per_class",
        "query_per_class",
        "systems_per_episode",
        "seed",
        "test_acc",
        "test_acc_source",
    ]
    ordered = [col for col in preferred_columns if col in df.columns]
    remaining = [col for col in df.columns if col not in ordered]
    return df[ordered + remaining]


def export_outputs(df: pd.DataFrame, csv_path: Path, markdown_path: Optional[Path]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if markdown_path:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        with markdown_path.open("w", encoding="utf-8") as handle:
            handle.write("# TSPN Hyperparameter Resummary\n\n")
            if df.empty:
                handle.write("_No runs discovered._\n")
            else:
                try:
                    handle.write(df.to_markdown(index=False))
                except ImportError:
                    handle.write(df.to_string(index=False))
            handle.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root: Path = args.root
    if not root.exists():
        print(f"[ERROR] Root directory not found: {root}")
        return 1

    run_dirs = discover_run_dirs(root, args.filter, args.max_runs)
    if not run_dirs:
        print(f"[WARN] No run directories discovered under {root}")
        return 0

    summaries: List[IterationSummary] = []
    for run_dir in run_dirs:
        iter_dirs = sorted(path for path in run_dir.glob("iter_*") if path.is_dir())
        if not iter_dirs:
            print(f"[WARN] Skipping {run_dir} (no iteration folders found)")
            continue
        for iter_dir in iter_dirs:
            summary = summarise_iteration(iter_dir)
            if summary:
                summaries.append(summary)

    df = build_dataframe(summaries, root)
    if df.empty:
        print("[WARN] No metrics recovered; nothing to write.")
        return 0

    output_csv = args.output or (root / "tspn_resummary.csv")
    export_outputs(df, output_csv, args.markdown)
    print(f"[INFO] Wrote CSV summary: {output_csv}")
    if args.markdown:
        print(f"[INFO] Wrote Markdown summary: {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

