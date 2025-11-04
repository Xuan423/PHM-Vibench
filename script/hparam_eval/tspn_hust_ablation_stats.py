"""Utility script to compute accuracy statistics for TSPN HUST ablation runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "save" / "hust_ablation"
DEFAULT_SUMMARY_NAME = "ablation_summary.csv"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate accuracy statistics from TSPN HUST ablation results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Path to ablation_summary.csv or containing directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write accuracy reports (defaults to the summary parent directory).",
    )
    parser.add_argument(
        "--metric-column",
        type=str,
        default=None,
        help="Exact column name to treat as accuracy. Overrides --metric-pattern.",
    )
    parser.add_argument(
        "--metric-pattern",
        type=str,
        default="test_acc",
        help="Substring used to auto-detect the accuracy column when --metric-column is not provided.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary_path = resolve_summary_path(args.input)
    output_dir = args.output_dir or summary_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)
    if df.empty:
        raise ValueError(f"Summary file {summary_path} is empty.")

    metric_column = select_metric_column(df, args.metric_column, args.metric_pattern)
    records = build_records(df, metric_column)
    if records.empty:
        raise ValueError("No accuracy records available after removing missing values.")

    summary = summarise_records(records)

    records_path = output_dir / "accuracy_records.csv"
    summary_path_out = output_dir / "accuracy_summary.csv"

    records.to_csv(records_path, index=False)
    summary.to_csv(summary_path_out, index=False)

    print(f"[INFO] Loaded summary from: {summary_path}")
    print(f"[INFO] Selected accuracy column: {metric_column}")
    print(f"[INFO] Wrote per-run accuracy to: {records_path}")
    print(f"[INFO] Wrote grouped statistics to: {summary_path_out}")
    print("[INFO] Accuracy summary:\n")
    try:
        print(summary.to_markdown(index=False))
    except (ImportError, AttributeError):
        print(summary)
    return 0


def resolve_summary_path(input_path: Path) -> Path:
    if input_path.is_dir():
        candidate = input_path / DEFAULT_SUMMARY_NAME
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Could not find {DEFAULT_SUMMARY_NAME} in directory {input_path}")
    if input_path.suffix.lower() not in {".csv"}:
        raise ValueError(f"Input path must point to a CSV file or directory, received: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Summary file not found: {input_path}")
    return input_path


def select_metric_column(df: pd.DataFrame, metric_column: Optional[str], metric_pattern: Optional[str]) -> str:
    if metric_column:
        if metric_column not in df.columns:
            raise KeyError(f"Specified metric column '{metric_column}' not found in summary.")
        return metric_column

    candidates: List[str] = []
    if metric_pattern:
        pattern = metric_pattern.lower()
        candidates = [col for col in df.columns if pattern in col.lower()]

    if not candidates:
        candidates = [col for col in df.columns if "acc" in col.lower()]

    if not candidates:
        raise KeyError(
            "Unable to detect an accuracy column. Specify it explicitly with --metric-column."
        )
    return candidates[0]


def build_records(df: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    accuracy_series = pd.to_numeric(df[metric_column], errors="coerce")
    records = pd.DataFrame(
        {
            "run_name": _column_or_default(df, "meta_run_name"),
            "variant": df["variant"],
            "base_variant": _column_or_default(df, "meta_base_variant", fallback=df["variant"]),
            "contrastive_loss_weight": _column_or_default(df, "meta_contrastive_loss_weight"),
            "source_domain_tag": _column_or_default(df, "meta_source_domain_tag"),
            "source_domain_ids": _column_or_default(df, "meta_source_domain_ids"),
            "accuracy": accuracy_series,
        }
    )
    records = records.dropna(subset=["accuracy"]).reset_index(drop=True)
    return records


def summarise_records(records: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "variant",
        "base_variant",
        "contrastive_loss_weight",
        "source_domain_tag",
        "source_domain_ids",
    ]
    group_columns = [col for col in group_columns if records[col].notna().any()]

    grouped = records.groupby(group_columns, dropna=False)["accuracy"]
    summary = grouped.agg(["count", "mean", "std", "min", "max"]).reset_index()
    summary = summary.rename(
        columns={
            "count": "num_runs",
            "mean": "accuracy_mean",
            "std": "accuracy_std",
            "min": "accuracy_min",
            "max": "accuracy_max",
        }
    )

    best_idx = grouped.idxmax()
    best_rows = records.loc[best_idx].reset_index(drop=True)
    best_rows = best_rows[group_columns + ["run_name", "accuracy"]].rename(
        columns={"run_name": "best_run_name", "accuracy": "best_accuracy"}
    )

    summary = summary.merge(best_rows, on=group_columns, how="left")
    ordered_columns = [
        *group_columns,
        "num_runs",
        "accuracy_mean",
        "accuracy_std",
        "accuracy_min",
        "accuracy_max",
        "best_run_name",
        "best_accuracy",
    ]
    # Ensure every expected column is present before reordering.
    for column in ordered_columns:
        if column not in summary.columns:
            summary[column] = pd.NA
    return summary[ordered_columns]


def _column_or_default(df: pd.DataFrame, column: str, *, fallback: Optional[pd.Series] = None) -> pd.Series:
    if column in df.columns:
        return df[column]
    if fallback is not None:
        return fallback
    return pd.Series([pd.NA] * len(df), index=df.index)


if __name__ == "__main__":
    raise SystemExit(main())
