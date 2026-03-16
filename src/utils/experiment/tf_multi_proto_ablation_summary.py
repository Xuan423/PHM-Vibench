from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_or_nan(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _flatten_health_values(health_items: Mapping[str, Mapping[str, Any]], field: str) -> List[float]:
    values: List[float] = []
    for head_payload in health_items.values():
        for item in head_payload.get(field, []):
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                continue
    return values


def _preferred_health_record(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not records:
        return None
    for stage in ("test", "val", "train"):
        stage_records = [record for record in records if str(record.get("stage")) == stage]
        if stage_records:
            return stage_records[-1]
    return records[-1]


def extract_diagnostics_summary(run_dir: str | Path) -> Dict[str, Any]:
    run_path = Path(run_dir)
    diagnostics_dir = run_path / "diagnostics"
    context = _load_json(diagnostics_dir / "diagnostics_context.json") or {}
    prototype_health_records = _load_json(diagnostics_dir / "prototype_health.json") or []
    prototype_cards_records = _load_json(diagnostics_dir / "prototype_cards.json") or []

    summary: Dict[str, Any] = {
        "diag_context_present": diagnostics_dir.exists(),
        "diag_variant_id": context.get("variant_id"),
        "diag_active_feature_count": context.get("active_feature_count"),
        "diag_active_branches": ",".join(context.get("active_components", {}).get("active_branches", [])),
        "diag_active_time_operators": ",".join(
            context.get("active_components", {}).get("active_operators", {}).get("time", [])
        ),
        "diag_active_freq_operators": ",".join(
            context.get("active_components", {}).get("active_operators", {}).get("freq", [])
        ),
        "diag_active_time_indicators": ",".join(
            context.get("active_components", {}).get("active_indicators", {}).get("time", [])
        ),
        "diag_active_freq_indicators": ",".join(
            context.get("active_components", {}).get("active_indicators", {}).get("freq", [])
        ),
        "diag_prototype_health_present": False,
        "diag_prototype_cards_present": bool(prototype_cards_records),
        "diag_max_assignment_ratio_mean": float("nan"),
        "diag_max_assignment_ratio_max": float("nan"),
        "diag_n_eff_mean": float("nan"),
        "diag_max_offdiag_cos_mean": float("nan"),
    }

    preferred = _preferred_health_record(prototype_health_records)
    if preferred is None:
        return summary

    health_items = preferred.get("items", {})
    if not isinstance(health_items, Mapping):
        return summary

    assignment_ratio_values = _flatten_health_values(health_items, "max_assignment_ratio")
    n_eff_values = _flatten_health_values(health_items, "n_eff")
    max_offdiag_values = _flatten_health_values(health_items, "max_offdiag_cos")

    summary.update(
        {
            "diag_prototype_health_present": True,
            "diag_max_assignment_ratio_mean": _mean_or_nan(assignment_ratio_values),
            "diag_max_assignment_ratio_max": max(assignment_ratio_values) if assignment_ratio_values else float("nan"),
            "diag_n_eff_mean": _mean_or_nan(n_eff_values),
            "diag_max_offdiag_cos_mean": _mean_or_nan(max_offdiag_values),
        }
    )
    return summary


def collect_study_results(
    item_spec: Mapping[str, Any],
    iteration_records: Sequence[Mapping[str, Any]],
    run_dirs: Sequence[str | Path],
    smoke: bool = False,
    num_epochs: int | None = None,
    task_spec: Mapping[str, Any] | None = None,
    study_type: str = "ablation",
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    task_spec = task_spec or {}
    item_id = str(item_spec.get("item_id", item_spec.get("variant_id", "")))
    for iteration, record in enumerate(iteration_records):
        row: Dict[str, Any] = {
            "study_type": study_type,
            "task_id": task_spec.get("task_id", ""),
            "task_description": task_spec.get("description", ""),
            "item_id": item_id,
            "variant_id": item_id,
            "group": item_spec.get("group", ""),
            "description": item_spec.get("description", ""),
            "iteration": int(iteration),
            "smoke": bool(smoke),
            "num_epochs": num_epochs,
            "diagnostics_expected": bool(item_spec.get("diagnostics_expected", False)),
            "success": bool(record.get("success", True)),
            "error_message": record.get("error_message"),
        }
        run_dir_value = run_dirs[iteration] if iteration < len(run_dirs) else None
        run_dir = Path(run_dir_value) if run_dir_value else None
        row["run_dir"] = str(run_dir) if run_dir is not None else None

        if run_dir is not None:
            row.update(extract_diagnostics_summary(run_dir))

        for key, value in record.items():
            if key in {"success", "error_message"}:
                continue
            if isinstance(value, bool):
                row[str(key)] = value
            elif isinstance(value, (int, float)):
                row[str(key)] = float(value)
            else:
                row[str(key)] = value
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_study_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    group_cols = [
        "study_type",
        "task_id",
        "task_description",
        "item_id",
        "variant_id",
        "group",
        "description",
        "smoke",
        "num_epochs",
        "diagnostics_expected",
    ]
    numeric_cols = [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
        and not pd.api.types.is_bool_dtype(df[column])
        and column not in {"iteration", "smoke", "success"}
    ]
    rows: List[Dict[str, Any]] = []
    for keys, group_df in df.groupby(group_cols, dropna=False):
        row = {column: value for column, value in zip(group_cols, keys)}
        row["num_iterations"] = int(group_df["iteration"].nunique())
        row["num_success"] = int(group_df["success"].sum()) if "success" in group_df else len(group_df)
        row["num_failed"] = int(len(group_df) - row["num_success"])
        if row["num_success"] < row["num_iterations"]:
            row["notes"] = "partial_or_failed_iterations"
        elif bool(row["smoke"]):
            row["notes"] = "smoke_run"
        else:
            row["notes"] = ""

        for column in numeric_cols:
            series = pd.to_numeric(group_df[column], errors="coerce")
            if series.isna().all():
                row[f"{column}_mean"] = float("nan")
                row[f"{column}_std"] = float("nan")
                row[f"{column}_min"] = float("nan")
                row[f"{column}_max"] = float("nan")
            else:
                row[f"{column}_mean"] = float(series.mean())
                row[f"{column}_std"] = float(series.std(ddof=0))
                row[f"{column}_min"] = float(series.min())
                row[f"{column}_max"] = float(series.max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["task_id", "group", "item_id"]).reset_index(drop=True)


def _format_markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return f"{value:.4f}"
    return str(value)


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format_markdown_cell(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def render_study_markdown_summary(
    runs_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    item_manifest: Sequence[Mapping[str, Any]],
    study_name: str,
    study_type: str,
    task_manifest: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    item_table = pd.DataFrame(
        [
            {
                "item_id": item.get("item_id", item.get("variant_id")),
                "group": item.get("group"),
                "description": item.get("description"),
                "diagnostics_expected": item.get("diagnostics_expected"),
            }
            for item in item_manifest
        ]
    )
    task_table = pd.DataFrame(task_manifest or [])

    summary_metric_cols = [
        column
        for column in summary_df.columns
        if column
        in {
            "study_type",
            "task_id",
            "item_id",
            "group",
            "num_iterations",
            "num_success",
            "num_failed",
            "notes",
        }
        or column.endswith("_mean")
    ]
    diag_cols = [
        column
        for column in summary_df.columns
        if column in {"task_id", "item_id", "group", "notes"}
        or column.startswith("diag_")
        and column.endswith("_mean")
    ]
    sections = [
        f"# {study_name}",
        "",
        f"Study type: `{study_type}`",
        "",
        "## Task Definitions",
        "",
        _df_to_markdown(task_table),
        "",
        "## Study Items",
        "",
        _df_to_markdown(item_table),
        "",
        "## Run-Level Records",
        "",
        _df_to_markdown(runs_df),
        "",
        "## Metric Summary",
        "",
        _df_to_markdown(summary_df[summary_metric_cols] if summary_metric_cols else summary_df),
        "",
        "## Diagnostics Summary",
        "",
        _df_to_markdown(summary_df[diag_cols] if diag_cols else pd.DataFrame()),
    ]
    return "\n".join(sections).strip() + "\n"


def write_study_summary_bundle(
    output_dir: str | Path,
    runs_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    markdown: str,
    item_manifest: Sequence[Mapping[str, Any]],
    task_manifest: Sequence[Mapping[str, Any]] | None = None,
    artifact_prefix: str = "study",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    (runs_df if not runs_df.empty else pd.DataFrame()).to_csv(output_path / f"{artifact_prefix}_runs.csv", index=False)
    (summary_df if not summary_df.empty else pd.DataFrame()).to_csv(
        output_path / f"{artifact_prefix}_summary.csv", index=False
    )
    (output_path / f"{artifact_prefix}_summary.md").write_text(markdown, encoding="utf-8")
    (output_path / f"{artifact_prefix}_manifest.json").write_text(
        json.dumps(
            {
                "items": list(item_manifest),
                "tasks": list(task_manifest or []),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def collect_variant_results(
    variant_spec: Mapping[str, Any],
    iteration_records: Sequence[Mapping[str, Any]],
    run_dirs: Sequence[str | Path],
    smoke: bool = False,
    num_epochs: int | None = None,
) -> pd.DataFrame:
    return collect_study_results(
        item_spec={**variant_spec, "item_id": variant_spec.get("variant_id")},
        iteration_records=iteration_records,
        run_dirs=run_dirs,
        smoke=smoke,
        num_epochs=num_epochs,
        study_type="ablation",
    )


def aggregate_variant_results(df: pd.DataFrame) -> pd.DataFrame:
    return aggregate_study_results(df)


def render_markdown_summary(
    runs_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    variant_manifest: Sequence[Mapping[str, Any]],
    study_name: str,
) -> str:
    return render_study_markdown_summary(
        runs_df=runs_df,
        summary_df=summary_df,
        item_manifest=[{**item, "item_id": item.get("variant_id")} for item in variant_manifest],
        study_name=study_name,
        study_type="ablation",
        task_manifest=[],
    )


def write_summary_bundle(
    output_dir: str | Path,
    runs_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    markdown: str,
    variant_manifest: Sequence[Mapping[str, Any]],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_study_summary_bundle(
        output_dir=output_path,
        runs_df=runs_df,
        summary_df=summary_df,
        markdown=markdown,
        item_manifest=[{**item, "item_id": item.get("variant_id")} for item in variant_manifest],
        task_manifest=[],
        artifact_prefix="ablation",
    )
    generated_manifest = output_path / "ablation_manifest.json"
    if generated_manifest.exists():
        generated_manifest.rename(output_path / "variant_manifest.json")
