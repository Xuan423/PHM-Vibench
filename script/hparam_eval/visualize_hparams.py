"""Visualise hyperparameter sweep results stored in save/hyparams_hust.xlsx.

The script avoids a runtime dependency on openpyxl by reading the XLSX archive
directly via :mod:`zipfile`. It aggregates the recorded ``test_acc`` values and
emits summary CSV/Markdown artefacts alongside heatmaps for every combination of
``contrastive_loss_weight``, ``domains_per_episode`` and ``classes_per_domain``.
"""

from __future__ import annotations

import argparse
import math
import textwrap
import zipfile
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    plt = None
    sns = None

import pandas as pd

HEADERS = (
    "contrastive_loss_weight",
    "domains_per_episode",
    "classes_per_domain",
    "support_per_class",
    "query_per_class",
    "test_acc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visualisations for contrastive hyperparameter sweeps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Example:
                python -m script.hparam_eval.visualize_hparams \\
                    --input save/hyparams_hust.xlsx \\
                    --output-dir save/hparam_viz
            """
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("save/hyparams_hust.xlsx"),
        help="Source Excel workbook containing sweep results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("save/hparam_viz"),
        help="Directory to store derived tables and figures.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="test_acc",
        help="Metric column to aggregate (defaults to 'test_acc').",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top-performing configurations to display in the summary plot.",
    )
    return parser.parse_args()


def load_excel_without_openpyxl(path: Path) -> pd.DataFrame:
    """Load the first worksheet of an XLSX file without using openpyxl."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    with zipfile.ZipFile(path) as archive:
        shared_strings = _parse_shared_strings(archive)
        rows = _parse_first_sheet(archive, shared_strings)

    if not rows:
        raise ValueError(f"No rows found in worksheet within {path}")
    headers, *data_rows = rows
    if headers[: len(HEADERS)] != list(HEADERS):
        raise ValueError(
            f"Unexpected headers {headers}. Expected first columns: {', '.join(HEADERS)}"
        )
    records = [row for row in data_rows if any(cell.strip() for cell in row)]
    df = pd.DataFrame(records, columns=headers)

    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except (ValueError, TypeError):
            # Leave non-numeric columns as-is.
            continue
    return df


def _parse_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    with archive.open("xl/sharedStrings.xml") as handle:
        root = ET.parse(handle).getroot()
        namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        values: list[str] = []
        for si in root.findall(f"{namespace}si"):
            text = "".join(node.text or "" for node in si.iter(f"{namespace}t"))
            values.append(text)
        return values


def _parse_first_sheet(archive: zipfile.ZipFile, shared_strings: Iterable[str]) -> list[list[str]]:
    with archive.open("xl/worksheets/sheet1.xml") as handle:
        root = ET.parse(handle).getroot()
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    sheet_data = root.find(f"{namespace}sheetData")
    if sheet_data is None:
        return []

    rows: list[list[str]] = []
    for row in sheet_data.findall(f"{namespace}row"):
        values: list[str] = []
        for cell in row.findall(f"{namespace}c"):
            value_node = cell.find(f"{namespace}v")
            if value_node is None:
                values.append("")
                continue
            text = value_node.text or ""
            if cell.get("t") == "s":
                values.append(shared_strings[int(text)])
            else:
                values.append(text)
        if not values:
            break
        rows.append(values)
    return rows


def aggregate_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Aggregate metric statistics per hyperparameter combination."""
    group_keys = [col for col in df.columns if col != metric]
    grouped = (
        df.groupby(group_keys)[metric]
        .agg(
            metric_mean="mean",
            metric_std="std",
            metric_min="min",
            metric_max="max",
            runs="size",
        )
        .reset_index()
    )
    return grouped


def save_tables(summary: pd.DataFrame, output_dir: Path, metric: str) -> None:
    csv_path = output_dir / f"{metric}_summary.csv"
    md_path = output_dir / f"{metric}_summary.md"

    summary.to_csv(csv_path, index=False)
    try:
        markdown = summary.to_markdown(index=False)
    except ImportError:
        header = " | ".join(summary.columns)
        separator = " | ".join("---" for _ in summary.columns)
        body = "\n".join(" | ".join(map(str, row)) for row in summary.to_numpy())
        markdown = "\n".join([header, separator, body])
    md_path.write_text(markdown, encoding="utf-8")


def render_summary_heatmap(summary: pd.DataFrame, metric_col: str, output_dir: Path) -> None:
    if plt is None or sns is None:
        print("[WARN] matplotlib/seaborn not available; skipping summary heatmap.")
        return
    summary = summary.copy()
    summary["row_label"] = summary.apply(
        lambda row: f"W{row['contrastive_loss_weight']}_D{row['domains_per_episode']}_C{row['classes_per_domain']}",
        axis=1,
    )
    summary["col_label"] = summary.apply(
        lambda row: f"S{row['support_per_class']}_Q{row['query_per_class']}", axis=1
    )

    pivot = summary.pivot(index="row_label", columns="col_label", values=metric_col)

    vmin = math.floor(summary[metric_col].min() * 1000) / 1000.0
    vmax = math.ceil(summary[metric_col].max() * 1000) / 1000.0

    plt.figure(figsize=(min(18, 1 + 0.5 * pivot.shape[1]), min(18, 1 + 0.35 * pivot.shape[0])))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": metric_col},
    )
    plt.title("Summary heatmap of contrastive hyperparameter sweep")
    plt.xlabel("Support / Query combinations")
    plt.ylabel("Weight / Domain / Classes")
    plt.tight_layout()
    plt.savefig(output_dir / "summary_heatmap.png", dpi=300)
    plt.close()


def render_topk_plot(summary: pd.DataFrame, metric_col: str, output_dir: Path, top_k: int) -> None:
    if plt is None or sns is None:
        print("[WARN] matplotlib/seaborn not available; skipping top-k plot.")
        return

    top = summary.sort_values(metric_col, ascending=False).head(top_k).copy()
    top["label"] = top.apply(
        lambda row: (
            f"W{row['contrastive_loss_weight']} | D{row['domains_per_episode']} | "
            f"C{row['classes_per_domain']} | S{row['support_per_class']} | Q{row['query_per_class']}"
        ),
        axis=1,
    )
    plt.figure(figsize=(8, max(4, 0.5 * len(top))))
    sns.barplot(data=top, x=metric_col, y="label", palette="viridis")
    plt.xlabel(metric_col)
    plt.ylabel("Top configurations")
    plt.title(f"Top {len(top)} configurations by {metric_col}")
    plt.tight_layout()
    plt.savefig(output_dir / "top_configs.png", dpi=300)
    plt.close()


def render_weight_summary(summary: pd.DataFrame, metric_col: str, output_dir: Path) -> None:
    if plt is None or sns is None:
        print("[WARN] matplotlib/seaborn not available; skipping weight summary plot.")
        return

    weight_stats = (
        summary.groupby("contrastive_loss_weight")[metric_col]
        .agg(["mean", "max", "min"])
        .reset_index()
        .rename(
            columns={
                "mean": f"{metric_col}_mean",
                "max": f"{metric_col}_max",
                "min": f"{metric_col}_min",
            }
        )
    )
    weight_stats.to_csv(output_dir / "weight_level_stats.csv", index=False)

    plt.figure(figsize=(6, 4))
    sns.lineplot(
        data=weight_stats,
        x="contrastive_loss_weight",
        y=f"{metric_col}_mean",
        marker="o",
        label="mean",
    )
    sns.scatterplot(
        data=weight_stats,
        x="contrastive_loss_weight",
        y=f"{metric_col}_max",
        marker="^",
        s=60,
        color="tab:green",
        label="max",
    )
    sns.scatterplot(
        data=weight_stats,
        x="contrastive_loss_weight",
        y=f"{metric_col}_min",
        marker="v",
        s=60,
        color="tab:red",
        label="min",
    )
    plt.ylabel(metric_col)
    plt.title("Contrastive loss weight vs. aggregated performance")
    plt.tight_layout()
    plt.savefig(output_dir / "weight_vs_metric.png", dpi=300)
    plt.close()


def render_parameter_trends(raw: pd.DataFrame, metric: str, output_dir: Path) -> None:
    if plt is None:
        print("[WARN] matplotlib not available; skipping combined trend plot.")
        return

    params = [
        "contrastive_loss_weight",
        "domains_per_episode",
        "classes_per_domain",
        "support_per_class",
        "query_per_class",
    ]

    plt.figure(figsize=(10, 6))
    for param in params:
        grouped = raw.groupby(param)[metric].agg(["mean", "std"])
        x = grouped.index.to_numpy()
        y = grouped["mean"].to_numpy()
        yerr = grouped["std"].to_numpy()
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            label=param,
            marker="o",
            capsize=4,
        )

    plt.xlabel("Hyperparameter value")
    plt.ylabel(metric)
    plt.title("Hyperparameter trends (mean ± std)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "combined_trends.png", dpi=300)
    plt.close()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_excel_without_openpyxl(args.input)
    if args.metric not in df.columns:
        raise KeyError(f"Metric column '{args.metric}' not present. Available: {list(df.columns)}")

    summary = aggregate_metric(df, args.metric)
    metric_col = "metric_mean"
    save_tables(summary, output_dir, args.metric)
    render_summary_heatmap(summary, metric_col, output_dir)
    render_topk_plot(summary, metric_col, output_dir, args.top_k)
    render_weight_summary(summary, metric_col, output_dir)
    render_parameter_trends(df, args.metric, output_dir)

    print(f"[INFO] Summary saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
