#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List


@dataclass
class CaseSpec:
    system_id: int
    target_domain: int
    variant: str

    @property
    def output_dir(self) -> Path:
        run_tag_map = {
            "full": "lsfix_5seed_20260402",
            "nocl": "lsfix_5seed_20260402",
            "single": "lsfix_5seed_20260402",
            "full_k4": "lsfix_5seed_20260402",
            "nocl_ls0": "lsfix_nocl_ls0_5seed_20260403",
        }
        return Path(
            f"results/demo/tfmpdg_lodo_t{self.system_id}_d{self.target_domain}_{self.variant}_{run_tag_map[self.variant]}"
        )

    @property
    def case_id(self) -> str:
        return f"t{self.system_id}_d{self.target_domain}_{self.variant}"


def iter_specs() -> Iterable[CaseSpec]:
    variants = ("full", "nocl", "nocl_ls0", "single")
    for td in (0, 1, 2):
        for v in variants:
            yield CaseSpec(system_id=27, target_domain=td, variant=v)
    for td in (14, 16, 22, 24):
        for v in (*variants, "full_k4"):
            yield CaseSpec(system_id=21, target_domain=td, variant=v)


def read_test_acc(csv_path: Path) -> float | None:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    acc_keys = [k for k in row.keys() if k.startswith("test_acc_")]
    if not acc_keys:
        return None
    return float(row[acc_keys[0]])


def collect_case_accs(root: Path) -> List[float]:
    accs: List[float] = []
    if not root.exists():
        return accs
    for p in sorted(root.glob("metadata.xlsx/M_TF_MultiProtoDG/**/test_result_*.csv")):
        val = read_test_acc(p)
        if val is not None:
            accs.append(val)
    return accs


def markdown_table(rows: list[list[str]]) -> str:
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    lines = [header, sep]
    for r in rows[1:]:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def summarize_group(
    acc_map: Dict[str, List[float]],
    case_done_map: Dict[str, int],
    system_id: int,
    variant: str,
) -> list[str]:
    key = f"{system_id}:{variant}"
    accs = acc_map.get(key, [])
    case_count = sum(
        1
        for spec in iter_specs()
        if spec.system_id == system_id and spec.variant == variant and case_done_map.get(spec.case_id, 0) > 0
    )
    if accs:
        return [
            str(system_id),
            variant,
            str(case_count),
            str(len(accs)),
            f"{mean(accs):.6f}",
            f"{pstdev(accs):.6f}" if len(accs) > 1 else "0.000000",
            f"{min(accs):.6f}",
            f"{max(accs):.6f}",
        ]
    return [str(system_id), variant, str(case_count), "0", "-", "-", "-", "-"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("paper/2025-12_TSPN_CL/2026_04/tfmpdg_lodo_27_21_lsfix_monitor.md"),
    )
    args = parser.parse_args()

    rows = [[
        "case",
        "system",
        "target_domain",
        "variant",
        "done/5",
        "mean_acc",
        "std_acc",
        "min_acc",
        "max_acc",
        "output_dir",
    ]]
    grouped_accs: Dict[str, List[float]] = {}
    case_done_map: Dict[str, int] = {}

    for spec in iter_specs():
        accs = collect_case_accs(spec.output_dir)
        grouped_accs.setdefault(f"{spec.system_id}:{spec.variant}", []).extend(accs)
        done = len(accs)
        case_done_map[spec.case_id] = done
        if done > 0:
            m = f"{mean(accs):.6f}"
            s = f"{pstdev(accs):.6f}" if done > 1 else "0.000000"
            mn = f"{min(accs):.6f}"
            mx = f"{max(accs):.6f}"
        else:
            m = s = mn = mx = "-"
        rows.append(
            [
                spec.case_id,
                str(spec.system_id),
                str(spec.target_domain),
                spec.variant,
                f"{done}/5",
                m,
                s,
                mn,
                mx,
                str(spec.output_dir),
            ]
        )

    group_rows = [[
        "system",
        "variant",
        "completed_cases",
        "seed_runs",
        "mean_acc",
        "std_acc",
        "min_acc",
        "max_acc",
    ]]
    for system_id, variants in (
        (27, ("full", "nocl", "nocl_ls0", "single")),
        (21, ("full", "nocl", "nocl_ls0", "single", "full_k4")),
    ):
        for variant in variants:
            group_rows.append(summarize_group(grouped_accs, case_done_map, system_id, variant))

    md = [
        "# TFMPDG LOO Monitor (sys27 + sys21)",
        "",
        f"- updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- setup: 5-seed, 100 epoch, patience 50, export_diagnostics=false",
        "- nocl_ls0: use_contrastive_head=false and task.label_smoothing=0",
        "",
        "## Aggregated",
        "",
        markdown_table(group_rows),
        "",
        "## Per Case",
        "",
        markdown_table(rows),
        "",
    ]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(md), encoding="utf-8")

    print(f"[monitor] wrote {args.markdown}")
    for r in rows[1:]:
        print(" | ".join(r[:9]))


if __name__ == "__main__":
    main()
