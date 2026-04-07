from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.experiment import run_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical taskset-driven TF_MultiProtoDG ablation or hparam studies."
    )
    parser.add_argument("--taskset", type=str, required=True, help="Path to the taskset YAML.")
    parser.add_argument("--study-config", type=str, required=True, help="Path to the ablation/hparam study YAML.")
    parser.add_argument("--local-config", type=str, default=None, help="Optional local override YAML.")
    parser.add_argument("--limit-items", nargs="*", default=None, help="Optional subset of study item ids.")
    parser.add_argument("--smoke", action="store_true", help="Use smoke iterations and smoke epochs.")
    parser.add_argument("--iterations", type=int, default=None, help="Override study iteration count.")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override trainer.num_epochs.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override study output directory.")
    parser.add_argument(
        "--enable-diagnostics",
        action="store_true",
        help="Opt in to model.export_diagnostics for the whole study run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_study(
        taskset_path=args.taskset,
        study_config_path=args.study_config,
        local_config=args.local_config,
        limit_items=args.limit_items,
        smoke=args.smoke,
        iterations_override=args.iterations,
        num_epochs_override=args.num_epochs,
        output_dir_override=args.output_dir,
        export_diagnostics_override=True if args.enable_diagnostics else None,
    )


if __name__ == "__main__":
    main()
