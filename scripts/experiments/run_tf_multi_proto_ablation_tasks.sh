#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TASKSET="configs/experiments/01_cross_domain/X_DG/tf_multi_proto_dg_batch/tasksets/sys27_t012.yaml"
STUDY="configs/experiments/01_cross_domain/X_DG/tf_multi_proto_dg_batch/ablation/study.yaml"
OUTPUT_DIR="results/experiments/tf_multi_proto_dg_batch/ablation"
LOCAL_CONFIG=""
PRINT_COMMAND=0

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --taskset)
      TASKSET="$2"
      shift 2
      ;;
    --study-config)
      STUDY="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --local-config)
      LOCAL_CONFIG="$2"
      shift 2
      ;;
    --print-command)
      PRINT_COMMAND=1
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

CMD=(
  "$PYTHON_BIN"
  "scripts/run_tf_multi_proto_dg_study.py"
  "--taskset" "$TASKSET"
  "--study-config" "$STUDY"
  "--output-dir" "$OUTPUT_DIR"
)

if [[ -n "$LOCAL_CONFIG" ]]; then
  CMD+=("--local-config" "$LOCAL_CONFIG")
fi

if [[ ${#ARGS[@]} -gt 0 ]]; then
  CMD+=("${ARGS[@]}")
fi

if [[ "$PRINT_COMMAND" -eq 1 ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
