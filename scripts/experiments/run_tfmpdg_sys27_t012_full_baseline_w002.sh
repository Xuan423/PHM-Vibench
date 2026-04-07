#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TASKSET="configs/experiments/01_cross_domain/X_DG/tf_multi_proto_dg_batch/tasksets/sys27_t012.yaml"
STUDY="configs/experiments/01_cross_domain/X_DG/tf_multi_proto_dg_batch/formal_sys27_t012_full_baseline_w002/study.yaml"

CMD=(
  "$PYTHON_BIN"
  "-u"
  "scripts/run_tf_multi_proto_dg_study.py"
  "--taskset" "$TASKSET"
  "--study-config" "$STUDY"
)

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

printf '[run] '
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
