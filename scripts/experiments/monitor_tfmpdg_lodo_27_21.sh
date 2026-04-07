#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/xuanli/work/PHM-Vibench"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-180}"
LOG_PATH="${LOG_PATH:-/tmp/tfmpdg_lodo_27_21_lsfix_monitor.log}"
MARKDOWN_PATH="${MARKDOWN_PATH:-paper/2025-12_TSPN_CL/2026_04/tfmpdg_lodo_27_21_lsfix_monitor.md}"

cd "${REPO_ROOT}"

while true; do
  {
    echo "============================================================"
    date '+%F %T'
    python scripts/experiments/summarize_tfmpdg_lodo_27_21.py --markdown "${MARKDOWN_PATH}" || true
    echo "[gpu]"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
  } >> "${LOG_PATH}" 2>&1
  sleep "${INTERVAL_SECONDS}"
done
