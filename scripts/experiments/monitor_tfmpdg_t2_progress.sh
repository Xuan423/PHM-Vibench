#!/usr/bin/env bash
set -euo pipefail

INTERVAL_SECONDS="${INTERVAL_SECONDS:-120}"
LOG_PATH="${LOG_PATH:-/tmp/tfmpdg_t2_progress.log}"

cd /home/xuanli/work/PHM-Vibench

while true; do
  {
    echo "============================================================"
    date '+%F %T'
    if [ -d "results/demo/tfmpdg_sys27_t012_formal_phase4_20260330" ]; then
      echo "[phase4_t012]"
      python scripts/experiments/summarize_tfmpdg_sys27_t012_phase4.py || true
    fi
    if [ -d "results/demo/tfmpdg_sys27_t012_formal_phase5_20260330" ]; then
      echo "[phase5_t012]"
      python scripts/experiments/summarize_tfmpdg_sys27_t012_phase4.py --root results/demo/tfmpdg_sys27_t012_formal_phase5_20260330 || true
    fi
    if [ -d "results/demo/tfmpdg_sys27_t012_formal_phase6_20260330" ]; then
      echo "[phase6_t012]"
      python scripts/experiments/summarize_tfmpdg_sys27_t012_phase4.py --root results/demo/tfmpdg_sys27_t012_formal_phase6_20260330 || true
    fi
    echo "[gpu]"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
  } >> "${LOG_PATH}" 2>&1
  sleep "${INTERVAL_SECONDS}"
done
