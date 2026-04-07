#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/xuanli/work/PHM-Vibench"
RUNNER_SESSION="${RUNNER_SESSION:-tfmpdg_lodo_27_21_runner}"
MONITOR_SESSION="${MONITOR_SESSION:-tfmpdg_lodo_27_21_monitor}"
MARKDOWN_PATH="${MARKDOWN_PATH:-paper/2025-12_TSPN_CL/2026_04/tfmpdg_lodo_27_21_lsfix_monitor.md}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-120}"

cd "${REPO_ROOT}"

while tmux has-session -t "${RUNNER_SESSION}" 2>/dev/null; do
  sleep "${INTERVAL_SECONDS}"
done

python scripts/experiments/summarize_tfmpdg_lodo_27_21.py --markdown "${MARKDOWN_PATH}"
tmux kill-session -t "${MONITOR_SESSION}" 2>/dev/null || true

echo "[$(date '+%F %T')] finalized ${MARKDOWN_PATH}"
