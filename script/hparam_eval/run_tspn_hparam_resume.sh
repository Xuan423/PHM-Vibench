#!/usr/bin/env bash
# Helper script for resuming partial TSPN hyperparameter sweeps.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

CONFIG_ROOT="${CONFIG_ROOT:-${REPO_ROOT}/configs/experiments/tspn_hparam_eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/save/hparam_eval}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

ARGS=("--config-root" "${CONFIG_ROOT}" "--output-root" "${OUTPUT_ROOT}")

if [[ -n "${TSPN_EVAL_DEVICES:-}" ]]; then
  ARGS+=("--devices" "${TSPN_EVAL_DEVICES}")
fi

if [[ -n "${TSPN_EVAL_DEVICE_POOL:-}" ]]; then
  ARGS+=("--device-pool" "${TSPN_EVAL_DEVICE_POOL}")
fi

if [[ -n "${TSPN_EVAL_PIPELINE:-}" ]]; then
  ARGS+=("--pipeline" "${TSPN_EVAL_PIPELINE}")
fi

if [[ -n "${TSPN_EVAL_TIMEOUT:-}" ]]; then
  ARGS+=("--timeout" "${TSPN_EVAL_TIMEOUT}")
fi

if [[ -n "${TSPN_EVAL_NOTES:-}" ]]; then
  ARGS+=("--notes" "${TSPN_EVAL_NOTES}")
fi

exec "${PYTHON_BIN}" -m script.hparam_eval.tspn_hparam_resume "${ARGS[@]}" "$@"
