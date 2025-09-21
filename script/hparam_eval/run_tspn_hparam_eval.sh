#!/usr/bin/env bash
# Wrapper around the TSPN hyperparameter sweep Python entrypoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIG_ROOT="${CONFIG_ROOT:-${REPO_ROOT}/configs/experiments/tspn_hparam_eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/save/hparam_eval}"

ARGS=("--config-root" "${CONFIG_ROOT}" "--output-root" "${OUTPUT_ROOT}")

if [[ -n "${TSPN_EVAL_DEVICES:-}" ]]; then
  ARGS+=("--devices" "${TSPN_EVAL_DEVICES}")
fi

if [[ -n "${TSPN_EVAL_PIPELINE:-}" ]]; then
  ARGS+=("--pipeline" "${TSPN_EVAL_PIPELINE}")
fi

if [[ -n "${TSPN_EVAL_TIMEOUT:-}" ]]; then
  ARGS+=("--timeout" "${TSPN_EVAL_TIMEOUT}")
fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/script/hparam_eval/tspn_hparam_eval.py" "${ARGS[@]}" "$@"
