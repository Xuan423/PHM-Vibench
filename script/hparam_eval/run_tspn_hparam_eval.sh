#!/usr/bin/env bash
# Wrapper around the TSPN hyperparameter sweep Python entrypoint.

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

# Ensure CONFIG_ROOT points to the correct directory
CONFIG_ROOT="${CONFIG_ROOT:-${REPO_ROOT}/configs/experiments/tspn_hparam_eval}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/save/hparam_eval}"

# Add the project root to PYTHONPATH to fix relative import issues
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" # Ensure PYTHONPATH is initialized

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

if [[ -n "${TSPN_EVAL_NOTES:-}" ]]; then
  ARGS+=("--notes" "${TSPN_EVAL_NOTES}")
fi

# Use python -m to run the orchestrator module
exec "${PYTHON_BIN}" -m script.hparam_eval.tspn_hparam_eval "${ARGS[@]}" "$@"
