#!/usr/bin/env bash
# Wrapper for the TSPN HUST ablation orchestrator.

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

CONFIG_ROOT="${CONFIG_ROOT:-${REPO_ROOT}/configs/experiments/tspn_hust_ablation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/save/hust_ablation}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

ARGS=("--config-root" "${CONFIG_ROOT}" "--output-root" "${OUTPUT_ROOT}")

if [[ -n "${HUST_ABLATION_VARIANTS:-}" ]]; then
  # shellcheck disable=SC2206
  VARIANT_LIST=(${HUST_ABLATION_VARIANTS})
  ARGS+=("--variants" "${VARIANT_LIST[@]}")
fi

if [[ -n "${HUST_ABLATION_DEVICES:-}" ]]; then
  ARGS+=("--devices" "${HUST_ABLATION_DEVICES}")
fi

if [[ -n "${HUST_ABLATION_DEVICE_POOL:-}" ]]; then
  ARGS+=("--device-pool" "${HUST_ABLATION_DEVICE_POOL}")
fi

if [[ -n "${HUST_ABLATION_PIPELINE:-}" ]]; then
  ARGS+=("--pipeline" "${HUST_ABLATION_PIPELINE}")
fi

if [[ -n "${HUST_ABLATION_TIMEOUT:-}" ]]; then
  ARGS+=("--timeout" "${HUST_ABLATION_TIMEOUT}")
fi

if [[ -n "${HUST_ABLATION_NOTES:-}" ]]; then
  ARGS+=("--notes" "${HUST_ABLATION_NOTES}")
fi

exec "${PYTHON_BIN}" -m script.hparam_eval.tspn_hust_ablation "${ARGS[@]}" "$@"
