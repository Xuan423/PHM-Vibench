#!/usr/bin/env bash
# Wrapper for the unified TSPN contrastive ablation orchestrator.

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

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/save/contrastive_ablation}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

ARGS=("--output-root" "${OUTPUT_ROOT}")

if [[ -n "${CONTRASTIVE_ABLATION_DATASETS:-}" ]]; then
  # shellcheck disable=SC2206
  DATASET_LIST=(${CONTRASTIVE_ABLATION_DATASETS})
  ARGS+=("--datasets" "${DATASET_LIST[@]}")
fi

if [[ -n "${CONTRASTIVE_ABLATION_VARIANTS:-}" ]]; then
  # shellcheck disable=SC2206
  VARIANT_LIST=(${CONTRASTIVE_ABLATION_VARIANTS})
  ARGS+=("--variants" "${VARIANT_LIST[@]}")
fi

if [[ -n "${CONTRASTIVE_ABLATION_DEVICES:-}" ]]; then
  ARGS+=("--devices" "${CONTRASTIVE_ABLATION_DEVICES}")
fi

if [[ -n "${CONTRASTIVE_ABLATION_DEVICE_POOL:-}" ]]; then
  ARGS+=("--device-pool" "${CONTRASTIVE_ABLATION_DEVICE_POOL}")
fi

if [[ -n "${CONTRASTIVE_ABLATION_PIPELINE:-}" ]]; then
  ARGS+=("--pipeline" "${CONTRASTIVE_ABLATION_PIPELINE}")
fi

if [[ -n "${CONTRASTIVE_ABLATION_TIMEOUT:-}" ]]; then
  ARGS+=("--timeout" "${CONTRASTIVE_ABLATION_TIMEOUT}")
fi

if [[ -n "${CONTRASTIVE_ABLATION_NOTES:-}" ]]; then
  ARGS+=("--notes" "${CONTRASTIVE_ABLATION_NOTES}")
fi

if [[ -n "${CONTRASTIVE_ABLATION_RESUME_FAILED:-}" ]]; then
  case "${CONTRASTIVE_ABLATION_RESUME_FAILED}" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On)
      ARGS+=("--resume-failed")
      ;;
  esac
fi

taskset -c 0-15 "${PYTHON_BIN}" -m script.hparam_eval.tspn_contrastive_ablation "${ARGS[@]}" "$@"
