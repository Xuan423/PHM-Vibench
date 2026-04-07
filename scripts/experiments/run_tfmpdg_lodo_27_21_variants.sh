#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/xuanli/work/PHM-Vibench"
cd "${REPO_ROOT}"

CONFIG="configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg_full_baseline_k2_w002_adapteffk.yaml"
RUN_TAG="lsfix_5seed_20260402"
LOG_ROOT="results/demo/tfmpdg_lodo_27_21_${RUN_TAG}_logs"
mkdir -p "${LOG_ROOT}"

COMMON_ARGS=(
  "--config" "${CONFIG}"
  "--override" "environment.iterations=5"
  "--override" "environment.seed=42"
  "--override" "trainer.num_epochs=100"
  "--override" "trainer.patience=50"
  "--override" "trainer.gpus=1"
  "--override" "trainer.device=cuda"
  "--override" "model.export_diagnostics=false"
)

run_case() {
  local case_id="$1"
  local project="$2"
  local output_dir="$3"
  local target_system_id="$4"
  local source_domains="$5"
  local target_domain="$6"
  local variant="$7"
  local log_file="${LOG_ROOT}/${case_id}.log"

  local -a variant_overrides=()
  if [[ "${variant}" == "nocl" ]]; then
    variant_overrides+=("--override" "model.use_contrastive_head=false")
  elif [[ "${variant}" == "single" ]]; then
    variant_overrides+=("--override" "model.num_prototypes_per_class=1")
  elif [[ "${variant}" == "full_k4" ]]; then
    variant_overrides+=("--override" "model.num_prototypes_per_class=4")
  fi

  echo "[$(date '+%F %T')] [start] ${case_id}" | tee -a "${log_file}"
  echo "[meta] system=${target_system_id} source=${source_domains} target=${target_domain} variant=${variant}" | tee -a "${log_file}"

  conda run -n phmbench python main.py \
    "${COMMON_ARGS[@]}" \
    --override "environment.project=${project}" \
    --override "environment.output_dir=${output_dir}" \
    --override "task.target_system_id=[${target_system_id}]" \
    --override "task.source_domain_id=[${source_domains}]" \
    --override "task.target_domain_id=[${target_domain}]" \
    "${variant_overrides[@]}" 2>&1 | tee -a "${log_file}"

  echo "[$(date '+%F %T')] [done] ${case_id}" | tee -a "${log_file}"
}

run_lodo_group() {
  local system_id="$1"
  local variants_csv="$2"
  shift
  shift
  local -a domains=("$@")
  IFS=',' read -r -a variants <<< "${variants_csv}"

  for target_domain in "${domains[@]}"; do
    local -a source_domains_arr=()
    for d in "${domains[@]}"; do
      if [[ "${d}" != "${target_domain}" ]]; then
        source_domains_arr+=("${d}")
      fi
    done
    local source_csv
    source_csv="$(IFS=,; echo "${source_domains_arr[*]}")"

    for variant in "${variants[@]}"; do
      local case_id="t${system_id}_d${target_domain}_${variant}"
      local project="demo_tfmpdg_lodo_t${system_id}_d${target_domain}_${variant}_${RUN_TAG}"
      local output_dir="results/demo/tfmpdg_lodo_t${system_id}_d${target_domain}_${variant}_${RUN_TAG}"
      run_case "${case_id}" "${project}" "${output_dir}" "${system_id}" "${source_csv}" "${target_domain}" "${variant}"
    done
  done
}

# sys27: 3-domain LOO
run_lodo_group 27 "full,nocl,single" 0 1 2
# sys21: user-specified 4-domain LOO
run_lodo_group 21 "full,nocl,single,full_k4" 14 16 22 24

echo "[$(date '+%F %T')] all LOO runs completed."
