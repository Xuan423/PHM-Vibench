# TF_MultiProtoDG Batch Study Smoke Runbook

## Environment

- Python environment: `phmbench`
- Dataset dependency: `/mnt/e/dataset/PHMbench-raw_data/metadata.xlsx`
- Verified target flow:
  - taskset-driven ablation study
  - taskset-driven hparam study

## Purpose

This runbook validates the second-stage batch experiment chain for `TF_MultiProtoDG`:

- reusable `taskset` loading
- independent `ablation` and `hparam` study loading
- generic study runner execution
- train / val / best-checkpoint load / test
- study-level summary generation
- default no-diagnostics behavior for bulk runs

## Smoke Scope

The reduced smoke uses:

- `iterations = 1`
- `num_epochs = 1`
- `data.num_workers = 0`
- demo taskset: `configs/experiments/01_cross_domain/X_DG/tf_multi_proto_dg_batch/tasksets/demo_custom_tasks.yaml`
- one ablation item: `baseline_full`
- one hparam item: the first expanded item from the hparam grid

## Verified Commands

### 1. Pytest-based environment smoke

```bash
/home/xuanli/miniforge/envs/phmbench/bin/python -m pytest test/test_tf_multi_proto_batch_phmbench_smoke.py -m slow
```

Expected result:

- `1 passed`

### 2. Generic CLI ablation smoke

```bash
/home/xuanli/miniforge/envs/phmbench/bin/python scripts/run_tf_multi_proto_dg_study.py \
  --taskset configs/experiments/01_cross_domain/X_DG/tf_multi_proto_dg_batch/tasksets/demo_custom_tasks.yaml \
  --study-config configs/experiments/01_cross_domain/X_DG/tf_multi_proto_dg_batch/ablation/study.yaml \
  --limit-items baseline_full \
  --smoke \
  --iterations 1 \
  --num-epochs 1 \
  --output-dir /tmp/tf_multi_proto_dg_batch_ablation_smoke
```

### 3. Generic CLI hparam smoke

```bash
/home/xuanli/miniforge/envs/phmbench/bin/python scripts/run_tf_multi_proto_dg_study.py \
  --taskset configs/experiments/01_cross_domain/X_DG/tf_multi_proto_dg_batch/tasksets/demo_custom_tasks.yaml \
  --study-config configs/experiments/01_cross_domain/X_DG/tf_multi_proto_dg_batch/hparam/study.yaml \
  --limit-items lcs_0p0__lce_0p1__tau_0p2__lr_0p001__wd_0p0 \
  --smoke \
  --iterations 1 \
  --num-epochs 1 \
  --output-dir /tmp/tf_multi_proto_dg_batch_hparam_smoke
```

### 4. Reusable shell launcher examples

```bash
bash scripts/experiments/run_tf_multi_proto_ablation_tasks.sh \
  --taskset configs/experiments/01_cross_domain/X_DG/tf_multi_proto_dg_batch/tasksets/leave_one_out.yaml \
  --smoke \
  --iterations 1 \
  --num-epochs 1 \
  --limit-items baseline_full
```

```bash
bash scripts/experiments/run_tf_multi_proto_hparam_tasks.sh \
  --taskset configs/experiments/01_cross_domain/X_DG/tf_multi_proto_dg_batch/tasksets/demo_custom_tasks.yaml \
  --smoke \
  --iterations 1 \
  --num-epochs 1 \
  --limit-items lcs_0p0__lce_0p1__tau_0p2__lr_0p001__wd_0p0
```

## Expected Artifacts

For ablation runs:

- `ablation_runs.csv`
- `ablation_summary.csv`
- `ablation_summary.md`
- `ablation_manifest.json`

For hparam runs:

- `hparam_runs.csv`
- `hparam_summary.csv`
- `hparam_summary.md`
- `hparam_manifest.json`

## Default Diagnostics Policy

- Bulk runs default `model.export_diagnostics = false`.
- Summary files are still generated without interpretable export files.
- Enable diagnostics only for targeted debugging runs by passing `--enable-diagnostics` to the generic CLI or shell launcher.

## Notes

- CUDA may hard-fallback to CPU if runtime initialization fails.
- The upstream `pynvml` deprecation warning may still appear in `phmbench`.
- Smoke mode intentionally uses low worker counts for stability.
