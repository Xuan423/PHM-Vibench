# TF_MultiProtoDG Canonical Ablation Smoke Runbook

## Environment

- Python environment: `phmbench`
- Verified on: `2026-04-07`
- Dataset dependency: `/mnt/e/dataset/PHMbench-raw_data/metadata.xlsx`

## Purpose

This runbook validates the canonical taskset-driven ablation chain for `TF_MultiProtoDG`:

- taskset loading
- study config loading
- item selection
- per-item pipeline execution
- train / val / best-checkpoint load / test
- study-level summary generation

## Smoke Scope

The reduced smoke uses:

- `iterations = 1`
- `num_epochs = 1`
- canonical taskset: `configs/experiments/01_cross_domain/X_DG/tf_multi_proto_dg_batch/tasksets/sys27_t012.yaml`
- local smoke override file: `/tmp/tfmpdg_smoke_local.yaml`
- two ablation items:
  - `full_model`
  - `tf_only_ce`

Smoke mode keeps `data.num_workers = 0` through the local override for stable CPU-side validation.

## Verified Commands

### 1. Pytest-based environment smoke

```bash
/home/xuanli/miniforge/envs/phmbench/bin/python -m pytest \
  test/test_tf_multi_proto_ablation_phmbench_smoke.py \
  -m slow
```

### 2. Operator-facing launcher smoke

```bash
/home/xuanli/miniforge/envs/phmbench/bin/bash scripts/experiments/run_tf_multi_proto_ablation_tasks.sh \
  --taskset configs/experiments/01_cross_domain/X_DG/tf_multi_proto_dg_batch/tasksets/sys27_t012.yaml \
  --local-config /tmp/tfmpdg_smoke_local.yaml \
  --limit-items full_model tf_only_ce \
  --smoke \
  --iterations 1 \
  --num-epochs 1 \
  --output-dir /tmp/tf_multi_proto_dg_ablation_cli_smoke2
```

## Expected Artifacts

Study-level outputs:

- `ablation_runs.csv`
- `ablation_summary.csv`
- `ablation_summary.md`
- `ablation_manifest.json`

Per-run outputs under each task/item directory:

- Lightning checkpoint files
- `test_result_0.csv`
- `test_result_mean.csv`

## Expected Summary Semantics

- `full_model`
  - `success=True`
  - prototype auxiliary metrics are present because the mainline keeps contrastive and prototype regularization enabled
- `tf_only_ce`
  - `success=True`
  - `test_contrastive_loss` stays `0`
  - prototype-assignment and prototype-update auxiliary behavior is disabled through the ablation loss-control mask

## Notes

- CUDA may hard-fallback to CPU if the runtime cannot initialize CUDA cleanly.
- The environment still emits the upstream `pynvml` deprecation warning.
- In smoke mode, low-worker warnings from Lightning are expected because the runbook intentionally uses `data.num_workers = 0`.
- The legacy standalone ablation runner has been retired in favor of the canonical taskset-driven launcher.
