# TF_MultiProtoDG Ablation Smoke Runbook

## Environment

- Python environment: `phmbench`
- Verified on: `2026-03-14`
- Dataset dependency: `/mnt/e/dataset/PHMbench-raw_data/metadata.xlsx`

## Purpose

This runbook validates the study-level ablation chain for `TF_MultiProtoDG`:

- study config loading
- variant selection
- per-variant pipeline execution
- train / val / best-checkpoint load / test
- diagnostics export
- study-level summary generation

The reduced smoke uses:

- `iterations = 1`
- `num_epochs = 1`
- two variants:
  - `branch_full_ce_only`
  - `proto_multi_no_contrast`

Smoke mode also forces `data.num_workers = 0` through the study config to avoid multiprocessing semaphore issues in restricted environments.

## Verified Commands

### 1. Pytest-based environment smoke

Verified command:

```bash
/home/xuanli/miniforge/envs/phmbench/bin/python -m pytest test/test_tf_multi_proto_ablation_phmbench_smoke.py -m slow
```

Observed result:

- `1 passed`

### 2. Operator-facing CLI smoke

Verified command:

```bash
/home/xuanli/miniforge/envs/phmbench/bin/python scripts/run_tf_multi_proto_dg_ablation.py \
  --study-config configs/experiments/01_cross_domain/X_DG/tf_multi_proto_dg_ablation/study.yaml \
  --limit-variants branch_full_ce_only proto_multi_no_contrast \
  --smoke \
  --iterations 1 \
  --num-epochs 1 \
  --output-dir /tmp/tf_multi_proto_dg_ablation_cli_smoke2
```

Observed result:

- both variants completed with `success=True`
- summary files were generated under `/tmp/tf_multi_proto_dg_ablation_cli_smoke2`

## Expected Artifacts

Study-level outputs:

- `ablation_runs.csv`
- `ablation_summary.csv`
- `ablation_summary.md`
- `variant_manifest.json`

Per-run outputs under each variant directory:

- Lightning checkpoint files
- `test_result_0.csv`
- `diagnostics/feature_map.json`
- `diagnostics/diagnostics_context.json`
- `diagnostics/prototype_cards.json` and `diagnostics/prototype_health.json` only when prototype diagnostics are enabled

## Expected Summary Semantics

- `branch_full_ce_only`
  - `success=True`
  - no prototype diagnostics summary values
  - `test_assignment_ratio` stays `0`
- `proto_multi_no_contrast`
  - `success=True`
  - prototype diagnostics summary values are present
  - `test_contrastive_loss` stays `0` because the contrastive weight is zeroed, while prototype state still participates

## Notes

- CUDA may hard-fallback to CPU if the runtime cannot initialize CUDA cleanly.
- The environment still emits the upstream `pynvml` deprecation warning.
- In smoke mode, low-worker warnings from Lightning are expected because the runbook intentionally uses `data.num_workers = 0`.
