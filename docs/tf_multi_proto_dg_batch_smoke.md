# TF_MultiProtoDG Batch Study Smoke Runbook

## Environment

- Python environment: `phmbench`
- Dataset dependency: `/mnt/e/dataset/PHMbench-raw_data/metadata.xlsx`
- Verified target flow:
  - canonical `main.py` entry
  - taskset-driven ablation study
  - taskset-driven hparam study

## Purpose

This runbook validates the canonical TF_MultiProtoDG execution chain:

- canonical `main.py` smoke execution
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
- canonical taskset: `configs/experiments/01_cross_domain/X_DG/tf_multi_proto_dg_batch/tasksets/sys27_t012.yaml`
- one ablation item: `basic_operators_only`
- one hparam item: `pcw_0p05__pctau_0p1__ls_0p02`
- local smoke override file: `/tmp/tfmpdg_smoke_local.yaml`

## Verified Commands

### 1. Canonical main.py smoke

```bash
/home/xuanli/miniforge/envs/phmbench/bin/python main.py \
  --config configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg.yaml \
  --override environment.output_dir=/tmp/tfmpdg_main_smoke \
  --override environment.iterations=1 \
  --override trainer.num_epochs=1 \
  --override trainer.device=cpu \
  --override trainer.gpus=0 \
  --override trainer.early_stopping=false \
  --override model.device=cpu \
  --override model.export_diagnostics=false \
  --override data.num_workers=0 \
  --override data.batch_size=8 \
  --override data.num_window=4
```

### 2. Generic CLI ablation smoke

```bash
/home/xuanli/miniforge/envs/phmbench/bin/python scripts/run_tf_multi_proto_dg_study.py \
  --taskset configs/experiments/01_cross_domain/X_DG/tf_multi_proto_dg_batch/tasksets/sys27_t012.yaml \
  --study-config configs/experiments/01_cross_domain/X_DG/tf_multi_proto_dg_batch/ablation/study.yaml \
  --local-config /tmp/tfmpdg_smoke_local.yaml \
  --limit-items basic_operators_only \
  --smoke \
  --iterations 1 \
  --num-epochs 1 \
  --output-dir /tmp/tf_multi_proto_dg_batch_ablation_smoke
```

### 3. Generic CLI hparam smoke

```bash
/home/xuanli/miniforge/envs/phmbench/bin/python scripts/run_tf_multi_proto_dg_study.py \
  --taskset configs/experiments/01_cross_domain/X_DG/tf_multi_proto_dg_batch/tasksets/sys27_t012.yaml \
  --study-config configs/experiments/01_cross_domain/X_DG/tf_multi_proto_dg_batch/hparam/study.yaml \
  --local-config /tmp/tfmpdg_smoke_local.yaml \
  --limit-items pcw_0p05__pctau_0p1__ls_0p02 \
  --smoke \
  --iterations 1 \
  --num-epochs 1 \
  --output-dir /tmp/tf_multi_proto_dg_batch_hparam_smoke
```

### 4. Reusable shell launcher examples

```bash
bash scripts/experiments/run_tf_multi_proto_ablation_tasks.sh \
  --taskset configs/experiments/01_cross_domain/X_DG/tf_multi_proto_dg_batch/tasksets/sys27_t012.yaml \
  --local-config /tmp/tfmpdg_smoke_local.yaml \
  --smoke \
  --iterations 1 \
  --num-epochs 1 \
  --limit-items basic_operators_only
```

```bash
bash scripts/experiments/run_tf_multi_proto_hparam_tasks.sh \
  --taskset configs/experiments/01_cross_domain/X_DG/tf_multi_proto_dg_batch/tasksets/sys27_t012.yaml \
  --local-config /tmp/tfmpdg_smoke_local.yaml \
  --smoke \
  --iterations 1 \
  --num-epochs 1 \
  --limit-items pcw_0p05__pctau_0p1__ls_0p02
```

### 5. Pytest-based environment smoke

```bash
/home/xuanli/miniforge/envs/phmbench/bin/python -m pytest \
  test/test_tf_multi_proto_batch_phmbench_smoke.py \
  test/test_tf_multi_proto_ablation_phmbench_smoke.py \
  -m slow
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
- The smoke local override keeps the canonical study/taskset path but forces CPU-friendly trainer/model/data settings.
