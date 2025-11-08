# Unified TSPN Contrastive Ablation Guide

This document describes the unified ablation workflow that extends the original HUST-only scripts to cover both HUST and SDUST datasets with consistent branch-weight handling, episodic batch chunking, and consolidated accuracy reporting via `tspn_resummarise.py`.

## Overview

- Entrypoint: `script/hparam_eval/tspn_contrastive_ablation.py`
- Shell helper: `script/hparam_eval/run_tspn_contrastive_ablation.sh`
- Config root: `configs/experiments/tspn_contrastive_ablation/`
  - `hust/`: inherits demo few-shot hyperparameters
  - `sdust/`: applies SDUST-specific few-shot overrides
- Branch toggles are controlled via `task.contrastive.branches` using `enabled` / `participates` fields. Disabling a branch now keeps the three-way weighting intact while zeroing the contribution in the total loss.
- `task.batch_size` is the single source of truth for episodic batching; it caps the number of query samples per Lightning iteration while each chunk continues to include the full support set.

## Dataset Recipes

### HUST

- Target system: `[19]`
- Domain tasks (target domain `10`):
  1. Sources `[0,1,2,3,4]` → target `[10]`
  2. Sources `[0,2,4,6,8]` → target `[10]`
  3. Sources `[1,3,5,7,9]` → target `[10]`
- Few-shot sampler: reuse demo defaults (episodes over 4 domains × 5 classes).

### SDUST

- Target system: `[21]`
- Domain tasks:
  1. `[0,4,9,13,17,21]` → `[37]`
  2. `[1,5,10,14,18,22]` → `[38]`
  3. `[2,6,11,15,19,23]` → `[39]`
  4. `[3,7,12,16,20,24]` → `[40]`
- Few-shot overrides in `sdust/base.yaml`:
  - `domains_per_episode=4`
  - `classes_per_domain=8`
  - `support_per_class=3`
  - `query_per_class=10`

## Variants

| Variant | Description | Branch participation |
|---------|-------------|----------------------|
| `contrastive_full` | All branches active (default `base.yaml`) | `support_support`, `support_query`, `query_query` participate |
| `contrastive_ssqq` | Support-query branch excluded from total loss | Branch remains enabled but `participates=false` |
| `contrastive_ss_only` | Only support-support branch contributes | `support_query` / `query_query` disabled |
| `contrastive_qq_only` | Only query-query branch contributes | `support_support` / `support_query` disabled |
| `contrastive_qs_only` | Only support-query contributes | `support_support` / `query_query` set to `participates=false` |
| `support_no_align` | Support prototype alignment & CE disabled | Support loss weight = 0; all branches participate |
| `contrastive_disabled` | Contrastive objective fully disabled | Branches disabled; loss weight fixed to 0 |
| `baseline_tspn` | No episodic sampling, no contrastive loss | Few-shot disabled, branches disabled |

Contrastive loss weights sweep over `{0.2, 0.3}` for all variants where `requires_weight_sweep=True`.

## Running Experiments

### Python module

```bash
python -m script.hparam_eval.tspn_contrastive_ablation \
  --datasets hust sdust \
  --variants contrastive_full support_no_align \
  --output-root save/contrastive_ablation \
  --max-parallel 2 \
  --device-pool 0,1
```

Key flags:

- `--datasets`: subset of `{hust, sdust}` (default: both)
- `--variants`: optional variant filter (default: all)
- `--notes`: appended to `environment.notes`
- Standard device / pipeline options match the previous runner

### Shell wrapper

```bash
CONTRASTIVE_ABLATION_DATASETS="hust sdust" \
CONTRASTIVE_ABLATION_VARIANTS="contrastive_full contrastive_qs_only" \
script/hparam_eval/run_tspn_contrastive_ablation.sh --max-parallel 2
```

Environment variables:

- `CONTRASTIVE_ABLATION_DATASETS` – whitespace separated dataset list
- `CONTRASTIVE_ABLATION_VARIANTS` – optional variant list
- `CONTRASTIVE_ABLATION_DEVICES`, `CONTRASTIVE_ABLATION_DEVICE_POOL` – GPU bindings
- `CONTRASTIVE_ABLATION_TIMEOUT`, `CONTRASTIVE_ABLATION_PIPELINE`, `CONTRASTIVE_ABLATION_NOTES` mirror CLI flags

## Outputs

Structure under `save/contrastive_ablation/`:

- `<dataset>/<run_name>/train.log`
- `<dataset>/<run_name>/run_summary.json`
- `<dataset>/<run_name>/resolved_config.yaml`
- `ablation_summary.csv` / `ablation_summary.md` (dataset column distinguishes runs)

`ablation_summary.csv` now contains:

- `dataset`, `variant`, runtime, status
- Metadata columns prefixed with `meta_` (domain label, branch participation, loss weight)
- Accuracy columns auto-detected (`test_acc`, `test_acc_HUST`, `test_acc_SDUST`, …)

## Accuracy Aggregation

Use the shared summariser to consolidate historical runs without relaunching experiments:

```bash
python script/hparam_eval/tspn_resummarise.py \
  --root save/contrastive_ablation \
  --output save/contrastive_ablation/tspn_resummary.csv \
  --markdown save/contrastive_ablation/tspn_resummary.md
```

`tspn_resummarise.py` scans every `iter_*` directory, records the effective `test_acc*` metric for each iteration, and preserves the hyperparameter columns (`loss_weight`, episodic sampler settings, seed) needed to compare the new ss-only / qq-only ablations with the baseline variants.

## Migration Notes

- Existing HUST-only configs remain available; new configs live in `configs/experiments/tspn_contrastive_ablation`.
- Branch toggles now specify `enabled` vs `participates`. The trainer computes weights from positive pairs before masking contributions, keeping comparative analysis consistent.
- Support-alignment ablation sets `support_loss.loss_weight = 0` to ensure the support CE term is removed alongside prototype alignment.
