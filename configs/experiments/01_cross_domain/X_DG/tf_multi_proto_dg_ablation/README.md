# TF_MultiProtoDG Ablation Study

This directory defines a study-level configuration for controlled ablations on the existing `TF_MultiProtoDG` model line.

## Files

- `study.yaml`: canonical ablation matrix for the current implementation.

## Scope

The study covers one reference run plus three ablation groups:

- Branch structure: `branch_time_only_ce`, `branch_freq_only_ce`, `branch_full_ce_only`
- Prototype / contrastive: `proto_single_with_contrast`, `proto_multi_no_contrast`
- Operator / indicator complexity: `ops_basic_only`, `indicators_basic_only`

## Control Strategy

- Structural ablations are expressed with `model.ablation.branch_masks`, `model.ablation.operator_masks.*`, and `model.ablation.indicator_masks.*`.
- Contrastive ablations are expressed with `model.ablation.loss_control.contrastive_scale`.
- Prototype participation is controlled with:
  - `model.ablation.loss_control.prototype_assignment_enabled`
  - `model.ablation.loss_control.prototype_update_enabled`

## Basic Components

The “basic only” variants are intentionally fixed to the minimum set required by the spec:

- Basic operators:
  - time: `Identity`
  - freq: `IdentitySpectrum`
- Basic indicators:
  - time: `Mean`
  - freq: `BandEnergy`

## Execution Knobs

`study.yaml` also defines study-wide execution defaults:

- `execution.iterations`: target number of repeated runs for final aggregation
- `execution.default_num_epochs`: default training length for formal runs
- `execution.smoke_iterations`: reduced iteration count for fast checks
- `execution.smoke_num_epochs`: reduced epoch count for fast checks
- `execution.smoke_num_workers`: reduced dataloader worker count for smoke stability
- `execution.diagnostics_extract`: whether the runner should extract diagnostics summaries

The dedicated ablation runner will consume this file directly and materialize per-variant overrides onto the base demo config `configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg.yaml`.
