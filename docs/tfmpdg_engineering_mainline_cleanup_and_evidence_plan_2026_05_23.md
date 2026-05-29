# TF_MultiProtoDG v4.1-Restored Mainline Validation Plan

Date: 2026-05-23

This document records the revised validation mainline after the no-local
cleanup produced a severe 15-seed regression. It is a development handoff for
the next long-run validation.

## Goal

Build a TF_MultiProtoDG mainline that can be tested against the v4.1 standard
while keeping the harmful direct-local decision paths disabled.

The 15-seed regression showed that the old no-local-off boundary was too strict:
the stable sys27 formal window depends on `local_anomaly_residual_mode=mlp`,
`proto_contrastive_weight=0.05`, `proto_contrastive_temperature=0.3`, and
`task.weight_decay=0.001`. The local anomaly residual MLP is therefore retained
as part of the fused concept `h`.

The 2026-05-24 anchor-only probe further split the dependency: final logits do
not need local/prototype residual votes to recover v4.1-level accuracy, but the
training-time fused `h` scaffold is needed to learn a strong prototype/class
anchor geometry. The clean mainline therefore uses `prototype_decision_output:
anchor_only`: public logits, CE loss, confidence, and reported predictions come
from the global class-anchor path, while local anomaly residuals remain only as
a training scaffold inside `h`.

The disabled paths remain:

- local prototype concept as the class concept,
- local prototype assignment/routing,
- local slot verification in prototype residual logits,
- local classifier residual inputs.

## Current Canonical Settings

Source of truth:

- `configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg.yaml`
- `src/model_factory/X_model/tf_multi_proto_dg/config_schema.py`
- `src/model_factory/X_model/tf_multi_proto_dg/model.py`
- `src/model_factory/X_model/tf_multi_proto_dg/prototype_head.py`

Canonical clean settings:

```yaml
data.num_workers: 4
data.pin_memory: false
task.weight_decay: 0.001
trainer.monitor: "val_total_loss"
model.prototype_concept_input: "h"
model.prototype_decision_output: "anchor_only"
model.prototype_assignment_input: "concept"
model.local_anomaly_residual_mode: "mlp"
model.prototype_residual_logit_mode: "agreement"
model.proto_contrastive_weight: 0.05
model.proto_contrastive_temperature: 0.3
model.classifier.residual_input: "semantic_stats"
```

The retained final classifier evidence is global-anchor based:

- class anchor scores,
- transparent time and frequency features feeding the anchor concept.

The retained training scaffold is:

- fused concept `h` with the local anomaly residual MLP,
- joint multi-prototype contrast and prototype assignment in the fused concept
  coordinate,
- prototype residual diagnostics for audit, not for the public final logits.

## Clean Boundary

The following former local-decision paths are rejected at config parsing:

- `prototype_concept_input=local_anomaly`
- `prototype_concept_input=relative/dual_relative/dual_relative_learned`
- `prototype_assignment_input=local_anomaly`
- `local_anomaly_residual_mode=direct/direct_delta`
- `prototype_residual_logit_mode=agreement_local_slot_verify`
- `prototype_residual_logit_mode=local_evidence`
- `prototype_residual_logit_mode=agreement_local_centered`
- `prototype_residual_logit_mode=local_competition`
- `prototype_residual_logit_mode=global_local_consensus`
- `prototype_residual_logit_mode=agreement_global_local_consensus`
- `classifier.residual_input=local_anomaly_summary`
- `classifier.residual_input=local_anomaly_profile`

`PrototypeHead` no longer accepts local residual-logit modes and no longer mixes
a local routing concept into prototype assignment. In the model wrapper,
`_run_proto_head(...)` only passes the fused concept and optional global anchor
concepts.

The 2026-05-24 cleanup also removed legacy local-slot compatibility outputs
from `PrototypeHead`: `local_proto_routing_scores`, `local_proto_pool_weights`,
and `local_slot_reliability` are no longer emitted. The classifier residual
path no longer has production-code branches for `local_anomaly_summary` or
`local_anomaly_profile`; these values are still computed only where needed for
post-hoc local evidence inspection and profile diagnostics.

With `prototype_decision_output=anchor_only`, `extras["logits"]` is replaced by
`joint_anchor_scores` before the task computes CE. This is the enforceable clean
boundary: local/prototype residual evidence may shape representation learning
through the fused concept and contrastive scaffold, but it cannot directly vote
for the class in the loss or in inference.

## Why The Boundary Changed

The no-local cleanup used `local_anomaly_residual_mode=off` plus
`proto_contrastive_weight=0.1` and `proto_contrastive_temperature=0.2`. In the
latest 15-seed screen this produced T0/T1 severe drops while final predictions
stayed almost identical to the anchor path, indicating that the representation
and prototype training geometry had degraded before any local logit correction
could matter.

The comparable historical `local_off_stage_sys27_t012_ep100` window kept
`prototype_residual_logit_mode=agreement` and did not use local slot verifier,
but it did retain `local_anomaly_residual_mode=mlp` and sys27 formal
regularization/prototype contrast settings. That is the dependency set restored
here.

```text
Can the restored clean-decision branch recover the v4.1-level 15-seed standard?
```

If long-run accuracy still drops materially, the next valid direction is a
factorized ablation over the restored dependencies (`mlp`, contrastive
temperature/weight, weight decay, and `prototype_decision_output`). Do not
re-enable `agreement_local_slot_verify` by default unless a separate diagnostic
branch proves it is stable across LOO tasks.

## 2026-05-24 Long-Run Evidence

The clean branch was validated with low-I/O 100epoch, 5-seed runs on sys27
T0/T1/T2 after enabling `prototype_decision_output=anchor_only`.

| run | output dir | mean acc | min acc | max acc | mean final-anchor disagreement |
| --- | --- | ---: | ---: | ---: | ---: |
| v4.1 T0 reference | `results/tmp/tfmpdg_current_ep100_t0_workers4_formal` | 0.9598 | 0.9353 | 0.9799 | 0.0004 |
| clean anchor-only T0 | `results/tmp/tfmpdg_anchor_only_decision_ep100_t0_workers0` | 0.9540 | 0.9286 | 0.9821 | 0.0000 |
| clean anchor-only T1 | `results/tmp/tfmpdg_clean_anchor_only_yaml_ep100_sys27_t1_workers0` | 0.9473 | 0.9152 | 0.9754 | 0.0000 |
| clean anchor-only T2 | `results/tmp/tfmpdg_clean_anchor_only_yaml_ep100_sys27_t2_workers0` | 0.9862 | 0.9732 | 1.0000 | 0.0000 |

Seed-level evidence:

- T0 clean anchor-only: `0.9821, 0.9286, 0.9621, 0.9688, 0.9286`.
- T1 clean anchor-only: `0.9710, 0.9754, 0.9241, 0.9509, 0.9152`.
- T2 clean anchor-only: `0.9888, 0.9866, 0.9821, 1.0000, 0.9732`.

The result closes the immediate regression question: the severe drop after the
earlier cleanup was caused by removing the training-time fused `h` scaffold and
changing prototype regularization, not by removing local/prototype residual
votes from final prediction. The current clean branch restores the useful
training geometry while enforcing that reported logits, CE, confidence, and
predictions are global-anchor only.

After the dead-field cleanup, a 1epoch T0 smoke at
`results/tmp/tfmpdg_clean_anchor_only_post_cleanup_smoke_ep1_t0_workers0`
matched the expected clean boundary: `test_acc == test_decision_anchor_acc ==
0.4486607` and `test_decision_anchor_final_disagree == 0.0`. This smoke is not
an accuracy proof; it verifies that removing the unused local-slot outputs did
not reopen or break the anchor-only decision path.

Short 30epoch screens remain useful for rejecting obvious failures, but they
underestimate this family. T2 30epoch had a low seed at `0.8616`; the same clean
setting at 100epoch recovered to a 5-seed minimum of `0.9732`. Therefore a
30epoch result around 0.90 is not by itself a rejection criterion for this
restored anchor-only family.

## Validation Protocol

Run low-I/O first:

```bash
/home/xuanli/miniforge/envs/phmbench/bin/python main.py \
  --config configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg.yaml \
  --override data.num_workers=0 \
  --override data.pin_memory=false \
  --override trainer.num_epochs=30 \
  --override trainer.patience=30 \
  --override environment.output_dir=results/tmp/tfmpdg_v41_restored_ep30
```

Escalate only if the 30-epoch screen is not an obvious reject:

```bash
/home/xuanli/miniforge/envs/phmbench/bin/python main.py \
  --config configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg.yaml \
  --override data.num_workers=0 \
  --override data.pin_memory=false \
  --override trainer.num_epochs=100 \
  --override trainer.patience=50 \
  --override environment.output_dir=results/tmp/tfmpdg_v41_restored_ep100
```

Acceptance should compare the same task family, same target-domain protocol,
same monitor `val_total_loss`, and the same iteration count used by the v4.1
baseline. Do not accept based on best single seed.

For sys27 LOO validation, use the explicit task overrides below.

T0:

```bash
/home/xuanli/miniforge/envs/phmbench/bin/python main.py \
  --config configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg.yaml \
  --override environment.seed=42 \
  --override environment.iterations=5 \
  --override environment.output_dir=results/tmp/tfmpdg_clean_anchor_only_yaml_ep100_sys27_t0_workers0 \
  --override task.target_system_id=[27] \
  --override task.source_domain_id=[1,2] \
  --override task.target_domain_id=[0] \
  --override data.num_workers=0 \
  --override data.pin_memory=false \
  --override trainer.num_workers=0 \
  --override trainer.monitor=val_total_loss \
  --override trainer.num_epochs=100 \
  --override trainer.patience=50 \
  --override trainer.save_last=false
```

T1:

```bash
/home/xuanli/miniforge/envs/phmbench/bin/python main.py \
  --config configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg.yaml \
  --override environment.seed=42 \
  --override environment.iterations=5 \
  --override environment.output_dir=results/tmp/tfmpdg_clean_anchor_only_yaml_ep100_sys27_t1_workers0 \
  --override task.target_system_id=[27] \
  --override task.source_domain_id=[0,2] \
  --override task.target_domain_id=[1] \
  --override data.num_workers=0 \
  --override data.pin_memory=false \
  --override trainer.num_workers=0 \
  --override trainer.monitor=val_total_loss \
  --override trainer.num_epochs=100 \
  --override trainer.patience=50 \
  --override trainer.save_last=false
```

T2:

```bash
/home/xuanli/miniforge/envs/phmbench/bin/python main.py \
  --config configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg.yaml \
  --override environment.seed=42 \
  --override environment.iterations=5 \
  --override environment.output_dir=results/tmp/tfmpdg_clean_anchor_only_yaml_ep100_sys27_t2_workers0 \
  --override task.target_system_id=[27] \
  --override task.source_domain_id=[0,1] \
  --override task.target_domain_id=[2] \
  --override data.num_workers=0 \
  --override data.pin_memory=false \
  --override trainer.num_workers=0 \
  --override trainer.monitor=val_total_loss \
  --override trainer.num_epochs=100 \
  --override trainer.patience=50 \
  --override trainer.save_last=false
```

## Engineering Local Role

Local evidence has two allowed roles:

- training scaffold: local anomaly residual MLP may help form the fused concept
  used by prototype contrastive learning;
- post-hoc engineering inspection:
  identify time patches or frequency bands with abnormal response and expose
  operator/indicator-level regions for review.

Local evidence is not allowed to:

- vote for the final class,
- alter confidence,
- enter local slot verification,
- enter local prototype routing,
- act as a standalone classifier.

Reports may label local findings as `reportable`, `review_only`, or `rejected`,
but those labels are explanatory audit outputs rather than decision inputs.

If a future local method is proposed, it must first pass a diagnostic proof that
local evidence has calibrated class-pair or prototype-slot identity and that
rescue exceeds hurt across LOO targets. Until then, local remains report-only.

## Regression Checks

Required before long-run launch:

```bash
/home/xuanli/miniforge/envs/phmbench/bin/python -m pytest \
  test/test_tf_multi_proto_ablation_config.py \
  test/test_tf_multi_proto_taskset_config.py \
  -q
```

The tests enforce the clean boundary at YAML and config-schema level. Passing
tests do not validate accuracy; they validate that the next accuracy run is
testing the intended clean branch.
