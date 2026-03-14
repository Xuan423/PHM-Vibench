# TF_MultiProtoDG Smoke Validation

## Target Environment

- Conda environment: `phmbench`
- Working directory: repository root
- Goal: verify the approved reduced chain from config loading to diagnostics export

## Commands

```bash
conda run -n phmbench python -m pytest \
  test/test_tf_multi_proto_tensor_ops.py \
  test/test_tf_multi_proto_operators.py \
  test/test_tf_multi_proto_indicators.py \
  test/test_tf_multi_proto_prototypes.py \
  test/test_tf_multi_proto_model_integration.py \
  test/test_interpretable_tf_diagnostics.py \
  test/test_tf_multi_proto_phmbench_smoke.py
```

## What the reduced smoke covers

1. Load the approved demo config `configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg.yaml`.
2. Build `TF_MultiProtoDG` with metadata-driven class and channel inference.
3. Run `Default_task` through train, val, and test step contracts on a reduced dummy batch.
4. Save and reload a checkpoint-shaped artifact.
5. Export `diagnostics/feature_map.json`, `diagnostics/prototype_cards.json`, and `diagnostics/prototype_health.json`.

## Expected artifacts

- `diagnostics/feature_map.json`
- `diagnostics/prototype_cards.json`
- `diagnostics/prototype_health.json`
- temporary smoke checkpoint file

## Notes

- This smoke validation is intentionally reduced in data scale and epoch count, but it preserves the real DG task, model, diagnostics, and checkpoint contracts.
- Full dataset runs should still use `python main.py --config configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg.yaml` inside `phmbench`.
