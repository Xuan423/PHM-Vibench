# TF_MultiProtoDG LSEP-Simplified Smoke Validation

Canonical environment:

```bash
conda activate phmbench
```

Verified unit and integration checks:

```bash
python -m pytest \
  test/test_tf_multi_proto_lsep_simplified_config.py \
  test/test_tf_multi_proto_local_channel_fusion.py \
  test/test_tf_multi_proto_role_compression.py \
  test/test_tf_multi_proto_cross_evidence_pooling.py \
  test/test_tf_multi_proto_lsep_prototypes.py \
  test/test_tf_multi_proto_lsep_model_integration.py \
  test/test_tf_multi_proto_lsep_diagnostics.py -q
```

Additional compatibility checks that were also run:

```bash
python -m pytest \
  test/test_tf_multi_proto_model_integration.py \
  test/test_interpretable_tf_diagnostics.py -q
```

Verified `main.py` smoke entry:

```bash
python main.py \
  --config configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg_lsep_simplified.yaml \
  --override environment.output_dir=<tmp-output-dir> \
  --override environment.iterations=1 \
  --override trainer.num_epochs=1 \
  --override trainer.device=cpu \
  --override trainer.gpus=0 \
  --override trainer.early_stopping=false \
  --override model.device=cpu \
  --override model.export_diagnostics=true \
  --override data.num_workers=0 \
  --override data.batch_size=8 \
  --override data.num_window=4
```

Expected artifacts under the generated `iter_0/` directory:

- `all_results.csv`
- `test_result_0.csv`
- `diagnostics/diagnostics_context.json`
- `diagnostics/feature_map.json`
- `diagnostics/prototype_cards.json`
- `diagnostics/prototype_health.json`
- `diagnostics/alpha_time_report.json`
- `diagnostics/alpha_freq_report.json`
- `diagnostics/role_basis_time.json`
- `diagnostics/role_basis_freq.json`
- `diagnostics/patch_band_similarity.json`
- `diagnostics/compact_concept_report.json`

Notes:

- The dedicated demo config is [configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg_lsep_simplified.yaml](/home/xuanli/work/PHM-Vibench/configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg_lsep_simplified.yaml).
- The default demo alias [configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg.yaml](/home/xuanli/work/PHM-Vibench/configs/demo/01_cross_domain/X_DG/tf_multi_proto_dg.yaml) now points to the same simplified structured-path family.
