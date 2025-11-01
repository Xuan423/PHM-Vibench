import json
from pathlib import Path

import pandas as pd

from script.hparam_eval import tspn_hust_ablation as ablation


def test_build_and_dry_run(tmp_path):
    variants = ablation.VARIANT_REGISTRY[:2]
    specs = ablation.build_run_specs(
        variants=variants,
        config_root=ablation.CONFIG_ROOT_DEFAULT,
        output_root=tmp_path,
        pipeline="Pipeline_01_default",
        global_notes="pytest-dry-run",
    )
    assert len(specs) == len(variants)
    assert all(spec.metadata["variant_label"] for spec in specs)

    results = ablation.execute_runs(
        specs,
        max_parallel=1,
        timeout=None,
        devices=None,
        device_pool=None,
        pipeline="Pipeline_01_default",
        dry_run=True,
    )

    for result in results:
        summary_path = result.spec.output_dir / "run_summary.json"
        assert summary_path.exists()
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert payload["status"] == "success"
        assert payload["metadata"]["variant_label"]

    ablation.summarise_results(results, tmp_path)
    summary_csv = tmp_path / "ablation_summary.csv"
    assert summary_csv.exists()
    df = pd.read_csv(summary_csv)
    assert "variant" in df.columns
    assert any(col.startswith("meta_variant_label") for col in df.columns)
