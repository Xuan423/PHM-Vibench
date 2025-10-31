import pandas as pd
import pytest

from script.hparam_eval.utils import build_override, launch_experiment, load_metrics, LaunchResult
from script.hparam_eval.tspn_hparam_eval import CONFIG_ROOT_DEFAULT, RunResult, RunSpec, build_run_specs, summarise_results

from src.configs.config_utils import load_config


BASE_CONFIG = CONFIG_ROOT_DEFAULT / "base_import.yaml"


def test_build_override_round_trip(tmp_path):
    overrides = {
        "environment.project": "pytest_build_override",
        "task.few_shot.enabled": True,
        "task.few_shot.support_per_class": 7,
    }
    override_path = build_override(BASE_CONFIG, overrides, "pytest-run", cache_dir=tmp_path)
    resolved = load_config(str(override_path))
    assert resolved.environment.project == "pytest_build_override"
    assert resolved.task.few_shot.enabled is True
    assert resolved.task.few_shot.support_per_class == 7


def test_launch_experiment_dry_run(tmp_path):
    overrides = {"environment.project": "pytest_dry_run"}
    config_path = build_override(BASE_CONFIG, overrides, "pytest-dry-run", cache_dir=tmp_path)
    log_path = tmp_path / "train.log"
    result = launch_experiment(
        config_path,
        log_path=log_path,
        timeout=10,
        dry_run=True,
    )
    assert result.returncode == 0
    assert log_path.exists()
    assert "Would launch main.py" in log_path.read_text(encoding="utf-8")


def test_load_metrics_summary(tmp_path):
    log_dir = tmp_path / "version_0"
    log_dir.mkdir()
    df = pd.DataFrame(
        {
            "step": [0, 1, 2],
            "val_acc": [0.1, 0.2, 0.3],
            "val_total_loss": [1.2, 0.9, 0.8],
        }
    )
    df.to_csv(log_dir / "metrics.csv", index=False)
    info = load_metrics(log_dir)
    assert info["summary"]["val_acc"] == pytest.approx(0.3)
    assert info["summary"]["val_total_loss"] == pytest.approx(0.8)


def test_summarise_results_creates_reports(tmp_path):
    success_spec = RunSpec(
        name="contrastive__loss_weight_0p1__domains_3",
        base_config=BASE_CONFIG,
        overrides={},
        hyperparams={
            "contrastive_loss_weight": 0.1,
            "domains_per_episode": 3,
            "classes_per_domain": 3,
            "support_per_class": 2,
            "query_per_class": 5,
        },
        output_dir=tmp_path / "contrastive" / "success",
    )
    success_result = RunResult(
        spec=success_spec,
        launch=LaunchResult(returncode=0, runtime=1.2),
        status="success",
        metrics={"summary": {"test_acc": 0.92, "val_total_loss": 0.7}},
        log_dir=None,
    )

    failed_spec = RunSpec(
        name="contrastive__loss_weight_0p3__domains_4",
        base_config=BASE_CONFIG,
        overrides={},
        hyperparams={
            "contrastive_loss_weight": 0.3,
            "domains_per_episode": 4,
            "classes_per_domain": 3,
            "support_per_class": 3,
            "query_per_class": 10,
        },
        output_dir=tmp_path / "contrastive" / "failed",
    )
    failed_result = RunResult(
        spec=failed_spec,
        launch=LaunchResult(returncode=1, runtime=0.8),
        status="failed",
        metrics={"summary": {}},
        log_dir=None,
    )

    summarise_results([success_result, failed_result], tmp_path)

    summary_csv = tmp_path / "contrastive_sweep_summary.csv"
    summary_md = tmp_path / "contrastive_sweep_summary.md"
    assert summary_csv.exists()
    assert summary_md.exists()

    df = pd.read_csv(summary_csv)
    assert "test_acc" in df.columns
    success_row = df.loc[df["run_name"] == success_spec.name].squeeze()
    assert success_row["status"] == "success"
    assert success_row["test_acc"] == pytest.approx(0.92)
    assert success_row["contrastive_loss_weight"] == pytest.approx(0.1)

    failed_row = df.loc[df["run_name"] == failed_spec.name].squeeze()
    assert failed_row["status"] == "failed"
    assert pytest.isna(failed_row["test_acc"])


def test_build_run_specs_respects_target_hyperparams(tmp_path):
    grid_path = CONFIG_ROOT_DEFAULT / "contrastive_grid.yaml"
    specs = build_run_specs(
        grid_path=grid_path,
        output_root=tmp_path,
        pipeline=None,
        global_notes="pytest",
        limit=3,
    )

    assert len(specs) == 3
    expected_keys = {
        "contrastive_loss_weight",
        "domains_per_episode",
        "classes_per_domain",
        "support_per_class",
        "query_per_class",
    }
    for spec in specs:
        assert set(spec.hyperparams.keys()) == expected_keys
        assert spec.output_dir.parent.name == "contrastive"
        assert spec.overrides["environment.output_dir"] == str(spec.output_dir)
        assert spec.overrides["environment.project"].startswith("contrastive_")
