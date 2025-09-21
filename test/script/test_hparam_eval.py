import pandas as pd
import pytest
from pathlib import Path

from script.hparam_eval.utils import build_override, launch_experiment, load_metrics, LaunchResult
from script.hparam_eval.tspn_hparam_eval import (
    CONFIG_ROOT_DEFAULT,
    OUTPUT_ROOT_DEFAULT,
    RunResult,
    RunSpec,
    summarise_results,
)

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


def test_summarise_results_creates_reports(tmp_path, monkeypatch):
    baseline_spec = RunSpec(
        name="baseline_original",
        group="baseline",
        base_config=BASE_CONFIG,
        overrides={},
        hyperparams={},
        output_dir=tmp_path / "baseline",
    )
    baseline_result = RunResult(
        spec=baseline_spec,
        launch=LaunchResult(returncode=0, runtime=1.0),
        status="success",
        metrics={"summary": {"val_acc": 0.9, "val_total_loss": 0.75}},
        log_dir=None,
    )

    contrastive_spec = RunSpec(
        name="contrastive_loss_0p1",
        group="contrastive",
        base_config=BASE_CONFIG,
        overrides={},
        hyperparams={"contrastive_loss_weight": 0.1},
        output_dir=tmp_path / "contrastive",
    )
    contrastive_result = RunResult(
        spec=contrastive_spec,
        launch=LaunchResult(returncode=0, runtime=1.2),
        status="success",
        metrics={"summary": {"val_acc": 0.95, "val_total_loss": 0.7}},
        log_dir=None,
    )

    summarise_results([baseline_result, contrastive_result], tmp_path)

    summary_csv = tmp_path / "summary_metrics.csv"
    summary_md = tmp_path / "summary_report.md"
    assert summary_csv.exists()
    assert summary_md.exists()

    df = pd.read_csv(summary_csv)
    delta = df.loc[
        (df["group"] == "contrastive") & (df["metric"] == "val_acc"), "baseline_delta"
    ].squeeze()
    assert delta == pytest.approx(0.05)
