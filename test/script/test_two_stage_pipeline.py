from pathlib import Path
from types import SimpleNamespace

import pytest

import importlib

two_stage_module = importlib.import_module("src.pipelines.tspn_two_stage.pipeline")
from src.pipelines.tspn_two_stage.config import StageConfigComposer
from src.pipelines.tspn_two_stage.stage_runner import StageResult


DEMO_CONFIG = "configs/demo/X_Single_DG/TSPN_FewShot/tspn_two_stage.yaml"


def _build_args(tmp_path, mode="both", stage1_ckpt=None):
    return SimpleNamespace(
        config_path=DEMO_CONFIG,
        notes="pytest",
        data_dir=None,
        set=[f"environment.output_dir={tmp_path}"],
        pipeline="pipelines.tspn_two_stage.pipeline",
        two_stage_mode=mode,
        stage1_checkpoint=stage1_ckpt,
        stage2_template=None,
        projector_log_dir=None,
    )


def test_pipeline_runs_both_stages_with_checkpoint_chaining(monkeypatch, tmp_path):
    calls = []

    class DummyRunner:
        def __init__(self, args, snapshot_root=None):
            self.snapshot_root = snapshot_root

        def run(self, config, stage_name, resume_from=None, compare_snapshot=None):
            calls.append(
                {
                    "stage": stage_name,
                    "resume": resume_from,
                    "compare": compare_snapshot,
                }
            )
            return StageResult(
                stage_name=stage_name,
                checkpoint_path=f"{stage_name}.ckpt",
                output_dir=str(tmp_path / stage_name),
                metrics={"val_acc": 0.5},
                projector_snapshot=f"{stage_name}_snapshot.json",
                manifest_path=str(tmp_path / f"{stage_name}_manifest.json"),
            )

    monkeypatch.setattr(two_stage_module, "StageRunner", DummyRunner)

    args = _build_args(tmp_path, mode="both")
    result = two_stage_module.pipeline(args)

    assert [call["stage"] for call in calls] == ["stage1", "stage2"]
    assert calls[1]["resume"] == "stage1.ckpt"
    assert calls[1]["compare"] == "stage1_snapshot.json"
    assert result["stages"] == ["stage1", "stage2"]
    assert Path(result["pipeline_manifest"]).exists()


def test_pipeline_requires_explicit_stage1_checkpoint(monkeypatch, tmp_path):
    calls = []

    class DummyRunner:
        def __init__(self, *_, **__):
            pass

        def run(self, *_, **__):
            calls.append("stage_invoked")
            return StageResult(
                stage_name="stage2",
                checkpoint_path="stage2.ckpt",
                output_dir=str(tmp_path / "stage2"),
                metrics={},
                projector_snapshot=None,
                manifest_path=str(tmp_path / "stage2_manifest.json"),
            )

    monkeypatch.setattr(two_stage_module, "StageRunner", DummyRunner)

    args = _build_args(tmp_path, mode="stage2", stage1_ckpt=None)
    with pytest.raises(RuntimeError):
        two_stage_module.pipeline(args)
    assert calls == []


def test_stage_config_composer_sets_stage_modes(tmp_path):
    overrides = {"environment.output_dir": str(tmp_path / "base_output")}
    composer = StageConfigComposer(
        base_config_path=DEMO_CONFIG,
        overrides=overrides,
        stage2_template_path="configs/pipelines/tspn_two_stage_stage2.yaml",
    )

    stage1 = composer.build_stage1()
    assert stage1.task.stage_mode == "stage1"
    assert stage1.task.few_shot.enabled is False
    assert stage1.task.contrastive.loss_weight == 0.0

    stage2 = composer.build_stage2()
    assert stage2.task.stage_mode == "stage2"
    assert stage2.task.few_shot.enabled is True


def test_stage2_rejects_explicit_episode_counts(tmp_path):
    config_path = tmp_path / "tspn_stage2_bad.yaml"
    config_path.write_text(
        """
environment:
  project: demo
data:
  data_dir: "."
  metadata_file: "meta.xlsx"
model:
  name: demo
  type: demo
task:
  name: contrastive_classification
  type: DG
  few_shot:
    enabled: true
    systems_per_episode: 1
pipeline:
  module: pipelines.tspn_two_stage.pipeline
  stage2_overrides:
    task:
      few_shot:
        enabled: true
        episodes_per_epoch: 2
"""
    )
    composer = StageConfigComposer(
        base_config_path=str(config_path),
    )
    with pytest.raises(ValueError):
        composer.build_stage2()


def test_stage2_rejects_warmup_flat_epochs(tmp_path):
    config_path = tmp_path / "tspn_stage2_warmup.yaml"
    config_path.write_text(
        """
environment:
  project: demo
data:
  data_dir: "."
  metadata_file: "meta.xlsx"
model:
  name: demo
  type: demo
task:
  name: contrastive_classification
  type: DG
  few_shot:
    enabled: true
pipeline:
  module: pipelines.tspn_two_stage.pipeline
  stage2_overrides:
    task:
      few_shot:
        enabled: true
        warmup_flat_epochs: 2
"""
    )
    composer = StageConfigComposer(
        base_config_path=str(config_path),
    )
    with pytest.raises(ValueError):
        composer.build_stage2()
