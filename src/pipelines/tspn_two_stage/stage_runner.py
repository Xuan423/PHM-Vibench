"""Stage execution utilities for the two-stage pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from ...configs.config_utils import ConfigWrapper, save_config, transfer_namespace
from ...data_factory import build_data
from ...model_factory import build_model
from ...task_factory import build_task
from ...trainer_factory import build_trainer
from ...trainer_factory.callbacks.projector_trajectory import ProjectorTrajectoryLogger
from ...utils.utils import close_lab, init_lab, load_best_model_checkpoint


@dataclass
class StageResult:
    stage_name: str
    checkpoint_path: Optional[str]
    output_dir: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    projector_snapshot: Optional[str] = None
    manifest_path: Optional[str] = None


class StageRunner:
    """Runs a single stage using the standard factory stack."""

    def __init__(
        self,
        cli_args: Any,
        *,
        snapshot_root: Optional[Path] = None,
    ) -> None:
        self.args = cli_args
        self.snapshot_root = snapshot_root

    # ------------------------------------------------------------------
    def run(
        self,
        configs: ConfigWrapper,
        *,
        stage_name: str,
        resume_from: Optional[str] = None,
        compare_snapshot: Optional[str] = None,
    ) -> StageResult:
        args_environment = transfer_namespace(getattr(configs, "environment", ConfigWrapper()))
        args_data = transfer_namespace(getattr(configs, "data", ConfigWrapper()))
        args_model = transfer_namespace(getattr(configs, "model", ConfigWrapper()))
        args_task = transfer_namespace(getattr(configs, "task", ConfigWrapper()))
        args_trainer = transfer_namespace(getattr(configs, "trainer", ConfigWrapper()))

        if resume_from:
            args_model.weights_path = resume_from

        iterations = max(1, int(getattr(args_environment, "iterations", 1)))
        aggregate_metrics: List[Dict[str, Any]] = []
        latest_checkpoint: Optional[str] = None
        latest_snapshot: Optional[str] = None
        manifest_entries: List[Dict[str, Any]] = []

        for iteration in range(iterations):
            run_dir = self._resolve_run_dir(args_environment, stage_name, iteration)
            config_dump = Path(run_dir) / "stage_config.yaml"
            save_config(configs, config_dump)

            experiment_name = f"{stage_name}_iter{iteration}"
            seed_everything(getattr(args_environment, "seed", 0) + iteration)
            init_lab(args_environment, self.args, experiment_name)

            data_factory = build_data(args_data, args_task)
            model = build_model(args_model, metadata=data_factory.get_metadata())
            task = build_task(
                args_task=args_task,
                network=model,
                args_data=args_data,
                args_model=args_model,
                args_trainer=args_trainer,
                args_environment=args_environment,
                metadata=data_factory.get_metadata(),
            )
            args_trainer.model = args_model
            trainer = build_trainer(args_environment, args_trainer, args_data, str(run_dir))

            projector_dir = self._resolve_snapshot_dir(stage_name, iteration, run_dir)
            projector_logger = ProjectorTrajectoryLogger(
                snapshot_dir=projector_dir,
                stage_name=stage_name,
                compare_snapshot=compare_snapshot if iteration == 0 else None,
            )
            trainer.callbacks.append(projector_logger)

            trainer.fit(
                task,
                data_factory.get_dataloader("train"),
                data_factory.get_dataloader("val"),
            )
            train_iterations = int(getattr(trainer, "num_training_batches", 0) or 0)

            task = load_best_model_checkpoint(task, trainer)
            test_results = trainer.test(task, data_factory.get_dataloader("test"))
            raw_metrics = test_results[0] if test_results else {}
            stage_metrics = self._to_serializable_dict(raw_metrics)
            ce_controller = getattr(task, "ce_weight_controller", None)
            ce_mode = getattr(ce_controller, "mode", None) if ce_controller else None
            if ce_mode:
                stage_metrics.setdefault("ce_weight_mode", ce_mode)
            if train_iterations:
                stage_metrics.setdefault("train_total_iterations", train_iterations)
                if stage_name == "stage2":
                    stage_metrics.setdefault("episode_iterations", train_iterations)
            aggregate_metrics.append(stage_metrics)

            result_df = pd.DataFrame([stage_metrics])
            result_df.to_csv(Path(run_dir) / f"test_result_{iteration}.csv", index=False)

            latest_checkpoint = self._extract_best_checkpoint(trainer)
            latest_snapshot = projector_logger.latest_snapshot

            entry = {
                "iteration": iteration,
                "checkpoint_path": latest_checkpoint,
                "metrics": stage_metrics,
                "snapshot_manifest": latest_snapshot,
            }
            if train_iterations and stage_name == "stage2":
                entry["episode_iterations"] = train_iterations
            manifest_entries.append(entry)

            data_ref = getattr(data_factory, "data", None)
            if data_ref is not None and hasattr(data_ref, "close"):
                data_ref.close()

            close_lab()

        stage_manifest_path = self._write_stage_manifest(
            args_environment,
            stage_name,
            manifest_entries,
        )

        averaged_metrics = self._average_metrics(aggregate_metrics)
        return StageResult(
            stage_name=stage_name,
            checkpoint_path=latest_checkpoint,
            output_dir=str(Path(getattr(args_environment, "output_dir", "."))),
            metrics=averaged_metrics,
            projector_snapshot=latest_snapshot,
            manifest_path=stage_manifest_path,
        )

    # ------------------------------------------------------------------
    def _resolve_run_dir(self, args_environment: ConfigWrapper, stage_name: str, iteration: int) -> Path:
        base = Path(getattr(args_environment, "output_dir", Path("save") / stage_name))
        run_dir = base / f"iter_{iteration}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _resolve_snapshot_dir(self, stage_name: str, iteration: int, run_dir: Path) -> Path:
        if self.snapshot_root:
            target = self.snapshot_root / stage_name / f"iter_{iteration}"
        else:
            target = Path(run_dir) / "projector"
        target.mkdir(parents=True, exist_ok=True)
        return target

    @staticmethod
    def _extract_best_checkpoint(trainer) -> Optional[str]:
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                return callback.best_model_path or None
        return None

    def _write_stage_manifest(
        self,
        args_environment: ConfigWrapper,
        stage_name: str,
        entries: Sequence[Dict[str, Any]],
    ) -> str:
        manifest = {
            "stage": stage_name,
            "project": getattr(args_environment, "project", ""),
            "output_dir": getattr(args_environment, "output_dir", ""),
            "entries": list(entries),
        }
        manifest_path = Path(getattr(args_environment, "output_dir", ".")) / "stage_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        return str(manifest_path)

    @staticmethod
    def _average_metrics(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {}
        keys = results[0].keys()
        averaged: Dict[str, Any] = {}
        for key in keys:
            values = [res[key] for res in results if key in res]
            if not values:
                continue
            try:
                averaged[key] = sum(float(v) for v in values) / len(values)
            except Exception:
                averaged[key] = values[-1]
        return averaged

    @staticmethod
    def _to_serializable_dict(metrics: Dict[str, Any]) -> Dict[str, Any]:
        clean: Dict[str, Any] = {}
        for key, value in metrics.items():
            if hasattr(value, "item"):
                clean[key] = float(value.item())
            elif isinstance(value, (float, int)):
                clean[key] = value
            else:
                clean[key] = value
        return clean
