"""Entry point for the two-stage TSPN contrastive pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...configs.config_utils import parse_set_args
from .config import StageConfigComposer
from .stage_runner import StageResult, StageRunner


def pipeline(args) -> Dict[str, Any]:
    overrides = _collect_overrides(args)

    composer = StageConfigComposer(
        base_config_path=args.config_path,
        overrides=overrides,
        stage2_template_path=getattr(args, "stage2_template", None),
    )

    snapshot_root = Path(args.projector_log_dir).expanduser() if getattr(args, "projector_log_dir", None) else None
    runner = StageRunner(args, snapshot_root=snapshot_root)

    mode = getattr(args, "two_stage_mode", "both")
    stage_results: List[StageResult] = []
    stage1_result: Optional[StageResult] = None

    if mode in ("both", "stage1"):
        stage1_config = composer.build_stage1()
        stage1_result = runner.run(stage1_config, stage_name="stage1")
        stage_results.append(stage1_result)

    if mode in ("both", "stage2"):
        stage2_config = composer.build_stage2()
        resume_from = None
        compare_snapshot = None

        if stage1_result:
            resume_from = stage1_result.checkpoint_path
            compare_snapshot = stage1_result.projector_snapshot
        else:
            resume_from = getattr(args, "stage1_checkpoint", None)
            compare_snapshot = None

        if not resume_from:
            raise RuntimeError("Stage2 requested but no Stage1 checkpoint provided. Use --stage1_checkpoint.")

        stage2_result = runner.run(
            stage2_config,
            stage_name="stage2",
            resume_from=resume_from,
            compare_snapshot=compare_snapshot,
        )
        stage_results.append(stage2_result)

    manifest_path = _write_pipeline_manifest(composer.stage_output_root, stage_results)
    return {
        "stages": [result.stage_name for result in stage_results],
        "manifests": [result.manifest_path for result in stage_results],
        "pipeline_manifest": manifest_path,
    }


# ----------------------------------------------------------------------
def _collect_overrides(args) -> Optional[Dict[str, Any]]:
    set_args: List[str] = []
    if getattr(args, "data_dir", None):
        set_args.append(f"data.data_dir={args.data_dir}")
    if getattr(args, "set", None):
        set_args.extend(args.set)
    return parse_set_args(set_args) if set_args else None


def _write_pipeline_manifest(root: Path, results: List[StageResult]) -> str:
    pipeline_manifest = {
        "stages": [result.__dict__ for result in results],
    }
    manifest_path = Path(root) / "two_stage_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(pipeline_manifest, handle, indent=2, ensure_ascii=False)
    return str(manifest_path)
