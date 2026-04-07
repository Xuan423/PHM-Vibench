"""Experiment utilities for study-level orchestration and summarization."""

from .tf_multi_proto_ablation_summary import (
    aggregate_study_results,
    aggregate_variant_results,
    collect_study_results,
    collect_variant_results,
    extract_diagnostics_summary,
    render_markdown_summary,
    render_study_markdown_summary,
    write_study_summary_bundle,
    write_summary_bundle,
)
from .tf_multi_proto_study_config import (
    ALLOWED_HPARAM_KEYS,
    SUPPORTED_SYSTEM_PRESETS,
    expand_hparam_grid,
    expand_tasks,
    get_system_family_preset,
    load_ablation_study,
    load_hparam_study,
    load_study_config,
    load_taskset,
    manifest_from_items,
    manifest_from_tasks,
    select_study_items,
)


def run_study(*args, **kwargs):
    from .tf_multi_proto_study_runner import run_study as _run_study

    return _run_study(*args, **kwargs)

__all__ = [
    "ALLOWED_HPARAM_KEYS",
    "SUPPORTED_SYSTEM_PRESETS",
    "aggregate_study_results",
    "aggregate_variant_results",
    "collect_study_results",
    "collect_variant_results",
    "expand_hparam_grid",
    "expand_tasks",
    "extract_diagnostics_summary",
    "get_system_family_preset",
    "load_ablation_study",
    "load_hparam_study",
    "load_study_config",
    "load_taskset",
    "manifest_from_items",
    "manifest_from_tasks",
    "render_markdown_summary",
    "render_study_markdown_summary",
    "run_study",
    "select_study_items",
    "write_study_summary_bundle",
    "write_summary_bundle",
]
